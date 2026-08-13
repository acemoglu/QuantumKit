import Foundation
import Metal

public enum DensityMatrixEngineError: Error {
    case deviceNotFound
    case commandQueueCreationFailed
    case commandBufferCreationFailed
    case commandBufferExecutionFailed(underlying: Error?)
    case functionNotFound(String)
    case libraryNotFound
    case qubitCountMismatch(circuit: Int, matrix: Int)
    case unsupportedGate(Gate)
    case nonUnitaryGateUnsupported(Gate)
    case zeroProbabilityMeasurement(qubit: Int)
    case invalidTraceForRenormalization(trace: Double)
}

public final class DensityMatrixEngine: @unchecked Sendable {
    struct Pipelines: @unchecked Sendable {
        let leftMultiplySingleQubit: MTLComputePipelineState
        let rightMultiplySingleQubitDagger: MTLComputePipelineState
        let applyKrausSingleQubit: MTLComputePipelineState
        let applyKrausTwoQubit: MTLComputePipelineState

        init(device: MTLDevice, library: MTLLibrary) throws {
            guard let leftFunc = library.makeFunction(name: "dm_left_multiply_single_qubit") else {
                throw DensityMatrixEngineError.functionNotFound("dm_left_multiply_single_qubit")
            }
            self.leftMultiplySingleQubit = try device.makeComputePipelineState(function: leftFunc)

            guard let rightFunc = library.makeFunction(name: "dm_right_multiply_single_qubit_dagger") else {
                throw DensityMatrixEngineError.functionNotFound("dm_right_multiply_single_qubit_dagger")
            }
            self.rightMultiplySingleQubitDagger = try device.makeComputePipelineState(function: rightFunc)

            guard let krausFunc = library.makeFunction(name: "dm_apply_kraus_single_qubit") else {
                throw DensityMatrixEngineError.functionNotFound("dm_apply_kraus_single_qubit")
            }
            self.applyKrausSingleQubit = try device.makeComputePipelineState(function: krausFunc)

            guard let kraus2Func = library.makeFunction(name: "dm_apply_two_qubit_kraus") else {
                throw DensityMatrixEngineError.functionNotFound("dm_apply_two_qubit_kraus")
            }
            self.applyKrausTwoQubit = try device.makeComputePipelineState(function: kraus2Func)
        }
    }

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let bufferPool: BufferPool
    let pipelines: Pipelines
    public let renormalizationInterval: Int

    public init(renormalizationInterval: Int = 50) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw DensityMatrixEngineError.deviceNotFound
        }
        self.device = device
        self.renormalizationInterval = max(0, renormalizationInterval)
        guard let commandQueue = device.makeCommandQueue() else {
            throw DensityMatrixEngineError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        self.bufferPool = BufferPool(device: device)
        self.pipelines = try Pipelines(device: device, library: try Self.loadLibrary(device: device))
    }

    public func execute(_ circuit: QuantumCircuit, on density: DensityMatrix, noise: NoiseModel? = nil) throws {
        var rng: QuantumRNG = .hardware
        _ = try executeRNG(circuit, on: density, rng: &rng, noise: noise)
    }

    /// RNG-injectable open-system execution supporting `swap`, mid-circuit `measure`, and `reset`
    /// in addition to unitary evolution and noise channels. Returns the classical outcomes of each
    /// `measure` gate in circuit order, mirroring ``QuantumEngine/executeRNG(_:on:rng:noise:)``.
    @discardableResult
    public func executeRNG(
        _ circuit: QuantumCircuit,
        on density: DensityMatrix,
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil
    ) throws -> CircuitExecutionResult {
        guard circuit.qubitCount == density.qubitCount else {
            throw DensityMatrixEngineError.qubitCountMismatch(circuit: circuit.qubitCount, matrix: density.qubitCount)
        }
        try circuit.requireFullyBound()

        let scratchBytes = density.elementCount * MemoryLayout<QFloat>.stride
        let scratchReal = try bufferPool.acquire(length: scratchBytes)
        let scratchImag = try bufferPool.acquire(length: scratchBytes)
        // The scratch buffers are shared by every gate/channel command buffer dispatched below. With
        // the per-gate `waitUntilCompleted` calls removed, returning them synchronously here could
        // hand a buffer still referenced by an in-flight command buffer back to the pool. Reclaim
        // them through a trailing command buffer on the serial, in-order queue: its completion
        // handler returns them to the pool only once every earlier dispatch — and therefore every use
        // of scratch — has finished. This is correct on both the normal and the early-`throw` exits.
        defer {
            if let reclaimBuffer = commandQueue.makeCommandBuffer() {
                bufferPool.release(scratchReal, after: reclaimBuffer)
                bufferPool.release(scratchImag, after: reclaimBuffer)
                reclaimBuffer.commit()
            } else {
                bufferPool.release(scratchReal)
                bufferPool.release(scratchImag)
            }
        }

        var measurementOutcomes: [[Int]] = []
        var appliedGateCount = 0
        var classicalMemory = ClassicalMemory(
            registerWidths: circuit.classicalRegisters.map(\.bitCount)
        )

        func executeRuntimeGate(_ gate: Gate, at gateIndex: Int) throws {
            switch gate {
            case .measure(let spec):
                let bits = try applyMeasurement(
                    qubits: spec.qubits,
                    on: density,
                    rng: &rng,
                    noise: noise,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
                measurementOutcomes.append(bits)
                let outcome = bits.enumerated().reduce(0) { partial, entry in
                    partial | (entry.element << entry.offset)
                }
                try classicalMemory.writeOutcome(
                    outcome,
                    measuredQubitCount: spec.qubits.count,
                    register: spec.classicalRegister,
                    bitOffset: spec.classicalBitOffset
                )
                if let noise, noise.hasLocalizedGateNoise {
                    try applyPointNoiseChannels(
                        after: gate,
                        at: gateIndex,
                        on: density,
                        noise: noise,
                        scratchReal: scratchReal,
                        scratchImag: scratchImag
                    )
                }

            case .reset(let qubit):
                try applyReset(
                    qubit: qubit,
                    on: density,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
                if let noise {
                    try applyPointNoiseChannels(
                        after: gate,
                        at: gateIndex,
                        on: density,
                        noise: noise,
                        scratchReal: scratchReal,
                        scratchImag: scratchImag
                    )
                }

            case .barrier, .delay:
                if let noise {
                    try applyPointNoiseChannels(
                        after: gate,
                        at: gateIndex,
                        on: density,
                        noise: noise,
                        scratchReal: scratchReal,
                        scratchImag: scratchImag
                    )
                }

            case .id:
                break

            case .swap(let q1, let q2):
                try applySwap(
                    q1: q1,
                    q2: q2,
                    on: density,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
                if let noise {
                    try applyNoiseChannels(
                        after: gate,
                        at: gateIndex,
                        on: density,
                        noise: noise,
                        scratchReal: scratchReal,
                        scratchImag: scratchImag
                    )
                }

            case .c_if(let classicalRegister, let expectedValue, let conditionedGate):
                if classicalMemory.value(ofRegister: classicalRegister) == expectedValue {
                    // Conditioned body reuses the enclosing instruction index for C7 targeting.
                    try executeRuntimeGate(conditionedGate, at: gateIndex)
                }

            case .initialize(let qubits, let amplitudes):
                try density.initialize(qubits: qubits, amplitudes: amplitudes)
                if let noise {
                    try applyPointNoiseChannels(
                        after: gate,
                        at: gateIndex,
                        on: density,
                        noise: noise,
                        scratchReal: scratchReal,
                        scratchImag: scratchImag
                    )
                }

            case .customUnitary(let matrix, let qubits) where qubits.count > 1:
                try applyHostCustomUnitary(
                    matrix: matrix,
                    qubits: qubits,
                    on: density
                )
                if let noise {
                    try applyNoiseChannels(
                        after: gate,
                        at: gateIndex,
                        on: density,
                        noise: noise,
                        scratchReal: scratchReal,
                        scratchImag: scratchImag
                    )
                }

            default:
                try applyUnitaryGate(gate, on: density, scratchReal: scratchReal, scratchImag: scratchImag)
                if let noise {
                    try applyNoiseChannels(
                        after: gate,
                        at: gateIndex,
                        on: density,
                        noise: noise,
                        scratchReal: scratchReal,
                        scratchImag: scratchImag
                    )
                }
            }

            appliedGateCount += 1
            if shouldRenormalize(afterAppliedGateCount: appliedGateCount) {
                try renormalizeTrace(of: density)
            }
        }

        for (gateIndex, gate) in circuit.gates.enumerated() {
            try executeRuntimeGate(gate, at: gateIndex)
        }

        // Every gate/channel above is committed without blocking the CPU. Drain the serial queue
        // once here so all GPU writes to the density buffers are complete and visible to the host
        // before the caller inspects them (e.g. via `probabilities`) after `executeRNG` returns.
        try drainPipeline()
        return CircuitExecutionResult(
            measurementOutcomes: measurementOutcomes,
            classicalMemory: classicalMemory
        )
    }

    func shouldRenormalize(afterAppliedGateCount gateCount: Int) -> Bool {
        renormalizationInterval > 0 && gateCount % renormalizationInterval == 0
    }

    /// Commits a single empty command buffer on the serial, in-order ``commandQueue`` and blocks
    /// until it — and therefore every gate/channel buffer committed before it — has completed.
    ///
    /// Because the queue is serial and executes in commit order, awaiting one trailing buffer
    /// guarantees all earlier work has finished. This is the single synchronization point the host
    /// needs before reading the shared density buffers, now that individual gate/channel dispatches
    /// no longer call `waitUntilCompleted`.
    func drainPipeline() throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw DensityMatrixEngineError.commandBufferCreationFailed
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DensityMatrixEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    /// Periodically restores unit trace by scaling ρ <- ρ / Tr(ρ). This is a host-side correction
    /// against accumulated Float32 rounding drift across deep Kraus/noise stacks.
    func renormalizeTrace(of density: DensityMatrix) throws {
        try drainPipeline()

        let real = density.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imag = density.imagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let stateCount = density.stateCount

        var trace = 0.0
        for i in 0..<stateCount {
            trace += Double(real[i * stateCount + i])
        }

        guard trace.isFinite, trace > 0 else {
            throw DensityMatrixEngineError.invalidTraceForRenormalization(trace: trace)
        }

        let invTrace = QFloat(1.0 / trace)
        for i in 0..<density.elementCount {
            real[i] *= invTrace
            imag[i] *= invTrace
        }
    }

    public func probabilities(of density: DensityMatrix) -> [QFloat] {
        // Synchronize before this host read. The drain only fences the queue (it encodes no kernels
        // of its own), so its only failure mode is command-buffer creation under extreme memory
        // pressure; there is no meaningful error to surface from this non-throwing accessor.
        try? drainPipeline()
        let real = density.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        return (0..<density.stateCount).map { basis in
            real[basis * density.stateCount + basis]
        }
    }

    private func applyUnitaryGate(
        _ gate: Gate,
        on density: DensityMatrix,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        if GateDecomposition.needsExecutionExpansion(gate) {
            for piece in try QuantumEngine.expandForExecution(gate) {
                try applyUnitaryGate(piece, on: density, scratchReal: scratchReal, scratchImag: scratchImag)
            }
            return
        }

        let encoded = try encodeSingleQubitUnitary(gate)
        let matrixBuffer = try acquirePooledMatrixBuffer(values: encoded.matrix)
        // Until the buffer is handed to a committed command buffer (via `release(_:after:)` below),
        // any early `throw` must still return it to the pool. A synchronous release is safe on that
        // path only because no command buffer has been committed yet, so the buffer is still idle.
        var unsubmittedMatrixBuffer: MTLBuffer? = matrixBuffer
        defer { if let buffer = unsubmittedMatrixBuffer { bufferPool.release(buffer) } }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DensityMatrixEngineError.commandBufferCreationFailed
        }

        var stateCount = UInt32(density.stateCount)
        var target = UInt32(encoded.target)
        var controlMask = encoded.controlMask

        // Pass A: temp = U * rho
        encoder.setComputePipelineState(pipelines.leftMultiplySingleQubit)
        encoder.setBuffer(density.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(density.imagBuffer, offset: 0, index: 1)
        encoder.setBuffer(scratchReal, offset: 0, index: 2)
        encoder.setBuffer(scratchImag, offset: 0, index: 3)
        encoder.setBytes(&stateCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&target, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&controlMask, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 7)
        dispatchMatrixElements(
            encoder: encoder,
            elementCount: density.elementCount,
            pipeline: pipelines.leftMultiplySingleQubit
        )

        // Pass B reads the scratch buffers that Pass A just wrote. Metal does not guarantee
        // ordering between dispatches in the same compute pass, so a buffer-scoped barrier is
        // required to ensure Pass B observes Pass A's writes rather than stale contents.
        if #available(macOS 10.14, iOS 12.0, tvOS 12.0, *) {
            encoder.memoryBarrier(scope: .buffers)
        }

        // Pass B: rho = temp * U†
        encoder.setComputePipelineState(pipelines.rightMultiplySingleQubitDagger)
        encoder.setBuffer(scratchReal, offset: 0, index: 0)
        encoder.setBuffer(scratchImag, offset: 0, index: 1)
        encoder.setBuffer(density.realBuffer, offset: 0, index: 2)
        encoder.setBuffer(density.imagBuffer, offset: 0, index: 3)
        encoder.setBytes(&stateCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&target, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&controlMask, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 7)
        dispatchMatrixElements(
            encoder: encoder,
            elementCount: density.elementCount,
            pipeline: pipelines.rightMultiplySingleQubitDagger
        )

        encoder.endEncoding()
        // Hand the gate matrix back to the pool only once this command buffer completes, then commit
        // without blocking. The serial queue keeps this gate ordered after all earlier gate work.
        bufferPool.release(matrixBuffer, after: commandBuffer)
        unsubmittedMatrixBuffer = nil
        commandBuffer.commit()
    }

    private func applyNoiseChannels(
        after gate: Gate,
        at gateIndex: Int,
        on density: DensityMatrix,
        noise: NoiseModel,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        // Distinct affected qubits, order preserved — mirrors the state-vector engine so both engines
        // see the same qubit list when deciding the depolarizing channel structure.
        var seen = Set<Int>()
        let affected = gate.affectedQubits.filter { seen.insert($0).inserted }

        // Depolarizing is keyed off the *number* of qubits the gate touches, exactly like the
        // state-vector engine: 1-qubit → single-qubit channel, 2-qubit → correlated 15-Pauli
        // channel, ≥3-qubit → independent single-qubit channels per qubit.
        if noise.appliesDepolarizing {
            try applyDepolarizingNoise(
                on: density,
                qubits: affected,
                probability: noise.depolarizingProbability,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        }

        for qubit in affected {
            if noise.appliesAmplitudeDamping {
                let gamma = noise.effectiveAmplitudeDampingProbability
                let keep = sqrt(max(0, 1 - gamma))
                let relax = sqrt(max(0, gamma))
                try applyKrausChannel(
                    on: density,
                    targetQubit: qubit,
                    kraus: [
                        [complex(1, 0), complex(0, 0), complex(0, 0), complex(keep, 0)],
                        [complex(0, 0), complex(relax, 0), complex(0, 0), complex(0, 0)],
                    ],
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

            if noise.appliesPhaseDamping {
                let lambda = noise.effectivePhaseDampingProbability
                let keep = sqrt(max(0, 1 - lambda))
                let dephase = sqrt(max(0, lambda))
                try applyKrausChannel(
                    on: density,
                    targetQubit: qubit,
                    kraus: [
                        [complex(1, 0), complex(0, 0), complex(0, 0), complex(keep, 0)],
                        [complex(0, 0), complex(0, 0), complex(0, 0), complex(dephase, 0)],
                    ],
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }
        }

        if noise.hasLocalizedGateNoise {
            try applyLocalizedNoiseChannels(
                after: gate,
                at: gateIndex,
                on: density,
                noise: noise,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        }
    }

    /// Localized (+ global reset/prep / idle) noise at non-unitary circuit points (C7 / C8 / C9).
    /// Skips global post-unitary dep/AD/PD — those remain unitary-gate-only.
    private func applyPointNoiseChannels(
        after gate: Gate,
        at gateIndex: Int,
        on density: DensityMatrix,
        noise: NoiseModel,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        if case .reset(let qubit) = gate, noise.resetErrorProbability > 0 {
            try applyPauliFlipChannel(
                on: density,
                qubit: qubit,
                axis: .x,
                probability: noise.resetErrorProbability,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        }

        if case .initialize(let qubits, _) = gate, noise.preparationErrorProbability > 0 {
            for qubit in qubits {
                try applyPauliFlipChannel(
                    on: density,
                    qubit: qubit,
                    axis: .x,
                    probability: noise.preparationErrorProbability,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }
        }

        if case .delay(let duration, let qubit) = gate, noise.thermalRelaxationOnDelay {
            try applyThermalRelaxation(
                on: density,
                qubit: qubit,
                duration: duration,
                t1: noise.t1,
                t2: noise.t2,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        }

        if noise.hasLocalizedGateNoise {
            try applyLocalizedNoiseChannels(
                after: gate,
                at: gateIndex,
                on: density,
                noise: noise,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        }
    }

    private func applyKrausChannel(
        on density: DensityMatrix,
        targetQubit: Int,
        kraus: [[SIMD2<QFloat>]],
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        let flat = kraus.flatMap { $0 }
        let krausBuffer = try acquirePooledMatrixBuffer(values: flat)
        // Until the buffer is handed to a committed command buffer (via `release(_:after:)` below),
        // an early `throw` must still return it to the pool. The synchronous release on that path is
        // safe only because no command buffer has been committed yet, so the buffer is still idle.
        var unsubmittedKrausBuffer: MTLBuffer? = krausBuffer
        defer { if let buffer = unsubmittedKrausBuffer { bufferPool.release(buffer) } }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DensityMatrixEngineError.commandBufferCreationFailed
        }

        var stateCount = UInt32(density.stateCount)
        var target = UInt32(targetQubit)
        var krausCount = UInt32(kraus.count)

        encoder.setComputePipelineState(pipelines.applyKrausSingleQubit)
        encoder.setBuffer(density.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(density.imagBuffer, offset: 0, index: 1)
        encoder.setBuffer(scratchReal, offset: 0, index: 2)
        encoder.setBuffer(scratchImag, offset: 0, index: 3)
        encoder.setBytes(&stateCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&target, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&krausCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(krausBuffer, offset: 0, index: 7)
        dispatchMatrixElements(
            encoder: encoder,
            elementCount: density.elementCount,
            pipeline: pipelines.applyKrausSingleQubit
        )
        encoder.endEncoding()

        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw DensityMatrixEngineError.commandBufferCreationFailed
        }
        blit.copy(from: scratchReal, sourceOffset: 0, to: density.realBuffer, destinationOffset: 0, size: density.realBuffer.length)
        blit.copy(from: scratchImag, sourceOffset: 0, to: density.imagBuffer, destinationOffset: 0, size: density.imagBuffer.length)
        blit.endEncoding()

        // Reclaim the Kraus matrix only once this command buffer completes, then commit without
        // blocking; the serial queue keeps the channel ordered after all earlier work.
        bufferPool.release(krausBuffer, after: commandBuffer)
        unsubmittedKrausBuffer = nil
        commandBuffer.commit()
    }

    /// Depolarizing noise, dispatched by the number of qubits the originating gate touches so that
    /// the density-matrix engine and the state-vector engine realize the *same* physical channel:
    /// - 1-qubit gates: the single-qubit depolarizing channel.
    /// - 2-qubit gates (`cx`, `cz`, `swap`): the *correlated* two-qubit depolarizing channel
    ///   (one identity branch plus the 15 non-identity two-qubit Paulis), not two independent
    ///   single-qubit channels.
    /// - ≥3-qubit gates: independent single-qubit depolarizing on each affected qubit.
    private func applyDepolarizingNoise(
        on density: DensityMatrix,
        qubits: [Int],
        probability p: QFloat,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        guard p > 0 else { return }

        switch qubits.count {
        case 0:
            return
        case 1:
            try applySingleQubitDepolarizing(
                on: density,
                qubit: qubits[0],
                probability: p,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        case 2:
            try applyTwoQubitDepolarizing(
                on: density,
                qubitA: qubits[0],
                qubitB: qubits[1],
                probability: p,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        default:
            for qubit in qubits {
                try applySingleQubitDepolarizing(
                    on: density,
                    qubit: qubit,
                    probability: p,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }
        }
    }

    /// Single-qubit depolarizing channel with Kraus operators
    /// `K0 = √(1-p)·I`, `K1 = √(p/3)·X`, `K2 = √(p/3)·Y`, `K3 = √(p/3)·Z` (∑ Kᵢ†Kᵢ = I).
    private func applySingleQubitDepolarizing(
        on density: DensityMatrix,
        qubit: Int,
        probability p: QFloat,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        let k0 = sqrt(max(0, 1 - p))
        let k1 = sqrt(p / 3)
        try applyKrausChannel(
            on: density,
            targetQubit: qubit,
            kraus: [
                [complex(k0, 0), complex(0, 0), complex(0, 0), complex(k0, 0)],
                [complex(0, 0), complex(k1, 0), complex(k1, 0), complex(0, 0)],
                [complex(0, 0), complex(0, -k1), complex(0, k1), complex(0, 0)],
                [complex(k1, 0), complex(0, 0), complex(0, 0), complex(-k1, 0)],
            ],
            scratchReal: scratchReal,
            scratchImag: scratchImag
        )
    }

    /// Correlated two-qubit depolarizing channel:
    ///
    ///   ρ → (1 − p)·ρ + (p/15)·Σ_{P ≠ I⊗I} (Pₐ ⊗ P_b) ρ (Pₐ ⊗ P_b)†
    ///
    /// i.e. the 16 Kraus operators are `√(1-p)·(I⊗I)` and `√(p/15)·(Pₐ ⊗ P_b)` over the 15
    /// non-identity two-qubit Paulis. This is the exact mixed-state analogue of the state-vector
    /// engine's two-qubit Pauli-jump unraveling, so both engines model the identical channel for a
    /// 2-qubit gate (rather than the DM engine applying two independent single-qubit channels).
    ///
    /// The whole 16-operator mixture is realized in a **single** `dm_apply_two_qubit_kraus` GPU
    /// dispatch: each output element depends only on the 4×4 subspace block of ρ, so there is no
    /// host-side accumulation over the full 4ⁿ matrix and no per-Pauli command-buffer round trip.
    private func applyTwoQubitDepolarizing(
        on density: DensityMatrix,
        qubitA: Int,
        qubitB: Int,
        probability p: QFloat,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        // Kraus weights: K0 = √(1-p)·(I⊗I), K_{1..15} = √(p/15)·(Pₐ ⊗ P_b).
        let identityWeight = max(0, 1 - p).squareRoot()
        let pauliWeight = max(0, p / 15).squareRoot()

        var kraus: [SIMD2<QFloat>] = []
        kraus.reserveCapacity(16 * 16)
        kraus.append(contentsOf: scaledMatrix(kron(singleQubitPauli(axis: 0), singleQubitPauli(axis: 0)), by: identityWeight))

        // 15 non-identity two-qubit Paulis: combined codes 1...15 with 2 bits per qubit
        // (0=I, 1=X, 2=Y, 3=Z), matching the state-vector engine's enumeration exactly.
        for combined in 1...15 {
            let axisA = combined / 4
            let axisB = combined % 4
            kraus.append(contentsOf: scaledMatrix(kron(singleQubitPauli(axis: axisA), singleQubitPauli(axis: axisB)), by: pauliWeight))
        }

        try applyTwoQubitKrausChannel(
            on: density,
            qubitA: qubitA,
            qubitB: qubitB,
            krausCount: 16,
            krausFlat: kraus,
            scratchReal: scratchReal,
            scratchImag: scratchImag
        )
    }

    /// Applies a two-qubit Kraus channel `ρ → Σ_k E_k ρ E_k†` in one GPU pass via
    /// `dm_apply_two_qubit_kraus`. `krausFlat` is the concatenation of `krausCount` operators, each a
    /// 4×4 row-major complex matrix (16 `SIMD2<QFloat>`, `qubitA` = high bit of the subspace code).
    /// The kernel writes the result into scratch, which is then blitted back over ρ.
    private func applyTwoQubitKrausChannel(
        on density: DensityMatrix,
        qubitA: Int,
        qubitB: Int,
        krausCount: Int,
        krausFlat: [SIMD2<QFloat>],
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        let krausBuffer = try acquirePooledMatrixBuffer(values: krausFlat)
        // Until the buffer is handed to a committed command buffer (via `release(_:after:)` below),
        // an early `throw` must still return it to the pool. The synchronous release on that path is
        // safe only because no command buffer has been committed yet, so the buffer is still idle.
        var unsubmittedKrausBuffer: MTLBuffer? = krausBuffer
        defer { if let buffer = unsubmittedKrausBuffer { bufferPool.release(buffer) } }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DensityMatrixEngineError.commandBufferCreationFailed
        }

        var stateCount = UInt32(density.stateCount)
        var qa = UInt32(qubitA)
        var qb = UInt32(qubitB)
        var count = UInt32(krausCount)

        encoder.setComputePipelineState(pipelines.applyKrausTwoQubit)
        encoder.setBuffer(density.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(density.imagBuffer, offset: 0, index: 1)
        encoder.setBuffer(scratchReal, offset: 0, index: 2)
        encoder.setBuffer(scratchImag, offset: 0, index: 3)
        encoder.setBytes(&stateCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&qa, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&qb, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBuffer(krausBuffer, offset: 0, index: 8)
        dispatchMatrixElements(
            encoder: encoder,
            elementCount: density.elementCount,
            pipeline: pipelines.applyKrausTwoQubit
        )
        encoder.endEncoding()

        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw DensityMatrixEngineError.commandBufferCreationFailed
        }
        blit.copy(from: scratchReal, sourceOffset: 0, to: density.realBuffer, destinationOffset: 0, size: density.realBuffer.length)
        blit.copy(from: scratchImag, sourceOffset: 0, to: density.imagBuffer, destinationOffset: 0, size: density.imagBuffer.length)
        blit.endEncoding()

        // Reclaim the Kraus matrix only once this command buffer completes, then commit without
        // blocking; the serial queue keeps the channel ordered after all earlier work.
        bufferPool.release(krausBuffer, after: commandBuffer)
        unsubmittedKrausBuffer = nil
        commandBuffer.commit()
    }

    /// Single-qubit Pauli as a 2×2 row-major matrix (axis 0=I, 1=X, 2=Y, 3=Z).
    private func singleQubitPauli(axis: Int) -> [SIMD2<QFloat>] {
        switch axis {
        case 1: return [complex(0, 0), complex(1, 0), complex(1, 0), complex(0, 0)]    // X
        case 2: return [complex(0, 0), complex(0, -1), complex(0, 1), complex(0, 0)]   // Y
        case 3: return [complex(1, 0), complex(0, 0), complex(0, 0), complex(-1, 0)]   // Z
        default: return [complex(1, 0), complex(0, 0), complex(0, 0), complex(1, 0)]   // I
        }
    }

    /// Kronecker product of two 2×2 row-major matrices into a 4×4 row-major matrix, with `a` as the
    /// high-order factor: `out[(iₐ·2+i_b)·4 + (jₐ·2+j_b)] = a[iₐ·2+jₐ] · b[i_b·2+j_b]`.
    private func kron(_ a: [SIMD2<QFloat>], _ b: [SIMD2<QFloat>]) -> [SIMD2<QFloat>] {
        let enforcesHighOrderBitA = true
        precondition(
            enforcesHighOrderBitA,
            "CRITICAL ARCHITECTURAL CONTRACT: The Metal kernel dm2_pair_index strictly assumes qubitA is the high-order bit. Do NOT modify this Kronecker product ordering without symmetrically updating the Metal shader bit shifts."
        )

        var out = [SIMD2<QFloat>](repeating: complex(0, 0), count: 16)
        for ia in 0..<2 {
            for ja in 0..<2 {
                let av = a[ia * 2 + ja]
                for ib in 0..<2 {
                    for jb in 0..<2 {
                        out[(ia * 2 + ib) * 4 + (ja * 2 + jb)] = complexMul(av, b[ib * 2 + jb])
                    }
                }
            }
        }
        return out
    }

    /// Scales every entry of a complex matrix by the real scalar `s`.
    private func scaledMatrix(_ matrix: [SIMD2<QFloat>], by s: QFloat) -> [SIMD2<QFloat>] {
        matrix.map { complex($0.x * s, $0.y * s) }
    }

    /// SWAP(q1, q2) decomposed into three CNOTs — CX(q1,q2)·CX(q2,q1)·CX(q1,q2) — each applied as a
    /// two-sided unitary update ρ → U ρ U†, reusing the existing controlled single-qubit path.
    private func applySwap(
        q1: Int,
        q2: Int,
        on density: DensityMatrix,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        try applyUnitaryGate(.cx(control: q1, target: q2), on: density, scratchReal: scratchReal, scratchImag: scratchImag)
        try applyUnitaryGate(.cx(control: q2, target: q1), on: density, scratchReal: scratchReal, scratchImag: scratchImag)
        try applyUnitaryGate(.cx(control: q1, target: q2), on: density, scratchReal: scratchReal, scratchImag: scratchImag)
    }

    /// Reset of `qubit` to |0⟩ as a CPTP map with Kraus operators K0 = |0⟩⟨0|, K1 = |0⟩⟨1|. Since
    /// K0†K0 + K1†K1 = I the map is trace-preserving (no renormalization needed); it is dispatched
    /// through the existing single-qubit Kraus pipeline.
    private func applyReset(
        qubit: Int,
        on density: DensityMatrix,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        try applyKrausChannel(
            on: density,
            targetQubit: qubit,
            kraus: [
                [complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)], // K0 = |0⟩⟨0|
                [complex(0, 0), complex(1, 0), complex(0, 0), complex(0, 0)], // K1 = |0⟩⟨1|
            ],
            scratchReal: scratchReal,
            scratchImag: scratchImag
        )
    }

    /// Computational-basis measurement with optional measurement-induced dephasing (C10).
    ///
    /// For each qubit: optional pre-measure phase damping, Born-rule sample, then either
    /// projective collapse (``MeasurementMode/projective``) or full Z-dephasing without
    /// collapse (``MeasurementMode/dephasingOnly``). Classical readout flips are applied last.
    private func applyMeasurement(
        qubits: [Int],
        on density: DensityMatrix,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws -> [Int] {
        guard !qubits.isEmpty else {
            throw QuantumMeasurementError.emptyQubitSelection
        }

        let mode = noise?.measurementMode ?? .projective
        let preDephasing = noise?.measurementDephasingProbability ?? 0

        var bits: [Int] = []
        bits.reserveCapacity(qubits.count)

        for qubit in qubits {
            if preDephasing > 0 {
                try applyPhaseDampingChannel(
                    on: density,
                    qubit: qubit,
                    lambda: preDephasing,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

            let p0 = try diagonalPopulation(of: density, qubit: qubit, bit: 0)
            let dice = rng.nextUnitDouble()
            let outcome = dice < p0 ? 0 : 1

            switch mode {
            case .projective:
                let probability = outcome == 0 ? p0 : 1 - p0
                guard probability > 0 else {
                    throw DensityMatrixEngineError.zeroProbabilityMeasurement(qubit: qubit)
                }

                let scaleFactor = 1 / probability.squareRoot()
                let maxFloat32 = Double(Float32.greatestFiniteMagnitude)
                let safeScaleFactor = min(scaleFactor, maxFloat32)
                let scale = QFloat(safeScaleFactor)
                let projector: [SIMD2<QFloat>] = outcome == 0
                    ? [complex(scale, 0), complex(0, 0), complex(0, 0), complex(0, 0)]
                    : [complex(0, 0), complex(0, 0), complex(0, 0), complex(scale, 0)]

                try applyKrausChannel(
                    on: density,
                    targetQubit: qubit,
                    kraus: [projector],
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )

            case .dephasingOnly:
                try applyPhaseDampingChannel(
                    on: density,
                    qubit: qubit,
                    lambda: 1,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

            bits.append(outcome)
        }

        if let noise {
            return noise.flipReadoutBits(bits, rng: &rng)
        }
        return bits
    }

    private func applyPhaseDampingChannel(
        on density: DensityMatrix,
        qubit: Int,
        lambda: QFloat,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        guard lambda > 0 else { return }
        let keep = sqrt(max(0, 1 - lambda))
        let dephase = sqrt(max(0, lambda))
        try applyKrausChannel(
            on: density,
            targetQubit: qubit,
            kraus: [
                [complex(1, 0), complex(0, 0), complex(0, 0), complex(keep, 0)],
                [complex(0, 0), complex(0, 0), complex(0, 0), complex(dephase, 0)],
            ],
            scratchReal: scratchReal,
            scratchImag: scratchImag
        )
    }

    /// Population of outcome `bit` on `qubit`: Σ_i ρ_ii over basis states i whose q-th bit is `bit`.
    /// The diagonal of a valid density matrix is real and non-negative, so the imaginary part is
    /// ignored; the fold is done in `Double` to avoid Float32 cancellation, then clamped to [0, 1].
    private func diagonalPopulation(of density: DensityMatrix, qubit: Int, bit: Int) throws -> Double {
        // This is a mid-circuit host read: the immediately preceding gates/channels (and any prior
        // collapse in this measurement) were committed without blocking, so drain the serial queue
        // before touching the shared density buffer to avoid reading stale, pre-dispatch contents.
        try drainPipeline()
        let real = density.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let stateCount = density.stateCount
        var sum = 0.0
        for i in 0..<stateCount where (i >> qubit) & 1 == bit {
            sum += Double(real[i * stateCount + i])
        }
        return min(max(sum, 0), 1)
    }

    private func dispatchMatrixElements(
        encoder: MTLComputeCommandEncoder,
        elementCount: Int,
        pipeline: MTLComputePipelineState
    ) {
        let threads = MTLSize(width: max(elementCount, 1), height: 1, depth: 1)
        let width = min(max(1, pipeline.threadExecutionWidth), pipeline.maxTotalThreadsPerThreadgroup)
        let group = MTLSize(width: width, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: group)
    }

    private struct EncodedSingleQubitUnitary {
        let target: Int
        let controlMask: UInt32
        let matrix: [SIMD2<QFloat>] // row-major [u00, u01, u10, u11]
    }

    private func encodeSingleQubitUnitary(_ gate: Gate) throws -> EncodedSingleQubitUnitary {
        // Reduce unbounded rotation/phase angles into [-π, π] (in Double) before the host-side
        // Float32 cos/sin below, matching the state-vector engine and avoiding precision loss.
        let gate = gate.angleWrapped
        switch gate {
        case .h(let target):
            let v = QFloat(0.5).squareRoot()
            return .init(target: target, controlMask: 0, matrix: [complex(v, 0), complex(v, 0), complex(v, 0), complex(-v, 0)])
        case .x(let target):
            return .init(target: target, controlMask: 0, matrix: [complex(0, 0), complex(1, 0), complex(1, 0), complex(0, 0)])
        case .y(let target):
            return .init(target: target, controlMask: 0, matrix: [complex(0, 0), complex(0, -1), complex(0, 1), complex(0, 0)])
        case .z(let target):
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(-1, 0)])
        case .s(let target):
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 1)])
        case .t(let target):
            let v = QFloat(0.5).squareRoot()
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(v, v)])
        case .sdg(let target):
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(0, -1)])
        case .tdg(let target):
            let v = QFloat(0.5).squareRoot()
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(v, -v)])
        case .sx(let target):
            return .init(
                target: target,
                controlMask: 0,
                matrix: [
                    complex(0.5, 0.5), complex(0.5, -0.5),
                    complex(0.5, -0.5), complex(0.5, 0.5),
                ]
            )
        case .sxdg(let target):
            return .init(
                target: target,
                controlMask: 0,
                matrix: [
                    complex(0.5, -0.5), complex(0.5, 0.5),
                    complex(0.5, 0.5), complex(0.5, -0.5),
                ]
            )
        case .p(let theta, let target):
            let angle = try theta.gpuAngle()
            let c = cos(angle)
            let s = sin(angle)
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(c, s)])
        case .u(let theta, let phi, let lambda, let target):
            let thetaValue = try theta.gpuAngle()
            let phiValue = try phi.gpuAngle()
            let lambdaValue = try lambda.gpuAngle()
            let c = cos(thetaValue * 0.5)
            let s = sin(thetaValue * 0.5)
            let eLambda = complex(cos(lambdaValue), sin(lambdaValue))
            let ePhi = complex(cos(phiValue), sin(phiValue))
            let ePhiLambda = complex(cos(phiValue + lambdaValue), sin(phiValue + lambdaValue))
            return .init(
                target: target,
                controlMask: 0,
                matrix: [
                    complex(c, 0),
                    complexMul(complex(-s, 0), eLambda),
                    complexMul(complex(s, 0), ePhi),
                    complexMul(complex(c, 0), ePhiLambda),
                ]
            )
        case .rx(let theta, let target):
            let angle = try theta.gpuAngle()
            let c = cos(angle * 0.5)
            let s = sin(angle * 0.5)
            return .init(target: target, controlMask: 0, matrix: [complex(c, 0), complex(0, -s), complex(0, -s), complex(c, 0)])
        case .ry(let theta, let target):
            let angle = try theta.gpuAngle()
            let c = cos(angle * 0.5)
            let s = sin(angle * 0.5)
            return .init(target: target, controlMask: 0, matrix: [complex(c, 0), complex(-s, 0), complex(s, 0), complex(c, 0)])
        case .rz(let theta, let target):
            let half = try theta.gpuAngle() * 0.5
            return .init(
                target: target,
                controlMask: 0,
                matrix: [
                    complex(cos(half), -sin(half)),
                    complex(0, 0),
                    complex(0, 0),
                    complex(cos(half), sin(half)),
                ]
            )
        case .cx(let control, let target):
            return .init(target: target, controlMask: bitMask(of: control), matrix: [complex(0, 0), complex(1, 0), complex(1, 0), complex(0, 0)])
        case .cz(let control, let target):
            return .init(target: target, controlMask: bitMask(of: control), matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(-1, 0)])
        case .ccx(let control1, let control2, let target):
            return .init(target: target, controlMask: bitMask(of: control1) | bitMask(of: control2), matrix: [complex(0, 0), complex(1, 0), complex(1, 0), complex(0, 0)])
        case .mcx(let controls, let target):
            return .init(target: target, controlMask: controls.reduce(0) { $0 | bitMask(of: $1) }, matrix: [complex(0, 0), complex(1, 0), complex(1, 0), complex(0, 0)])
        case .mcz(let controls, let target):
            return .init(target: target, controlMask: controls.reduce(0) { $0 | bitMask(of: $1) }, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(-1, 0)])
        case .crx(let theta, let control, let target):
            let angle = try theta.gpuAngle()
            let c = cos(angle * 0.5)
            let s = sin(angle * 0.5)
            return .init(target: target, controlMask: bitMask(of: control), matrix: [complex(c, 0), complex(0, -s), complex(0, -s), complex(c, 0)])
        case .cry(let theta, let control, let target):
            let angle = try theta.gpuAngle()
            let c = cos(angle * 0.5)
            let s = sin(angle * 0.5)
            return .init(target: target, controlMask: bitMask(of: control), matrix: [complex(c, 0), complex(-s, 0), complex(s, 0), complex(c, 0)])
        case .crz(let theta, let control, let target):
            let half = try theta.gpuAngle() * 0.5
            return .init(
                target: target,
                controlMask: bitMask(of: control),
                matrix: [
                    complex(cos(half), -sin(half)),
                    complex(0, 0),
                    complex(0, 0),
                    complex(cos(half), sin(half)),
                ]
            )
        case .cp(let theta, let control, let target):
            let angle = try theta.gpuAngle()
            return .init(
                target: target,
                controlMask: bitMask(of: control),
                matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(cos(angle), sin(angle))]
            )
        case .unitary1(let matrix, let target):
            return .init(
                target: target,
                controlMask: 0,
                matrix: matrix.map { complex($0.real, $0.imaginary) }
            )
        case .customUnitary(let matrix, let qubits):
            guard qubits.count == 1, let target = qubits.first else {
                throw DensityMatrixEngineError.unsupportedGate(gate)
            }
            return .init(
                target: target,
                controlMask: 0,
                matrix: matrix.map { complex($0.real, $0.imaginary) }
            )
        case .initialize, .swap, .measure, .reset, .c_if, .barrier, .delay, .id:
            throw DensityMatrixEngineError.unsupportedGate(gate)
        case .iswap, .ecr, .rxx, .ryy, .rzz, .dcx, .cswap:
            // Composites must be expanded in applyUnitaryGate before encoding.
            throw DensityMatrixEngineError.unsupportedGate(gate)
        }
    }

    private func complex(_ re: QFloat, _ im: QFloat) -> SIMD2<QFloat> {
        SIMD2<QFloat>(re, im)
    }

    private func complexMul(_ a: SIMD2<QFloat>, _ b: SIMD2<QFloat>) -> SIMD2<QFloat> {
        complex((a.x * b.x) - (a.y * b.y), (a.x * b.y) + (a.y * b.x))
    }

    private func bitMask(of qubit: Int) -> UInt32 {
        UInt32(1) << UInt32(qubit)
    }

    /// Acquires a shared scratch buffer from the ``BufferPool`` sized to `values` and copies them in.
    /// The caller must return it to the pool with ``BufferPool/release(_:after:)`` keyed to the
    /// command buffer that consumes it, so it re-enters the free list only once the GPU has finished
    /// — gate/Kraus dispatches are no longer awaited synchronously. This still eliminates the
    /// per-gate `makeBuffer` allocation storm; under pipelining the pool simply grows to the number
    /// of buffers in flight before stabilizing.
    private func acquirePooledMatrixBuffer<T>(values: [T]) throws -> MTLBuffer {
        let byteCount = values.count * MemoryLayout<T>.stride
        let buffer = try bufferPool.acquire(length: max(byteCount, 1))
        guard byteCount > 0 else { return buffer }
        values.withUnsafeBytes { source in
            buffer.contents().copyMemory(from: source.baseAddress!, byteCount: byteCount)
        }
        return buffer
    }

    static func loadLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let library = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return library
        }

        let bundle = Bundle.module
        let candidateURLs = [
            bundle.url(forResource: "DensityMatrixKernels", withExtension: "metalsrc", subdirectory: "Metal"),
            bundle.url(forResource: "DensityMatrixKernels", withExtension: "metalsrc"),
        ]
        guard let sourceURL = candidateURLs.compactMap({ $0 }).first else {
            throw DensityMatrixEngineError.libraryNotFound
        }
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        return try device.makeLibrary(source: source, options: nil)
    }
}

// MARK: - Localized noise channels

extension DensityMatrixEngine {

    func applyLocalizedNoiseChannels(
        after gate: Gate,
        at gateIndex: Int,
        on density: DensityMatrix,
        noise: NoiseModel,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        var seen = Set<Int>()
        let affected = gate.affectedQubits.filter { seen.insert($0).inserted }

        for rule in noise.matchingLocalizedRules(
            for: gate,
            affectedQubits: affected,
            gateIndex: gateIndex
        ) {
            let qubits = rule.target.applicationQubits(gate: gate, affectedQubits: affected)
            guard !qubits.isEmpty else { continue }
            try applyLocalized(
                channel: rule.channel,
                after: gate,
                on: density,
                qubits: qubits,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        }
    }

    private func applyLocalized(
        channel: QuantumChannel,
        after gate: Gate,
        on density: DensityMatrix,
        qubits: [Int],
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        switch channel {
        case .depolarizing(let probability):
            try applyDepolarizingNoise(
                on: density,
                qubits: qubits,
                probability: probability,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )

        case .amplitudeDamping(let gamma):
            let keep = sqrt(max(0, 1 - gamma))
            let relax = sqrt(max(0, gamma))
            for qubit in qubits {
                try applyKrausChannel(
                    on: density,
                    targetQubit: qubit,
                    kraus: [
                        [complex(1, 0), complex(0, 0), complex(0, 0), complex(keep, 0)],
                        [complex(0, 0), complex(relax, 0), complex(0, 0), complex(0, 0)],
                    ],
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

        case .phaseDamping(let lambda):
            let keep = sqrt(max(0, 1 - lambda))
            let dephase = sqrt(max(0, lambda))
            for qubit in qubits {
                try applyKrausChannel(
                    on: density,
                    targetQubit: qubit,
                    kraus: [
                        [complex(1, 0), complex(0, 0), complex(0, 0), complex(keep, 0)],
                        [complex(0, 0), complex(0, 0), complex(0, 0), complex(dephase, 0)],
                    ],
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

        case .pauliXFlip(let probability):
            for qubit in qubits {
                try applyPauliFlipChannel(
                    on: density,
                    qubit: qubit,
                    axis: .x,
                    probability: probability,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

        case .pauliYFlip(let probability):
            for qubit in qubits {
                try applyPauliFlipChannel(
                    on: density,
                    qubit: qubit,
                    axis: .y,
                    probability: probability,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

        case .pauliZFlip(let probability):
            for qubit in qubits {
                try applyPauliFlipChannel(
                    on: density,
                    qubit: qubit,
                    axis: .z,
                    probability: probability,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

        case .coherentOverRotation(let axis, let angle):
            guard abs(angle) > 0 else { return }
            for qubit in qubits {
                try applyUnitaryGate(
                    Self.rotationGate(axis: axis, angle: angle, target: qubit),
                    on: density,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

        case .coherentUnitaryError(let axis, let angle, let probability):
            guard probability > 0, abs(angle) > 0 else { return }
            for qubit in qubits {
                try applyCoherentUnitaryMixture(
                    on: density,
                    qubit: qubit,
                    axis: axis,
                    angle: angle,
                    probability: probability,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

        case .idleThermalRelaxation(let t1, let t2):
            guard case .delay(let duration, _) = gate else { return }
            for qubit in qubits {
                try applyThermalRelaxation(
                    on: density,
                    qubit: qubit,
                    duration: duration,
                    t1: t1,
                    t2: t2,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

        case .correlatedPauli(let axis, let probability):
            guard qubits.count == 2, probability > 0 else { return }
            try applyCorrelatedPauli(
                on: density,
                qubitA: qubits[0],
                qubitB: qubits[1],
                axis: axis,
                probability: probability,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )

        case .correlatedZZ(let angle):
            guard qubits.count == 2, abs(angle) > 0 else { return }
            try applyUnitaryGate(
                .rzz(theta: QFloatExpr(angle), q1: qubits[0], q2: qubits[1]),
                on: density,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        }
    }

    /// Exact channel `(1-p)ρ + p (P⊗P)ρ(P⊗P)` on a qubit pair (C5).
    private func applyCorrelatedPauli(
        on density: DensityMatrix,
        qubitA: Int,
        qubitB: Int,
        axis: CoherentRotationAxis,
        probability p: QFloat,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        if p >= 1 {
            let gate: Gate
            switch axis {
            case .x: gate = .x(target: qubitA)
            case .y: gate = .y(target: qubitA)
            case .z: gate = .z(target: qubitA)
            }
            let gateB: Gate
            switch axis {
            case .x: gateB = .x(target: qubitB)
            case .y: gateB = .y(target: qubitB)
            case .z: gateB = .z(target: qubitB)
            }
            try applyUnitaryGate(gate, on: density, scratchReal: scratchReal, scratchImag: scratchImag)
            try applyUnitaryGate(gateB, on: density, scratchReal: scratchReal, scratchImag: scratchImag)
            return
        }

        let axisCode: Int
        switch axis {
        case .x: axisCode = 1
        case .y: axisCode = 2
        case .z: axisCode = 3
        }

        let identityWeight = max(0, 1 - p).squareRoot()
        let pauliWeight = max(0, p).squareRoot()
        let pauli = singleQubitPauli(axis: axisCode)
        var kraus: [SIMD2<QFloat>] = []
        kraus.reserveCapacity(2 * 16)
        kraus.append(contentsOf: scaledMatrix(
            kron(singleQubitPauli(axis: 0), singleQubitPauli(axis: 0)),
            by: identityWeight
        ))
        kraus.append(contentsOf: scaledMatrix(kron(pauli, pauli), by: pauliWeight))

        try applyTwoQubitKrausChannel(
            on: density,
            qubitA: qubitA,
            qubitB: qubitB,
            krausCount: 2,
            krausFlat: kraus,
            scratchReal: scratchReal,
            scratchImag: scratchImag
        )
    }

    /// Apply AD then pure-dephasing for an idle interval of `duration` (C8).
    private func applyThermalRelaxation(
        on density: DensityMatrix,
        qubit: Int,
        duration: QFloat,
        t1: QFloat,
        t2: QFloat,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        let gamma = Self.amplitudeDampingProbability(t1: t1, duration: duration)
        if gamma > 0 {
            let keep = sqrt(max(0, 1 - gamma))
            let relax = sqrt(max(0, gamma))
            try applyKrausChannel(
                on: density,
                targetQubit: qubit,
                kraus: [
                    [complex(1, 0), complex(0, 0), complex(0, 0), complex(keep, 0)],
                    [complex(0, 0), complex(relax, 0), complex(0, 0), complex(0, 0)],
                ],
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        }

        let lambda = Self.phaseDampingProbability(t1: t1, t2: t2, duration: duration)
        if lambda > 0 {
            let keep = sqrt(max(0, 1 - lambda))
            let dephase = sqrt(max(0, lambda))
            try applyKrausChannel(
                on: density,
                targetQubit: qubit,
                kraus: [
                    [complex(1, 0), complex(0, 0), complex(0, 0), complex(keep, 0)],
                    [complex(0, 0), complex(0, 0), complex(0, 0), complex(dephase, 0)],
                ],
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
        }
    }

    private static func amplitudeDampingProbability(t1: QFloat, duration: QFloat) -> QFloat {
        guard t1 > 0, duration > 0 else { return 0 }
        return min(max(1 - exp(-duration / t1), 0), 1)
    }

    private static func phaseDampingProbability(t1: QFloat, t2: QFloat, duration: QFloat) -> QFloat {
        guard t2 > 0, duration > 0 else { return 0 }
        let inverseT2 = 1.0 / Double(t2)
        let inversePureDephasing = t1 > 0
            ? inverseT2 - 1.0 / (2.0 * Double(t1))
            : inverseT2
        guard inversePureDephasing > 0 else { return 0 }
        let lambda = 1.0 - exp(-2.0 * Double(duration) * inversePureDephasing)
        return min(max(QFloat(lambda), 0), 1)
    }

    private static func rotationGate(
        axis: CoherentRotationAxis,
        angle: QFloat,
        target: Int
    ) -> Gate {
        let theta = QFloatExpr(angle)
        switch axis {
        case .x: return .rx(theta: theta, target: target)
        case .y: return .ry(theta: theta, target: target)
        case .z: return .rz(theta: theta, target: target)
        }
    }

    /// Exact channel `(1-p)ρ + p UρU†` via Kraus `{√(1-p) I, √p U}`.
    private func applyCoherentUnitaryMixture(
        on density: DensityMatrix,
        qubit: Int,
        axis: CoherentRotationAxis,
        angle: QFloat,
        probability p: QFloat,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        if p >= 1 {
            try applyUnitaryGate(
                Self.rotationGate(axis: axis, angle: angle, target: qubit),
                on: density,
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
            return
        }

        let half = angle / 2
        let c = cos(half)
        let s = sin(half)
        let k0 = sqrt(max(0, 1 - p))
        let k1 = sqrt(max(0, p))

        let u: [SIMD2<QFloat>]
        switch axis {
        case .x:
            u = [complex(c, 0), complex(0, -s), complex(0, -s), complex(c, 0)]
        case .y:
            u = [complex(c, 0), complex(-s, 0), complex(s, 0), complex(c, 0)]
        case .z:
            u = [complex(cos(half), -sin(half)), complex(0, 0), complex(0, 0), complex(cos(half), sin(half))]
        }

        let kraus: [[SIMD2<QFloat>]] = [
            [complex(k0, 0), complex(0, 0), complex(0, 0), complex(k0, 0)],
            u.map { complex($0.x * k1, $0.y * k1) },
        ]

        try applyKrausChannel(
            on: density,
            targetQubit: qubit,
            kraus: kraus,
            scratchReal: scratchReal,
            scratchImag: scratchImag
        )
    }

    private enum PauliFlipAxis {
        case x, y, z
    }

    private func applyPauliFlipChannel(
        on density: DensityMatrix,
        qubit: Int,
        axis: PauliFlipAxis,
        probability p: QFloat,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        guard p > 0 else { return }
        let k0 = sqrt(max(0, 1 - p))
        let k1 = sqrt(max(0, p))

        let kraus: [[SIMD2<QFloat>]]
        switch axis {
        case .x:
            kraus = [
                [complex(k0, 0), complex(0, 0), complex(0, 0), complex(k0, 0)],
                [complex(0, 0), complex(k1, 0), complex(k1, 0), complex(0, 0)],
            ]
        case .y:
            kraus = [
                [complex(k0, 0), complex(0, 0), complex(0, 0), complex(k0, 0)],
                [complex(0, 0), complex(0, -k1), complex(0, k1), complex(0, 0)],
            ]
        case .z:
            kraus = [
                [complex(k0, 0), complex(0, 0), complex(0, 0), complex(k0, 0)],
                [complex(k1, 0), complex(0, 0), complex(0, 0), complex(-k1, 0)],
            ]
        }

        try applyKrausChannel(
            on: density,
            targetQubit: qubit,
            kraus: kraus,
            scratchReal: scratchReal,
            scratchImag: scratchImag
        )
    }

    private func applyHostCustomUnitary(
        matrix: [ComplexAmplitude],
        qubits: [Int],
        on density: DensityMatrix
    ) throws {
        let subDimension = 1 << qubits.count
        guard matrix.count == subDimension * subDimension else { return }

        let dimension = density.stateCount
        let targetMask = qubits.reduce(0) { $0 | (1 << $1) }
        let passiveMask = ((1 << density.qubitCount) - 1) & ~targetMask

        var unitary = [ComplexAmplitude](
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: dimension * dimension
        )
        for row in 0..<dimension {
            for column in 0..<dimension {
                guard (row & passiveMask) == (column & passiveMask) else { continue }
                let subRow = QubitIndexing.partialOutcomeIndex(stateIndex: row, qubits: qubits)
                let subColumn = QubitIndexing.partialOutcomeIndex(stateIndex: column, qubits: qubits)
                unitary[row * dimension + column] = matrix[subRow * subDimension + subColumn]
            }
        }

        let real = density.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imag = density.imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        var rho = [[(re: Double, im: Double)]](
            repeating: Array(repeating: (0.0, 0.0), count: dimension),
            count: dimension
        )
        for row in 0..<dimension {
            for column in 0..<dimension {
                let index = row * dimension + column
                rho[row][column] = (Double(real[index]), Double(imag[index]))
            }
        }

        func complexMul(_ a: (re: Double, im: Double), _ b: (re: Double, im: Double)) -> (re: Double, im: Double) {
            (a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
        }

        func dagger(_ value: ComplexAmplitude) -> (re: Double, im: Double) {
            (Double(value.real), -Double(value.imaginary))
        }

        var temp = rho
        for row in 0..<dimension {
            for column in 0..<dimension {
                var sum = (0.0, 0.0)
                for k in 0..<dimension {
                    let u = (Double(unitary[row * dimension + k].real), Double(unitary[row * dimension + k].imaginary))
                    sum = (
                        sum.0 + complexMul(u, rho[k][column]).0,
                        sum.1 + complexMul(u, rho[k][column]).1
                    )
                }
                temp[row][column] = sum
            }
        }

        var updated = temp
        for row in 0..<dimension {
            for column in 0..<dimension {
                var sum = (0.0, 0.0)
                for k in 0..<dimension {
                    let uDagger = dagger(unitary[column * dimension + k])
                    sum = (
                        sum.0 + complexMul(temp[row][k], uDagger).0,
                        sum.1 + complexMul(temp[row][k], uDagger).1
                    )
                }
                updated[row][column] = sum
            }
        }

        for row in 0..<dimension {
            for column in 0..<dimension {
                let index = row * dimension + column
                real[index] = QFloat(updated[row][column].0)
                imag[index] = QFloat(updated[row][column].1)
            }
        }
    }
}
