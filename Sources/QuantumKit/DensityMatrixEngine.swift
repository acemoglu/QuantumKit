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
}

public final class DensityMatrixEngine: @unchecked Sendable {
    struct Pipelines: @unchecked Sendable {
        let leftMultiplySingleQubit: MTLComputePipelineState
        let rightMultiplySingleQubitDagger: MTLComputePipelineState
        let applyKrausSingleQubit: MTLComputePipelineState

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
        }
    }

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let bufferPool: BufferPool
    let pipelines: Pipelines

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw DensityMatrixEngineError.deviceNotFound
        }
        self.device = device
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

        let scratchBytes = density.elementCount * MemoryLayout<QFloat>.stride
        let scratchReal = try bufferPool.acquire(length: scratchBytes)
        let scratchImag = try bufferPool.acquire(length: scratchBytes)
        defer {
            bufferPool.release(scratchReal)
            bufferPool.release(scratchImag)
        }

        var measurementOutcomes: [[Int]] = []

        for gate in circuit.gates {
            switch gate {
            case .measure(let qubits):
                let bits = try applyMeasurement(
                    qubits: qubits,
                    on: density,
                    rng: &rng,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
                measurementOutcomes.append(bits)

            case .reset(let qubit):
                try applyReset(
                    qubit: qubit,
                    on: density,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )

            case .swap(let q1, let q2):
                try applySwap(
                    q1: q1,
                    q2: q2,
                    on: density,
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
                if let noise {
                    try applyNoiseChannels(after: gate, on: density, noise: noise, scratchReal: scratchReal, scratchImag: scratchImag)
                }

            default:
                try applyUnitaryGate(gate, on: density, scratchReal: scratchReal, scratchImag: scratchImag)
                if let noise {
                    try applyNoiseChannels(after: gate, on: density, noise: noise, scratchReal: scratchReal, scratchImag: scratchImag)
                }
            }
        }

        return CircuitExecutionResult(measurementOutcomes: measurementOutcomes)
    }

    public func probabilities(of density: DensityMatrix) -> [QFloat] {
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
        let encoded = try encodeSingleQubitUnitary(gate)
        let matrixBuffer = try acquirePooledMatrixBuffer(values: encoded.matrix)
        defer { bufferPool.release(matrixBuffer) }

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
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DensityMatrixEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    private func applyNoiseChannels(
        after gate: Gate,
        on density: DensityMatrix,
        noise: NoiseModel,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        let affected = Set(gate.affectedQubits)
        for qubit in affected {
            if noise.appliesDepolarizing {
                let p = noise.depolarizingProbability
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
                let lambda = noise.phaseDampingProbability
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
        defer { bufferPool.release(krausBuffer) }

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

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DensityMatrixEngineError.commandBufferExecutionFailed(underlying: error)
        }
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

    /// Projective computational-basis measurement with state collapse. Each listed qubit is measured
    /// independently in circuit order; on distinct qubits the projectors commute, so sequential
    /// collapse reproduces the correct joint distribution.
    ///
    /// For qubit q the outcome probability is p_b = Tr(Π_b ρ), the sum of the diagonal populations
    /// over basis states whose q-th bit equals b. After sampling b with the engine's RNG, the
    /// selected projector is folded with its own normalization into a single Kraus operator
    /// Π_b / √p_b, so the existing Kraus pipeline performs ρ → Π_b ρ Π_b / p_b in one pass and the
    /// post-measurement matrix already has unit trace.
    private func applyMeasurement(
        qubits: [Int],
        on density: DensityMatrix,
        rng: inout QuantumRNG,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws -> [Int] {
        guard !qubits.isEmpty else {
            throw QuantumMeasurementError.emptyQubitSelection
        }

        var bits: [Int] = []
        bits.reserveCapacity(qubits.count)

        for qubit in qubits {
            let p0 = diagonalPopulation(of: density, qubit: qubit, bit: 0)
            let dice = rng.nextUnitDouble()
            let outcome = dice < p0 ? 0 : 1
            let probability = outcome == 0 ? p0 : 1 - p0
            guard probability > 0 else {
                throw DensityMatrixEngineError.zeroProbabilityMeasurement(qubit: qubit)
            }

            let scale = QFloat(1 / probability.squareRoot())
            let projector: [SIMD2<QFloat>] = outcome == 0
                ? [complex(scale, 0), complex(0, 0), complex(0, 0), complex(0, 0)] // Π0 / √p0
                : [complex(0, 0), complex(0, 0), complex(0, 0), complex(scale, 0)] // Π1 / √p1

            try applyKrausChannel(
                on: density,
                targetQubit: qubit,
                kraus: [projector],
                scratchReal: scratchReal,
                scratchImag: scratchImag
            )
            bits.append(outcome)
        }

        return bits
    }

    /// Population of outcome `bit` on `qubit`: Σ_i ρ_ii over basis states i whose q-th bit is `bit`.
    /// The diagonal of a valid density matrix is real and non-negative, so the imaginary part is
    /// ignored; the fold is done in `Double` to avoid Float32 cancellation, then clamped to [0, 1].
    private func diagonalPopulation(of density: DensityMatrix, qubit: Int, bit: Int) -> Double {
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
            let c = cos(theta)
            let s = sin(theta)
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(c, s)])
        case .u(let theta, let phi, let lambda, let target):
            let c = cos(theta * 0.5)
            let s = sin(theta * 0.5)
            let eLambda = complex(cos(lambda), sin(lambda))
            let ePhi = complex(cos(phi), sin(phi))
            let ePhiLambda = complex(cos(phi + lambda), sin(phi + lambda))
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
            let c = cos(theta * 0.5)
            let s = sin(theta * 0.5)
            return .init(target: target, controlMask: 0, matrix: [complex(c, 0), complex(0, -s), complex(0, -s), complex(c, 0)])
        case .ry(let theta, let target):
            let c = cos(theta * 0.5)
            let s = sin(theta * 0.5)
            return .init(target: target, controlMask: 0, matrix: [complex(c, 0), complex(-s, 0), complex(s, 0), complex(c, 0)])
        case .rz(let theta, let target):
            let half = theta * 0.5
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
            let c = cos(theta * 0.5)
            let s = sin(theta * 0.5)
            return .init(target: target, controlMask: bitMask(of: control), matrix: [complex(c, 0), complex(0, -s), complex(0, -s), complex(c, 0)])
        case .cry(let theta, let control, let target):
            let c = cos(theta * 0.5)
            let s = sin(theta * 0.5)
            return .init(target: target, controlMask: bitMask(of: control), matrix: [complex(c, 0), complex(-s, 0), complex(s, 0), complex(c, 0)])
        case .crz(let theta, let control, let target):
            let half = theta * 0.5
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
            return .init(
                target: target,
                controlMask: bitMask(of: control),
                matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(cos(theta), sin(theta))]
            )
        case .swap, .measure, .reset:
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
    /// The caller must `release` it back to the pool. Because every gate/Kraus dispatch here is
    /// awaited synchronously (`waitUntilCompleted`), a released buffer is guaranteed idle and is
    /// reused on the next gate — eliminating the per-gate `makeBuffer` allocation storm.
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
