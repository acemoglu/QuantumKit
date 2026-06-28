import Foundation
import Metal

extension QuantumEngine {

    /// Depolarizing noise via Pauli-jump unraveling.
    ///
    /// - 1-qubit gates: with probability `p` a jump occurs and one of the 3 non-identity
    ///   single-qubit Paulis (X, Y, Z) is applied uniformly.
    /// - 2-qubit gates (e.g. `cx`, `cz`, `swap`): the channel is the *two-qubit* depolarizing
    ///   channel, so with probability `p` a single jump occurs and one of the 15 non-identity
    ///   two-qubit Paulis (IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ) is applied
    ///   uniformly — rather than (incorrectly) drawing an independent single-qubit Pauli per qubit.
    /// - ≥3-qubit gates: fall back to independent per-qubit single-qubit depolarizing.
    func applyDepolarizingNoise(
        after gate: Gate,
        on state: StateVector,
        probability: QFloat,
        rng: inout QuantumRNG
    ) throws {
        guard probability > 0 else { return }

        // Distinct affected qubits, order preserved.
        var seen = Set<Int>()
        let qubits = gate.affectedQubits.filter { seen.insert($0).inserted }

        switch qubits.count {
        case 0:
            return

        case 1:
            guard rng.nextUnitFloat() < probability else { return }
            try executeUnitaryGate(randomSingleQubitPauli(on: qubits[0], rng: &rng), on: state)

        case 2:
            guard rng.nextUnitFloat() < probability else { return }
            for pauli in randomTwoQubitPauli(on: qubits[0], and: qubits[1], rng: &rng) {
                try executeUnitaryGate(pauli, on: state)
            }

        default:
            for qubit in qubits {
                guard rng.nextUnitFloat() < probability else { continue }
                try executeUnitaryGate(randomSingleQubitPauli(on: qubit, rng: &rng), on: state)
            }
        }
    }

    /// Uniformly samples one of the 3 non-identity single-qubit Paulis {X, Y, Z}.
    private func randomSingleQubitPauli(on qubit: Int, rng: inout QuantumRNG) -> Gate {
        let pauliRoll = rng.nextUnitFloat()
        if pauliRoll < (1.0 / 3.0) {
            return .x(target: qubit)
        } else if pauliRoll < (2.0 / 3.0) {
            return .y(target: qubit)
        } else {
            return .z(target: qubit)
        }
    }

    /// Uniformly samples one of the 15 non-identity two-qubit Paulis `P_a ⊗ P_b` (excluding I⊗I)
    /// and returns its non-identity single-qubit factors (which act on distinct qubits and so
    /// commute, making application order irrelevant).
    private func randomTwoQubitPauli(on qubitA: Int, and qubitB: Int, rng: inout QuantumRNG) -> [Gate] {
        // Enumerate the 16 combinations 0..15 with 2 bits per qubit (0=I,1=X,2=Y,3=Z) and skip I⊗I
        // by shifting a uniform 0..14 draw into 1..15.
        let index = min(Int(rng.nextUnitFloat() * 15), 14)
        let combined = index + 1
        let axisA = combined / 4
        let axisB = combined % 4

        var paulis: [Gate] = []
        if let a = pauliGate(axis: axisA, on: qubitA) { paulis.append(a) }
        if let b = pauliGate(axis: axisB, on: qubitB) { paulis.append(b) }
        return paulis
    }

    /// Maps an axis code (0=I, 1=X, 2=Y, 3=Z) to a Pauli gate, or `nil` for identity.
    private func pauliGate(axis: Int, on qubit: Int) -> Gate? {
        switch axis {
        case 1: return .x(target: qubit)
        case 2: return .y(target: qubit)
        case 3: return .z(target: qubit)
        default: return nil
        }
    }

    /// Amplitude damping via quantum-jump (Monte-Carlo wavefunction) unraveling.
    ///
    /// For each affected qubit with channel strength `gamma`, the jump probability is
    /// state-dependent: `p_jump = gamma * P(qubit = |1⟩)`. With probability `p_jump` the
    /// relaxation Kraus operator `K1 = √gamma · |0⟩⟨1|` is applied; otherwise the no-jump
    /// operator `K0 = diag(1, √(1-gamma))` damps the excited amplitude. Both branches are
    /// renormalized so the ensemble average reproduces the amplitude damping channel.
    func applyAmplitudeDamping(
        after gate: Gate,
        on state: StateVector,
        probability gamma: QFloat,
        rng: inout QuantumRNG
    ) throws {
        guard gamma > 0 else { return }

        for qubit in Set(gate.affectedQubits) {
            let jumpProbability = try trajectoryJumpProbability(
                on: state,
                qubit: qubit,
                weight: gamma
            )
            guard jumpProbability > 0 else { continue }

            if rng.nextUnitFloat() < jumpProbability {
                try dispatchAmplitudeDamping(on: state, qubit: qubit, pipeline: pipelines.amplitudeDampingJump)
            } else {
                let factor = (1 - gamma).squareRoot()
                try dispatchAmplitudeDamping(
                    on: state,
                    qubit: qubit,
                    pipeline: pipelines.amplitudeDampingNoJump,
                    factor: factor
                )
            }
            try normalizeState(on: state)
        }
    }

    func dispatchAmplitudeDamping(
        on state: StateVector,
        qubit: Int,
        pipeline: MTLComputePipelineState,
        factor: QFloat? = nil
    ) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        var targetQubit = UInt32(qubit)
        var factorValue = factor ?? 0
        dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipeline, state: state) { encoder in
            encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            if factor != nil {
                encoder.setBytes(&factorValue, length: MemoryLayout<QFloat>.stride, index: 3)
            }
        }

        computeEncoder.endEncoding()
        // Commit the damping (jump / no-jump) Kraus dispatch without draining the GPU. It uses only
        // the long-lived state buffers, and the serial command queue keeps it ordered before the
        // normalization that follows in the caller — so no per-channel CPU stall is needed.
        commandBuffer.commit()
    }

    /// Born-rule probability that `qubit` is measured in state |1⟩ given the current amplitudes.
    ///
    /// Computed with a GPU tree reduction (`masked_population_reduce`): each threadgroup reduces
    /// 256 squared amplitudes restricted to the `qubit == |1⟩` subspace into one partial, and the
    /// host sums the `2ⁿ / 256` partials, avoiding an `O(2ⁿ)` host-side scan of the amplitudes.
    func qubitOnePopulation(on state: StateVector, qubit: Int) throws -> QFloat {
        let blockSize = Self.scanBlockSize
        let elementCount = state.stateCount
        let blockCount = max((elementCount + blockSize - 1) / blockSize, 1)

        // One partial sum per threadgroup; the kernel writes each block's reduced total here and the
        // host sums them below (no float atomics, deterministic order).
        let partialsBuffer = try bufferPool.acquire(length: blockCount * MemoryLayout<QFloat>.stride)
        defer { bufferPool.release(partialsBuffer) }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        var targetQubit = UInt32(qubit)
        var countValue = UInt32(elementCount)

        computeEncoder.setComputePipelineState(pipelines.maskedPopulationReduce)
        computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 3)
        computeEncoder.setBuffer(partialsBuffer, offset: 0, index: 4)

        computeEncoder.dispatchThreadgroups(
            MTLSize(width: blockCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1)
        )

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }

        let partials = partialsBuffer.contents().assumingMemoryBound(to: QFloat.self)
        var total = 0.0
        for index in 0..<blockCount {
            total += Double(partials[index])
        }

        return QFloat(total)
    }

    /// Phase damping realized as its equivalent phase-flip (random Z) channel.
    ///
    /// Uses a jump unraveling with Kraus operators:
    /// `E0 = |0⟩⟨0| + sqrt(1-λ)|1⟩⟨1|`, `E1 = sqrt(λ)|1⟩⟨1|`.
    /// Jump probability is `p_jump = λ * P(qubit = |1⟩)`.
    func applyPhaseDamping(
        after gate: Gate,
        on state: StateVector,
        flipProbability lambda: QFloat,
        rng: inout QuantumRNG
    ) throws {
        guard lambda > 0 else { return }

        for qubit in Set(gate.affectedQubits) {
            let jumpProbability = try trajectoryJumpProbability(
                on: state,
                qubit: qubit,
                weight: lambda
            )
            guard jumpProbability > 0 else { continue }
            if rng.nextUnitFloat() < jumpProbability {
                try dispatchAmplitudeDamping(
                    on: state,
                    qubit: qubit,
                    pipeline: pipelines.phaseDampingJump
                )
            } else {
                let factor = (1 - lambda).squareRoot()
                try dispatchAmplitudeDamping(
                    on: state,
                    qubit: qubit,
                    pipeline: pipelines.phaseDampingNoJump,
                    factor: factor
                )
            }
            try normalizeState(on: state)
        }
    }

    /// Compensated GPU estimate of `p = weight * P(qubit = |1⟩)`, where `weight` is typically
    /// a channel parameter (e.g. `gamma` or `lambda`) and the population term is
    /// `<psi|1_q><1_q|psi>`.
    func trajectoryJumpProbability(
        on state: StateVector,
        qubit: Int,
        weight: QFloat
    ) throws -> QFloat {
        guard weight > 0 else { return 0 }
        let blockSize = Self.scanBlockSize
        let stateCount = state.stateCount
        var partialCount = max((stateCount + blockSize - 1) / blockSize, 1)

        let maxPartials = partialCount
        let bytes = maxPartials * MemoryLayout<QFloat>.stride
        let hiA = try bufferPool.acquire(length: bytes)
        let loA = try bufferPool.acquire(length: bytes)
        let hiB = try bufferPool.acquire(length: bytes)
        let loB = try bufferPool.acquire(length: bytes)
        defer {
            bufferPool.release(hiA)
            bufferPool.release(loA)
            bufferPool.release(hiB)
            bufferPool.release(loB)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        var targetQubit = UInt32(qubit)
        var weightValue = weight
        var elementCount = UInt32(stateCount)
        computeEncoder.setComputePipelineState(pipelines.trajectoryWeightedPopulationPartial)
        computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.setBytes(&weightValue, length: MemoryLayout<QFloat>.stride, index: 3)
        computeEncoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.stride, index: 4)
        computeEncoder.setBuffer(hiA, offset: 0, index: 5)
        computeEncoder.setBuffer(loA, offset: 0, index: 6)
        computeEncoder.dispatchThreadgroups(
            MTLSize(width: partialCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1)
        )

        var readHi = hiA
        var readLo = loA
        var writeHi = hiB
        var writeLo = loB

        while partialCount > 1 {
            let nextCount = max((partialCount + blockSize - 1) / blockSize, 1)
            var partialElementCount = UInt32(partialCount)
            computeEncoder.setComputePipelineState(pipelines.renormCompensatedPartial)
            computeEncoder.setBuffer(readHi, offset: 0, index: 0)
            computeEncoder.setBuffer(readLo, offset: 0, index: 1)
            computeEncoder.setBytes(&partialElementCount, length: MemoryLayout<UInt32>.stride, index: 2)
            computeEncoder.setBuffer(writeHi, offset: 0, index: 3)
            computeEncoder.setBuffer(writeLo, offset: 0, index: 4)
            computeEncoder.dispatchThreadgroups(
                MTLSize(width: nextCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1)
            )

            swap(&readHi, &writeHi)
            swap(&readLo, &writeLo)
            partialCount = nextCount
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }

        let hiPtr = readHi.contents().assumingMemoryBound(to: QFloat.self)
        let loPtr = readLo.contents().assumingMemoryBound(to: QFloat.self)
        let probability = Double(hiPtr[0]) + Double(loPtr[0])
        return QFloat(min(max(probability, 0), 1))
    }
}
