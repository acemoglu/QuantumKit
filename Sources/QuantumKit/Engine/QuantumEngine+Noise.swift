import Foundation
import Metal

extension QuantumEngine {

    func applyDepolarizingNoise(
        after gate: Gate,
        on state: StateVector,
        probability: QFloat,
        rng: inout QuantumRNG
    ) throws {
        guard probability > 0 else { return }

        for qubit in Set(gate.affectedQubits) {
            guard rng.nextUnitFloat() < probability else { continue }

            let pauliRoll = rng.nextUnitFloat()
            let pauliGate: Gate
            if pauliRoll < (1.0 / 3.0) {
                pauliGate = .x(target: qubit)
            } else if pauliRoll < (2.0 / 3.0) {
                pauliGate = .y(target: qubit)
            } else {
                pauliGate = .z(target: qubit)
            }

            try executeUnitaryGate(pauliGate, on: state)
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
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
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
