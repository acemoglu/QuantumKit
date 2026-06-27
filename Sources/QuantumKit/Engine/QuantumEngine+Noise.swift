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
            let excitedPopulation = try qubitOnePopulation(on: state, qubit: qubit)
            guard excitedPopulation > 0 else { continue }

            let jumpProbability = gamma * excitedPopulation
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
    /// The phase damping channel of strength λ is exactly the phase-flip channel
    /// `ρ → (1-p)ρ + p·ZρZ` with `p = (1-√(1-λ))/2`. In the trajectory picture this means
    /// applying a Pauli-Z with probability `flipProbability` to each affected qubit.
    func applyPhaseDamping(
        after gate: Gate,
        on state: StateVector,
        flipProbability: QFloat,
        rng: inout QuantumRNG
    ) throws {
        guard flipProbability > 0 else { return }

        for qubit in Set(gate.affectedQubits) {
            guard rng.nextUnitFloat() < flipProbability else { continue }
            try executeUnitaryGate(.z(target: qubit), on: state)
        }
    }
}
