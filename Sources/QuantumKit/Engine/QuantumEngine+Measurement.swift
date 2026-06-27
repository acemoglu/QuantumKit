import Foundation
import Metal

extension QuantumEngine {

    public func executePartialMeasurementCollapse(
        on state: StateVector,
        qubits: [Int],
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil
    ) throws -> Int {
        guard !qubits.isEmpty else {
            throw QuantumMeasurementError.emptyQubitSelection
        }

        let diceRoll = rng.nextUnitDouble()
        let outcome = try samplePartialOutcome(on: state, qubits: qubits, diceRoll: diceRoll)

        let qubitIndices = qubits.map { UInt32($0) }
        let qubitBuffer = try makeSharedBuffer(length: qubitIndices.count * MemoryLayout<UInt32>.stride)
        qubitBuffer.contents().copyMemory(
            from: qubitIndices,
            byteCount: qubitIndices.count * MemoryLayout<UInt32>.stride
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        var measuredQubitCount = UInt32(qubits.count)
        var outcomeValue = UInt32(outcome)

        dispatchFullStateKernel(encoder: computeEncoder, pipeline: pipelines.partialCollapse, state: state) { encoder in
            encoder.setBuffer(qubitBuffer, offset: 0, index: 2)
            encoder.setBytes(&measuredQubitCount, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&outcomeValue, length: MemoryLayout<UInt32>.stride, index: 4)
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }

        try normalizeState(on: state)

        if let noise {
            return noise.flipReadoutOutcome(outcome, measuredQubitCount: qubits.count, rng: &rng)
        }
        return outcome
    }

    public func executeMeasurementCollapse(
        on state: StateVector,
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil
    ) throws -> Int {
        let diceRoll = rng.nextUnitDouble()
        var collapsedIndex = try executeMeasurementCollapse(on: state, dice: diceRoll)
        if let noise {
            collapsedIndex = noise.flipReadoutOutcome(
                collapsedIndex,
                measuredQubitCount: state.qubitCount,
                rng: &rng
            )
        }
        return collapsedIndex
    }

    public func executeResetQubit(on state: StateVector, qubit: Int) throws {
        var rng: QuantumRNG = .hardware
        try executeResetQubit(on: state, qubit: qubit, rng: &rng)
    }

    /// Resets `qubit` to |0⟩ by measuring it in the computational basis and flipping it back to
    /// |0⟩ when the (random) outcome is |1⟩.
    ///
    /// This is the standard state-vector reset realized as a measurement trajectory: collapsing
    /// first keeps the post-reset vector pure (a valid single trajectory) and, unlike a bare
    /// projection onto |0⟩, it never annihilates a qubit that is fully in |1⟩.
    func executeResetQubit(on state: StateVector, qubit: Int, rng: inout QuantumRNG) throws {
        let outcome = try executePartialMeasurementCollapse(
            on: state,
            qubits: [qubit],
            rng: &rng,
            noise: nil
        )

        if outcome & 1 == 1 {
            try executeUnitaryGate(.x(target: qubit), on: state)
        }
    }

    func normalizeState(on state: StateVector) throws {
        let stateCount = state.stateCount
        let probabilityBytes = stateCount * MemoryLayout<QFloat>.stride
        let probHi = try bufferPool.acquire(length: probabilityBytes)
        let probLo = try bufferPool.acquire(length: probabilityBytes)
        let aux = try makePrefixSumAuxBuffers(stateCount: stateCount, pooled: true)
        // Safe to recycle at scope exit: both command buffers below are waited on synchronously.
        defer {
            bufferPool.release(probHi)
            bufferPool.release(probLo)
            releasePrefixSumAuxBuffers(aux)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        dispatchFullStateKernel(encoder: computeEncoder, pipeline: pipelines.probabilities, state: state) { encoder in
            encoder.setBuffer(probHi, offset: 0, index: 2)
        }

        try encodeInclusivePrefixSum(
            encoder: computeEncoder,
            hiBuffer: probHi,
            loBuffer: probLo,
            elementCount: stateCount,
            auxiliaryHi: aux.hi,
            auxiliaryLo: aux.lo
        )

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }

        // The final CDF element is the total probability; sum its (hi, lo) halves in Double so the
        // compensated GPU scan is not re-truncated to Float before taking the norm.
        let hiPointer = probHi.contents().assumingMemoryBound(to: QFloat.self)
        let loPointer = probLo.contents().assumingMemoryBound(to: QFloat.self)
        let totalProbability = Double(hiPointer[stateCount - 1]) + Double(loPointer[stateCount - 1])
        guard totalProbability > 0 else {
            throw QuantumEngineError.zeroStateNorm
        }

        let invNorm = 1 / totalProbability.squareRoot()

        guard let normalizeCommandBuffer = commandQueue.makeCommandBuffer(),
              let normalizeEncoder = normalizeCommandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        var invNormValue = QFloat(invNorm)
        dispatchFullStateKernel(encoder: normalizeEncoder, pipeline: pipelines.normalize, state: state) { encoder in
            encoder.setBytes(&invNormValue, length: MemoryLayout<QFloat>.stride, index: 2)
        }

        normalizeEncoder.endEncoding()
        normalizeCommandBuffer.commit()
        normalizeCommandBuffer.waitUntilCompleted()
        if let error = normalizeCommandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    func samplePartialOutcome(on state: StateVector, qubits: [Int], diceRoll: Double) throws -> Int {
        let byteCount = state.stateCount * MemoryLayout<QFloat>.stride
        let buffer = try bufferPool.acquire(length: byteCount)
        defer { bufferPool.release(buffer) }

        try executeProbabilityKernel(on: state, outputBuffer: buffer)
        let pointer = buffer.contents().assumingMemoryBound(to: QFloat.self)

        // Accumulate the marginal in Double: the O(2ⁿ) host fold over per-state float probabilities
        // is exactly where Float32 cancellation would otherwise erase sub-ulp contributions.
        var marginal = [Double](repeating: 0, count: 1 << qubits.count)
        for stateIndex in 0..<state.stateCount {
            let outcome = partialOutcomeIndex(stateIndex: stateIndex, qubits: qubits)
            marginal[outcome] += Double(pointer[stateIndex])
        }

        var cumulative = 0.0
        for (index, probability) in marginal.enumerated() {
            cumulative += probability
            if diceRoll < cumulative {
                return index
            }
        }

        return marginal.count - 1
    }

    func partialOutcomeIndex(stateIndex: Int, qubits: [Int]) -> Int {
        var outcome = 0
        for (position, qubit) in qubits.enumerated() {
            let bit = (stateIndex >> qubit) & 1
            outcome |= bit << position
        }
        return outcome
    }

    func measuredBits(outcome: Int, qubits: [Int]) -> [Int] {
        qubits.indices.map { position in
            (outcome >> position) & 1
        }
    }

    public func executeMeasurementCollapse(on state: StateVector, diceRoll: Float) throws -> Int {
        try executeMeasurementCollapse(on: state, dice: Double(diceRoll))
    }

    /// Collapses `state` to a computational-basis index sampled at the (high-resolution) `dice`
    /// position of the cumulative distribution.
    ///
    /// The CDF is built with the compensated (double-single) GPU prefix sum and searched in the
    /// same precision, and `dice` is a 53-bit `Double` split into a `(hi, lo)` float pair — together
    /// these let the sampler resolve states whose probability is below the float32 ulp of the
    /// running total, eliminating the high-qubit CDF plateaus.
    func executeMeasurementCollapse(on state: StateVector, dice: Double) throws -> Int {
        let stateCount = state.stateCount
        let probabilityBytes = stateCount * MemoryLayout<QFloat>.stride

        let probHi = try bufferPool.acquire(length: probabilityBytes)
        let probLo = try bufferPool.acquire(length: probabilityBytes)
        let collapsedBuffer = try makeSharedBuffer(length: MemoryLayout<UInt32>.stride)
        let aux = try makePrefixSumAuxBuffers(stateCount: stateCount, pooled: true)
        // Safe to recycle at scope exit: both command buffers below are waited on synchronously.
        defer {
            bufferPool.release(probHi)
            bufferPool.release(probLo)
            releasePrefixSumAuxBuffers(aux)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        dispatchFullStateKernel(encoder: computeEncoder, pipeline: pipelines.probabilities, state: state) { encoder in
            encoder.setBuffer(probHi, offset: 0, index: 2)
        }

        try encodeInclusivePrefixSum(
            encoder: computeEncoder,
            hiBuffer: probHi,
            loBuffer: probLo,
            elementCount: stateCount,
            auxiliaryHi: aux.hi,
            auxiliaryLo: aux.lo
        )

        // Split the Double dice roll into a double-single (hi, lo) float pair for the search.
        let diceHi = Float(dice)
        var dicePair = SIMD2<Float>(diceHi, Float(dice - Double(diceHi)))
        var elementCount = UInt32(stateCount)
        computeEncoder.setComputePipelineState(pipelines.collapseSearch)
        computeEncoder.setBuffer(probHi, offset: 0, index: 0)
        computeEncoder.setBuffer(probLo, offset: 0, index: 1)
        computeEncoder.setBytes(&dicePair, length: MemoryLayout<SIMD2<Float>>.stride, index: 2)
        computeEncoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.stride, index: 3)
        computeEncoder.setBuffer(collapsedBuffer, offset: 0, index: 4)
        computeEncoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }

        let collapsedPointer = collapsedBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let collapsedIndex = collapsedPointer[0]

        guard let collapseCommandBuffer = commandQueue.makeCommandBuffer(),
              let collapseEncoder = collapseCommandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        dispatchFullStateKernel(encoder: collapseEncoder, pipeline: pipelines.collapseState, state: state) { encoder in
            var index = collapsedIndex
            encoder.setBytes(&index, length: MemoryLayout<UInt32>.stride, index: 2)
        }

        collapseEncoder.endEncoding()
        collapseCommandBuffer.commit()
        collapseCommandBuffer.waitUntilCompleted()
        if let error = collapseCommandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }

        return Int(collapsedIndex)
    }

    public func executeProbabilityKernel(on state: StateVector, outputBuffer: MTLBuffer) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        dispatchFullStateKernel(encoder: computeEncoder, pipeline: pipelines.probabilities, state: state) { encoder in
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }
}
