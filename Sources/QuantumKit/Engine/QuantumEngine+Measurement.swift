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
        // The reset can finish with an asynchronously-committed X gate; drain once so a direct host
        // read of the state buffers after this standalone call sees the settled amplitudes.
        try drainPipeline()
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
        let maxFloat32 = Double(Float32.greatestFiniteMagnitude)
        let safeInvNorm = min(invNorm, maxFloat32)

        guard let normalizeCommandBuffer = commandQueue.makeCommandBuffer(),
              let normalizeEncoder = normalizeCommandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        var invNormValue = QFloat(safeInvNorm)
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

    /// Largest measured-qubit count routed through the GPU marginal reduction. The leaf stage emits
    /// `blockCount · 2ᵏ` partials, so capping `k ≤ 8` (`binCount ≤ 256`) keeps that scratch at or
    /// below one probability buffer (`2ⁿ`). Wider simultaneous measurements (rare) fall back to the
    /// host fold, which is `O(2ⁿ)` either way — no regression versus the previous behavior.
    static let maxGPUMarginalQubits = 8

    func samplePartialOutcome(on state: StateVector, qubits: [Int], diceRoll: Double) throws -> Int {
        if qubits.count <= Self.maxGPUMarginalQubits {
            return try samplePartialOutcomeOnGPU(on: state, qubits: qubits, diceRoll: diceRoll)
        }
        return try samplePartialOutcomeOnHost(on: state, qubits: qubits, diceRoll: diceRoll)
    }

    /// GPU marginal sampler: buckets per-state Born probabilities into the `2ᵏ` outcome bins with a
    /// single leaf histogram pass and a compensated (double-single) cross-block reduction, replacing
    /// the former `O(2ⁿ)` host fold over every amplitude with a `2ᵏ`-element host CDF search.
    private func samplePartialOutcomeOnGPU(on state: StateVector, qubits: [Int], diceRoll: Double) throws -> Int {
        let blockSize = Self.scanBlockSize
        let stateCount = state.stateCount
        let binCount = 1 << qubits.count
        var blockCount = max((stateCount + blockSize - 1) / blockSize, 1)

        let qubitIndices = qubits.map { UInt32($0) }
        let qubitBuffer = try makeSharedBuffer(length: qubitIndices.count * MemoryLayout<UInt32>.stride)
        qubitBuffer.contents().copyMemory(
            from: qubitIndices,
            byteCount: qubitIndices.count * MemoryLayout<UInt32>.stride
        )

        // hiA holds the (largest) leaf histogram; the reduction outputs shrink by 256× each pass, so
        // the lo/ping-pong buffers only need the first-reduce footprint.
        let firstReduceBlocks = max((blockCount + blockSize - 1) / blockSize, 1)
        let leafBytes = blockCount * binCount * MemoryLayout<QFloat>.stride
        let reduceBytes = firstReduceBlocks * binCount * MemoryLayout<QFloat>.stride
        let hiA = try bufferPool.acquire(length: leafBytes)
        let loA = try bufferPool.acquire(length: reduceBytes)
        let hiB = try bufferPool.acquire(length: reduceBytes)
        let loB = try bufferPool.acquire(length: reduceBytes)
        // Safe to recycle at scope exit: the command buffer below is awaited synchronously.
        defer {
            bufferPool.release(hiA)
            bufferPool.release(loA)
            bufferPool.release(hiB)
            bufferPool.release(loB)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        var measuredCount = UInt32(qubits.count)
        var binCountValue = UInt32(binCount)
        var elementCount = UInt32(stateCount)
        encoder.setComputePipelineState(pipelines.marginalLeafHistogram)
        encoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        encoder.setBuffer(qubitBuffer, offset: 0, index: 2)
        encoder.setBytes(&measuredCount, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&binCountValue, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBuffer(hiA, offset: 0, index: 6)
        encoder.setThreadgroupMemoryLength(binCount * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: blockCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1)
        )

        var readHi = hiA
        var readLo = loA
        var writeHi = hiB
        var writeLo = loB
        var producedLo = false

        while blockCount > 1 {
            let nextBlockCount = (blockCount + blockSize - 1) / blockSize
            var inBlockCountValue = UInt32(blockCount)
            var binCountReduce = UInt32(binCount)
            // WARNING: Pooled buffers contain uninitialized memory from prior runs. The readLoFlag MUST be preserved to prevent probability corruption.
            var readLoFlag: UInt32 = producedLo ? 1 : 0
            encoder.setComputePipelineState(pipelines.marginalPartialsReduce)
            encoder.setBuffer(readHi, offset: 0, index: 0)
            encoder.setBuffer(readLo, offset: 0, index: 1)
            encoder.setBytes(&inBlockCountValue, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&binCountReduce, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&readLoFlag, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBuffer(writeHi, offset: 0, index: 5)
            encoder.setBuffer(writeLo, offset: 0, index: 6)
            let outElements = nextBlockCount * binCount
            encoder.dispatchThreads(
                MTLSize(width: outElements, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(blockSize, outElements), height: 1, depth: 1)
            )

            swap(&readHi, &writeHi)
            swap(&readLo, &writeLo)
            producedLo = true
            blockCount = nextBlockCount
        }

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }

        let hiPointer = readHi.contents().assumingMemoryBound(to: QFloat.self)
        let loPointer = readLo.contents().assumingMemoryBound(to: QFloat.self)
        var cumulative = 0.0
        for bin in 0..<binCount {
            let probability = Double(hiPointer[bin]) + (producedLo ? Double(loPointer[bin]) : 0.0)
            cumulative += probability
            if diceRoll < cumulative {
                return bin
            }
        }
        return binCount - 1
    }

    /// Host-side marginal fold for measurements too wide for the GPU path (`k > maxGPUMarginalQubits`).
    /// Accumulates in `Double` so the `O(2ⁿ)` sum over per-state float probabilities does not lose
    /// sub-ulp contributions to Float32 cancellation.
    private func samplePartialOutcomeOnHost(on state: StateVector, qubits: [Int], diceRoll: Double) throws -> Int {
        let byteCount = state.stateCount * MemoryLayout<QFloat>.stride
        let buffer = try bufferPool.acquire(length: byteCount)
        defer { bufferPool.release(buffer) }

        try executeProbabilityKernel(on: state, outputBuffer: buffer)
        let pointer = buffer.contents().assumingMemoryBound(to: QFloat.self)

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

    /// Compensated GPU evaluation of ⟨ψ|P|ψ⟩ for a general Pauli tensor product `P`.
    ///
    /// `P|j⟩ = phase(j)·|j ⊕ flipMask⟩`, where `phase(j) = (−1)^popcount(j & signMask)·phaseBase`
    /// and `phaseBase = phaseBaseReal + i·phaseBaseImag = i^{#Y}` is the global complex factor.
    /// Each basis state's real contribution `Re[ conj(a_{j⊕flipMask})·phase(j)·a_j ]` is summed by a
    /// double-single leaf pass plus the ping-pong compensated reduction, so the host reads a single
    /// `(hi, lo)` pair instead of folding `O(2ⁿ)` amplitudes. Mirrors ``trajectoryJumpProbability``.
    func pauliExpectation(
        on state: StateVector,
        flipMask: Int,
        signMask: Int,
        phaseBaseReal: QFloat,
        phaseBaseImag: QFloat
    ) throws -> QFloat {
        let blockSize = Self.scanBlockSize
        let stateCount = state.stateCount
        var partialCount = max((stateCount + blockSize - 1) / blockSize, 1)

        let maxPartials = partialCount
        let bytes = maxPartials * MemoryLayout<QFloat>.stride
        let hiA = try bufferPool.acquire(length: bytes)
        let loA = try bufferPool.acquire(length: bytes)
        let hiB = try bufferPool.acquire(length: bytes)
        let loB = try bufferPool.acquire(length: bytes)
        // Safe to recycle at scope exit: the command buffer below is awaited synchronously.
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

        var flipMaskValue = UInt32(flipMask)
        var signMaskValue = UInt32(signMask)
        var phaseReal = phaseBaseReal
        var phaseImag = phaseBaseImag
        var elementCount = UInt32(stateCount)
        computeEncoder.setComputePipelineState(pipelines.pauliExpectationPartial)
        computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&flipMaskValue, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.setBytes(&signMaskValue, length: MemoryLayout<UInt32>.stride, index: 3)
        computeEncoder.setBytes(&phaseReal, length: MemoryLayout<QFloat>.stride, index: 4)
        computeEncoder.setBytes(&phaseImag, length: MemoryLayout<QFloat>.stride, index: 5)
        computeEncoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.stride, index: 6)
        computeEncoder.setBuffer(hiA, offset: 0, index: 7)
        computeEncoder.setBuffer(loA, offset: 0, index: 8)
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

        // Combine the surviving (hi, lo) pair in Double so the compensated GPU sum is not
        // re-truncated to Float before being returned.
        let hiPtr = readHi.contents().assumingMemoryBound(to: QFloat.self)
        let loPtr = readLo.contents().assumingMemoryBound(to: QFloat.self)
        return QFloat(Double(hiPtr[0]) + Double(loPtr[0]))
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
