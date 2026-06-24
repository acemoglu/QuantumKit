//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation
import Metal

public enum QuantumEngineError: Error {

    case deviceNotFound
    case libraryNotFound
    case functionNotFound(String)
    case pipelineStateCreationFailed
    case commandBufferCreationFailed
    case commandQueueCreationFailed
    case qubitCountMismatch(circuit: Int, state: Int)
    case bufferAllocationFailed(requiredBytes: Int)
    case commandBufferExecutionFailed(underlying: Error?)
    case prefixSumBufferLevelMissing(level: Int)
    case zeroStateNorm

}

public struct Pipelines {

    let hadamard: MTLComputePipelineState
    let pauliX: MTLComputePipelineState
    let pauliY: MTLComputePipelineState
    let pauliZ: MTLComputePipelineState
    let cnot: MTLComputePipelineState
    let ccx: MTLComputePipelineState
    let rotX: MTLComputePipelineState
    let rotZ: MTLComputePipelineState

    let probabilities: MTLComputePipelineState
    let prefixSum: MTLComputePipelineState
    let collapseSearch: MTLComputePipelineState
    let collapseState: MTLComputePipelineState
    let partialCollapse: MTLComputePipelineState
    let resetQubit: MTLComputePipelineState
    let normalize: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let hFunc = library.makeFunction(name: "hadamard_gate") else { throw QuantumEngineError.functionNotFound("hadamard_gate") }
        self.hadamard = try device.makeComputePipelineState(function: hFunc)

        guard let xFunc = library.makeFunction(name: "pauli_x_gate") else { throw QuantumEngineError.functionNotFound("pauli_x_gate") }
        self.pauliX = try device.makeComputePipelineState(function: xFunc)

        guard let yFunc = library.makeFunction(name: "pauli_y_gate") else { throw QuantumEngineError.functionNotFound("pauli_y_gate") }
        self.pauliY = try device.makeComputePipelineState(function: yFunc)

        guard let zFunc = library.makeFunction(name: "pauli_z_gate") else { throw QuantumEngineError.functionNotFound("pauli_z_gate") }
        self.pauliZ = try device.makeComputePipelineState(function: zFunc)

        guard let cxFunc = library.makeFunction(name: "cnot_gate") else { throw QuantumEngineError.functionNotFound("cnot_gate") }
        self.cnot = try device.makeComputePipelineState(function: cxFunc)

        guard let ccxFunc = library.makeFunction(name: "ccx_gate") else { throw QuantumEngineError.functionNotFound("ccx_gate") }
        self.ccx = try device.makeComputePipelineState(function: ccxFunc)

        guard let rxFunc = library.makeFunction(name: "rx_gate") else { throw QuantumEngineError.functionNotFound("rx_gate") }
        self.rotX = try device.makeComputePipelineState(function: rxFunc)

        guard let rzFunc = library.makeFunction(name: "rz_gate") else { throw QuantumEngineError.functionNotFound("rz_gate") }
        self.rotZ = try device.makeComputePipelineState(function: rzFunc)

        guard let probFunc = library.makeFunction(name: "compute_probabilities") else { throw QuantumEngineError.functionNotFound("compute_probabilities") }
        self.probabilities = try device.makeComputePipelineState(function: probFunc)

        guard let prefixSumFunc = library.makeFunction(name: "prefix_sum_probabilities") else { throw QuantumEngineError.functionNotFound("prefix_sum_probabilities") }
        self.prefixSum = try device.makeComputePipelineState(function: prefixSumFunc)

        guard let collapseFunc = library.makeFunction(name: "find_collapsed_state") else { throw QuantumEngineError.functionNotFound("find_collapsed_state") }
        self.collapseSearch = try device.makeComputePipelineState(function: collapseFunc)

        guard let collapseStateFunc = library.makeFunction(name: "collapse_state_vector") else { throw QuantumEngineError.functionNotFound("collapse_state_vector") }
        self.collapseState = try device.makeComputePipelineState(function: collapseStateFunc)

        guard let partialCollapseFunc = library.makeFunction(name: "partial_collapse_state_vector") else { throw QuantumEngineError.functionNotFound("partial_collapse_state_vector") }
        self.partialCollapse = try device.makeComputePipelineState(function: partialCollapseFunc)

        guard let resetQubitFunc = library.makeFunction(name: "reset_qubit_state_vector") else { throw QuantumEngineError.functionNotFound("reset_qubit_state_vector") }
        self.resetQubit = try device.makeComputePipelineState(function: resetQubitFunc)

        guard let normalizeFunc = library.makeFunction(name: "normalize_state_vector") else { throw QuantumEngineError.functionNotFound("normalize_state_vector") }
        self.normalize = try device.makeComputePipelineState(function: normalizeFunc)

    }
}

public class QuantumEngine {

    private static let scanBlockSize = 256

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelines: Pipelines

    private static func loadMetalLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let library = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return library
        }

        if let library = device.makeDefaultLibrary() {
            return library
        }

        let bundle = Bundle.module
        let candidateURLs = [
            bundle.url(forResource: "Gates", withExtension: "metal", subdirectory: "Metal"),
            bundle.url(forResource: "Gates", withExtension: "metal"),
        ]

        guard let metalURL = candidateURLs.compactMap({ $0 }).first else {
            throw QuantumEngineError.libraryNotFound
        }

        let source = try String(contentsOf: metalURL, encoding: .utf8)
        return try device.makeLibrary(source: source, options: nil)
    }

    public init() throws {
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else { throw QuantumEngineError.deviceNotFound }
        self.device = defaultDevice

        guard let queue = device.makeCommandQueue() else { throw QuantumEngineError.commandQueueCreationFailed }
        self.commandQueue = queue

        let library = try Self.loadMetalLibrary(device: device)
        self.pipelines = try Pipelines(device: device, library: library)
    }

    private func dispatchPairwiseGate(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        state: StateVector,
        configure: (MTLComputeCommandEncoder) -> Void
    ) {
        let pairCount = state.stateCount / 2

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        configure(encoder)

        let threadsPerGrid = MTLSize(width: pairCount, height: 1, depth: 1)
        let executionWidth = pipeline.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: min(executionWidth, pairCount), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    private func dispatchFullStateKernel(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        state: StateVector,
        configure: (MTLComputeCommandEncoder) -> Void
    ) {
        let threadCount = max(state.stateCount, 1)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        configure(encoder)

        let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
        let executionWidth = pipeline.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: min(executionWidth, threadCount), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    public func execute(_ circuit: QuantumCircuit, on state: StateVector) throws {
        var rng: QuantumRNG = .hardware
        _ = try executeRNG(circuit, on: state, rng: &rng)
    }

    @discardableResult
    public func executeRNG(
        _ circuit: QuantumCircuit,
        on state: StateVector,
        rng: inout QuantumRNG
    ) throws -> CircuitExecutionResult {
        guard circuit.qubitCount == state.qubitCount else {
            throw QuantumEngineError.qubitCountMismatch(circuit: circuit.qubitCount, state: state.qubitCount)
        }

        var measurementOutcomes: [[Int]] = []
        var pendingUnitaryGates: [Gate] = []

        for gate in circuit.gates {
            switch gate {
            case .measure(let qubits):
                try flushUnitaryGates(pendingUnitaryGates, on: state)
                pendingUnitaryGates.removeAll(keepingCapacity: true)

                let outcome = try executePartialMeasurementCollapse(on: state, qubits: qubits, rng: &rng)
                measurementOutcomes.append(measuredBits(outcome: outcome, qubits: qubits))

            case .reset(let qubit):
                try flushUnitaryGates(pendingUnitaryGates, on: state)
                pendingUnitaryGates.removeAll(keepingCapacity: true)
                try executeResetQubit(on: state, qubit: qubit)

            default:
                pendingUnitaryGates.append(gate)
            }
        }

        try flushUnitaryGates(pendingUnitaryGates, on: state)
        return CircuitExecutionResult(measurementOutcomes: measurementOutcomes)
    }

    private func flushUnitaryGates(_ gates: [Gate], on state: StateVector) throws {
        guard !gates.isEmpty else { return }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        for gate in gates {
            switch gate {
            case .h(let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.hadamard, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                }

            case .x(let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.pauliX, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                }

            case .y(let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.pauliY, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                }

            case .z(let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.pauliZ, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                }

            case .cx(let control, let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.cnot, state: state) { encoder in
                    var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                    encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                }

            case .ccx(let control1, let control2, let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.ccx, state: state) { encoder in
                    var qubits = SIMD3<UInt32>(x: UInt32(control1), y: UInt32(control2), z: UInt32(target))
                    encoder.setBytes(&qubits, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
                }

            case .rz(let theta, let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.rotZ, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                    var thetaValue = Float(theta)
                    encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
                }

            case .rx(let theta, let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.rotX, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                    var thetaValue = Float(theta)
                    encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
                }

            case .measure, .reset:
                break
            }
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    public func executePartialMeasurementCollapse(
        on state: StateVector,
        qubits: [Int],
        rng: inout QuantumRNG
    ) throws -> Int {
        guard !qubits.isEmpty else {
            throw QuantumMeasurementError.emptyQubitSelection
        }

        let diceRoll = rng.nextUnitFloat()
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
        return outcome
    }

    public func executeResetQubit(on state: StateVector, qubit: Int) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        var targetQubit = UInt32(qubit)
        dispatchFullStateKernel(encoder: computeEncoder, pipeline: pipelines.resetQubit, state: state) { encoder in
            encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }

        try normalizeState(on: state)
    }

    private func normalizeState(on state: StateVector) throws {
        let stateCount = state.stateCount
        let probabilityBytes = stateCount * MemoryLayout<QFloat>.stride
        let probBuffer = try makeSharedBuffer(length: probabilityBytes)
        var auxiliaryBuffers: [MTLBuffer] = []

        var currentCount = stateCount
        while currentCount > 1 {
            let blockCount = max((currentCount + Self.scanBlockSize - 1) / Self.scanBlockSize, 1)
            let byteCount = blockCount * MemoryLayout<QFloat>.stride
            auxiliaryBuffers.append(try makeSharedBuffer(length: byteCount))
            currentCount = blockCount
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        dispatchFullStateKernel(encoder: computeEncoder, pipeline: pipelines.probabilities, state: state) { encoder in
            encoder.setBuffer(probBuffer, offset: 0, index: 2)
        }

        if stateCount > 1 {
            try encodeInclusivePrefixSum(
                encoder: computeEncoder,
                buffer: probBuffer,
                elementCount: stateCount,
                auxiliaryBuffers: auxiliaryBuffers
            )
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }

        let probabilityPointer = probBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let totalProbability = stateCount == 1 ? probabilityPointer[0] : probabilityPointer[stateCount - 1]
        guard totalProbability > 0 else {
            throw QuantumEngineError.zeroStateNorm
        }

        let invNorm = 1 / sqrt(totalProbability)

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

    private func samplePartialOutcome(on state: StateVector, qubits: [Int], diceRoll: Float) throws -> Int {
        let byteCount = state.stateCount * MemoryLayout<QFloat>.stride
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw QuantumEngineError.bufferAllocationFailed(requiredBytes: byteCount)
        }

        try executeProbabilityKernel(on: state, outputBuffer: buffer)
        let pointer = buffer.contents().assumingMemoryBound(to: QFloat.self)
        let fullDistribution = Array(UnsafeBufferPointer(start: pointer, count: state.stateCount))

        var marginal = [QFloat](repeating: 0, count: 1 << qubits.count)
        for (stateIndex, probability) in fullDistribution.enumerated() {
            let outcome = partialOutcomeIndex(stateIndex: stateIndex, qubits: qubits)
            marginal[outcome] += probability
        }

        var cumulative: QFloat = 0
        for (index, probability) in marginal.enumerated() {
            cumulative += probability
            if diceRoll < cumulative {
                return index
            }
        }

        return marginal.count - 1
    }

    private func partialOutcomeIndex(stateIndex: Int, qubits: [Int]) -> Int {
        var outcome = 0
        for (position, qubit) in qubits.enumerated() {
            let bit = (stateIndex >> qubit) & 1
            outcome |= bit << position
        }
        return outcome
    }

    private func measuredBits(outcome: Int, qubits: [Int]) -> [Int] {
        qubits.indices.map { position in
            (outcome >> position) & 1
        }
    }

    private func makeSharedBuffer(length: Int) throws -> MTLBuffer {
        guard length > 0,
              let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw QuantumEngineError.bufferAllocationFailed(requiredBytes: max(length, 0))
        }
        return buffer
    }

    private func dispatchPrefixSumPhase(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        dataBuffer: MTLBuffer,
        blockSumsBuffer: MTLBuffer,
        elementCount: Int,
        phase: UInt32
    ) {
        let blockSize = Self.scanBlockSize
        let threadCount = max(elementCount, 1)
        var countValue = UInt32(elementCount)
        var phaseValue = phase

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(dataBuffer, offset: 0, index: 0)
        encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&phaseValue, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: min(blockSize, threadCount), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    private func encodeInclusivePrefixSum(
        encoder: MTLComputeCommandEncoder,
        buffer: MTLBuffer,
        elementCount: Int,
        auxiliaryBuffers: [MTLBuffer],
        level: Int = 0
    ) throws {
        guard elementCount > 1 else { return }
        guard level < auxiliaryBuffers.count else {
            throw QuantumEngineError.prefixSumBufferLevelMissing(level: level)
        }

        let blockSize = Self.scanBlockSize
        let numBlocks = (elementCount + blockSize - 1) / blockSize
        let blockSumsBuffer = auxiliaryBuffers[level]

        dispatchPrefixSumPhase(
            encoder: encoder,
            pipeline: pipelines.prefixSum,
            dataBuffer: buffer,
            blockSumsBuffer: blockSumsBuffer,
            elementCount: elementCount,
            phase: 0
        )

        if numBlocks > 1 {
            try encodeInclusivePrefixSum(
                encoder: encoder,
                buffer: blockSumsBuffer,
                elementCount: numBlocks,
                auxiliaryBuffers: auxiliaryBuffers,
                level: level + 1
            )

            dispatchPrefixSumPhase(
                encoder: encoder,
                pipeline: pipelines.prefixSum,
                dataBuffer: buffer,
                blockSumsBuffer: blockSumsBuffer,
                elementCount: elementCount,
                phase: 2
            )
        }
    }

    public func executeMeasurementCollapse(on state: StateVector, diceRoll: Float) throws -> Int {
        let stateCount = state.stateCount
        let probabilityBytes = stateCount * MemoryLayout<QFloat>.stride

        let probBuffer = try makeSharedBuffer(length: probabilityBytes)
        let collapsedBuffer = try makeSharedBuffer(length: MemoryLayout<UInt32>.stride)
        var auxiliaryBuffers: [MTLBuffer] = []

        var currentCount = stateCount
        while currentCount > 1 {
            let blockCount = max((currentCount + Self.scanBlockSize - 1) / Self.scanBlockSize, 1)
            let byteCount = blockCount * MemoryLayout<QFloat>.stride
            auxiliaryBuffers.append(try makeSharedBuffer(length: byteCount))
            currentCount = blockCount
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        dispatchFullStateKernel(encoder: computeEncoder, pipeline: pipelines.probabilities, state: state) { encoder in
            encoder.setBuffer(probBuffer, offset: 0, index: 2)
        }

        try encodeInclusivePrefixSum(
            encoder: computeEncoder,
            buffer: probBuffer,
            elementCount: stateCount,
            auxiliaryBuffers: auxiliaryBuffers
        )

        var diceRollValue = diceRoll
        var elementCount = UInt32(stateCount)
        computeEncoder.setComputePipelineState(pipelines.collapseSearch)
        computeEncoder.setBuffer(probBuffer, offset: 0, index: 0)
        computeEncoder.setBytes(&diceRollValue, length: MemoryLayout<Float>.stride, index: 1)
        computeEncoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.setBuffer(collapsedBuffer, offset: 0, index: 3)
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
