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
    case circuitNotUnitaryOnly

}

public struct Pipelines {

    let hadamard: MTLComputePipelineState
    let pauliX: MTLComputePipelineState
    let pauliY: MTLComputePipelineState
    let pauliZ: MTLComputePipelineState
    let phaseS: MTLComputePipelineState
    let phaseT: MTLComputePipelineState
    let cnot: MTLComputePipelineState
    let ccx: MTLComputePipelineState
    let rotX: MTLComputePipelineState
    let rotY: MTLComputePipelineState
    let rotZ: MTLComputePipelineState

    let probabilities: MTLComputePipelineState
    let maskedPopulationReduce: MTLComputePipelineState
    let prefixSum: MTLComputePipelineState
    let collapseSearch: MTLComputePipelineState
    let collapseState: MTLComputePipelineState
    let partialCollapse: MTLComputePipelineState
    let resetQubit: MTLComputePipelineState
    let amplitudeDampingJump: MTLComputePipelineState
    let amplitudeDampingNoJump: MTLComputePipelineState
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

        guard let sFunc = library.makeFunction(name: "s_gate") else { throw QuantumEngineError.functionNotFound("s_gate") }
        self.phaseS = try device.makeComputePipelineState(function: sFunc)

        guard let tFunc = library.makeFunction(name: "t_gate") else { throw QuantumEngineError.functionNotFound("t_gate") }
        self.phaseT = try device.makeComputePipelineState(function: tFunc)

        guard let cxFunc = library.makeFunction(name: "cnot_gate") else { throw QuantumEngineError.functionNotFound("cnot_gate") }
        self.cnot = try device.makeComputePipelineState(function: cxFunc)

        guard let ccxFunc = library.makeFunction(name: "ccx_gate") else { throw QuantumEngineError.functionNotFound("ccx_gate") }
        self.ccx = try device.makeComputePipelineState(function: ccxFunc)

        guard let rxFunc = library.makeFunction(name: "rx_gate") else { throw QuantumEngineError.functionNotFound("rx_gate") }
        self.rotX = try device.makeComputePipelineState(function: rxFunc)

        guard let ryFunc = library.makeFunction(name: "ry_gate") else { throw QuantumEngineError.functionNotFound("ry_gate") }
        self.rotY = try device.makeComputePipelineState(function: ryFunc)

        guard let rzFunc = library.makeFunction(name: "rz_gate") else { throw QuantumEngineError.functionNotFound("rz_gate") }
        self.rotZ = try device.makeComputePipelineState(function: rzFunc)

        guard let probFunc = library.makeFunction(name: "compute_probabilities") else { throw QuantumEngineError.functionNotFound("compute_probabilities") }
        self.probabilities = try device.makeComputePipelineState(function: probFunc)

        guard let maskedPopFunc = library.makeFunction(name: "masked_population_reduce") else { throw QuantumEngineError.functionNotFound("masked_population_reduce") }
        self.maskedPopulationReduce = try device.makeComputePipelineState(function: maskedPopFunc)

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

        guard let ampDampingJumpFunc = library.makeFunction(name: "amplitude_damping_jump") else { throw QuantumEngineError.functionNotFound("amplitude_damping_jump") }
        self.amplitudeDampingJump = try device.makeComputePipelineState(function: ampDampingJumpFunc)

        guard let ampDampingNoJumpFunc = library.makeFunction(name: "amplitude_damping_no_jump") else { throw QuantumEngineError.functionNotFound("amplitude_damping_no_jump") }
        self.amplitudeDampingNoJump = try device.makeComputePipelineState(function: ampDampingNoJumpFunc)

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
        _ = try executeRNG(circuit, on: state, rng: &rng, noise: nil)
    }

    /// Applies a unitary-only circuit to many states in one GPU command buffer.
    public func executeUnitaryBatch(
        _ circuit: QuantumCircuit,
        on states: [StateVector]
    ) throws {
        guard !states.isEmpty else { return }

        guard circuit.isUnitaryOnly else {
            throw QuantumEngineError.circuitNotUnitaryOnly
        }

        let qubitCount = circuit.qubitCount
        guard states.allSatisfy({ $0.qubitCount == qubitCount }) else {
            throw QuantumEngineError.qubitCountMismatch(circuit: qubitCount, state: states[0].qubitCount)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        for gate in circuit.gates {
            for state in states {
                encodeUnitaryGate(gate, encoder: computeEncoder, state: state)
            }
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    @discardableResult
    public func executeRNG(
        _ circuit: QuantumCircuit,
        on state: StateVector,
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil
    ) throws -> CircuitExecutionResult {
        guard circuit.qubitCount == state.qubitCount else {
            throw QuantumEngineError.qubitCountMismatch(circuit: circuit.qubitCount, state: state.qubitCount)
        }

        let noiseEnabled = noise?.hasGateNoise == true
        var measurementOutcomes: [[Int]] = []
        var pendingUnitaryGates: [Gate] = []

        for gate in circuit.gates {
            switch gate {
            case .measure(let qubits):
                try flushUnitaryGates(pendingUnitaryGates, on: state)
                pendingUnitaryGates.removeAll(keepingCapacity: true)

                let outcome = try executePartialMeasurementCollapse(
                    on: state,
                    qubits: qubits,
                    rng: &rng,
                    noise: noise
                )
                measurementOutcomes.append(measuredBits(outcome: outcome, qubits: qubits))

            case .reset(let qubit):
                try flushUnitaryGates(pendingUnitaryGates, on: state)
                pendingUnitaryGates.removeAll(keepingCapacity: true)
                try executeResetQubit(on: state, qubit: qubit)

            default:
                if noiseEnabled, let noise {
                    try flushUnitaryGates(pendingUnitaryGates, on: state)
                    pendingUnitaryGates.removeAll(keepingCapacity: true)
                    try executeUnitaryGate(gate, on: state)
                    if noise.appliesDepolarizing {
                        try applyDepolarizingNoise(
                            after: gate,
                            on: state,
                            probability: noise.depolarizingProbability,
                            rng: &rng
                        )
                    }
                    if noise.appliesAmplitudeDamping {
                        try applyAmplitudeDamping(
                            after: gate,
                            on: state,
                            probability: noise.effectiveAmplitudeDampingProbability,
                            rng: &rng
                        )
                    }
                    if noise.appliesPhaseDamping {
                        try applyPhaseDamping(
                            after: gate,
                            on: state,
                            flipProbability: noise.effectivePhaseFlipProbability,
                            rng: &rng
                        )
                    }
                } else {
                    pendingUnitaryGates.append(gate)
                }
            }
        }

        try flushUnitaryGates(pendingUnitaryGates, on: state)
        return CircuitExecutionResult(measurementOutcomes: measurementOutcomes)
    }

    private func executeUnitaryGate(_ gate: Gate, on state: StateVector) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        encodeUnitaryGate(gate, encoder: computeEncoder, state: state)

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    private func applyDepolarizingNoise(
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
    private func applyAmplitudeDamping(
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

    private func dispatchAmplitudeDamping(
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
    /// 256 squared amplitudes restricted to the `qubit == |1⟩` subspace and atomically accumulates
    /// its partial into a single result, avoiding an `O(2ⁿ)` host-side scan.
    private func qubitOnePopulation(on state: StateVector, qubit: Int) throws -> QFloat {
        let resultBuffer = try makeSharedBuffer(length: MemoryLayout<QFloat>.stride)
        resultBuffer.contents().assumingMemoryBound(to: QFloat.self)[0] = 0

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        let blockSize = Self.scanBlockSize
        let elementCount = state.stateCount
        let blockCount = max((elementCount + blockSize - 1) / blockSize, 1)

        var targetQubit = UInt32(qubit)
        var countValue = UInt32(elementCount)

        computeEncoder.setComputePipelineState(pipelines.maskedPopulationReduce)
        computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
        computeEncoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 3)
        computeEncoder.setBuffer(resultBuffer, offset: 0, index: 4)

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

        return resultBuffer.contents().assumingMemoryBound(to: QFloat.self)[0]
    }

    /// Phase damping realized as its equivalent phase-flip (random Z) channel.
    ///
    /// The phase damping channel of strength λ is exactly the phase-flip channel
    /// `ρ → (1-p)ρ + p·ZρZ` with `p = (1-√(1-λ))/2`. In the trajectory picture this means
    /// applying a Pauli-Z with probability `flipProbability` to each affected qubit.
    private func applyPhaseDamping(
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

    private func flushUnitaryGates(_ gates: [Gate], on state: StateVector) throws {
        guard !gates.isEmpty else { return }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        for gate in gates {
            encodeUnitaryGate(gate, encoder: computeEncoder, state: state)
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    private func encodeUnitaryGate(
        _ gate: Gate,
        encoder: MTLComputeCommandEncoder,
        state: StateVector
    ) {
        switch gate {
        case .h(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.hadamard, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .x(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.pauliX, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .y(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.pauliY, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .z(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.pauliZ, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .s(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.phaseS, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .t(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.phaseT, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .cx(let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cnot, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .ccx(let control1, let control2, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.ccx, state: state) { encoder in
                var qubits = SIMD3<UInt32>(x: UInt32(control1), y: UInt32(control2), z: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
            }

        case .rz(let theta, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.rotZ, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .rx(let theta, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.rotX, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .ry(let theta, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.rotY, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .measure, .reset:
            break
        }
    }

    public func executePartialMeasurementCollapse(
        on state: StateVector,
        qubits: [Int],
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil
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
        let diceRoll = rng.nextUnitFloat()
        var collapsedIndex = try executeMeasurementCollapse(on: state, diceRoll: diceRoll)
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
