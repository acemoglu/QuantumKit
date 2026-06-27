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

public struct Pipelines: @unchecked Sendable {

    let hadamard: MTLComputePipelineState
    let pauliX: MTLComputePipelineState
    let pauliY: MTLComputePipelineState
    let pauliZ: MTLComputePipelineState
    let phaseS: MTLComputePipelineState
    let phaseT: MTLComputePipelineState
    let phaseSDagger: MTLComputePipelineState
    let phaseTDagger: MTLComputePipelineState
    let sqrtX: MTLComputePipelineState
    let sqrtXDagger: MTLComputePipelineState
    let phase: MTLComputePipelineState
    let universal: MTLComputePipelineState
    let cnot: MTLComputePipelineState
    let cz: MTLComputePipelineState
    let swapGate: MTLComputePipelineState
    let ccx: MTLComputePipelineState
    let rotX: MTLComputePipelineState
    let rotY: MTLComputePipelineState
    let rotZ: MTLComputePipelineState
    let cRotX: MTLComputePipelineState
    let cRotY: MTLComputePipelineState
    let cRotZ: MTLComputePipelineState
    let cPhase: MTLComputePipelineState
    let mcx: MTLComputePipelineState
    let mcz: MTLComputePipelineState

    let probabilities: MTLComputePipelineState
    let maskedPopulationReduce: MTLComputePipelineState
    let prefixSum: MTLComputePipelineState
    let prefixSumNaive: MTLComputePipelineState
    let collapseSearch: MTLComputePipelineState
    let collapseState: MTLComputePipelineState
    let partialCollapse: MTLComputePipelineState
    let resetQubit: MTLComputePipelineState
    let amplitudeDampingJump: MTLComputePipelineState
    let amplitudeDampingNoJump: MTLComputePipelineState
    let normalize: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary, preciseLibrary: MTLLibrary) throws {
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

        guard let sdgFunc = library.makeFunction(name: "s_dagger_gate") else { throw QuantumEngineError.functionNotFound("s_dagger_gate") }
        self.phaseSDagger = try device.makeComputePipelineState(function: sdgFunc)

        guard let tdgFunc = library.makeFunction(name: "t_dagger_gate") else { throw QuantumEngineError.functionNotFound("t_dagger_gate") }
        self.phaseTDagger = try device.makeComputePipelineState(function: tdgFunc)

        guard let sxFunc = library.makeFunction(name: "sx_gate") else { throw QuantumEngineError.functionNotFound("sx_gate") }
        self.sqrtX = try device.makeComputePipelineState(function: sxFunc)

        guard let sxdgFunc = library.makeFunction(name: "sx_dagger_gate") else { throw QuantumEngineError.functionNotFound("sx_dagger_gate") }
        self.sqrtXDagger = try device.makeComputePipelineState(function: sxdgFunc)

        guard let phaseFunc = library.makeFunction(name: "phase_gate") else { throw QuantumEngineError.functionNotFound("phase_gate") }
        self.phase = try device.makeComputePipelineState(function: phaseFunc)

        guard let uFunc = library.makeFunction(name: "u_gate") else { throw QuantumEngineError.functionNotFound("u_gate") }
        self.universal = try device.makeComputePipelineState(function: uFunc)

        guard let cxFunc = library.makeFunction(name: "cnot_gate") else { throw QuantumEngineError.functionNotFound("cnot_gate") }
        self.cnot = try device.makeComputePipelineState(function: cxFunc)

        guard let czFunc = library.makeFunction(name: "cz_gate") else { throw QuantumEngineError.functionNotFound("cz_gate") }
        self.cz = try device.makeComputePipelineState(function: czFunc)

        guard let swapFunc = library.makeFunction(name: "swap_gate") else { throw QuantumEngineError.functionNotFound("swap_gate") }
        self.swapGate = try device.makeComputePipelineState(function: swapFunc)

        guard let ccxFunc = library.makeFunction(name: "ccx_gate") else { throw QuantumEngineError.functionNotFound("ccx_gate") }
        self.ccx = try device.makeComputePipelineState(function: ccxFunc)

        guard let rxFunc = library.makeFunction(name: "rx_gate") else { throw QuantumEngineError.functionNotFound("rx_gate") }
        self.rotX = try device.makeComputePipelineState(function: rxFunc)

        guard let ryFunc = library.makeFunction(name: "ry_gate") else { throw QuantumEngineError.functionNotFound("ry_gate") }
        self.rotY = try device.makeComputePipelineState(function: ryFunc)

        guard let rzFunc = library.makeFunction(name: "rz_gate") else { throw QuantumEngineError.functionNotFound("rz_gate") }
        self.rotZ = try device.makeComputePipelineState(function: rzFunc)

        guard let crxFunc = library.makeFunction(name: "crx_gate") else { throw QuantumEngineError.functionNotFound("crx_gate") }
        self.cRotX = try device.makeComputePipelineState(function: crxFunc)

        guard let cryFunc = library.makeFunction(name: "cry_gate") else { throw QuantumEngineError.functionNotFound("cry_gate") }
        self.cRotY = try device.makeComputePipelineState(function: cryFunc)

        guard let crzFunc = library.makeFunction(name: "crz_gate") else { throw QuantumEngineError.functionNotFound("crz_gate") }
        self.cRotZ = try device.makeComputePipelineState(function: crzFunc)

        guard let cphaseFunc = library.makeFunction(name: "cphase_gate") else { throw QuantumEngineError.functionNotFound("cphase_gate") }
        self.cPhase = try device.makeComputePipelineState(function: cphaseFunc)

        guard let mcxFunc = library.makeFunction(name: "mcx_gate") else { throw QuantumEngineError.functionNotFound("mcx_gate") }
        self.mcx = try device.makeComputePipelineState(function: mcxFunc)

        guard let mczFunc = library.makeFunction(name: "mcz_gate") else { throw QuantumEngineError.functionNotFound("mcz_gate") }
        self.mcz = try device.makeComputePipelineState(function: mczFunc)

        guard let probFunc = library.makeFunction(name: "compute_probabilities") else { throw QuantumEngineError.functionNotFound("compute_probabilities") }
        self.probabilities = try device.makeComputePipelineState(function: probFunc)

        guard let maskedPopFunc = library.makeFunction(name: "masked_population_reduce") else { throw QuantumEngineError.functionNotFound("masked_population_reduce") }
        self.maskedPopulationReduce = try device.makeComputePipelineState(function: maskedPopFunc)

        // Precision-critical: compiled with fast math off so the compensated summation is preserved.
        guard let prefixSumFunc = preciseLibrary.makeFunction(name: "prefix_sum_probabilities") else { throw QuantumEngineError.functionNotFound("prefix_sum_probabilities") }
        self.prefixSum = try device.makeComputePipelineState(function: prefixSumFunc)

        // Benchmark-only Float32 baseline (loaded from the fast default library).
        guard let prefixSumNaiveFunc = library.makeFunction(name: "prefix_sum_probabilities_naive") else { throw QuantumEngineError.functionNotFound("prefix_sum_probabilities_naive") }
        self.prefixSumNaive = try device.makeComputePipelineState(function: prefixSumNaiveFunc)

        guard let collapseFunc = preciseLibrary.makeFunction(name: "find_collapsed_state") else { throw QuantumEngineError.functionNotFound("find_collapsed_state") }
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

/// A thread-safe free list of reusable `MTLBuffer`s, keyed by byte length.
///
/// Measurement and normalization need large (2ⁿ-sized) scratch buffers on every call; re-allocating
/// them per shot or per mid-circuit measurement is expensive (~1 GB at 28 qubits). The pool hands
/// out a buffer of the requested size — reused if one is free, freshly allocated otherwise — and
/// takes it back when the caller is done. Because every GPU dispatch here is synchronous
/// (`waitUntilCompleted`), a released buffer is guaranteed idle.
///
/// An acquired buffer holds arbitrary stale contents, so callers must fully overwrite (or clear) it
/// before reading. The free list is lock-guarded so the owning engine stays safe to share across
/// threads.
private final class BufferPool: @unchecked Sendable {
    private let device: MTLDevice
    private let lock = NSLock()
    private var freeBuffersByLength: [Int: [MTLBuffer]] = [:]

    init(device: MTLDevice) {
        self.device = device
    }

    func acquire(length: Int) throws -> MTLBuffer {
        lock.lock()
        if var bucket = freeBuffersByLength[length], let reused = bucket.popLast() {
            freeBuffersByLength[length] = bucket
            lock.unlock()
            return reused
        }
        lock.unlock()

        guard length > 0,
              let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw QuantumEngineError.bufferAllocationFailed(requiredBytes: max(length, 0))
        }
        return buffer
    }

    func release(_ buffer: MTLBuffer) {
        lock.lock()
        freeBuffersByLength[buffer.length, default: []].append(buffer)
        lock.unlock()
    }
}

/// GPU-backed executor for quantum circuits.
///
/// `QuantumEngine` is safe to share across threads: it holds immutable Metal objects (`MTLDevice`,
/// a thread-safe `MTLCommandQueue`, and immutable pipeline states) plus a lock-guarded
/// ``BufferPool`` for scratch reuse. Each call allocates its own command buffers, so distinct
/// ``StateVector`` instances may be executed concurrently from different threads using the same
/// engine.
///
/// - Important: A *single* ``StateVector`` must not be operated on from multiple threads
///   simultaneously; serialize access to a given state yourself if you share one.
public final class QuantumEngine: @unchecked Sendable {

    private static let scanBlockSize = 256

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelines: Pipelines
    private let bufferPool: BufferPool

    private static func loadMetalLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let library = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return library
        }

        if let library = device.makeDefaultLibrary() {
            return library
        }

        let source = try metalShaderSource()
        return try device.makeLibrary(source: source, options: nil)
    }

    /// Compiles the shader source with **fast math disabled** so the compensated (double-single)
    /// summation in `prefix_sum_probabilities`/`find_collapsed_state` survives.
    ///
    /// The default library keeps fast math on for maximum gate throughput; algebraic reassociation
    /// there is harmless. But that same reassociation cancels the error term of a Kahan/two-sum
    /// (`err = (a - (s - bb)) + (b - bb)` collapses to `0`), silently degrading the CDF back to naive
    /// Float32 and reintroducing the high-qubit plateau. Only the two precision-critical kernels are
    /// taken from this library; every gate still runs from the fast default library.
    private static func loadPreciseLibrary(device: MTLDevice) throws -> MTLLibrary {
        let source = try metalShaderSource()
        let options = MTLCompileOptions()
        if #available(macOS 15.0, iOS 18.0, tvOS 18.0, *) {
            options.mathMode = .safe
        } else {
            options.fastMathEnabled = false
        }
        return try device.makeLibrary(source: source, options: options)
    }

    private static func metalShaderSource() throws -> String {
        let bundle = Bundle.module
        let candidateURLs = [
            bundle.url(forResource: "Gates", withExtension: "metal", subdirectory: "Metal"),
            bundle.url(forResource: "Gates", withExtension: "metal"),
        ]

        guard let metalURL = candidateURLs.compactMap({ $0 }).first else {
            throw QuantumEngineError.libraryNotFound
        }

        return try String(contentsOf: metalURL, encoding: .utf8)
    }

    public init() throws {
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else { throw QuantumEngineError.deviceNotFound }
        self.device = defaultDevice

        guard let queue = device.makeCommandQueue() else { throw QuantumEngineError.commandQueueCreationFailed }
        self.commandQueue = queue
        self.bufferPool = BufferPool(device: defaultDevice)

        let library = try Self.loadMetalLibrary(device: device)
        let preciseLibrary = try Self.loadPreciseLibrary(device: device)
        self.pipelines = try Pipelines(device: device, library: library, preciseLibrary: preciseLibrary)
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
        // Fill the threadgroup up to the pipeline's hardware maximum (≈1024) rather than a single
        // SIMD width (≈32); these gate kernels use no threadgroup memory and index purely off
        // thread_position_in_grid, so a wider group only raises occupancy.
        let threadgroupWidth = min(pipeline.maxTotalThreadsPerThreadgroup, max(pairCount, 1))
        let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
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
        // See `dispatchPairwiseGate`: these kernels carry no threadgroup state, so we size the group
        // at the pipeline maximum for occupancy instead of a single execution width.
        let threadgroupWidth = min(pipeline.maxTotalThreadsPerThreadgroup, threadCount)
        let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
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
                try executeResetQubit(on: state, qubit: qubit, rng: &rng)

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
    /// 256 squared amplitudes restricted to the `qubit == |1⟩` subspace into one partial, and the
    /// host sums the `2ⁿ / 256` partials, avoiding an `O(2ⁿ)` host-side scan of the amplitudes.
    private func qubitOnePopulation(on state: StateVector, qubit: Int) throws -> QFloat {
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

    /// Packs a list of control qubit indices into a bitmask (bit `q` set ⟺ qubit `q` is a control).
    private static func controlMask(_ controls: [Int]) -> UInt32 {
        var mask: UInt32 = 0
        for control in controls {
            mask |= UInt32(1) << UInt32(control)
        }
        return mask
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

        case .sdg(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.phaseSDagger, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .tdg(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.phaseTDagger, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .sx(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.sqrtX, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .sxdg(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.sqrtXDagger, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .p(let theta, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.phase, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .u(let theta, let phi, let lambda, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.universal, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                var angles = SIMD3<Float>(x: Float(theta), y: Float(phi), z: Float(lambda))
                encoder.setBytes(&angles, length: MemoryLayout<SIMD3<Float>>.stride, index: 3)
            }

        case .cx(let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cnot, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .cz(let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cz, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .swap(let q1, let q2):
            dispatchFullStateKernel(encoder: encoder, pipeline: pipelines.swapGate, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(q1), y: UInt32(q2))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .ccx(let control1, let control2, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.ccx, state: state) { encoder in
                var qubits = SIMD3<UInt32>(x: UInt32(control1), y: UInt32(control2), z: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
            }

        case .crx(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cRotX, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .cry(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cRotY, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .crz(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cRotZ, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .cp(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cPhase, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .mcx(let controls, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.mcx, state: state) { encoder in
                var packed = SIMD2<UInt32>(x: Self.controlMask(controls), y: UInt32(target))
                encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .mcz(let controls, let target):
            dispatchFullStateKernel(encoder: encoder, pipeline: pipelines.mcz, state: state) { encoder in
                var fullMask = Self.controlMask(controls) | (UInt32(1) << UInt32(target))
                encoder.setBytes(&fullMask, length: MemoryLayout<UInt32>.stride, index: 2)
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

    private func normalizeState(on state: StateVector) throws {
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

    private func samplePartialOutcome(on state: StateVector, qubits: [Int], diceRoll: Double) throws -> Int {
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

    /// Allocates the per-level block-sum scratch buffers (hi and lo halves) for a compensated scan
    /// over `stateCount` elements. When `pooled` is true the buffers come from ``bufferPool`` and the
    /// caller must hand them back with ``releasePrefixSumAuxBuffers(_:)``.
    private func makePrefixSumAuxBuffers(
        stateCount: Int,
        pooled: Bool = false
    ) throws -> (hi: [MTLBuffer], lo: [MTLBuffer]) {
        var hi: [MTLBuffer] = []
        var lo: [MTLBuffer] = []

        var currentCount = stateCount
        while currentCount > 1 {
            let blockCount = max((currentCount + Self.scanBlockSize - 1) / Self.scanBlockSize, 1)
            let byteCount = blockCount * MemoryLayout<QFloat>.stride
            hi.append(pooled ? try bufferPool.acquire(length: byteCount) : try makeSharedBuffer(length: byteCount))
            lo.append(pooled ? try bufferPool.acquire(length: byteCount) : try makeSharedBuffer(length: byteCount))
            currentCount = blockCount
        }

        return (hi, lo)
    }

    /// Returns the buffers from a pooled ``makePrefixSumAuxBuffers(stateCount:pooled:)`` call.
    private func releasePrefixSumAuxBuffers(_ aux: (hi: [MTLBuffer], lo: [MTLBuffer])) {
        for buffer in aux.hi { bufferPool.release(buffer) }
        for buffer in aux.lo { bufferPool.release(buffer) }
    }

    private func dispatchPrefixSumPhase(
        encoder: MTLComputeCommandEncoder,
        dataHi: MTLBuffer,
        dataLo: MTLBuffer,
        blockHi: MTLBuffer,
        blockLo: MTLBuffer,
        elementCount: Int,
        phase: UInt32,
        readInputLo: UInt32
    ) {
        let blockSize = Self.scanBlockSize
        let threadCount = max(elementCount, 1)
        var countValue = UInt32(elementCount)
        var phaseValue = phase
        var readInputLoValue = readInputLo

        encoder.setComputePipelineState(pipelines.prefixSum)
        encoder.setBuffer(dataHi, offset: 0, index: 0)
        encoder.setBuffer(dataLo, offset: 0, index: 1)
        encoder.setBuffer(blockHi, offset: 0, index: 2)
        encoder.setBuffer(blockLo, offset: 0, index: 3)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&phaseValue, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&readInputLoValue, length: MemoryLayout<UInt32>.stride, index: 6)

        let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: min(blockSize, threadCount), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    /// Compensated (double-single) inclusive prefix sum over `(hiBuffer, loBuffer)`. The leaf level
    /// (`level == 0`) feeds raw float probabilities (no `lo` term yet); recursive block-sum passes
    /// carry the compensation through.
    private func encodeInclusivePrefixSum(
        encoder: MTLComputeCommandEncoder,
        hiBuffer: MTLBuffer,
        loBuffer: MTLBuffer,
        elementCount: Int,
        auxiliaryHi: [MTLBuffer],
        auxiliaryLo: [MTLBuffer],
        level: Int = 0
    ) throws {
        guard elementCount > 1 else { return }
        guard level < auxiliaryHi.count else {
            throw QuantumEngineError.prefixSumBufferLevelMissing(level: level)
        }

        let blockSize = Self.scanBlockSize
        let numBlocks = (elementCount + blockSize - 1) / blockSize
        let blockHi = auxiliaryHi[level]
        let blockLo = auxiliaryLo[level]
        let readInputLo: UInt32 = level == 0 ? 0 : 1

        dispatchPrefixSumPhase(
            encoder: encoder,
            dataHi: hiBuffer,
            dataLo: loBuffer,
            blockHi: blockHi,
            blockLo: blockLo,
            elementCount: elementCount,
            phase: 0,
            readInputLo: readInputLo
        )

        if numBlocks > 1 {
            try encodeInclusivePrefixSum(
                encoder: encoder,
                hiBuffer: blockHi,
                loBuffer: blockLo,
                elementCount: numBlocks,
                auxiliaryHi: auxiliaryHi,
                auxiliaryLo: auxiliaryLo,
                level: level + 1
            )

            dispatchPrefixSumPhase(
                encoder: encoder,
                dataHi: hiBuffer,
                dataLo: loBuffer,
                blockHi: blockHi,
                blockLo: blockLo,
                elementCount: elementCount,
                phase: 2,
                readInputLo: readInputLo
            )
        }
    }

    // MARK: - Benchmark baseline (naive Float32 scan)

    private func dispatchPrefixSumPhaseNaive(
        encoder: MTLComputeCommandEncoder,
        dataBuffer: MTLBuffer,
        blockSumsBuffer: MTLBuffer,
        elementCount: Int,
        phase: UInt32
    ) {
        let blockSize = Self.scanBlockSize
        let threadCount = max(elementCount, 1)
        var countValue = UInt32(elementCount)
        var phaseValue = phase

        encoder.setComputePipelineState(pipelines.prefixSumNaive)
        encoder.setBuffer(dataBuffer, offset: 0, index: 0)
        encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&phaseValue, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: min(blockSize, threadCount), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    private func encodeInclusivePrefixSumNaive(
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

        dispatchPrefixSumPhaseNaive(
            encoder: encoder,
            dataBuffer: buffer,
            blockSumsBuffer: blockSumsBuffer,
            elementCount: elementCount,
            phase: 0
        )

        if numBlocks > 1 {
            try encodeInclusivePrefixSumNaive(
                encoder: encoder,
                buffer: blockSumsBuffer,
                elementCount: numBlocks,
                auxiliaryBuffers: auxiliaryBuffers,
                level: level + 1
            )

            dispatchPrefixSumPhaseNaive(
                encoder: encoder,
                dataBuffer: buffer,
                blockSumsBuffer: blockSumsBuffer,
                elementCount: elementCount,
                phase: 2
            )
        }
    }

    /// Timing comparison of the CDF prefix-sum stage only: the uncompensated Float32 baseline versus
    /// the shipped compensated (double-single) scan, averaged over `iterations` GPU runs.
    public struct ScanBenchmarkResult: Sendable {
        public let stateCount: Int
        public let iterations: Int
        public let naiveMillisecondsAverage: Double
        public let compensatedMillisecondsAverage: Double

        /// Extra wall-clock cost of the compensated scan, as a percentage of the naive scan.
        public var overheadPercent: Double {
            guard naiveMillisecondsAverage > 0 else { return 0 }
            return (compensatedMillisecondsAverage - naiveMillisecondsAverage) / naiveMillisecondsAverage * 100
        }
    }

    /// Benchmarks only the CDF/scan stage for a `qubitCount`-wide probability array.
    ///
    /// Both variants run the identical hierarchical scan structure over the same uniform input; only
    /// the per-element work (and the extra `lo` compensation buffer traffic) differs. Buffer refills
    /// are excluded from the timing, so the measurement reflects the GPU scan alone.
    public func benchmarkPrefixSumScan(qubitCount: Int, iterations: Int) throws -> ScanBenchmarkResult {
        precondition(qubitCount > 0 && iterations > 0)
        let stateCount = 1 << qubitCount
        let bytes = stateCount * MemoryLayout<QFloat>.stride

        let source = try makeSharedBuffer(length: bytes)
        let sourcePointer = source.contents().assumingMemoryBound(to: QFloat.self)
        let uniform = QFloat(1.0 / Double(stateCount))
        for index in 0..<stateCount { sourcePointer[index] = uniform }

        let naiveData = try makeSharedBuffer(length: bytes)
        var naiveAux: [MTLBuffer] = []
        var remaining = stateCount
        while remaining > 1 {
            let blockCount = max((remaining + Self.scanBlockSize - 1) / Self.scanBlockSize, 1)
            naiveAux.append(try makeSharedBuffer(length: blockCount * MemoryLayout<QFloat>.stride))
            remaining = blockCount
        }

        let compensatedHi = try makeSharedBuffer(length: bytes)
        let compensatedLo = try makeSharedBuffer(length: bytes)
        let compensatedAux = try makePrefixSumAuxBuffers(stateCount: stateCount)

        func refill(_ buffer: MTLBuffer) {
            buffer.contents().copyMemory(from: source.contents(), byteCount: bytes)
        }

        func runNaiveOnce() throws {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw QuantumEngineError.commandBufferCreationFailed
            }
            try encodeInclusivePrefixSumNaive(
                encoder: encoder,
                buffer: naiveData,
                elementCount: stateCount,
                auxiliaryBuffers: naiveAux
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
            }
        }

        func runCompensatedOnce() throws {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw QuantumEngineError.commandBufferCreationFailed
            }
            try encodeInclusivePrefixSum(
                encoder: encoder,
                hiBuffer: compensatedHi,
                loBuffer: compensatedLo,
                elementCount: stateCount,
                auxiliaryHi: compensatedAux.hi,
                auxiliaryLo: compensatedAux.lo
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
            }
        }

        // Warm-up (pipeline/caches) — untimed.
        refill(naiveData); try runNaiveOnce()
        refill(compensatedHi); try runCompensatedOnce()

        var naiveTotal = 0.0
        for _ in 0..<iterations {
            refill(naiveData)
            let start = CFAbsoluteTimeGetCurrent()
            try runNaiveOnce()
            naiveTotal += CFAbsoluteTimeGetCurrent() - start
        }

        var compensatedTotal = 0.0
        for _ in 0..<iterations {
            refill(compensatedHi)
            let start = CFAbsoluteTimeGetCurrent()
            try runCompensatedOnce()
            compensatedTotal += CFAbsoluteTimeGetCurrent() - start
        }

        return ScanBenchmarkResult(
            stateCount: stateCount,
            iterations: iterations,
            naiveMillisecondsAverage: naiveTotal / Double(iterations) * 1000,
            compensatedMillisecondsAverage: compensatedTotal / Double(iterations) * 1000
        )
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
