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
    let phaseDampingJump: MTLComputePipelineState
    let phaseDampingNoJump: MTLComputePipelineState
    let trajectoryWeightedPopulationPartial: MTLComputePipelineState
    let normalize: MTLComputePipelineState
    let renormStateNormPartial: MTLComputePipelineState
    let renormCompensatedPartial: MTLComputePipelineState
    let renormScale: MTLComputePipelineState

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

        guard let phaseDampingJumpFunc = library.makeFunction(name: "phase_damping_jump") else { throw QuantumEngineError.functionNotFound("phase_damping_jump") }
        self.phaseDampingJump = try device.makeComputePipelineState(function: phaseDampingJumpFunc)

        guard let phaseDampingNoJumpFunc = library.makeFunction(name: "phase_damping_no_jump") else { throw QuantumEngineError.functionNotFound("phase_damping_no_jump") }
        self.phaseDampingNoJump = try device.makeComputePipelineState(function: phaseDampingNoJumpFunc)

        guard let trajectoryWeightedPopFunc = preciseLibrary.makeFunction(name: "trajectory_weighted_population_partial") else { throw QuantumEngineError.functionNotFound("trajectory_weighted_population_partial") }
        self.trajectoryWeightedPopulationPartial = try device.makeComputePipelineState(function: trajectoryWeightedPopFunc)

        guard let normalizeFunc = library.makeFunction(name: "normalize_state_vector") else { throw QuantumEngineError.functionNotFound("normalize_state_vector") }
        self.normalize = try device.makeComputePipelineState(function: normalizeFunc)

        guard let renormStatePartialFunc = preciseLibrary.makeFunction(name: "renorm_state_norm_partial") else { throw QuantumEngineError.functionNotFound("renorm_state_norm_partial") }
        self.renormStateNormPartial = try device.makeComputePipelineState(function: renormStatePartialFunc)

        guard let renormCompensatedPartialFunc = preciseLibrary.makeFunction(name: "renorm_compensated_partial") else { throw QuantumEngineError.functionNotFound("renorm_compensated_partial") }
        self.renormCompensatedPartial = try device.makeComputePipelineState(function: renormCompensatedPartialFunc)

        guard let renormScaleFunc = library.makeFunction(name: "renorm_scale_state") else { throw QuantumEngineError.functionNotFound("renorm_scale_state") }
        self.renormScale = try device.makeComputePipelineState(function: renormScaleFunc)

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
final class BufferPool: @unchecked Sendable {
    let device: MTLDevice
    let lock = NSLock()
    var freeBuffersByLength: [Int: [MTLBuffer]] = [:]

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

    static let scanBlockSize = 256
    static let renormalizationBlockSize = 256

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelines: Pipelines
    let bufferPool: BufferPool
    public let renormalizationInterval: Int

    static func loadMetalLibrary(device: MTLDevice) throws -> MTLLibrary {
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
    static func loadPreciseLibrary(device: MTLDevice) throws -> MTLLibrary {
        let source = try metalShaderSource()
        let options = MTLCompileOptions()
        if #available(macOS 15.0, iOS 18.0, tvOS 18.0, *) {
            options.mathMode = .safe
        } else {
            options.fastMathEnabled = false
        }
        return try device.makeLibrary(source: source, options: options)
    }

    /// The Metal shader sources are split across multiple `.metal` files for readability
    /// (`Gates.metal` holds the precision-critical scan/collapse path and the `df_*` helpers;
    /// `GateKernels.metal` holds the unitary gate kernels). Both libraries are compiled at runtime
    /// from these sources, so they are concatenated into a single translation unit here. Duplicate
    /// `#include <metal_stdlib>` / `using namespace metal;` lines across the files are harmless.
    static func metalShaderSource() throws -> String {
        let metalFileNames = ["Gates", "GateKernels", "Renormalization", "Trajectories"]
        let bundle = Bundle.module

        var combined = ""
        for name in metalFileNames {
            let candidateURLs = [
                bundle.url(forResource: name, withExtension: "metalsrc", subdirectory: "Metal"),
                bundle.url(forResource: name, withExtension: "metalsrc"),
            ]
            guard let metalURL = candidateURLs.compactMap({ $0 }).first else {
                throw QuantumEngineError.libraryNotFound
            }
            combined += try String(contentsOf: metalURL, encoding: .utf8)
            combined += "\n"
        }

        return combined
    }

    public init(renormalizationInterval: Int = 50) throws {
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else { throw QuantumEngineError.deviceNotFound }
        self.device = defaultDevice
        self.renormalizationInterval = max(0, renormalizationInterval)

        guard let queue = device.makeCommandQueue() else { throw QuantumEngineError.commandQueueCreationFailed }
        self.commandQueue = queue
        self.bufferPool = BufferPool(device: defaultDevice)

        let library = try Self.loadMetalLibrary(device: device)
        let preciseLibrary = try Self.loadPreciseLibrary(device: device)
        self.pipelines = try Pipelines(device: device, library: library, preciseLibrary: preciseLibrary)
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

        guard var commandBuffer = commandQueue.makeCommandBuffer(),
              var computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }
        var gateCounters = [Int](repeating: 0, count: states.count)
        var renormScratchByState: [ObjectIdentifier: RenormalizationScratch] = [:]
        defer {
            for scratch in renormScratchByState.values {
                releaseRenormalizationScratch(scratch)
            }
        }

        // A single MTLCommandBuffer holds a bounded command stream; encoding tens of thousands of
        // dispatches (a deep circuit fanned out across many states) overflows the driver and
        // silently truncates or faults. Once the encoded dispatch count crosses this threshold we
        // flush and continue on a fresh buffer. Flushes happen only at gate boundaries — never
        // mid-renormalization, whose multi-pass reduction relies on intra-encoder ordering.
        let maxDispatchesPerCommandBuffer = 1000
        // Conservative upper bound on the dispatches a single renormalization expands into (one
        // initial reduction + ≤3 compensated reduction passes + the scale kernel for ≤28 qubits).
        let renormDispatchUpperBound = 8
        var encodedDispatches = 0

        func flushAndRestart() throws {
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
            }
            guard let nextBuffer = commandQueue.makeCommandBuffer(),
                  let nextEncoder = nextBuffer.makeComputeCommandEncoder() else {
                throw QuantumEngineError.commandBufferCreationFailed
            }
            commandBuffer = nextBuffer
            computeEncoder = nextEncoder
            encodedDispatches = 0
        }

        for gate in circuit.gates {
            for (index, state) in states.enumerated() {
                encodeUnitaryGate(gate, encoder: computeEncoder, state: state)
                encodedDispatches += 1
                gateCounters[index] += 1
                if shouldRenormalize(afterAppliedGateCount: gateCounters[index]) {
                    let key = ObjectIdentifier(state)
                    let scratch = try renormScratchByState[key] ?? makeRenormalizationScratch(stateCount: state.stateCount)
                    renormScratchByState[key] = scratch
                    try encodeStateRenormalization(encoder: computeEncoder, state: state, scratch: scratch)
                    encodedDispatches += renormDispatchUpperBound
                }

                if encodedDispatches >= maxDispatchesPerCommandBuffer {
                    try flushAndRestart()
                }
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
        var unitaryGateCounter = 0

        for gate in circuit.gates {
            switch gate {
            case .measure(let qubits):
                try flushUnitaryGates(pendingUnitaryGates, on: state, gateCounter: &unitaryGateCounter)
                pendingUnitaryGates.removeAll(keepingCapacity: true)

                let outcome = try executePartialMeasurementCollapse(
                    on: state,
                    qubits: qubits,
                    rng: &rng,
                    noise: noise
                )
                measurementOutcomes.append(measuredBits(outcome: outcome, qubits: qubits))

            case .reset(let qubit):
                try flushUnitaryGates(pendingUnitaryGates, on: state, gateCounter: &unitaryGateCounter)
                pendingUnitaryGates.removeAll(keepingCapacity: true)
                try executeResetQubit(on: state, qubit: qubit, rng: &rng)

            default:
                if noiseEnabled, let noise {
                    try flushUnitaryGates(pendingUnitaryGates, on: state, gateCounter: &unitaryGateCounter)
                    pendingUnitaryGates.removeAll(keepingCapacity: true)
                    try executeUnitaryGate(gate, on: state, gateCounter: &unitaryGateCounter)
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
                            flipProbability: noise.phaseDampingProbability,
                            rng: &rng
                        )
                    }
                } else {
                    pendingUnitaryGates.append(gate)
                }
            }
        }

        try flushUnitaryGates(pendingUnitaryGates, on: state, gateCounter: &unitaryGateCounter)
        return CircuitExecutionResult(measurementOutcomes: measurementOutcomes)
    }
}
