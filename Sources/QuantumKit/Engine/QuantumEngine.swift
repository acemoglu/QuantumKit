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
    case localizedNoiseRequiresDensityMatrixBackend
    case nonProjectiveMeasurementRequiresDensityMatrixBackend
    case unsupportedGateEncoding(Gate)

}

/// Metal compute pipeline cache for ``QuantumEngine``. Package-internal (H4 audit).
struct Pipelines: @unchecked Sendable {

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
    let customUnitary1Q: MTLComputePipelineState

    let probabilities: MTLComputePipelineState
    let maskedPopulationReduce: MTLComputePipelineState
    let prefixSum: MTLComputePipelineState
    let prefixSumNaive: MTLComputePipelineState
    let collapseSearch: MTLComputePipelineState
    let collapseSearchBatch: MTLComputePipelineState
    let collapseState: MTLComputePipelineState
    let partialCollapse: MTLComputePipelineState
    let resetQubit: MTLComputePipelineState
    let amplitudeDampingJump: MTLComputePipelineState
    let amplitudeDampingNoJump: MTLComputePipelineState
    let phaseDampingJump: MTLComputePipelineState
    let phaseDampingNoJump: MTLComputePipelineState
    let trajectoryWeightedPopulationPartial: MTLComputePipelineState
    let pauliExpectationPartial: MTLComputePipelineState
    let marginalLeafHistogram: MTLComputePipelineState
    let marginalPartialsReduce: MTLComputePipelineState
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

        guard let customUnitary1QFunc = library.makeFunction(name: "custom_unitary_1q_gate") else {
            throw QuantumEngineError.functionNotFound("custom_unitary_1q_gate")
        }
        self.customUnitary1Q = try device.makeComputePipelineState(function: customUnitary1QFunc)

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

        guard let collapseBatchFunc = preciseLibrary.makeFunction(name: "find_collapsed_states") else { throw QuantumEngineError.functionNotFound("find_collapsed_states") }
        self.collapseSearchBatch = try device.makeComputePipelineState(function: collapseBatchFunc)

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

        // Compensated leaf for general Pauli expectation: its in-block (hi, lo) tree reduction uses
        // the tj_two_sum/tj_add helpers, so it is taken from the precise library to keep the error
        // terms from collapsing under fast-math reassociation (same rationale as the trajectory leaf).
        guard let pauliExpectationPartialFunc = preciseLibrary.makeFunction(name: "pauli_expectation_partial") else { throw QuantumEngineError.functionNotFound("pauli_expectation_partial") }
        self.pauliExpectationPartial = try device.makeComputePipelineState(function: pauliExpectationPartialFunc)

        // Leaf bucketing runs from the fast default library (a per-block float histogram of <=256
        // terms needs no compensation); the cross-block reduction is taken from the precise library
        // so its double-single two-sum survives fast-math reassociation.
        guard let marginalLeafFunc = library.makeFunction(name: "marginal_leaf_histogram") else { throw QuantumEngineError.functionNotFound("marginal_leaf_histogram") }
        self.marginalLeafHistogram = try device.makeComputePipelineState(function: marginalLeafFunc)

        guard let marginalReduceFunc = preciseLibrary.makeFunction(name: "marginal_partials_reduce") else { throw QuantumEngineError.functionNotFound("marginal_partials_reduce") }
        self.marginalPartialsReduce = try device.makeComputePipelineState(function: marginalReduceFunc)

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
/// takes it back when the caller is done.
///
/// - Important: The execution pipelines no longer block on `waitUntilCompleted` after every
///   dispatch, so a buffer handed back on the CPU thread may still be referenced by an in-flight
///   GPU command buffer. Recycling it synchronously would be a use-after-free. Callers that release
///   a buffer tied to GPU work must use ``release(_:after:)``, which defers the (lock-guarded)
///   return to the buffer's `addCompletedHandler` so it only re-enters the free list once the GPU
///   has genuinely finished. The synchronous ``release(_:)`` is reserved for buffers whose owning
///   command buffer has already been awaited.
///
/// An acquired buffer holds arbitrary stale contents, so callers must fully overwrite (or clear) it
/// before reading. The free list is lock-guarded so the owning engine stays safe to share across
/// threads, including from the completion handlers that run on Metal's private queues.
final class BufferPool: @unchecked Sendable {
    let device: MTLDevice
    let lock = NSLock()
    var freeBuffersByLength: [Int: [MTLBuffer]] = [:]

    init(device: MTLDevice) {
        self.device = device
    }

    func acquire(length: Int, zero: Bool = true) throws -> MTLBuffer {
        lock.lock()
        if var bucket = freeBuffersByLength[length], let reused = bucket.popLast() {
            freeBuffersByLength[length] = bucket
            lock.unlock()
            if zero {
                memset(reused.contents(), 0, length)
            }
            return reused
        }
        lock.unlock()

        guard length > 0,
              let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw QuantumEngineError.bufferAllocationFailed(requiredBytes: max(length, 0))
        }
        if zero {
            memset(buffer.contents(), 0, length)
        }
        return buffer
    }

    /// Synchronously returns `buffer` to the free list. Only safe once the GPU work that used the
    /// buffer has been awaited (e.g. behind a `waitUntilCompleted` at a host-readback site).
    func release(_ buffer: MTLBuffer) {
        lock.lock()
        freeBuffersByLength[buffer.length, default: []].append(buffer)
        lock.unlock()
    }

    /// Async-safe release: returns `buffer` to the free list only after `commandBuffer` has
    /// completed on the GPU. The completion handler hops onto the same `NSLock`, so concurrent
    /// completions and `acquire`/`release` calls stay serialized.
    func release(_ buffer: MTLBuffer, after commandBuffer: MTLCommandBuffer) {
        commandBuffer.addCompletedHandler { [self] _ in
            self.release(buffer)
        }
    }
}

/// GPU-backed executor for quantum circuits.
///
/// Construct with ``init(renormalizationInterval:)`` — the shared device comes from
/// ``MetalRuntime``; callers need not import Metal or pass an ``MTLDevice``.
///
/// `QuantumEngine` is safe to share across threads: it holds immutable Metal objects (device,
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

    /// The Metal shader sources are split across multiple `.metalsrc` files for readability
    /// (`Gates.metalsrc` holds the precision-critical scan/collapse path and the `df_*` helpers;
    /// `GateKernels.metalsrc` holds the unitary gate kernels). Both libraries are compiled at runtime
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

    /// Creates an engine on ``MetalRuntime/sharedDevice()`` (no caller Metal imports).
    public init(renormalizationInterval: Int = 50) throws {
        let scanBlockSize = Self.scanBlockSize
        precondition(scanBlockSize > 0 && (scanBlockSize & (scanBlockSize - 1)) == 0, "Metal threadgroup dimensions MUST be exact powers of two for binary tree reductions to work.")
        let renormalizationBlockSize = Self.renormalizationBlockSize
        precondition(renormalizationBlockSize > 0 && (renormalizationBlockSize & (renormalizationBlockSize - 1)) == 0, "Metal threadgroup dimensions MUST be exact powers of two for binary tree reductions to work.")

        let defaultDevice = try MetalRuntime.sharedDevice()
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

    /// Blocks the CPU until every command buffer previously committed to ``commandQueue`` has
    /// finished on the GPU.
    ///
    /// With the per-gate/per-channel `waitUntilCompleted` calls removed, gate and noise dispatches
    /// are committed without draining the queue. Because the queue is serial and in-order, committing
    /// one trailing command buffer and awaiting it guarantees all earlier work has completed — the
    /// single synchronization point needed before classical state (amplitudes / probabilities) is
    /// read back directly from the shared buffers on the host.
    func drainPipeline() throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
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
                try encodeUnitaryGate(gate, encoder: computeEncoder, state: state)
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

    /// After ``CircuitExecutionCancellationError``, `state` is undefined and must not be reused;
    /// allocate a fresh ``StateVector`` for any later run. The Metal command queue is drained
    /// before this returns (success, cancel, or other error).
    @discardableResult
    public func executeRNG(
        _ circuit: QuantumCircuit,
        on state: StateVector,
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil,
        runState: CircuitRunState = .full,
        cancellationCheck: (() throws -> Void)? = nil
    ) throws -> CircuitExecutionResult {
        guard circuit.qubitCount == state.qubitCount else {
            throw QuantumEngineError.qubitCountMismatch(circuit: circuit.qubitCount, state: state.qubitCount)
        }
        try circuit.requireFullyBound()
        let instructionRange = try CircuitRunStateValidation.resolvedRange(
            gateCount: circuit.gates.count,
            runState: runState
        )
        // Always drain on exit (success, cancel, or other error) so in-flight command buffers
        // complete before buffer-pool reuse or host inspection.
        defer { try? drainPipeline() }

        if let noise, noise.hasLocalizedGateNoise {
            throw QuantumEngineError.localizedNoiseRequiresDensityMatrixBackend
        }
        if let noise, noise.measurementMode != .projective {
            throw QuantumEngineError.nonProjectiveMeasurementRequiresDensityMatrixBackend
        }

        let noiseEnabled = noise?.hasGateNoise == true
        var measurementOutcomes = runState.measurementOutcomes
        var pendingUnitaryGates: [Gate] = []
        // Top-level instruction tally shared with CPU engines for resume / CircuitCheckpoint.
        var appliedGateCount = runState.appliedGateCount
        // Unitary-piece counter drives GPU renorm cadence (expanded gates may be many pieces).
        var renormCounter = runState.unitaryRenormCount ?? runState.appliedGateCount
        var classicalMemory = runState.classicalMemory
            ?? ClassicalMemory(
            registerWidths: circuit.classicalRegisters.map(\.bitCount)
        )
        if let seededPhase = runState.cumulativeGlobalPhaseRadians {
            state.setCumulativeGlobalPhaseRadians(seededPhase)
        }

        func flushPendingUnitaryGates() throws {
            try flushUnitaryGates(
                pendingUnitaryGates,
                on: state,
                gateCounter: &renormCounter,
                renormalize: true
            )
            pendingUnitaryGates.removeAll(keepingCapacity: true)
        }

        func applyNoise(after gate: Gate) throws {
            guard noiseEnabled, let noise else { return }
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
                    flipProbability: noise.effectivePhaseDampingProbability,
                    rng: &rng
                )
            }
        }

        func executeRuntimeGate(_ gate: Gate, countsTowardApplied: Bool = true) throws {
            switch gate {
            case .measure(let spec):
                try flushPendingUnitaryGates()

                if let noise, noise.measurementDephasingProbability > 0 {
                    try applyPhaseDamping(
                        after: gate,
                        on: state,
                        flipProbability: noise.measurementDephasingProbability,
                        rng: &rng
                    )
                }

                let outcome = try executePartialMeasurementCollapse(
                    on: state,
                    qubits: spec.qubits,
                    rng: &rng,
                    noise: noise
                )
                measurementOutcomes.append(measuredBits(outcome: outcome, qubits: spec.qubits))
                try classicalMemory.writeOutcome(
                    outcome,
                    measuredQubitCount: spec.qubits.count,
                    register: spec.classicalRegister,
                    bitOffset: spec.classicalBitOffset
                )

            case .reset(let qubit):
                try flushPendingUnitaryGates()
                try executeResetQubit(on: state, qubit: qubit, rng: &rng)
                if let noise, noise.resetErrorProbability > 0,
                   rng.nextUnitFloat() < noise.resetErrorProbability {
                    try executeUnitaryGate(
                        .x(target: qubit),
                        on: state,
                        gateCounter: &renormCounter,
                        renormalize: true,
                        trackGlobalPhase: false
                    )
                }

            case .barrier:
                try flushPendingUnitaryGates()

            case .delay(let duration, _):
                try flushPendingUnitaryGates()
                if let noise, noise.thermalRelaxationOnDelay {
                    let gamma = noise.amplitudeDampingProbability(forDuration: duration)
                    if gamma > 0 {
                        try applyAmplitudeDamping(
                            after: gate,
                            on: state,
                            probability: gamma,
                            rng: &rng
                        )
                    }
                    let lambda = noise.phaseDampingProbability(forDuration: duration)
                    if lambda > 0 {
                        try applyPhaseDamping(
                            after: gate,
                            on: state,
                            flipProbability: lambda,
                            rng: &rng
                        )
                    }
                }

            case .id:
                break

            case .c_if(let classicalRegister, let expectedValue, let conditionedGate):
                try flushPendingUnitaryGates()
                if classicalMemory.value(ofRegister: classicalRegister) == expectedValue {
                    // Nested body is not a top-level circuit instruction.
                    try executeRuntimeGate(conditionedGate, countsTowardApplied: false)
                }

            case .while_c(let classicalRegister, let expectedValue, let body, let maxIterations):
                try flushPendingUnitaryGates()
                var iterations = 0
                while classicalMemory.value(ofRegister: classicalRegister) == expectedValue {
                    guard iterations < maxIterations else {
                        throw QuantumCircuitError.maxLoopIterationsExceeded(maxIterations: maxIterations)
                    }
                    for nested in body {
                        try executeRuntimeGate(nested, countsTowardApplied: false)
                    }
                    iterations += 1
                }

            case .unitary1:
                try flushPendingUnitaryGates()
                try executeUnitaryGate(
                    gate,
                    on: state,
                    gateCounter: &renormCounter,
                    renormalize: true
                )
                try applyNoise(after: gate)

            case .initialize(let qubits, let amplitudes):
                try flushPendingUnitaryGates()
                // initialize writes shared Metal buffers on the host.
                try drainPipeline()
                try state.initialize(qubits: qubits, amplitudes: amplitudes)
                if let noise, noise.preparationErrorProbability > 0 {
                    for qubit in qubits where rng.nextUnitFloat() < noise.preparationErrorProbability {
                        try executeUnitaryGate(
                            .x(target: qubit),
                            on: state,
                            gateCounter: &renormCounter,
                            renormalize: true,
                            trackGlobalPhase: false
                        )
                    }
                }

            case .customUnitary(let matrix, let qubits) where qubits.count > 1:
                try flushPendingUnitaryGates()
                try applyHostCustomUnitary(matrix, qubits: qubits, on: state)
                try state.accumulateGlobalPhase(of: gate)
                renormCounter += 1
                if shouldRenormalize(afterAppliedGateCount: renormCounter) {
                    try normalizeState(on: state)
                }
                try applyNoise(after: gate)

            default:
                if noiseEnabled {
                    try flushPendingUnitaryGates()
                    try executeUnitaryGate(
                        gate,
                        on: state,
                        gateCounter: &renormCounter,
                        renormalize: true
                    )
                    try applyNoise(after: gate)
                } else {
                    pendingUnitaryGates.append(gate)
                }
            }

            if countsTowardApplied {
                appliedGateCount += 1
            }
        }

        // Metal SV pending-flushes unitaries after this loop, then `drainPipeline`. Timing each
        // instruction would sample `append` rather than encode+wait, so this engine never records
        // per-gate samples (`gateTimings` stays nil). Backend `evolve` / `sample` phases wrap
        // executeRNG and therefore include the flush+drain already paid here. Do not add extra
        // waits when profiling is on.
        for index in instructionRange {
            try cancellationCheck?()
            try executeRuntimeGate(circuit.gates[index])
        }

        try flushPendingUnitaryGates()
        // Gate/noise dispatches above are now committed asynchronously, so drain the queue once
        // here — at the end of the circuit run — before the caller reads classical state back from
        // the shared buffers on the host.
        try drainPipeline()
        return CircuitExecutionResult(
            measurementOutcomes: measurementOutcomes,
            classicalMemory: classicalMemory,
            appliedGateCount: appliedGateCount,
            unitaryRenormCount: renormCounter,
            cumulativeGlobalPhaseRadians: state.cumulativeGlobalPhaseRadians
        )
    }

    /// Drains GPU work, then copies ``StateVector`` amplitudes to a host snapshot.
    public func snapshot(_ state: StateVector) throws -> StateVectorSnapshot {
        try drainPipeline()
        return state.snapshotHostAmplitudes()
    }

    /// Restores amplitudes from a host snapshot (caller should not have in-flight GPU work on `state`).
    public func restore(_ state: StateVector, from snapshot: StateVectorSnapshot) throws {
        try drainPipeline()
        try state.restoreHostAmplitudes(from: snapshot)
    }
}
