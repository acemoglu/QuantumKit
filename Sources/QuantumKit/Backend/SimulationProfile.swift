import Foundation

/// Opt-in host-side simulation telemetry. Disabled by default; does not affect amplitudes,
/// RNG consumption, or shot outcomes when enabled.
public struct SimulationProfilingOptions: Sendable, Equatable {
    /// Master switch. When `false`, ``QuantumResult/profile`` is `nil`.
    public var enabled: Bool
    /// Host wall time around each top-level instruction, **when that engine instruments gates**.
    ///
    /// - CPU statevector / CPU density-matrix: one sample per circuit index, **summed across
    ///   shots** (serial loops **and** concurrent independent-shot workers). Under shot
    ///   parallelism the sum is aggregate host CPU-ns and can exceed the `sample` phase wall
    ///   time; it is not a wall-clock exclusive duration.
    /// - Metal density-matrix: host-encode time per instruction (GPU drain is in the phase).
    /// - Metal statevector: **not instrumented**. Pending-unitary flush + `drainPipeline` run
    ///   after the gate loop; per-gate samples would be “append to pending,” not encode+wait.
    ///   ``SimulationProfile/gateTimings`` is `nil`. Phase `evolve` / `sample` includes flush+drain.
    /// - Batched Metal shots (`executeUnitaryBatch`): not instrumented per gate (`nil`), same reason.
    ///
    /// No extra Metal kernel timestamps are taken.
    public var recordGateTimings: Bool
    /// Named phases: `evolve`, `sample`, `estimate`, `gradient`. Recorded only by the
    /// **installing task** (`usingRecorder` nesting depth 0 on the task that created the
    /// recorder). Nested `backend.run` / primitive calls, and inherited child Tasks, may
    /// `timeGate` but do not emit phase rows or call ``SimulationProfiling/finishProfile``.
    /// Shot-parallel workers must only `timeGate` (or ``SimulationProfiling/withWorkerRecorder``).
    /// `nil` on the profile when this flag is on but the owner never entered a `timePhase`.
    public var recordPhaseTimings: Bool

    public init(
        enabled: Bool = false,
        recordGateTimings: Bool = false,
        recordPhaseTimings: Bool = false
    ) {
        self.enabled = enabled
        self.recordGateTimings = enabled && recordGateTimings
        self.recordPhaseTimings = enabled && recordPhaseTimings
    }

    /// Profiling off (default).
    public static let disabled = SimulationProfilingOptions()
    /// Per-run wall time and estimated peak memory only.
    public static let enabled = SimulationProfilingOptions(enabled: true)
    /// Per-run plus per-gate (where instrumented) and per-phase host timings.
    public static let detailed = SimulationProfilingOptions(
        enabled: true,
        recordGateTimings: true,
        recordPhaseTimings: true
    )

    var needsRecorder: Bool {
        enabled && (recordGateTimings || recordPhaseTimings)
    }
}

/// How ``SimulationProfile/peakMemoryBytes`` was obtained.
///
/// QuantumKit reports an **estimate** from state-buffer size plus known scratch / shot-batch
/// pools. Process RSS (`getrusage` / Mach `task_info`) is not used: it is process-wide, not
/// per-run, and would not be comparable across CPU vs Metal buffer allocations.
public enum SimulationMemorySource: String, Codable, Sendable, Equatable {
    /// Formula: primary state bytes + known workspace (renorm / DM scratch / live batch copies).
    case estimated
}

/// Host wall time for one top-level circuit instruction.
///
/// ``index`` is unique in a finished profile: nanoseconds are **summed** across every
/// application of that instruction (each shot — including concurrent CPU workers — and each
/// nested `executeRNG` in the same run). Concurrent shots therefore accumulate aggregate
/// CPU-ns that can exceed the owning phase’s wall clock.
public struct SimulationGateTiming: Codable, Sendable, Equatable {
    /// Index into ``QuantumCircuit/gates``.
    public let index: Int
    /// Cumulative host nanoseconds for this index (sum over shots / re-executions; not
    /// wall-clock exclusive under ``ShotExecutionPolicy/canBatch`` parallelism).
    public let wallClockNanoseconds: UInt64

    public init(index: Int, wallClockNanoseconds: UInt64) {
        self.index = index
        self.wallClockNanoseconds = wallClockNanoseconds
    }
}

/// Host wall time for a named run phase (`evolve`, `sample`, `estimate`, or `gradient`).
public struct SimulationPhaseTiming: Codable, Sendable, Equatable {
    public let name: String
    public let wallClockNanoseconds: UInt64

    public init(name: String, wallClockNanoseconds: UInt64) {
        self.name = name
        self.wallClockNanoseconds = wallClockNanoseconds
    }
}

/// Correctness-neutral telemetry for one ``QuantumBackend/run(circuit:options:)`` (or primitive
/// that builds ``QuantumResultMetadata`` from the same options).
public struct SimulationProfile: Codable, Sendable, Equatable {
    /// Wall time of the profiled invocation (matches ``QuantumResultMetadata/wallClockNanoseconds``).
    public let wallClockNanoseconds: UInt64
    /// Primary quantum state footprint: SV `2^n` complex amplitudes, or DM `4^n` elements.
    public let stateBytes: Int
    /// High-water estimate: ``stateBytes`` × live copies (Metal ``StateVectorBatch`` or CPU
    /// independent-shot worker pool) + one scratch copy.
    public let peakMemoryBytes: Int
    /// Always ``SimulationMemorySource/estimated`` in this release (not OS RSS).
    public let memorySource: SimulationMemorySource
    /// Per-instruction host timings.
    ///
    /// - `nil` when ``SimulationProfilingOptions/recordGateTimings`` is off, **or** when it is
    ///   on but this path never instrumented gates (Metal SV pending-flush, batched
    ///   `executeUnitaryBatch`, primitives that only timed a phase, empty recorder, **shot
    ///   Estimator** — basis-changed measure circuits are not the user circuit).
    /// - Non-`nil` (possibly `[]`) only when an engine started gate instrumentation for this
    ///   run. `[]` means the instrumented instruction range was empty, not “Metal batch skipped.”
    public let gateTimings: [SimulationGateTiming]?
    /// Named phase timings from the **installing owner task** only.
    ///
    /// - `nil` when ``SimulationProfilingOptions/recordPhaseTimings`` is off, **or** when no
    ///   owner `timePhase` ran (uninstrumented / nested or inherited-worker `finishProfile`).
    /// - Non-`nil` when at least one owner phase was timed. Empty array is reserved for a started
    ///   phase session with zero samples (not used by current backends).
    public let phaseTimings: [SimulationPhaseTiming]?

    public init(
        wallClockNanoseconds: UInt64,
        stateBytes: Int,
        peakMemoryBytes: Int,
        memorySource: SimulationMemorySource = .estimated,
        gateTimings: [SimulationGateTiming]? = nil,
        phaseTimings: [SimulationPhaseTiming]? = nil
    ) {
        self.wallClockNanoseconds = wallClockNanoseconds
        self.stateBytes = stateBytes
        self.peakMemoryBytes = peakMemoryBytes
        self.memorySource = memorySource
        self.gateTimings = gateTimings
        self.phaseTimings = phaseTimings
    }
}

/// Host-side timing sink installed for the duration of a profiled run via ``TaskLocal``.
///
/// Mutation is serialized with `NSLock` so inherited TaskLocal child tasks and GCD shot
/// workers cannot data-race the dictionaries. Prefer **one recorder per run**; do not append
/// from unsynchronized shared state. Concurrent `timeGate` bodies may run in parallel; only
/// the accounting update is locked.
///
/// Phase rows and ``SimulationProfiling/finishProfile`` belong to the **installing task**
/// (the Swift Task that called ``SimulationProfiling/usingRecorder``, or the sync context
/// with no task). Inherited child Tasks share this object for `timeGate` merge only.
final class SimulationProfileRecorder: @unchecked Sendable {
    let options: SimulationProfilingOptions
    private let lock = NSLock()
    private var gateNanos: [Int: UInt64] = [:]
    private var phaseTimingsStorage: [SimulationPhaseTiming] = []
    private var gateInstrumentationStarted = false
    private var phaseInstrumentationStarted = false
    private var inFlightGateTimings = 0
    /// Captured at init; compared in ``isInstallationOwnerTask``. The installing task stays
    /// alive for the `usingRecorder` body, which is the only window these closures run.
    private let ownerTaskEquals: () -> Bool

    init(options: SimulationProfilingOptions) {
        self.options = options
        let owner = withUnsafeCurrentTask { $0 }
        self.ownerTaskEquals = {
            withUnsafeCurrentTask { current in
                switch (owner, current) {
                case (.none, .none):
                    return true
                case let (.some(left), .some(right)):
                    return left == right
                default:
                    return false
                }
            }
        }
    }

    var recordsGates: Bool { options.recordGateTimings }

    /// `true` on the Swift Task (or task-less sync context) that constructed this recorder.
    var isInstallationOwnerTask: Bool { ownerTaskEquals() }

    /// Marks that this run's engine will (or did) time gates. Call before an empty instruction
    /// range so ``SimulationProfile/gateTimings`` is `[]` rather than `nil`.
    func markGateInstrumentationStarted() {
        lock.lock()
        gateInstrumentationStarted = true
        lock.unlock()
    }

    func timeGate(index: Int, _ body: () throws -> Void) throws {
        lock.lock()
        inFlightGateTimings += 1
        lock.unlock()
        defer {
            lock.lock()
            inFlightGateTimings -= 1
            lock.unlock()
        }
        let started = DispatchTime.now()
        try body()
        let elapsed = DispatchTime.now().uptimeNanoseconds &- started.uptimeNanoseconds
        lock.lock()
        gateInstrumentationStarted = true
        gateNanos[index, default: 0] &+= elapsed
        lock.unlock()
    }

    func timePhase<T>(_ name: String, _ body: () throws -> T) throws -> T {
        let started = DispatchTime.now()
        let value = try body()
        let elapsed = DispatchTime.now().uptimeNanoseconds &- started.uptimeNanoseconds
        lock.lock()
        phaseInstrumentationStarted = true
        phaseTimingsStorage.append(SimulationPhaseTiming(name: name, wallClockNanoseconds: elapsed))
        lock.unlock()
        return value
    }

    func snapshotGateTimings() -> [SimulationGateTiming]? {
        lock.lock()
        defer { lock.unlock() }
        // In-flight `timeGate` bodies have not been added yet. Callers that spawn workers
        // must join them before ``SimulationProfiling/finishProfile``.
        assert(
            inFlightGateTimings == 0,
            "finishProfile/snapshot while timeGate is in flight; join workers first"
        )
        guard gateInstrumentationStarted else { return nil }
        return gateNanos.keys.sorted().map { index in
            SimulationGateTiming(index: index, wallClockNanoseconds: gateNanos[index] ?? 0)
        }
    }

    func snapshotPhaseTimings() -> [SimulationPhaseTiming]? {
        lock.lock()
        defer { lock.unlock() }
        guard phaseInstrumentationStarted else { return nil }
        return phaseTimingsStorage
    }

    /// Test hook: unique circuit indices recorded so far (nil if instrumentation never started).
    func recordedGateIndices() -> [Int]? {
        lock.lock()
        defer { lock.unlock() }
        guard gateInstrumentationStarted else { return nil }
        return gateNanos.keys.sorted()
    }
}

enum SimulationProfiling {
    @TaskLocal static var recorder: SimulationProfileRecorder?
    /// Depth under ``usingRecorder``: `0` = this scope installed the recorder;
    /// `>0` = nested reuse on the **same** task. Named phases and ``finishProfile``
    /// require depth 0 **and** the installing task (inherited child Tasks are not owners).
    @TaskLocal static var recorderNestingDepth: Int = 0
    /// When `true`, engines skip `timeGate` / `markGateInstrumentationStarted` (shot Estimator
    /// basis-changed circuits must not land in the owner's ``SimulationProfile/gateTimings``).
    @TaskLocal static var gateRecordingSuppressed: Bool = false

    /// `true` when a recorder is active because an outer `usingRecorder` already owns this run
    /// (e.g. ``GradientCalculator`` wrapping ``Estimator``, or ``Sampler`` wrapping `backend.run`).
    /// Nested same-task calls still accumulate gate samples on the outer recorder unless
    /// ``withGateRecordingSuppressed`` is active.
    static var isNestedUnderOuterRecorder: Bool {
        recorder != nil && recorderNestingDepth > 0
    }

    /// Installing task, nesting depth 0. Only this context records named phases and
    /// ``finishProfile``. Inherited child Tasks share the recorder for `timeGate` only.
    static var isPhaseOwner: Bool {
        guard let recorder, recorderNestingDepth == 0 else { return false }
        return recorder.isInstallationOwnerTask
    }

    /// Active recorder that should receive per-gate samples (options + not suppressed).
    static var gateRecorder: SimulationProfileRecorder? {
        guard !gateRecordingSuppressed, let recorder, recorder.recordsGates else { return nil }
        return recorder
    }

    /// Installs a recorder when gate or phase timings were requested. Reentrant: if a recorder
    /// is already installed, **always** increments nesting depth (even when inner `options`
    /// would not install one) so inner `timePhase` / `finishProfile` cannot publish.
    ///
    /// The TaskLocal value is a single object for the Swift Task (and inherited child tasks).
    /// Shot-parallel workers must **only** `timeGate` (never `timePhase`, `usingRecorder`, or
    /// ``finishProfile``) unless they go through ``withWorkerRecorder``. Join those workers
    /// before ``finishProfile`` — an in-flight `timeGate` body is not yet in the snapshot.
    static func usingRecorder<T>(for options: QuantumRunOptions, _ body: () throws -> T) rethrows -> T {
        if recorder != nil {
            return try $recorderNestingDepth.withValue(recorderNestingDepth + 1, operation: body)
        }
        guard options.profiling.needsRecorder else {
            return try body()
        }
        let created = SimulationProfileRecorder(options: options.profiling)
        return try $recorder.withValue(created) {
            try $recorderNestingDepth.withValue(0, operation: body)
        }
    }

    /// Same-task helper that disables phase publishing and ``finishProfile`` for `body`.
    /// Inherited child Tasks are already non-owners (installing-task identity); use this
    /// when spawning work that stays on the owner task but must not publish phases.
    static func withWorkerRecorder<T>(_ body: () throws -> T) rethrows -> T {
        try $recorderNestingDepth.withValue(recorderNestingDepth + 1, operation: body)
    }

    /// Runs `body` without per-gate samples on the active recorder. Shot Estimator uses this
    /// so basis-changed measure circuits cannot leak indices into an outer Gradient profile.
    static func withGateRecordingSuppressed<T>(_ body: () throws -> T) rethrows -> T {
        try $gateRecordingSuppressed.withValue(true, operation: body)
    }

    /// Times a named phase **only when this task is the recorder owner**. Nested `backend.run`
    /// and inherited workers still execute `body` but do not append a phase row (Estimator must
    /// not inherit Trajectory's `sample`; Sampler / Trajectory CPU must `timePhase` at the owner).
    static func timePhase<T>(_ name: String, _ body: () throws -> T) throws -> T {
        guard isPhaseOwner, let recorder, recorder.options.recordPhaseTimings else {
            return try body()
        }
        return try recorder.timePhase(name, body)
    }

    /// Builds the profile for the **installing owner**. Nested `usingRecorder` and inherited
    /// child Tasks return `nil`.
    ///
    /// Callers that spawn concurrent `timeGate` work **must join** those workers before this
    /// returns: an in-flight `timeGate` body is excluded from the snapshot (debug-asserted).
    static func finishProfile(
        options: QuantumRunOptions,
        circuit: QuantumCircuit,
        method: QuantumSimulationMethod,
        isCPU: Bool,
        elapsed: UInt64,
        effectiveShots: Int? = nil,
        includeGateTimings: Bool = true
    ) -> SimulationProfile? {
        guard options.profiling.enabled else { return nil }
        // Nested `backend.run` / inherited workers must not publish a partial snapshot.
        // `.enabled` (no recorder) still publishes wall/memory-only telemetry.
        if recorder != nil {
            guard isPhaseOwner else { return nil }
        }
        let (stateBytes, peakBytes) = SimulationMemoryFootprint.estimate(
            qubitCount: circuit.qubitCount,
            method: method,
            isCPU: isCPU,
            shots: effectiveShots ?? options.shots,
            batchSize: options.sampleOptions.batchSize,
            circuit: circuit,
            noise: options.noise
        )
        let rec = recorder
        let gateTimings: [SimulationGateTiming]?
        if options.profiling.recordGateTimings, includeGateTimings {
            gateTimings = rec?.snapshotGateTimings()
        } else {
            gateTimings = nil
        }
        let phaseTimings: [SimulationPhaseTiming]?
        if options.profiling.recordPhaseTimings {
            phaseTimings = rec?.snapshotPhaseTimings()
        } else {
            phaseTimings = nil
        }
        return SimulationProfile(
            wallClockNanoseconds: elapsed,
            stateBytes: stateBytes,
            peakMemoryBytes: peakBytes,
            memorySource: .estimated,
            gateTimings: gateTimings,
            phaseTimings: phaseTimings
        )
    }
}

enum SimulationMemoryFootprint {
    /// State bytes plus known scratch. Matches the live working set of this run (batch pool
    /// size, CPU Double vs Metal Float32), not ``ResourceEstimate``'s policy-level heuristic.
    ///
    /// `shots` must be the **effective** shot count of this invocation (backend `options.shots`,
    /// Sampler shots, or Estimator-resolved shots), not `nil` merely because the field on
    /// ``QuantumRunOptions`` was left unset.
    static func estimate(
        qubitCount: Int,
        method: QuantumSimulationMethod,
        isCPU: Bool,
        shots: Int?,
        batchSize: Int,
        circuit: QuantumCircuit,
        noise: NoiseModel?
    ) -> (stateBytes: Int, peakBytes: Int) {
        let complexBytes = isCPU
            ? 2 * MemoryLayout<Double>.stride
            : 2 * MemoryLayout<Float32>.stride
        let dim = 1 << qubitCount
        let stateBytes: Int
        switch method {
        case .statevector, .trajectory:
            stateBytes = dim * complexBytes
        case .densityMatrix:
            stateBytes = dim * dim * complexBytes
        case .stabilizer:
            let rows = 2 * qubitCount + 1
            let bitCells = rows * 2 * qubitCount + rows
            stateBytes = max((bitCells + 7) / 8, 64)
        }

        let liveCopies = liveCopiesForEstimate(
            method: method,
            isCPU: isCPU,
            shots: shots,
            batchSize: batchSize,
            circuit: circuit,
            noise: noise
        )

        // One extra state-sized workspace: Metal renorm/measure or DM scratch buffers.
        let peakBytes = stateBytes * liveCopies + stateBytes
        return (stateBytes, peakBytes)
    }

    static func liveCopiesForEstimate(
        method: QuantumSimulationMethod,
        isCPU: Bool,
        shots: Int?,
        batchSize: Int,
        circuit: QuantumCircuit,
        noise: NoiseModel?
    ) -> Int {
        switch method {
        case .densityMatrix:
            return 1
        case .statevector, .trajectory:
            return ShotExecutionPolicy.liveStateCopies(
                circuit: circuit,
                noise: noise,
                shots: shots,
                batchSize: batchSize,
                isCPU: isCPU
            )
        case .stabilizer:
            // Independent stabilizer shots may keep one tableau per worker.
            guard let shots, shots > 0 else { return 1 }
            if ShotExecutionPolicy.mustSerial(circuit: circuit, noise: noise) {
                return 1
            }
            return min(max(batchSize, 1), shots, max(ProcessInfo.processInfo.activeProcessorCount, 1))
        }
    }
}
