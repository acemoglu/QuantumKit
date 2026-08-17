import Foundation

/// CPU statevector shot sampling. Independent shots (``ShotExecutionPolicy/canBatch``) may
/// run concurrently — one ``CPUStateVector`` and one RNG stream per worker. Coupled shots
/// (``ShotExecutionPolicy/mustSerial``) stay on a single sequential ``QuantumRNG``.
///
/// **RNG contract:** when ``ShotExecutionPolicy/canBatch(circuit:noise:)``, each shot uses
/// ``QuantumRNG/independentShotStream(seed:shotIndex:)`` (or `.hardware` if no seed).
/// Stream root priority: ``seed`` argument if non-`nil`, else the initial state of a
/// ``QuantumRNG/seeded`` `rng`, else hardware entropy per shot. The `rng` value is **never
/// advanced** on that path — do not rely on post-call `rng` mutation. `batchSize` only sizes
/// the worker pool and does **not** change histograms *within that contract*.
/// ``mustSerial`` consumes `rng` as one sequential stream.
///
/// This per-shot stream schedule intentionally differs from a single ``QuantumRNG/seeded``
/// advanced across shots (legacy CPU) and from Metal’s sequential measurement RNG.
enum CPUShotSampler {

    /// Test hook: fired once on the owner thread when independent-shot sampling begins
    /// (before the first worker wave). Cleared by the callee after invoke.
    nonisolated(unsafe) static var onIndependentSamplingStarted: (() -> Void)?
    /// Test hook: fired once from a GCD worker on the first independent shot body
    /// (after pool allocation, inside ``concurrentPerform``). Cleared after invoke.
    nonisolated(unsafe) static var onIndependentWorkerShotStarted: (() -> Void)?
    private static let testHookLock = NSLock()

    private static func consumeIndependentSamplingStartedHook() {
        testHookLock.lock()
        let started = onIndependentSamplingStarted
        onIndependentSamplingStarted = nil
        testHookLock.unlock()
        started?()
    }

    private static func consumeIndependentWorkerShotStartedHook() {
        testHookLock.lock()
        let started = onIndependentWorkerShotStarted
        onIndependentWorkerShotStarted = nil
        testHookLock.unlock()
        started?()
    }

    static func runSampleCountsRNG(
        circuit: QuantumCircuit,
        engine: CPUStatevectorEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        options: SampleCountOptions,
        seed: UInt64?,
        cancellationCheck: (() throws -> Void)? = nil
    ) throws -> ShotCounts {
        guard shots > 0 else {
            throw QuantumMeasurementError.invalidShotCount(shots)
        }

        if options.preferPreparedSampling,
           circuit.allowsPreparedStatevectorShotSampling(noise: noise),
           let prefix = circuit.preparedShotUnitaryPrefix() {
            try cancellationCheck?()
            let state = try CPUStateVector(qubitCount: circuit.qubitCount)
            _ = try engine.executeRNG(
                prefix,
                on: state,
                rng: &rng,
                noise: nil,
                cancellationCheck: cancellationCheck
            )
            return try DensityMatrixShotSampler.sampleTerminalShots(
                probabilities: state.probabilities(),
                shots: shots,
                measuredQubitCount: circuit.qubitCount,
                rng: &rng,
                noise: noise,
                cancellationCheck: cancellationCheck
            )
        }

        // Independent shots always use per-shot streams so batchSize 1 vs N match.
        if ShotExecutionPolicy.canBatch(circuit: circuit, noise: noise) {
            let streamSeed = Self.independentStreamRoot(seed: seed, rng: rng)
            // Do not advance `rng` on this path (seed / seeded-root / hardware is authoritative).
            _ = rng
            let poolSize = ShotExecutionPolicy.cpuWorkerPoolSize(
                circuit: circuit,
                noise: noise,
                requested: options.batchSize,
                shots: shots
            )
            return try runIndependentShots(
                circuit: circuit,
                engine: engine,
                shots: shots,
                noise: noise,
                poolSize: poolSize,
                seed: streamSeed,
                cancellationCheck: cancellationCheck
            )
        }

        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, 1 << min(circuit.qubitCount, 16)))
        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        for _ in 0..<shots {
            try cancellationCheck?()
            let outcome = try runOneShot(
                circuit: circuit,
                engine: engine,
                state: state,
                rng: &rng,
                noise: noise,
                cancellationCheck: cancellationCheck
            )
            histogram[outcome, default: 0] += 1
        }
        return ShotCounts(shots: shots, counts: histogram)
    }

    /// Root for ``independentShotStream``: explicit `seed`, else initial ``seeded`` state of
    /// `rng`, else `nil` (hardware per shot).
    private static func independentStreamRoot(seed: UInt64?, rng: QuantumRNG) -> UInt64? {
        if let seed { return seed }
        if case .seeded(let state) = rng { return state }
        return nil
    }

    private static func runIndependentShots(
        circuit: QuantumCircuit,
        engine: CPUStatevectorEngine,
        shots: Int,
        noise: NoiseModel?,
        poolSize: Int,
        seed: UInt64?,
        cancellationCheck: (() throws -> Void)?
    ) throws -> ShotCounts {
        consumeIndependentSamplingStartedHook()

        let workers = min(poolSize, max(ProcessInfo.processInfo.activeProcessorCount, 1), shots)
        let ping = ShotParallelRuntime.makeCancellationPing(cancellationCheck)

        if workers <= 1 {
            return try runIndependentSerial(
                circuit: circuit,
                engine: engine,
                shots: shots,
                noise: noise,
                seed: seed,
                ping: ping
            )
        }

        let pool = try (0..<workers).map { _ in
            try CPUStateVector(qubitCount: circuit.qubitCount)
        }
        let recorder = SimulationProfiling.recorder
        let gateSuppressed = SimulationProfiling.gateRecordingSuppressed

        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, 1 << min(circuit.qubitCount, 16)))
        var completed = 0

        while completed < shots {
            try ping.call()
            let active = min(workers, shots - completed)
            let work = IndependentShotWork(
                circuit: circuit,
                engine: engine,
                noise: noise,
                seed: seed,
                shotBase: completed,
                pool: pool,
                recorder: recorder,
                gateRecordingSuppressed: gateSuppressed,
                ping: ping
            )
            let outcomes = try ShotParallelRuntime.concurrentMap(count: active) { offset in
                try work.run(offset: offset)
            }
            for outcome in outcomes {
                histogram[outcome, default: 0] += 1
            }
            completed += active
        }

        return ShotCounts(shots: shots, counts: histogram)
    }

    private static func runIndependentSerial(
        circuit: QuantumCircuit,
        engine: CPUStatevectorEngine,
        shots: Int,
        noise: NoiseModel?,
        seed: UInt64?,
        ping: CancellationPing
    ) throws -> ShotCounts {
        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, 1 << min(circuit.qubitCount, 16)))
        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        for shotIndex in 0..<shots {
            try ping.call()
            var rng: QuantumRNG
            if let seed {
                rng = QuantumRNG.independentShotStream(seed: seed, shotIndex: shotIndex)
            } else {
                rng = .hardware
            }
            let outcome = try runOneShot(
                circuit: circuit,
                engine: engine,
                state: state,
                rng: &rng,
                noise: noise,
                cancellationCheck: { try ping.call() }
            )
            histogram[outcome, default: 0] += 1
        }
        return ShotCounts(shots: shots, counts: histogram)
    }

    static func runOneShot(
        circuit: QuantumCircuit,
        engine: CPUStatevectorEngine,
        state: CPUStateVector,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        cancellationCheck: (() throws -> Void)?
    ) throws -> Int {
        try cancellationCheck?()
        state.resetToZero()
        _ = try engine.executeRNG(
            circuit,
            on: state,
            rng: &rng,
            noise: noise,
            cancellationCheck: cancellationCheck
        )
        return try engine.measureCollapse(
            on: state,
            qubits: Array(0..<circuit.qubitCount),
            rng: &rng,
            noise: noise
        )
    }

    static func notifyIndependentWorkerShotStartedForTests() {
        consumeIndependentWorkerShotStartedHook()
    }
}

/// Captures shot work for `DispatchQueue.concurrentPerform`.
private final class IndependentShotWork: @unchecked Sendable {
    let circuit: QuantumCircuit
    let engine: CPUStatevectorEngine
    let noise: NoiseModel?
    let seed: UInt64?
    let shotBase: Int
    let pool: [CPUStateVector]
    let recorder: SimulationProfileRecorder?
    let gateRecordingSuppressed: Bool
    let ping: CancellationPing

    init(
        circuit: QuantumCircuit,
        engine: CPUStatevectorEngine,
        noise: NoiseModel?,
        seed: UInt64?,
        shotBase: Int,
        pool: [CPUStateVector],
        recorder: SimulationProfileRecorder?,
        gateRecordingSuppressed: Bool,
        ping: CancellationPing
    ) {
        self.circuit = circuit
        self.engine = engine
        self.noise = noise
        self.seed = seed
        self.shotBase = shotBase
        self.pool = pool
        self.recorder = recorder
        self.gateRecordingSuppressed = gateRecordingSuppressed
        self.ping = ping
    }

    func run(offset: Int) throws -> Int {
        try ping.call()
        CPUShotSampler.notifyIndependentWorkerShotStartedForTests()
        return try ShotParallelRuntime.installWorkerRecorder(
            recorder: recorder,
            gateRecordingSuppressed: gateRecordingSuppressed
        ) {
            let shotIndex = shotBase + offset
            var rng: QuantumRNG
            if let seed {
                rng = QuantumRNG.independentShotStream(seed: seed, shotIndex: shotIndex)
            } else {
                rng = .hardware
            }
            return try CPUShotSampler.runOneShot(
                circuit: circuit,
                engine: engine,
                state: pool[offset],
                rng: &rng,
                noise: noise,
                cancellationCheck: { try self.ping.call() }
            )
        }
    }
}
