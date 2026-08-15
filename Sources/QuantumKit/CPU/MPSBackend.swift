import Foundation

/// Host CPU matrix-product-state backend (B18 MVP, 1D open chain).
///
/// Construct explicitly via ``QuantumBackendFactory/makeMPS(configuration:)``.
/// Width-only ``QuantumBackendFactory/recommendMethod(qubitCount:noise:policy:)`` never
/// selects ``QuantumSimulationMethod/mps``.
///
/// **χ / SVD:** ``MPSConfiguration/maxBondDimension``; relative cutoff
/// ``MPSConfiguration/svdTruncationThreshold``. Evolution is always local adjacent SVD
/// updates with on-the-fly χ truncation. Amplitude export contracts sites when
/// `n ≤ maxAmplitudeExportQubits`. Terminal Z shots sample the MPS bond-by-bond
/// (no full amplitude vector required).
///
/// **Topology:** non-adjacent two-qubit gates use a **SWAP bubble chain** (documented), not
/// a hard reject. See ``MPSEngine`` for the supported gate subset.
///
/// Thread-safety: share the backend; each ``run`` allocates its own ``MPSState``.
public final class MPSBackend: QuantumBackend, @unchecked Sendable {
    public let engine: MPSEngine
    public let configuration: MPSConfiguration
    public var method: QuantumSimulationMethod { .mps }

    public init(configuration: MPSConfiguration = .default) {
        self.configuration = configuration
        self.engine = MPSEngine(configuration: configuration)
    }

    public init(engine: MPSEngine) {
        self.engine = engine
        self.configuration = engine.configuration
    }

    public func run(
        circuit: QuantumCircuit,
        options: QuantumRunOptions = QuantumRunOptions()
    ) throws -> QuantumResult {
        try executeRun(circuit: circuit, options: options, cancellationCheck: nil)
    }

    func executeRun(
        circuit: QuantumCircuit,
        options: QuantumRunOptions,
        cancellationCheck: (() throws -> Void)?
    ) throws -> QuantumResult {
        if circuit.qubitCount > configuration.maxQubitCount {
            throw MPSError.qubitCountExceedsLimit(
                max: configuration.maxQubitCount,
                requested: circuit.qubitCount
            )
        }
        try circuit.requireFullyBound()
        if let noise = options.noise, noise.hasAnyChannel {
            throw MPSError.noiseNotSupported
        }

        let started = DispatchTime.now()
        return try SimulationProfiling.usingRecorder(for: options) {
            var rng = makeCPURNG(seed: options.seed)

            if let shots = options.shots {
                guard shots > 0 else { throw QuantumMeasurementError.invalidShotCount(shots) }
                let counts = try SimulationProfiling.timePhase("sample") {
                    try MPSShotSampler.runSampleCounts(
                        circuit: circuit,
                        engine: engine,
                        shots: shots,
                        rng: &rng,
                        seed: options.seed,
                        options: options.sampleOptions,
                        cancellationCheck: cancellationCheck
                    )
                }
                return QuantumResult(
                    metadata: makeCPUMetadata(
                        circuit: circuit,
                        options: options,
                        started: started,
                        method: .mps
                    ),
                    shotCounts: counts
                )
            }

            let execution = try SimulationProfiling.timePhase("evolve") {
                var state = try MPSState(
                    qubitCount: circuit.qubitCount,
                    configuration: configuration
                )
                return try engine.execute(circuit, on: &state)
            }
            return QuantumResult(
                metadata: makeCPUMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .mps
                ),
                execution: execution
            )
        }
    }
}

enum MPSShotSampler {
    static func runSampleCounts(
        circuit: QuantumCircuit,
        engine: MPSEngine,
        shots: Int,
        rng: inout QuantumRNG,
        seed: UInt64?,
        options: SampleCountOptions,
        cancellationCheck: (() throws -> Void)?
    ) throws -> ShotCounts {
        // Mid-circuit measure is unsupported, so circuits are always independent.
        let streamSeed: UInt64?
        if let seed {
            streamSeed = seed
        } else if case .seeded(let s) = rng {
            streamSeed = s
        } else {
            streamSeed = nil
        }
        _ = rng

        let workers = min(
            max(options.batchSize, 1),
            shots,
            max(ProcessInfo.processInfo.activeProcessorCount, 1)
        )
        if workers <= 1 {
            var histogram: [Int: Int] = [:]
            var state = try MPSState(
                qubitCount: circuit.qubitCount,
                configuration: engine.configuration
            )
            for shotIndex in 0..<shots {
                try cancellationCheck?()
                var shotRNG: QuantumRNG
                if let streamSeed {
                    shotRNG = QuantumRNG.independentShotStream(seed: streamSeed, shotIndex: shotIndex)
                } else {
                    shotRNG = .hardware
                }
                state.resetToZero()
                let outcome = try engine.sampleTerminalOutcome(
                    circuit: circuit,
                    state: &state,
                    rng: &shotRNG
                )
                histogram[outcome, default: 0] += 1
            }
            return ShotCounts(shots: shots, counts: histogram)
        }

        let ping = ShotParallelRuntime.makeCancellationPing(cancellationCheck)
        var histogram: [Int: Int] = [:]
        var completed = 0
        while completed < shots {
            try ping.call()
            let active = min(workers, shots - completed)
            let shotBase = completed
            let outcomes = try ShotParallelRuntime.concurrentMap(count: active) { offset -> Int in
                let shotIndex = shotBase + offset
                var state = try MPSState(
                    qubitCount: circuit.qubitCount,
                    configuration: engine.configuration
                )
                var shotRNG: QuantumRNG
                if let streamSeed {
                    shotRNG = QuantumRNG.independentShotStream(seed: streamSeed, shotIndex: shotIndex)
                } else {
                    shotRNG = .hardware
                }
                return try engine.sampleTerminalOutcome(
                    circuit: circuit,
                    state: &state,
                    rng: &shotRNG
                )
            }
            for outcome in outcomes {
                histogram[outcome, default: 0] += 1
            }
            completed += active
        }
        return ShotCounts(shots: shots, counts: histogram)
    }
}
