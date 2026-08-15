import Foundation

/// Host CPU stabilizer / tableau backend for Clifford+measure circuits (B19).
///
/// Construct explicitly via ``QuantumBackendFactory/makeStabilizer()``. Width-only
/// ``QuantumBackendFactory/recommendMethod(qubitCount:noise:policy:)`` never selects this
/// method. Opt-in auto-selection requires ``SimulationPolicy/preferStabilizerWhenClifford``
/// and the circuit-aware recommender.
///
/// Supported gates: see ``StabilizerCircuitValidator/supportedCliffordGatesDescription``.
/// Non-Clifford gates and any noise model throw ``StabilizerError`` (no silent SV fallback).
///
/// Thread-safety: share the backend freely; each ``run`` uses its own tableau(s).
public final class StabilizerBackend: QuantumBackend, @unchecked Sendable {
    public let engine: StabilizerEngine
    public let maxQubitCount: Int
    public var method: QuantumSimulationMethod { .stabilizer }

    public init(
        engine: StabilizerEngine = StabilizerEngine(),
        maxQubitCount: Int = StabilizerTableau.maxQubitCount
    ) {
        self.engine = engine
        self.maxQubitCount = max(1, min(maxQubitCount, StabilizerTableau.maxQubitCount))
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
        if circuit.qubitCount > maxQubitCount {
            throw StabilizerError.qubitCountExceedsLimit(
                max: maxQubitCount,
                requested: circuit.qubitCount
            )
        }
        try circuit.requireFullyBound()
        try StabilizerCircuitValidator.validate(circuit)
        if let noise = options.noise, noise.hasAnyChannel {
            throw StabilizerError.noiseNotSupported
        }
        if let noise = options.noise, noise.measurementMode != .projective {
            throw StabilizerError.nonProjectiveMeasurementNotSupported
        }

        let started = DispatchTime.now()
        return try SimulationProfiling.usingRecorder(for: options) {
            var rng = makeCPURNG(seed: options.seed)

            if let shots = options.shots {
                guard shots > 0 else { throw QuantumMeasurementError.invalidShotCount(shots) }
                let counts = try SimulationProfiling.timePhase("sample") {
                    try StabilizerShotSampler.runSampleCounts(
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
                        method: .stabilizer
                    ),
                    shotCounts: counts
                )
            }

            let execution = try SimulationProfiling.timePhase("evolve") {
                var tableau = try StabilizerTableau(qubitCount: circuit.qubitCount)
                return try engine.execute(circuit, on: &tableau, rng: &rng)
            }
            return QuantumResult(
                metadata: makeCPUMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .stabilizer
                ),
                execution: execution
            )
        }
    }
}

/// Shot sampling for ``StabilizerBackend``. Mirrors ``CPUShotSampler`` RNG contracts:
/// independent circuits use ``QuantumRNG/independentShotStream(seed:shotIndex:)``;
/// mid-circuit measure / `c_if` consume one sequential stream.
enum StabilizerShotSampler {

    static func runSampleCounts(
        circuit: QuantumCircuit,
        engine: StabilizerEngine,
        shots: Int,
        rng: inout QuantumRNG,
        seed: UInt64?,
        options: SampleCountOptions,
        cancellationCheck: (() throws -> Void)?
    ) throws -> ShotCounts {
        if ShotExecutionPolicy.canBatch(circuit: circuit, noise: nil) {
            let streamSeed = independentStreamRoot(seed: seed, rng: rng)
            _ = rng
            return try runIndependentShots(
                circuit: circuit,
                engine: engine,
                shots: shots,
                seed: streamSeed,
                poolSize: min(
                    max(options.batchSize, 1),
                    shots,
                    max(ProcessInfo.processInfo.activeProcessorCount, 1)
                ),
                cancellationCheck: cancellationCheck
            )
        }

        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, 1 << min(circuit.qubitCount, 16)))
        var tableau = try StabilizerTableau(qubitCount: circuit.qubitCount)
        for _ in 0..<shots {
            try cancellationCheck?()
            tableau.resetToZero()
            let outcome = try engine.sampleTerminalOutcome(
                circuit: circuit,
                tableau: &tableau,
                rng: &rng
            )
            histogram[outcome, default: 0] += 1
        }
        return ShotCounts(shots: shots, counts: histogram)
    }

    private static func independentStreamRoot(seed: UInt64?, rng: QuantumRNG) -> UInt64? {
        if let seed { return seed }
        if case .seeded(let state) = rng { return state }
        return nil
    }

    private static func runIndependentShots(
        circuit: QuantumCircuit,
        engine: StabilizerEngine,
        shots: Int,
        seed: UInt64?,
        poolSize: Int,
        cancellationCheck: (() throws -> Void)?
    ) throws -> ShotCounts {
        let workers = min(poolSize, shots)
        if workers <= 1 {
            var histogram: [Int: Int] = [:]
            var tableau = try StabilizerTableau(qubitCount: circuit.qubitCount)
            for shotIndex in 0..<shots {
                try cancellationCheck?()
                var rng: QuantumRNG
                if let seed {
                    rng = QuantumRNG.independentShotStream(seed: seed, shotIndex: shotIndex)
                } else {
                    rng = .hardware
                }
                tableau.resetToZero()
                let outcome = try engine.sampleTerminalOutcome(
                    circuit: circuit,
                    tableau: &tableau,
                    rng: &rng
                )
                histogram[outcome, default: 0] += 1
            }
            return ShotCounts(shots: shots, counts: histogram)
        }

        // Concurrent independent shots in waves (same pattern as ``CPUShotSampler``).
        let ping = ShotParallelRuntime.makeCancellationPing(cancellationCheck)
        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, 1 << min(circuit.qubitCount, 16)))
        var completed = 0
        while completed < shots {
            try ping.call()
            let active = min(workers, shots - completed)
            let shotBase = completed
            let outcomes = try ShotParallelRuntime.concurrentMap(count: active) { offset -> Int in
                let shotIndex = shotBase + offset
                var tableau = try StabilizerTableau(qubitCount: circuit.qubitCount)
                var rng: QuantumRNG
                if let seed {
                    rng = QuantumRNG.independentShotStream(seed: seed, shotIndex: shotIndex)
                } else {
                    rng = .hardware
                }
                return try engine.sampleTerminalOutcome(
                    circuit: circuit,
                    tableau: &tableau,
                    rng: &rng
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
