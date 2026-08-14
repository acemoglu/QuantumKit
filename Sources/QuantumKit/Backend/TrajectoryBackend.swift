import Foundation

public enum TrajectoryBackendError: Error, Equatable {
    /// Ensemble semantics require a positive ``QuantumRunOptions/shots`` count.
    case shotsRequired
    /// ``TrajectoryBackend`` can only wrap Metal or CPU statevector backends.
    case unsupportedUnderlyingBackend
}

/// Monte-Carlo statevector trajectory ensemble backend.
///
/// Wraps a statevector engine (Metal or CPU) and tags results as
/// ``QuantumSimulationMethod/trajectory``. Rejects localized gate noise and
/// ``MeasurementMode/dephasingOnly`` — the same constraints as SV engines.
///
/// ``run(circuit:options:)`` **requires** ``QuantumRunOptions/shots``; a single
/// unraveling path is not a mixed-state trajectory result. Use
/// ``averageProbabilities(circuit:trajectories:seed:noise:)`` for explicit ensembles
/// without a histogram.
///
/// Thread-safety: same as the wrapped statevector backend (distinct per-shot states;
/// CPU independent shots may run concurrently — one state and RNG stream per worker).
public final class TrajectoryBackend: QuantumBackend, @unchecked Sendable {
    public var method: QuantumSimulationMethod { .trajectory }
    /// Device of the wrapped statevector backend, not merely whether Metal exists on the host.
    public var deviceName: String {
        if metal != nil {
            return MetalRuntime.deviceName ?? "Metal"
        }
        return "CPU"
    }

    private let metal: StatevectorBackend?
    private let cpu: CPUStatevectorBackend?

    public init(wrapping backend: any QuantumBackend) throws {
        if let metal = backend as? StatevectorBackend {
            self.metal = metal
            self.cpu = nil
        } else if let cpu = backend as? CPUStatevectorBackend {
            self.metal = nil
            self.cpu = cpu
        } else if let nested = backend as? TrajectoryBackend {
            self.metal = nested.metal
            self.cpu = nested.cpu
        } else {
            throw TrajectoryBackendError.unsupportedUnderlyingBackend
        }
    }

    public init(engine: QuantumEngine) {
        self.metal = StatevectorBackend(engine: engine)
        self.cpu = nil
    }

    public init(engine: CPUStatevectorEngine) {
        self.metal = nil
        self.cpu = CPUStatevectorBackend(engine: engine)
    }

    public func run(circuit: QuantumCircuit, options: QuantumRunOptions = QuantumRunOptions()) throws -> QuantumResult {
        try executeRun(circuit: circuit, options: options, cancellationCheck: nil)
    }

    /// Async shot ensemble with cooperative cancellation between shots / gates.
    public func runAsync(
        circuit: QuantumCircuit,
        options: QuantumRunOptions = QuantumRunOptions()
    ) async throws -> QuantumResult {
        try CircuitCancellation.mapCancellation {
            try self.runCancellable(circuit: circuit, options: options)
        }
    }

    func runCancellable(
        circuit: QuantumCircuit,
        options: QuantumRunOptions
    ) throws -> QuantumResult {
        try executeRun(
            circuit: circuit,
            options: options,
            cancellationCheck: { try CircuitCancellation.check() }
        )
    }

    func executeRun(
        circuit: QuantumCircuit,
        options: QuantumRunOptions,
        cancellationCheck: (() throws -> Void)?
    ) throws -> QuantumResult {
        try validateTrajectoryNoise(options.noise)
        guard let shots = options.shots, shots > 0 else {
            throw TrajectoryBackendError.shotsRequired
        }
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        return try SimulationProfiling.usingRecorder(for: options) {
            let counts = try SimulationProfiling.timePhase("sample") { () -> ShotCounts in
                if let metal {
                    var rng = makeRNG(seed: options.seed)
                    return try QuantumMeasurement.runSampleCountsRNG(
                        circuit: circuit,
                        engine: metal.engine,
                        shots: shots,
                        rng: &rng,
                        noise: options.noise,
                        options: options.sampleOptions,
                        cancellationCheck: cancellationCheck
                    )
                }
                guard let cpu else {
                    throw TrajectoryBackendError.unsupportedUnderlyingBackend
                }
                let underlying: QuantumResult
                if cancellationCheck != nil {
                    underlying = try cpu.runCancellable(circuit: circuit, options: options)
                } else {
                    underlying = try cpu.run(circuit: circuit, options: options)
                }
                return underlying.shotCounts ?? ShotCounts(shots: shots, counts: [:])
            }
            if metal != nil {
                return QuantumResult(
                    metadata: makeMetadata(
                        circuit: circuit,
                        options: options,
                        started: started,
                        method: .trajectory
                    ),
                    shotCounts: counts
                )
            }
            return QuantumResult(
                metadata: makeCPUMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .trajectory
                ),
                execution: nil,
                shotCounts: counts
            )
        }
    }

    /// Ensemble-averaged computational-basis probabilities (Phase11-style).
    public func averageProbabilities(
        circuit: QuantumCircuit,
        trajectories: Int,
        seed: UInt64,
        noise: NoiseModel? = nil
    ) throws -> [QFloat] {
        try validateTrajectoryNoise(noise)
        guard trajectories > 0 else {
            throw QuantumMeasurementError.invalidShotCount(trajectories)
        }
        try circuit.requireFullyBound()

        let dim = 1 << circuit.qubitCount
        var sums = Array(repeating: Double(0), count: dim)
        var rng: QuantumRNG = .seeded(seed)

        if let metal {
            for _ in 0..<trajectories {
                let state = try StateVector(qubitCount: circuit.qubitCount)
                _ = try metal.engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
                let probs = try QuantumMeasurement.probabilities(state: state, engine: metal.engine)
                for index in 0..<dim {
                    sums[index] += Double(probs[index])
                }
            }
        } else if let cpu {
            for _ in 0..<trajectories {
                let state = try CPUStateVector(qubitCount: circuit.qubitCount)
                _ = try cpu.engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
                let probs = state.probabilitiesDouble()
                for index in 0..<dim {
                    sums[index] += probs[index]
                }
            }
        }

        let scale = Double(trajectories)
        return sums.map { QFloat($0 / scale) }
    }

    private func validateTrajectoryNoise(_ noise: NoiseModel?) throws {
        guard let noise else { return }
        if noise.hasLocalizedGateNoise {
            if metal != nil {
                throw QuantumEngineError.localizedNoiseRequiresDensityMatrixBackend
            }
            throw CPUEngineError.localizedNoiseRequiresDensityMatrixBackend
        }
        if noise.measurementMode != .projective {
            if metal != nil {
                throw QuantumEngineError.nonProjectiveMeasurementRequiresDensityMatrixBackend
            }
            throw CPUEngineError.nonProjectiveMeasurementRequiresDensityMatrixBackend
        }
    }
}
