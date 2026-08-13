import Foundation

/// Monte-Carlo statevector trajectory ensemble backend.
///
/// Wraps a statevector engine (Metal or CPU) and tags results as
/// ``QuantumSimulationMethod/trajectory``. Rejects localized gate noise and
/// ``MeasurementMode/dephasingOnly`` — the same constraints as SV engines.
///
/// Thread-safety: same as the wrapped statevector backend (distinct per-shot states).
public final class TrajectoryBackend: QuantumBackend, @unchecked Sendable {
    public var method: QuantumSimulationMethod { .trajectory }

    private let metal: StatevectorBackend?
    private let cpu: CPUStatevectorBackend?

    public init(wrapping backend: any QuantumBackend) {
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
            preconditionFailure("TrajectoryBackend requires a statevector backend")
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
        try validateTrajectoryNoise(options.noise)
        let started = DispatchTime.now()
        try circuit.requireFullyBound()

        if let metal {
            if let shots = options.shots {
                var rng = makeRNG(seed: options.seed)
                let counts = try QuantumMeasurement.runSampleCountsRNG(
                    circuit: circuit,
                    engine: metal.engine,
                    shots: shots,
                    rng: &rng,
                    noise: options.noise,
                    options: options.sampleOptions
                )
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

            let state = try StateVector(qubitCount: circuit.qubitCount)
            var rng = makeRNG(seed: options.seed)
            let execution = try metal.engine.executeRNG(
                circuit,
                on: state,
                rng: &rng,
                noise: options.noise
            )
            return QuantumResult(
                metadata: makeMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .trajectory
                ),
                execution: execution
            )
        }

        guard let cpu else {
            preconditionFailure("TrajectoryBackend missing underlying engine")
        }

        // Reuse CPU SV shot / execute paths, then retag metadata as trajectory.
        let underlying = try cpu.run(circuit: circuit, options: options)
        return QuantumResult(
            metadata: QuantumResultMetadata(
                method: .trajectory,
                seed: options.seed,
                deviceName: "CPU",
                wallClockNanoseconds: DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds,
                qubitCount: circuit.qubitCount,
                gateCount: circuit.gates.count,
                noiseSnapshot: options.noise,
                pipelineHash: PipelineFingerprint.hash(
                    circuit: circuit,
                    method: .trajectory,
                    options: options
                )
            ),
            execution: underlying.execution,
            shotCounts: underlying.shotCounts
        )
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
