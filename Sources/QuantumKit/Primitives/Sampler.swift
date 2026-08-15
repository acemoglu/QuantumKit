import Foundation

public enum SamplerError: Error, Equatable {
    case unsupportedBackend
}

/// Standardized measurement distribution returned by ``Sampler``.
public struct SamplerResult: Sendable, Equatable {
    public let metadata: QuantumResultMetadata
    public let qubitCount: Int
    /// Bitstring → probability (exact Born rule) or empirical frequency (shots).
    ///
    /// Keys use ``QubitBitOrdering/bitstringMSB`` (leftmost = qubit `n-1`). Integer
    /// ``shotCounts`` keys remain ``QubitBitOrdering/engineLSB``.
    public let quasiProbabilities: [String: QFloat]
    /// Histogram when `shots` were requested; `nil` for exact distributions.
    public let shotCounts: ShotCounts?

    public init(
        metadata: QuantumResultMetadata,
        qubitCount: Int,
        quasiProbabilities: [String: QFloat],
        shotCounts: ShotCounts? = nil
    ) {
        self.metadata = metadata
        self.qubitCount = qubitCount
        self.quasiProbabilities = quasiProbabilities
        self.shotCounts = shotCounts
    }
}

/// High-level primitive for extracting measurement distributions from a circuit.
///
/// Without `shots`, returns exact Born-rule / diagonal probabilities. With `shots`, returns a
/// shot histogram plus empirical frequencies. Supported on statevector, density-matrix
/// (prepared-ρ batching when possible), trajectory, and host MPS backends.
///
/// When ``QuantumRunOptions/resilience`` includes ``ResilienceOptions/readoutMitigation``,
/// host-side inverse readout correction is applied to shot histograms after sampling
/// (exact Born paths are unchanged). ``ResilienceOptions/zne`` and ``ResilienceOptions/pec``
/// are Estimator-only and are ignored here (also omitted from ``pipelineHash``).
public struct Sampler: Sendable {
    public init() {}

    public func run(
        circuit: QuantumCircuit,
        backend: any QuantumBackend,
        options: QuantumRunOptions = QuantumRunOptions()
    ) throws -> SamplerResult {
        try SimulationProfiling.usingRecorder(for: options) {
            if let statevectorBackend = backend as? StatevectorBackend {
                return try sample(circuit: circuit, backend: statevectorBackend, options: options)
            }
            if let densityBackend = backend as? DensityMatrixBackend {
                return try sample(circuit: circuit, backend: densityBackend, options: options)
            }
            if let cpuDensity = backend as? CPUDensityMatrixBackend {
                return try sample(circuit: circuit, backend: cpuDensity, options: options)
            }
            if let trajectory = backend as? TrajectoryBackend {
                return try sample(circuit: circuit, backend: trajectory, options: options)
            }
            if let cpuSV = backend as? CPUStatevectorBackend {
                return try sample(circuit: circuit, backend: cpuSV, options: options)
            }
            if let mps = backend as? MPSBackend {
                return try sample(circuit: circuit, backend: mps, options: options)
            }
            throw SamplerError.unsupportedBackend
        }
    }

    /// Apply opt-in readout mitigation and build quasi-probabilities from the kept shots.
    private func mitigatedShotResult(
        counts: ShotCounts,
        circuit: QuantumCircuit,
        options: QuantumRunOptions,
        started: DispatchTime,
        method: QuantumSimulationMethod,
        deviceName: String? = nil,
        pipelineHash: String? = nil
    ) throws -> SamplerResult {
        let shots = counts.shots
        let finalCounts = try applyResilienceIfNeeded(
            counts,
            qubitCount: circuit.qubitCount,
            resilience: options.resilience
        )
        let bitstrings = finalCounts.bitstringCounts(qubitCount: circuit.qubitCount)
        let quasiProbabilities = bitstrings.mapValues { QFloat($0) / QFloat(max(shots, 1)) }
        let resolvedHash = pipelineHash ?? PipelineFingerprint.hash(
            circuit: circuit,
            method: method,
            // Sampler applies readout mitigation only — do not hash Estimator-only ZNE/PEC.
            options: {
                var fingerprintOptions = options
                fingerprintOptions.resilience = options.resilience.readoutOnly
                return fingerprintOptions
            }()
        )
        return SamplerResult(
            metadata: makeSamplerMetadata(
                circuit: circuit,
                options: options,
                started: started,
                method: method,
                deviceName: deviceName,
                effectiveShots: shots,
                pipelineHash: resolvedHash
            ),
            qubitCount: circuit.qubitCount,
            quasiProbabilities: quasiProbabilities,
            shotCounts: finalCounts
        )
    }

    private func sample(
        circuit: QuantumCircuit,
        backend: StatevectorBackend,
        options: QuantumRunOptions
    ) throws -> SamplerResult {
        let started = DispatchTime.now()

        if let shots = options.shots {
            var rng = makePrimitiveRNG(seed: options.seed)
            let counts = try SimulationProfiling.timePhase("sample") {
                try QuantumMeasurement.runSampleCountsRNG(
                    circuit: circuit,
                    engine: backend.engine,
                    shots: shots,
                    rng: &rng,
                    noise: options.noise,
                    options: options.sampleOptions
                )
            }
            return try mitigatedShotResult(
                counts: counts,
                circuit: circuit,
                options: options,
                started: started,
                method: .statevector
            )
        }

        let quasiProbabilities = try SimulationProfiling.timePhase("evolve") {
            let state = try StateVector(qubitCount: circuit.qubitCount)
            var rng = makePrimitiveRNG(seed: options.seed)
            _ = try backend.engine.executeRNG(
                circuit,
                on: state,
                rng: &rng,
                noise: options.noise
            )

            let probabilities = try QuantumMeasurement.probabilities(state: state, engine: backend.engine)
            return makeBitstringProbabilities(
                probabilities: probabilities,
                qubitCount: circuit.qubitCount
            )
        }

        return SamplerResult(
            metadata: makeSamplerMetadata(
                circuit: circuit,
                options: options,
                started: started,
                method: .statevector
            ),
            qubitCount: circuit.qubitCount,
            quasiProbabilities: quasiProbabilities
        )
    }

    private func sample(
        circuit: QuantumCircuit,
        backend: DensityMatrixBackend,
        options: QuantumRunOptions
    ) throws -> SamplerResult {
        let started = DispatchTime.now()

        if let shots = options.shots {
            var rng = makePrimitiveRNG(seed: options.seed)
            let counts = try SimulationProfiling.timePhase("sample") {
                try DensityMatrixShotSampler.runSampleCountsRNG(
                    circuit: circuit,
                    engine: backend.engine,
                    shots: shots,
                    rng: &rng,
                    noise: options.noise
                )
            }
            return try mitigatedShotResult(
                counts: counts,
                circuit: circuit,
                options: options,
                started: started,
                method: .densityMatrix
            )
        }

        let quasiProbabilities = try SimulationProfiling.timePhase("evolve") {
            let density = try DensityMatrix(qubitCount: circuit.qubitCount)
            var rng = makePrimitiveRNG(seed: options.seed)
            _ = try backend.engine.executeRNG(
                circuit,
                on: density,
                rng: &rng,
                noise: options.noise
            )
            return makeBitstringProbabilities(
                probabilities: backend.engine.probabilities(of: density),
                qubitCount: circuit.qubitCount
            )
        }

        return SamplerResult(
            metadata: makeSamplerMetadata(
                circuit: circuit,
                options: options,
                started: started,
                method: .densityMatrix
            ),
            qubitCount: circuit.qubitCount,
            quasiProbabilities: quasiProbabilities
        )
    }

    private func sample(
        circuit: QuantumCircuit,
        backend: CPUDensityMatrixBackend,
        options: QuantumRunOptions
    ) throws -> SamplerResult {
        let started = DispatchTime.now()

        if let shots = options.shots {
            var rng = makePrimitiveRNG(seed: options.seed)
            let counts = try SimulationProfiling.timePhase("sample") {
                try DensityMatrixShotSampler.runSampleCountsRNG(
                    circuit: circuit,
                    engine: backend.engine,
                    shots: shots,
                    rng: &rng,
                    noise: options.noise
                )
            }
            return try mitigatedShotResult(
                counts: counts,
                circuit: circuit,
                options: options,
                started: started,
                method: .densityMatrix,
                deviceName: "CPU"
            )
        }

        let quasiProbabilities = try SimulationProfiling.timePhase("evolve") {
            let density = try CPUDensityMatrix(qubitCount: circuit.qubitCount)
            var rng = makePrimitiveRNG(seed: options.seed)
            _ = try backend.engine.executeRNG(circuit, on: density, rng: &rng, noise: options.noise)
            return makeBitstringProbabilities(
                probabilities: density.probabilities(),
                qubitCount: circuit.qubitCount
            )
        }
        return SamplerResult(
            metadata: makeSamplerMetadata(
                circuit: circuit,
                options: options,
                started: started,
                method: .densityMatrix,
                deviceName: "CPU"
            ),
            qubitCount: circuit.qubitCount,
            quasiProbabilities: quasiProbabilities
        )
    }

    private func sample(
        circuit: QuantumCircuit,
        backend: TrajectoryBackend,
        options: QuantumRunOptions
    ) throws -> SamplerResult {
        guard options.shots != nil else {
            throw TrajectoryBackendError.shotsRequired
        }
        return try sampleShotsFromBackendRun(
            circuit: circuit,
            backend: backend,
            options: options,
            method: .trajectory,
            deviceName: backend.deviceName
        )
    }

    private func sample(
        circuit: QuantumCircuit,
        backend: CPUStatevectorBackend,
        options: QuantumRunOptions
    ) throws -> SamplerResult {
        let started = DispatchTime.now()
        if options.shots != nil {
            return try sampleShotsFromBackendRun(
                circuit: circuit,
                backend: backend,
                options: options,
                method: .statevector,
                deviceName: "CPU"
            )
        }
        let quasiProbabilities = try SimulationProfiling.timePhase("evolve") {
            let state = try CPUStateVector(qubitCount: circuit.qubitCount)
            var rng = makePrimitiveRNG(seed: options.seed)
            _ = try backend.engine.executeRNG(circuit, on: state, rng: &rng, noise: options.noise)
            return makeBitstringProbabilities(
                probabilities: state.probabilities(),
                qubitCount: circuit.qubitCount
            )
        }
        return SamplerResult(
            metadata: makeSamplerMetadata(
                circuit: circuit,
                options: options,
                started: started,
                method: .statevector,
                deviceName: "CPU"
            ),
            qubitCount: circuit.qubitCount,
            quasiProbabilities: quasiProbabilities
        )
    }

    private func sample(
        circuit: QuantumCircuit,
        backend: MPSBackend,
        options: QuantumRunOptions
    ) throws -> SamplerResult {
        if options.shots != nil {
            return try sampleShotsFromBackendRun(
                circuit: circuit,
                backend: backend,
                options: options,
                method: .mps,
                deviceName: "CPU"
            )
        }
        let started = DispatchTime.now()
        let quasiProbabilities = try SimulationProfiling.timePhase("evolve") {
            var state = try MPSState(
                qubitCount: circuit.qubitCount,
                configuration: backend.configuration
            )
            _ = try backend.engine.execute(circuit, on: &state)
            return makeBitstringProbabilities(
                probabilities: try state.probabilities(),
                qubitCount: circuit.qubitCount
            )
        }
        return SamplerResult(
            metadata: makeSamplerMetadata(
                circuit: circuit,
                options: options,
                started: started,
                method: .mps,
                deviceName: "CPU"
            ),
            qubitCount: circuit.qubitCount,
            quasiProbabilities: quasiProbabilities
        )
    }

    /// Owner-level `sample` phase around `backend.run`. Nested backend `timePhase` is a no-op;
    /// this call records the phase, then ``finishProfile`` publishes it (inner profile is nil).
    private func sampleShotsFromBackendRun(
        circuit: QuantumCircuit,
        backend: any QuantumBackend,
        options: QuantumRunOptions,
        method: QuantumSimulationMethod,
        deviceName: String
    ) throws -> SamplerResult {
        guard let shots = options.shots else {
            throw TrajectoryBackendError.shotsRequired
        }
        let started = DispatchTime.now()
        let result = try SimulationProfiling.timePhase("sample") {
            try backend.run(circuit: circuit, options: options)
        }
        // Successful `backend.run` always sets `shotCounts` when `options.shots` is set.
        // The empty histogram is a defensive fallback, not a physics path for a missing sample.
        let counts = result.shotCounts ?? ShotCounts(shots: shots, counts: [:])
        // Do not reuse backend.run's pipelineHash: it folds full QuantumRunOptions.resilience
        // (including Estimator-only ZNE/PEC). Sampler recomputes with readoutOnly.
        return try mitigatedShotResult(
            counts: counts,
            circuit: circuit,
            options: options,
            started: started,
            method: method,
            deviceName: deviceName
        )
    }
}

func applyResilienceIfNeeded(
    _ counts: ShotCounts,
    qubitCount: Int,
    resilience: ResilienceOptions
) throws -> ShotCounts {
    guard let matrix = resilience.readoutMitigation else { return counts }
    return try ReadoutMitigation.apply(to: counts, matrix: matrix, qubitCount: qubitCount)
}

private func makeSamplerMetadata(
    circuit: QuantumCircuit,
    options: QuantumRunOptions,
    started: DispatchTime,
    method: QuantumSimulationMethod,
    deviceName: String? = nil,
    effectiveShots: Int? = nil,
    pipelineHash: String? = nil
) -> QuantumResultMetadata {
    let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
    return QuantumResultMetadata(
        method: method,
        seed: options.seed,
        deviceName: deviceName ?? MetalRuntime.deviceName,
        wallClockNanoseconds: elapsed,
        qubitCount: circuit.qubitCount,
        gateCount: circuit.gates.count,
        noiseSnapshot: options.noise,
        pipelineHash: pipelineHash,
        profile: SimulationProfiling.finishProfile(
            options: options,
            circuit: circuit,
            method: method,
            isCPU: deviceName == "CPU",
            elapsed: elapsed,
            effectiveShots: effectiveShots ?? options.shots
        )
    )
}

private func makeBitstringProbabilities(
    probabilities: [QFloat],
    qubitCount: Int
) -> [String: QFloat] {
    var result: [String: QFloat] = [:]
    result.reserveCapacity(probabilities.count)

    for (index, probability) in probabilities.enumerated() where probability != 0 {
        let key = (try? QubitBitOrdering.bitstringMSB.bitstring(forIndex: index, qubitCount: qubitCount))
            ?? String(repeating: "0", count: qubitCount)
        result[key] = probability
    }
    return result
}
