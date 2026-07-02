import Foundation

public enum SamplerError: Error, Equatable {
    case unsupportedBackend
    case shotsNotSupportedForDensityMatrix
}

/// Standardized measurement distribution returned by ``Sampler``.
public struct SamplerResult: Sendable, Equatable {
    public let metadata: QuantumResultMetadata
    public let qubitCount: Int
    /// MSB-first bitstring → probability (exact Born rule) or empirical frequency (shots).
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
/// Without `shots`, returns exact Born-rule probabilities. With `shots`, returns a
/// shot histogram plus empirical frequencies. Sampling is supported on ``StatevectorBackend``;
/// ``DensityMatrixBackend`` provides exact diagonal probabilities only.
public struct Sampler: Sendable {
    public init() {}

    public func run(
        circuit: QuantumCircuit,
        backend: any QuantumBackend,
        options: QuantumRunOptions = QuantumRunOptions()
    ) throws -> SamplerResult {
        if let statevectorBackend = backend as? StatevectorBackend {
            return try sample(circuit: circuit, backend: statevectorBackend, options: options)
        }
        if let densityBackend = backend as? DensityMatrixBackend {
            return try sample(circuit: circuit, backend: densityBackend, options: options)
        }
        throw SamplerError.unsupportedBackend
    }

    private func sample(
        circuit: QuantumCircuit,
        backend: StatevectorBackend,
        options: QuantumRunOptions
    ) throws -> SamplerResult {
        let started = DispatchTime.now()

        if let shots = options.shots {
            var rng = makePrimitiveRNG(seed: options.seed)
            let counts = try QuantumMeasurement.runSampleCountsRNG(
                circuit: circuit,
                engine: backend.engine,
                shots: shots,
                rng: &rng,
                noise: options.noise,
                options: options.sampleOptions
            )

            let bitstrings = counts.bitstringCounts(qubitCount: circuit.qubitCount)
            let quasiProbabilities = bitstrings.mapValues { QFloat($0) / QFloat(shots) }

            return SamplerResult(
                metadata: makeSamplerMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .statevector
                ),
                qubitCount: circuit.qubitCount,
                quasiProbabilities: quasiProbabilities,
                shotCounts: counts
            )
        }

        let state = try StateVector(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try backend.engine.executeRNG(
            circuit,
            on: state,
            rng: &rng,
            noise: options.noise
        )

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: backend.engine)
        let quasiProbabilities = makeBitstringProbabilities(
            probabilities: probabilities,
            qubitCount: circuit.qubitCount
        )

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
        if options.shots != nil {
            throw SamplerError.shotsNotSupportedForDensityMatrix
        }

        let started = DispatchTime.now()
        let density = try DensityMatrix(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try backend.engine.executeRNG(
            circuit,
            on: density,
            rng: &rng,
            noise: options.noise
        )

        let probabilities = backend.engine.probabilities(of: density)
        let quasiProbabilities = makeBitstringProbabilities(
            probabilities: probabilities,
            qubitCount: circuit.qubitCount
        )

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
}

private func makeSamplerMetadata(
    circuit: QuantumCircuit,
    options: QuantumRunOptions,
    started: DispatchTime,
    method: QuantumSimulationMethod
) -> QuantumResultMetadata {
    let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
    return QuantumResultMetadata(
        method: method,
        seed: options.seed,
        deviceName: MetalRuntime.deviceName,
        wallClockNanoseconds: elapsed,
        qubitCount: circuit.qubitCount,
        gateCount: circuit.gates.count,
        noiseSnapshot: options.noise
    )
}

private func makeBitstringProbabilities(
    probabilities: [QFloat],
    qubitCount: Int
) -> [String: QFloat] {
    var result: [String: QFloat] = [:]
    result.reserveCapacity(probabilities.count)

    for (index, probability) in probabilities.enumerated() where probability != 0 {
        result[bitstring(for: index, qubitCount: qubitCount)] = probability
    }
    return result
}

private func bitstring(for index: Int, qubitCount: Int) -> String {
    (0..<qubitCount)
        .reversed()
        .map { position in
            ((index >> position) & 1) == 1 ? "1" : "0"
        }
        .joined()
}
