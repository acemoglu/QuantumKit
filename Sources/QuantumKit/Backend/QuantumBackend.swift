import Foundation

/// Simulation method selected by a ``QuantumBackend`` implementation.
public enum QuantumSimulationMethod: String, Codable, Sendable, Equatable {
    case statevector
    case densityMatrix
}

/// Options for a single ``QuantumBackend/run(circuit:options:)`` invocation.
public struct QuantumRunOptions: Sendable, Equatable {
    public var noise: NoiseModel?
    public var seed: UInt64?
    /// When set, the backend samples terminal measurement outcomes instead of only evolving state.
    public var shots: Int?
    public var sampleOptions: SampleCountOptions

    public init(
        noise: NoiseModel? = nil,
        seed: UInt64? = nil,
        shots: Int? = nil,
        sampleOptions: SampleCountOptions = SampleCountOptions()
    ) {
        self.noise = noise
        self.seed = seed
        self.shots = shots
        self.sampleOptions = sampleOptions
    }
}

/// Reproducibility and execution provenance for a backend run.
public struct QuantumResultMetadata: Codable, Sendable, Equatable {
    public let quantumKitVersion: String
    public let method: QuantumSimulationMethod
    public let seed: UInt64?
    public let deviceName: String?
    public let wallClockNanoseconds: UInt64
    public let qubitCount: Int
    public let gateCount: Int
    public let noiseSnapshot: NoiseModel?

    public init(
        quantumKitVersion: String = QuantumKitInfo.version,
        method: QuantumSimulationMethod,
        seed: UInt64?,
        deviceName: String?,
        wallClockNanoseconds: UInt64,
        qubitCount: Int,
        gateCount: Int,
        noiseSnapshot: NoiseModel?
    ) {
        self.quantumKitVersion = quantumKitVersion
        self.method = method
        self.seed = seed
        self.deviceName = deviceName
        self.wallClockNanoseconds = wallClockNanoseconds
        self.qubitCount = qubitCount
        self.gateCount = gateCount
        self.noiseSnapshot = noiseSnapshot
    }
}

/// Unified result envelope returned by ``QuantumBackend`` implementations.
public struct QuantumResult: Sendable, Equatable {
    public let metadata: QuantumResultMetadata
    public let execution: CircuitExecutionResult?
    public let shotCounts: ShotCounts?

    public init(
        metadata: QuantumResultMetadata,
        execution: CircuitExecutionResult? = nil,
        shotCounts: ShotCounts? = nil
    ) {
        self.metadata = metadata
        self.execution = execution
        self.shotCounts = shotCounts
    }
}

/// Common entry point for state-vector and density-matrix simulation backends.
public protocol QuantumBackend: Sendable {
    var method: QuantumSimulationMethod { get }
    func run(circuit: QuantumCircuit, options: QuantumRunOptions) throws -> QuantumResult
}

/// Factory helpers for the supported simulation backends.
public enum QuantumBackendFactory {
    public static func makeStatevector(renormalizationInterval: Int = 50) throws -> StatevectorBackend {
        try StatevectorBackend(renormalizationInterval: renormalizationInterval)
    }

    public static func makeDensityMatrix(renormalizationInterval: Int = 50) throws -> DensityMatrixBackend {
        try DensityMatrixBackend(renormalizationInterval: renormalizationInterval)
    }
}

/// GPU state-vector backend backed by ``QuantumEngine``.
public final class StatevectorBackend: QuantumBackend, @unchecked Sendable {
    public let engine: QuantumEngine
    public var method: QuantumSimulationMethod { .statevector }

    public init(renormalizationInterval: Int = 50) throws {
        self.engine = try QuantumEngine(renormalizationInterval: renormalizationInterval)
    }

    public init(engine: QuantumEngine) {
        self.engine = engine
    }

    public func run(circuit: QuantumCircuit, options: QuantumRunOptions = QuantumRunOptions()) throws -> QuantumResult {
        let started = DispatchTime.now()
        try circuit.requireFullyBound()

        if let shots = options.shots {
            var rng = makeRNG(seed: options.seed)
            let counts = try QuantumMeasurement.runSampleCountsRNG(
                circuit: circuit,
                engine: engine,
                shots: shots,
                rng: &rng,
                noise: options.noise,
                options: options.sampleOptions
            )

            return QuantumResult(
                metadata: makeMetadata(
                    circuit: circuit,
                    options: options,
                    started: started
                ),
                shotCounts: counts
            )
        }

        let state = try StateVector(qubitCount: circuit.qubitCount)
        var rng = makeRNG(seed: options.seed)
        let execution = try engine.executeRNG(
            circuit,
            on: state,
            rng: &rng,
            noise: options.noise
        )

        return QuantumResult(
            metadata: makeMetadata(
                circuit: circuit,
                options: options,
                started: started
            ),
            execution: execution
        )
    }
}

/// GPU density-matrix backend backed by ``DensityMatrixEngine``.
public final class DensityMatrixBackend: QuantumBackend, @unchecked Sendable {
    public let engine: DensityMatrixEngine
    public var method: QuantumSimulationMethod { .densityMatrix }

    public init(renormalizationInterval: Int = 50) throws {
        self.engine = try DensityMatrixEngine(renormalizationInterval: renormalizationInterval)
    }

    public init(engine: DensityMatrixEngine) {
        self.engine = engine
    }

    public func run(circuit: QuantumCircuit, options: QuantumRunOptions = QuantumRunOptions()) throws -> QuantumResult {
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        let density = try DensityMatrix(qubitCount: circuit.qubitCount)
        var rng = makeRNG(seed: options.seed)
        let execution = try engine.executeRNG(
            circuit,
            on: density,
            rng: &rng,
            noise: options.noise
        )

        return QuantumResult(
            metadata: makeMetadata(
                circuit: circuit,
                options: options,
                started: started,
                method: .densityMatrix
            ),
            execution: execution
        )
    }
}

private func makeRNG(seed: UInt64?) -> QuantumRNG {
    if let seed {
        return .seeded(seed)
    }
    return .hardware
}

private func makeMetadata(
    circuit: QuantumCircuit,
    options: QuantumRunOptions,
    started: DispatchTime,
    method: QuantumSimulationMethod = .statevector
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
