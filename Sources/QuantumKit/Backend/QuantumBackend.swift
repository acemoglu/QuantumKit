import Foundation

/// Simulation method selected by a ``QuantumBackend`` implementation.
public enum QuantumSimulationMethod: String, Codable, Sendable, Equatable {
    case statevector
    case densityMatrix
    /// Monte-Carlo statevector unraveling ensemble (shared global noise channels).
    case trajectory
    /// Host CPU stabilizer / tableau simulation for Clifford+measure circuits (B19).
    ///
    /// Not selected by ``QuantumBackendFactory/recommendMethod(qubitCount:noise:policy:)``.
    /// Construct with ``QuantumBackendFactory/makeStabilizer()``, or opt in via
    /// ``SimulationPolicy/preferStabilizerWhenClifford`` and the circuit-aware
    /// ``QuantumBackendFactory/recommendMethod(circuit:noise:policy:)``.
    case stabilizer
    /// Host CPU matrix-product-state simulation for 1D-ish circuits (B18 MVP).
    ///
    /// Not selected by ``recommendMethod``. Construct with
    /// ``QuantumBackendFactory/makeMPS(configuration:)``.
    case mps
}

/// Options for a single ``QuantumBackend/run(circuit:options:)`` invocation.
public struct QuantumRunOptions: Sendable, Equatable {
    public var noise: NoiseModel?
    public var seed: UInt64?
    /// When set, the backend samples terminal measurement outcomes instead of only evolving state.
    public var shots: Int?
    public var sampleOptions: SampleCountOptions
    /// Host-side telemetry. Default off; enabling it must not change shot histograms or amplitudes.
    public var profiling: SimulationProfilingOptions
    /// Opt-in resilience (default ``ResilienceOptions/disabled``). Sampler applies
    /// ``ResilienceOptions/readoutMitigation`` to shot histograms when set.
    public var resilience: ResilienceOptions

    public init(
        noise: NoiseModel? = nil,
        seed: UInt64? = nil,
        shots: Int? = nil,
        sampleOptions: SampleCountOptions = SampleCountOptions(),
        profiling: SimulationProfilingOptions = .disabled,
        resilience: ResilienceOptions = .disabled
    ) {
        self.noise = noise
        self.seed = seed
        self.shots = shots
        self.sampleOptions = sampleOptions
        self.profiling = profiling
        self.resilience = resilience
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
    /// SHA-256 hex digest of circuit + run options (version, method, seed, shots, noise).
    public let pipelineHash: String?
    /// Present when ``QuantumRunOptions/profiling`` is enabled. Nil otherwise.
    public let profile: SimulationProfile?

    public init(
        quantumKitVersion: String = QuantumKitInfo.version,
        method: QuantumSimulationMethod,
        seed: UInt64?,
        deviceName: String?,
        wallClockNanoseconds: UInt64,
        qubitCount: Int,
        gateCount: Int,
        noiseSnapshot: NoiseModel?,
        pipelineHash: String? = nil,
        profile: SimulationProfile? = nil
    ) {
        self.quantumKitVersion = quantumKitVersion
        self.method = method
        self.seed = seed
        self.deviceName = deviceName
        self.wallClockNanoseconds = wallClockNanoseconds
        self.qubitCount = qubitCount
        self.gateCount = gateCount
        self.noiseSnapshot = noiseSnapshot
        self.pipelineHash = pipelineHash
        self.profile = profile
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

    /// Bitstring histogram when shots were requested (``QubitBitOrdering/bitstringMSB``).
    /// Integer ``shotCounts`` keys use ``QubitBitOrdering/engineLSB``.
    public var bitstringCounts: [String: Int]? {
        shotCounts?.bitstringCounts(qubitCount: metadata.qubitCount)
    }

    /// Hex-encoded outcome histogram for the packed ``engineLSB`` index
    /// (e.g. `"0x3"` for bitstring `11` on 2 qubits).
    public var hexCounts: [String: Int]? {
        shotCounts?.hexCounts(qubitCount: metadata.qubitCount)
    }

    /// Empirical probabilities from shot counts (``bitstringMSB`` keys), when available.
    public var probabilities: [String: QFloat]? {
        shotCounts?.probabilities(qubitCount: metadata.qubitCount)
    }

    /// Classical register values after mid-circuit measurement, when available.
    public var memorySlots: [Int]? {
        execution.map { $0.classicalMemory.memorySlots }
    }

    /// Host-side telemetry when ``QuantumRunOptions/profiling`` was enabled.
    public var profile: SimulationProfile? { metadata.profile }
}

/// Common entry point for state-vector and density-matrix simulation backends.
public protocol QuantumBackend: Sendable {
    var method: QuantumSimulationMethod { get }
    func run(circuit: QuantumCircuit, options: QuantumRunOptions) throws -> QuantumResult
}

/// Factory helpers for the supported simulation backends.
public enum QuantumBackendFactory {
    /// Metal when a GPU is available, otherwise CPU (``SimulationDevicePreference/automatic``).
    ///
    /// For explicit device selection or a memory budget, use
    /// ``makeStatevector(renormalizationInterval:devicePreference:qubitCount:policy:)``.
    public static func makeStatevector(renormalizationInterval: Int = 50) throws -> any QuantumBackend {
        try makeStatevector(
            renormalizationInterval: renormalizationInterval,
            devicePreference: .automatic
        )
    }

    /// Metal when a GPU is available, otherwise CPU (``SimulationDevicePreference/automatic``).
    ///
    /// For explicit device selection or a memory budget, use
    /// ``makeDensityMatrix(renormalizationInterval:devicePreference:qubitCount:policy:)``.
    public static func makeDensityMatrix(renormalizationInterval: Int = 50) throws -> any QuantumBackend {
        try makeDensityMatrix(
            renormalizationInterval: renormalizationInterval,
            devicePreference: .automatic
        )
    }
}

/// GPU state-vector backend backed by ``QuantumEngine``.
///
/// Construct with ``init(renormalizationInterval:)`` or ``QuantumBackendFactory`` — no
/// ``MTLDevice`` is required. Explicit Metal device selection is an advanced path reserved
/// for H6b / interop; prefer ``MetalRuntime`` when sharing a device is unavoidable.
///
/// Thread-safety: safe to share the backend/engine across threads. Do not mutate one
/// ``StateVector`` concurrently; concurrent runs on distinct states (including batched shots)
/// are supported.
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
        try executeRun(circuit: circuit, options: options, cancellationCheck: nil)
    }

    func runCancellable(circuit: QuantumCircuit, options: QuantumRunOptions) throws -> QuantumResult {
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
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        return try SimulationProfiling.usingRecorder(for: options) {
            if let shots = options.shots {
                var rng = makeRNG(seed: options.seed)
                let counts = try SimulationProfiling.timePhase("sample") {
                    try QuantumMeasurement.runSampleCountsRNG(
                        circuit: circuit,
                        engine: engine,
                        shots: shots,
                        rng: &rng,
                        noise: options.noise,
                        options: options.sampleOptions,
                        cancellationCheck: cancellationCheck
                    )
                }

                return QuantumResult(
                    metadata: makeMetadata(
                        circuit: circuit,
                        options: options,
                        started: started
                    ),
                    shotCounts: counts
                )
            }

            let execution = try SimulationProfiling.timePhase("evolve") {
                let state = try StateVector(qubitCount: circuit.qubitCount)
                var rng = makeRNG(seed: options.seed)
                return try engine.executeRNG(
                    circuit,
                    on: state,
                    rng: &rng,
                    noise: options.noise,
                    cancellationCheck: cancellationCheck
                )
            }

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
}

/// GPU density-matrix backend backed by ``DensityMatrixEngine``.
///
/// Construct with ``init(renormalizationInterval:)`` or ``QuantumBackendFactory`` — no
/// ``MTLDevice`` is required.
///
/// Thread-safety: share the backend across threads only with distinct ``DensityMatrix``
/// instances per concurrent run.
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
        try executeRun(circuit: circuit, options: options, cancellationCheck: nil)
    }

    func runCancellable(circuit: QuantumCircuit, options: QuantumRunOptions) throws -> QuantumResult {
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
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        return try SimulationProfiling.usingRecorder(for: options) {
            var rng = makeRNG(seed: options.seed)

            if let shots = options.shots {
                let counts = try SimulationProfiling.timePhase("sample") {
                    try DensityMatrixShotSampler.runSampleCountsRNG(
                        circuit: circuit,
                        engine: engine,
                        shots: shots,
                        rng: &rng,
                        noise: options.noise,
                        cancellationCheck: cancellationCheck
                    )
                }
                return QuantumResult(
                    metadata: makeMetadata(
                        circuit: circuit,
                        options: options,
                        started: started,
                        method: .densityMatrix
                    ),
                    shotCounts: counts
                )
            }

            let execution = try SimulationProfiling.timePhase("evolve") {
                let density = try DensityMatrix(qubitCount: circuit.qubitCount)
                return try engine.executeRNG(
                    circuit,
                    on: density,
                    rng: &rng,
                    noise: options.noise,
                    cancellationCheck: cancellationCheck
                )
            }

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
}

func makeRNG(seed: UInt64?) -> QuantumRNG {
    if let seed {
        return .seeded(seed)
    }
    return .hardware
}

func makeMetadata(
    circuit: QuantumCircuit,
    options: QuantumRunOptions,
    started: DispatchTime,
    method: QuantumSimulationMethod = .statevector,
    deviceName: String? = nil,
    isCPU: Bool = false
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
        pipelineHash: PipelineFingerprint.hash(circuit: circuit, method: method, options: options),
        profile: SimulationProfiling.finishProfile(
            options: options,
            circuit: circuit,
            method: method,
            isCPU: isCPU,
            elapsed: elapsed
        )
    )
}
