import Foundation

extension MetalRuntime {
    /// `true` when a system Metal device can be acquired.
    public static var isAvailable: Bool {
        (try? sharedDevice()) != nil
    }
}

extension QuantumBackendFactory {

    /// Builds a statevector backend for the requested device preference.
    public static func makeStatevector(
        renormalizationInterval: Int = 50,
        devicePreference: SimulationDevicePreference,
        qubitCount: Int? = nil,
        policy: SimulationPolicy = .default
    ) throws -> any QuantumBackend {
        try requireBudgetedQubitCount(
            qubitCount,
            method: .statevector,
            noise: nil,
            policy: policy
        )
        switch try resolveDevice(devicePreference, precision: policy.precision) {
        case .metal:
            return try StatevectorBackend(renormalizationInterval: renormalizationInterval)
        case .cpu:
            let limit = policy.cpuStatevectorQubitLimit
            if let qubitCount, qubitCount > limit {
                throw CPUEngineError.qubitCountExceedsLimit(
                    max: limit,
                    requested: qubitCount
                )
            }
            return CPUStatevectorBackend(
                renormalizationInterval: renormalizationInterval,
                maxQubitCount: limit
            )
        }
    }

    /// Builds a density-matrix backend for the requested device preference.
    public static func makeDensityMatrix(
        renormalizationInterval: Int = 50,
        devicePreference: SimulationDevicePreference,
        qubitCount: Int? = nil,
        policy: SimulationPolicy = .default
    ) throws -> any QuantumBackend {
        try requireBudgetedQubitCount(
            qubitCount,
            method: .densityMatrix,
            noise: nil,
            policy: policy
        )
        switch try resolveDevice(devicePreference, precision: policy.precision) {
        case .metal:
            return try DensityMatrixBackend(renormalizationInterval: renormalizationInterval)
        case .cpu:
            let limit = policy.cpuDensityMatrixQubitLimit
            if let qubitCount, qubitCount > limit {
                throw CPUEngineError.qubitCountExceedsLimit(
                    max: limit,
                    requested: qubitCount
                )
            }
            return CPUDensityMatrixBackend(
                renormalizationInterval: renormalizationInterval,
                maxQubitCount: limit
            )
        }
    }

    private enum ResolvedDevice {
        case metal
        case cpu
    }

    private static func resolveDevice(
        _ preference: SimulationDevicePreference,
        precision: SimulationPrecision = .float32
    ) throws -> ResolvedDevice {
        if precision == .float64 {
            // Metal shaders/buffers are Float32-only; Float64 is CPU Double.
            switch preference {
            case .metal:
                throw SimulationPrecisionError.metalFloat64Unsupported
            case .cpu, .automatic:
                return .cpu
            }
        }
        switch preference {
        case .cpu:
            return .cpu
        case .metal:
            guard MetalRuntime.isAvailable else {
                throw QuantumEngineError.deviceNotFound
            }
            return .metal
        case .automatic:
            return MetalRuntime.isAvailable ? .metal : .cpu
        }
    }
}

/// CPU statevector backend backed by ``CPUStatevectorEngine``.
///
/// Thread-safety: backend/engine may be shared; do not run concurrent shots that mutate a shared
/// ``CPUStateVector``. Distinct per-shot states (as in ``run``) are safe across concurrent backends.
public final class CPUStatevectorBackend: QuantumBackend, @unchecked Sendable {
    public let engine: CPUStatevectorEngine
    /// Soft width cap from the constructing ``SimulationPolicy`` (≤ ``CPUStateVector/maxQubitCount``).
    public let maxQubitCount: Int
    public var method: QuantumSimulationMethod { .statevector }

    public init(
        renormalizationInterval: Int = 50,
        maxQubitCount: Int = CPUStateVector.maxQubitCount
    ) {
        self.engine = CPUStatevectorEngine(renormalizationInterval: renormalizationInterval)
        self.maxQubitCount = max(1, min(maxQubitCount, CPUStateVector.maxQubitCount))
    }

    public init(
        engine: CPUStatevectorEngine,
        maxQubitCount: Int = CPUStateVector.maxQubitCount
    ) {
        self.engine = engine
        self.maxQubitCount = max(1, min(maxQubitCount, CPUStateVector.maxQubitCount))
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
        try requireQubitCount(circuit.qubitCount)
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        return try SimulationProfiling.usingRecorder(for: options) {
            var rng = makeCPURNG(seed: options.seed)

            if let shots = options.shots {
                guard shots > 0 else { throw QuantumMeasurementError.invalidShotCount(shots) }
                let counts = try SimulationProfiling.timePhase("sample") {
                    var histogram: [Int: Int] = [:]
                    for _ in 0..<shots {
                        try cancellationCheck?()
                        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
                        _ = try engine.executeRNG(
                            circuit,
                            on: state,
                            rng: &rng,
                            noise: options.noise,
                            cancellationCheck: cancellationCheck
                        )
                        let outcome = try engine.measureCollapse(
                            on: state,
                            qubits: Array(0..<circuit.qubitCount),
                            rng: &rng,
                            noise: options.noise
                        )
                        histogram[outcome, default: 0] += 1
                    }
                    return ShotCounts(shots: shots, counts: histogram)
                }
                return QuantumResult(
                    metadata: makeCPUMetadata(
                        circuit: circuit,
                        options: options,
                        started: started,
                        method: .statevector
                    ),
                    shotCounts: counts
                )
            }

            let execution = try SimulationProfiling.timePhase("evolve") {
                let state = try CPUStateVector(qubitCount: circuit.qubitCount)
                return try engine.executeRNG(
                    circuit,
                    on: state,
                    rng: &rng,
                    noise: options.noise,
                    cancellationCheck: cancellationCheck
                )
            }
            return QuantumResult(
                metadata: makeCPUMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .statevector
                ),
                execution: execution
            )
        }
    }

    func requireQubitCount(_ qubitCount: Int) throws {
        if qubitCount > maxQubitCount {
            throw CPUEngineError.qubitCountExceedsLimit(max: maxQubitCount, requested: qubitCount)
        }
    }
}

/// CPU density-matrix backend backed by ``CPUDensityMatrixEngine``.
///
/// Thread-safety: share freely across threads only when each call uses a distinct
/// ``CPUDensityMatrix`` (``run`` allocates one per invocation).
public final class CPUDensityMatrixBackend: QuantumBackend, @unchecked Sendable {
    public let engine: CPUDensityMatrixEngine
    /// Soft width cap from the constructing ``SimulationPolicy`` (≤ ``CPUDensityMatrix/maxQubitCount``).
    public let maxQubitCount: Int
    public var method: QuantumSimulationMethod { .densityMatrix }

    public init(
        renormalizationInterval: Int = 50,
        maxQubitCount: Int = CPUDensityMatrix.maxQubitCount
    ) {
        self.engine = CPUDensityMatrixEngine(renormalizationInterval: renormalizationInterval)
        self.maxQubitCount = max(1, min(maxQubitCount, CPUDensityMatrix.maxQubitCount))
    }

    public init(
        engine: CPUDensityMatrixEngine,
        maxQubitCount: Int = CPUDensityMatrix.maxQubitCount
    ) {
        self.engine = engine
        self.maxQubitCount = max(1, min(maxQubitCount, CPUDensityMatrix.maxQubitCount))
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
        try requireQubitCount(circuit.qubitCount)
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        return try SimulationProfiling.usingRecorder(for: options) {
            var rng = makeCPURNG(seed: options.seed)

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
                    metadata: makeCPUMetadata(
                        circuit: circuit,
                        options: options,
                        started: started,
                        method: .densityMatrix
                    ),
                    shotCounts: counts
                )
            }

            let execution = try SimulationProfiling.timePhase("evolve") {
                let density = try CPUDensityMatrix(qubitCount: circuit.qubitCount)
                return try engine.executeRNG(
                    circuit,
                    on: density,
                    rng: &rng,
                    noise: options.noise,
                    cancellationCheck: cancellationCheck
                )
            }
            return QuantumResult(
                metadata: makeCPUMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .densityMatrix
                ),
                execution: execution
            )
        }
    }

    func requireQubitCount(_ qubitCount: Int) throws {
        if qubitCount > maxQubitCount {
            throw CPUEngineError.qubitCountExceedsLimit(max: maxQubitCount, requested: qubitCount)
        }
    }
}

func makeCPURNG(seed: UInt64?) -> QuantumRNG {
    if let seed { return .seeded(seed) }
    return .hardware
}

func makeCPUMetadata(
    circuit: QuantumCircuit,
    options: QuantumRunOptions,
    started: DispatchTime,
    method: QuantumSimulationMethod
) -> QuantumResultMetadata {
    makeMetadata(
        circuit: circuit,
        options: options,
        started: started,
        method: method,
        deviceName: "CPU",
        isCPU: true
    )
}
