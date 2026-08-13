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
        switch try resolveDevice(devicePreference) {
        case .metal:
            return try StatevectorBackend(renormalizationInterval: renormalizationInterval)
        case .cpu:
            if let qubitCount, qubitCount > policy.cpuStatevectorQubitLimit {
                throw CPUEngineError.qubitCountExceedsLimit(
                    max: policy.cpuStatevectorQubitLimit,
                    requested: qubitCount
                )
            }
            return CPUStatevectorBackend(renormalizationInterval: renormalizationInterval)
        }
    }

    /// Builds a density-matrix backend for the requested device preference.
    public static func makeDensityMatrix(
        renormalizationInterval: Int = 50,
        devicePreference: SimulationDevicePreference,
        qubitCount: Int? = nil,
        policy: SimulationPolicy = .default
    ) throws -> any QuantumBackend {
        switch try resolveDevice(devicePreference) {
        case .metal:
            return try DensityMatrixBackend(renormalizationInterval: renormalizationInterval)
        case .cpu:
            if let qubitCount, qubitCount > policy.cpuDensityMatrixQubitLimit {
                throw CPUEngineError.qubitCountExceedsLimit(
                    max: policy.cpuDensityMatrixQubitLimit,
                    requested: qubitCount
                )
            }
            return CPUDensityMatrixBackend(renormalizationInterval: renormalizationInterval)
        }
    }

    private enum ResolvedDevice {
        case metal
        case cpu
    }

    private static func resolveDevice(_ preference: SimulationDevicePreference) throws -> ResolvedDevice {
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
public final class CPUStatevectorBackend: QuantumBackend, @unchecked Sendable {
    public let engine: CPUStatevectorEngine
    public var method: QuantumSimulationMethod { .statevector }

    public init(renormalizationInterval: Int = 50) {
        self.engine = CPUStatevectorEngine(renormalizationInterval: renormalizationInterval)
    }

    public init(engine: CPUStatevectorEngine) {
        self.engine = engine
    }

    public func run(circuit: QuantumCircuit, options: QuantumRunOptions = QuantumRunOptions()) throws -> QuantumResult {
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        var rng = makeCPURNG(seed: options.seed)

        if let shots = options.shots {
            guard shots > 0 else { throw QuantumMeasurementError.invalidShotCount(shots) }
            var histogram: [Int: Int] = [:]
            for _ in 0..<shots {
                let state = try CPUStateVector(qubitCount: circuit.qubitCount)
                _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: options.noise)
                let outcome = try engine.measureCollapse(
                    on: state,
                    qubits: Array(0..<circuit.qubitCount),
                    rng: &rng,
                    noise: options.noise
                )
                histogram[outcome, default: 0] += 1
            }
            return QuantumResult(
                metadata: makeCPUMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .statevector
                ),
                shotCounts: ShotCounts(shots: shots, counts: histogram)
            )
        }

        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        let execution = try engine.executeRNG(circuit, on: state, rng: &rng, noise: options.noise)
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

/// CPU density-matrix backend backed by ``CPUDensityMatrixEngine``.
public final class CPUDensityMatrixBackend: QuantumBackend, @unchecked Sendable {
    public let engine: CPUDensityMatrixEngine
    public var method: QuantumSimulationMethod { .densityMatrix }

    public init(renormalizationInterval: Int = 50) {
        self.engine = CPUDensityMatrixEngine(renormalizationInterval: renormalizationInterval)
    }

    public init(engine: CPUDensityMatrixEngine) {
        self.engine = engine
    }

    public func run(circuit: QuantumCircuit, options: QuantumRunOptions = QuantumRunOptions()) throws -> QuantumResult {
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        let density = try CPUDensityMatrix(qubitCount: circuit.qubitCount)
        var rng = makeCPURNG(seed: options.seed)
        let execution = try engine.executeRNG(
            circuit,
            on: density,
            rng: &rng,
            noise: options.noise
        )
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

private func makeCPURNG(seed: UInt64?) -> QuantumRNG {
    if let seed { return .seeded(seed) }
    return .hardware
}

private func makeCPUMetadata(
    circuit: QuantumCircuit,
    options: QuantumRunOptions,
    started: DispatchTime,
    method: QuantumSimulationMethod
) -> QuantumResultMetadata {
    let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
    return QuantumResultMetadata(
        method: method,
        seed: options.seed,
        deviceName: "CPU",
        wallClockNanoseconds: elapsed,
        qubitCount: circuit.qubitCount,
        gateCount: circuit.gates.count,
        noiseSnapshot: options.noise,
        pipelineHash: PipelineFingerprint.hash(circuit: circuit, method: method, options: options)
    )
}
