import Foundation

public enum SimulationPolicyError: Error, Equatable {
    case qubitCountExceedsAllLimits(requested: Int, statevectorMax: Int, densityMatrixMax: Int)
    case densityMatrixRequiredButTooWide(requested: Int, max: Int)
    case trajectoryRequiredButTooWide(requested: Int, max: Int)
    case estimatedMemoryExceedsBudget(estimated: Int, budget: Int)
    /// ``maxPeakMemoryBytes`` is set but the factory call omitted ``qubitCount``.
    case memoryBudgetRequiresQubitCount
}

/// Prefers Metal GPU engines when available, otherwise host CPU fallbacks.
public enum SimulationDevicePreference: String, Sendable, Equatable, Codable {
    /// Use Metal when a device exists; otherwise fall back to CPU.
    case automatic
    /// Require Metal; fail if unavailable.
    case metal
    /// Force host CPU engines.
    case cpu
}

/// Resolved device hint returned by resource estimation (may differ from the request when
/// ``SimulationPrecision/float64`` forces CPU, or Metal is unavailable under `.automatic`).
public enum SimulationDeviceHint: String, Sendable, Equatable, Codable {
    case metal
    case cpu
}

/// Tiering policy for automatic backend / method selection (B6 / F1).
public struct SimulationPolicy: Sendable, Equatable {
    public var statevectorQubitLimit: Int
    public var densityMatrixQubitLimit: Int
    /// When noise has channels, prefer density-matrix if width fits. When `false`, prefer
    /// statevector unraveling whenever the noise is trajectory-compatible and SV fits.
    public var preferDensityMatrixWhenNoisy: Bool
    /// When noisy width exceeds the DM limit and noise is trajectory-compatible,
    /// recommend ``QuantumSimulationMethod/trajectory`` (never while DM still fits).
    public var preferTrajectoryWhenDensityMatrixTooWide: Bool
    /// Metal vs CPU engine selection.
    public var devicePreference: SimulationDevicePreference
    /// Soft width cap for CPU statevector backends.
    public var cpuStatevectorQubitLimit: Int
    /// Soft width cap for CPU density-matrix backends.
    public var cpuDensityMatrixQubitLimit: Int
    /// Float32 (default / Metal) vs Float64 (CPU Double). See ``SimulationPrecision``.
    public var precision: SimulationPrecision
    /// Optional hard cap on estimated peak memory. ``estimateResources``, ``makeRecommended``,
    /// and factory ``makeStatevector`` / ``makeDensityMatrix`` / ``makeTrajectory`` fail early
    /// when a qubit count is provided and the footprint would exceed this budget.
    /// Direct ``CPUStatevectorBackend`` / ``StatevectorBackend`` (and DM) constructors are
    /// unconstrained; the factory + policy path is the budgeted API.
    /// A non-`nil` budget without a `qubitCount:` argument throws
    /// ``SimulationPolicyError/memoryBudgetRequiresQubitCount``.
    public var maxPeakMemoryBytes: Int?
    /// Default ensemble size used for trajectory resource / runtime heuristics.
    public var defaultTrajectoryShots: Int

    public init(
        statevectorQubitLimit: Int = StateVector.maxQubitCount,
        densityMatrixQubitLimit: Int = DensityMatrix.maxQubitCount,
        preferDensityMatrixWhenNoisy: Bool = true,
        preferTrajectoryWhenDensityMatrixTooWide: Bool = true,
        devicePreference: SimulationDevicePreference = .automatic,
        cpuStatevectorQubitLimit: Int = CPUStateVector.maxQubitCount,
        cpuDensityMatrixQubitLimit: Int = CPUDensityMatrix.maxQubitCount,
        precision: SimulationPrecision = .float32,
        maxPeakMemoryBytes: Int? = nil,
        defaultTrajectoryShots: Int = 1024
    ) {
        self.statevectorQubitLimit = statevectorQubitLimit
        self.densityMatrixQubitLimit = densityMatrixQubitLimit
        self.preferDensityMatrixWhenNoisy = preferDensityMatrixWhenNoisy
        self.preferTrajectoryWhenDensityMatrixTooWide = preferTrajectoryWhenDensityMatrixTooWide
        self.devicePreference = devicePreference
        self.cpuStatevectorQubitLimit = max(1, min(cpuStatevectorQubitLimit, CPUStateVector.maxQubitCount))
        self.cpuDensityMatrixQubitLimit = max(1, min(cpuDensityMatrixQubitLimit, CPUDensityMatrix.maxQubitCount))
        self.precision = precision
        self.maxPeakMemoryBytes = maxPeakMemoryBytes
        self.defaultTrajectoryShots = max(1, defaultTrajectoryShots)
    }

    public static let `default` = SimulationPolicy()
}

/// Lightweight pre-execution resource estimate (supports F1 / I13-lite).
public struct ResourceEstimate: Sendable, Equatable {
    public let qubitCount: Int
    public let recommendedMethod: QuantumSimulationMethod
    /// Bytes for the primary quantum state buffer (SV amplitudes or DM elements).
    public let estimatedStateBytes: Int
    /// Peak working-set estimate including scratch / ensemble overhead.
    public let estimatedPeakMemoryBytes: Int
    /// Rough wall-time heuristic in nanoseconds (order-of-magnitude only).
    public let estimatedRuntimeHintNanoseconds: UInt64
    /// Device the factory would prefer for this request.
    public let recommendedDevice: SimulationDeviceHint
    public let statevectorLimit: Int
    public let densityMatrixLimit: Int
    /// Ensemble size assumed for trajectory peak-memory / runtime heuristics.
    public let assumedTrajectoryShots: Int?

    public init(
        qubitCount: Int,
        recommendedMethod: QuantumSimulationMethod,
        estimatedStateBytes: Int,
        estimatedPeakMemoryBytes: Int,
        estimatedRuntimeHintNanoseconds: UInt64,
        recommendedDevice: SimulationDeviceHint,
        statevectorLimit: Int,
        densityMatrixLimit: Int,
        assumedTrajectoryShots: Int? = nil
    ) {
        self.qubitCount = qubitCount
        self.recommendedMethod = recommendedMethod
        self.estimatedStateBytes = estimatedStateBytes
        self.estimatedPeakMemoryBytes = estimatedPeakMemoryBytes
        self.estimatedRuntimeHintNanoseconds = estimatedRuntimeHintNanoseconds
        self.recommendedDevice = recommendedDevice
        self.statevectorLimit = statevectorLimit
        self.densityMatrixLimit = densityMatrixLimit
        self.assumedTrajectoryShots = assumedTrajectoryShots
    }
}

extension QuantumBackendFactory {

    /// Recommends SV / DM / trajectory from width and optional noise (no backend construction).
    public static func recommendMethod(
        qubitCount: Int,
        noise: NoiseModel? = nil,
        policy: SimulationPolicy = .default
    ) throws -> QuantumSimulationMethod {
        guard qubitCount > 0 else {
            throw StateVectorError.invalidQubitCount(qubitCount)
        }

        let device = try resolveDeviceHint(policy: policy)
        // Width caps must match the device that will actually run. On CPU, also honor any
        // tighter Metal-scale limits the caller set on the shared policy fields.
        let dmLimit = device == .cpu
            ? min(policy.cpuDensityMatrixQubitLimit, policy.densityMatrixQubitLimit)
            : policy.densityMatrixQubitLimit
        let svLimit = device == .cpu
            ? min(policy.cpuStatevectorQubitLimit, policy.statevectorQubitLimit)
            : policy.statevectorQubitLimit

        let noisy = noise?.hasAnyChannel == true
        let trajectoryCompatible = noise?.supportsTrajectorySimulation ?? true
        let dmFits = qubitCount <= dmLimit
        let svFits = qubitCount <= svLimit

        if noisy {
            if policy.preferDensityMatrixWhenNoisy && dmFits {
                return .densityMatrix
            }
            // Unraveling SV when the flag is off, SV fits, and noise is trajectory-compatible.
            if !policy.preferDensityMatrixWhenNoisy && svFits && trajectoryCompatible {
                return .statevector
            }
            // Trajectory only when DM cannot fit — never as a substitute while DM still fits.
            if !dmFits
                && policy.preferTrajectoryWhenDensityMatrixTooWide
                && trajectoryCompatible
                && svFits
            {
                return .trajectory
            }
            if dmFits {
                return .densityMatrix
            }
            if !trajectoryCompatible || !policy.preferTrajectoryWhenDensityMatrixTooWide {
                throw SimulationPolicyError.densityMatrixRequiredButTooWide(
                    requested: qubitCount,
                    max: dmLimit
                )
            }
            throw SimulationPolicyError.trajectoryRequiredButTooWide(
                requested: qubitCount,
                max: svLimit
            )
        }

        if svFits {
            return .statevector
        }
        if dmFits {
            return .densityMatrix
        }

        throw SimulationPolicyError.qubitCountExceedsAllLimits(
            requested: qubitCount,
            statevectorMax: svLimit,
            densityMatrixMax: dmLimit
        )
    }

    /// Builds the backend recommended by ``recommendMethod`` using ``SimulationPolicy/devicePreference``.
    public static func makeRecommended(
        qubitCount: Int,
        noise: NoiseModel? = nil,
        policy: SimulationPolicy = .default,
        renormalizationInterval: Int = 50
    ) throws -> any QuantumBackend {
        let estimate = try estimateResources(qubitCount: qubitCount, noise: noise, policy: policy)
        if let budget = policy.maxPeakMemoryBytes, estimate.estimatedPeakMemoryBytes > budget {
            throw SimulationPolicyError.estimatedMemoryExceedsBudget(
                estimated: estimate.estimatedPeakMemoryBytes,
                budget: budget
            )
        }

        switch estimate.recommendedMethod {
        case .statevector:
            return try makeStatevector(
                renormalizationInterval: renormalizationInterval,
                devicePreference: policy.devicePreference,
                qubitCount: qubitCount,
                policy: policy
            )
        case .densityMatrix:
            return try makeDensityMatrix(
                renormalizationInterval: renormalizationInterval,
                devicePreference: policy.devicePreference,
                qubitCount: qubitCount,
                policy: policy
            )
        case .trajectory:
            return try makeTrajectory(
                renormalizationInterval: renormalizationInterval,
                devicePreference: policy.devicePreference,
                qubitCount: qubitCount,
                policy: policy,
                noise: noise
            )
        }
    }

    /// Builds a trajectory (SV Monte-Carlo ensemble) backend for the requested device preference.
    ///
    /// When `devicePreference` is omitted, ``SimulationPolicy/devicePreference`` is used.
    /// Pass `noise` so the memory-budget check matches noisy (serial) vs noiseless (batchable)
    /// trajectory peak estimates.
    public static func makeTrajectory(
        renormalizationInterval: Int = 50,
        devicePreference: SimulationDevicePreference? = nil,
        qubitCount: Int? = nil,
        policy: SimulationPolicy = .default,
        noise: NoiseModel? = nil
    ) throws -> any QuantumBackend {
        let resolvedPreference = devicePreference ?? policy.devicePreference
        try requireBudgetedQubitCount(
            qubitCount,
            method: .trajectory,
            noise: noise,
            policy: policy
        )
        let underlying = try makeStatevector(
            renormalizationInterval: renormalizationInterval,
            devicePreference: resolvedPreference,
            qubitCount: qubitCount,
            policy: policy
        )
        return try TrajectoryBackend(wrapping: underlying)
    }

    /// Estimates memory footprint, rough runtime, and recommended method without allocating buffers.
    public static func estimateResources(
        qubitCount: Int,
        noise: NoiseModel? = nil,
        policy: SimulationPolicy = .default,
        gateCount: Int = 0,
        shots: Int? = nil
    ) throws -> ResourceEstimate {
        let method = try recommendMethod(qubitCount: qubitCount, noise: noise, policy: policy)
        let device = try resolveDeviceHint(policy: policy)
        let dmLimit = device == .cpu
            ? min(policy.cpuDensityMatrixQubitLimit, policy.densityMatrixQubitLimit)
            : policy.densityMatrixQubitLimit
        let svLimit = device == .cpu
            ? min(policy.cpuStatevectorQubitLimit, policy.statevectorQubitLimit)
            : policy.statevectorQubitLimit
        let (stateBytes, peakBytes, trajShots) = peakMemoryBreakdown(
            qubitCount: qubitCount,
            method: method,
            device: device,
            noise: noise,
            policy: policy,
            shots: shots
        )

        if let budget = policy.maxPeakMemoryBytes, peakBytes > budget {
            throw SimulationPolicyError.estimatedMemoryExceedsBudget(
                estimated: peakBytes,
                budget: budget
            )
        }

        let gates = max(gateCount, 1)
        let runtimeHint = roughRuntimeHintNanoseconds(
            method: method,
            qubitCount: qubitCount,
            gateCount: gates,
            trajectoryShots: trajShots ?? 1,
            device: device
        )

        return ResourceEstimate(
            qubitCount: qubitCount,
            recommendedMethod: method,
            estimatedStateBytes: stateBytes,
            estimatedPeakMemoryBytes: peakBytes,
            estimatedRuntimeHintNanoseconds: runtimeHint,
            recommendedDevice: device,
            statevectorLimit: svLimit,
            densityMatrixLimit: dmLimit,
            assumedTrajectoryShots: trajShots
        )
    }

    static func requireBudgetedQubitCount(
        _ qubitCount: Int?,
        method: QuantumSimulationMethod,
        noise: NoiseModel?,
        policy: SimulationPolicy,
        shots: Int? = nil
    ) throws {
        guard policy.maxPeakMemoryBytes != nil else { return }
        guard let qubitCount else {
            throw SimulationPolicyError.memoryBudgetRequiresQubitCount
        }
        try enforceMemoryBudget(
            qubitCount: qubitCount,
            method: method,
            noise: noise,
            policy: policy,
            shots: shots
        )
    }

    static func enforceMemoryBudget(
        qubitCount: Int,
        method: QuantumSimulationMethod,
        noise: NoiseModel?,
        policy: SimulationPolicy,
        shots: Int? = nil
    ) throws {
        guard let budget = policy.maxPeakMemoryBytes else { return }
        let device = try resolveDeviceHint(policy: policy)
        let (_, peakBytes, _) = peakMemoryBreakdown(
            qubitCount: qubitCount,
            method: method,
            device: device,
            noise: noise,
            policy: policy,
            shots: shots
        )
        if peakBytes > budget {
            throw SimulationPolicyError.estimatedMemoryExceedsBudget(
                estimated: peakBytes,
                budget: budget
            )
        }
    }

    static func peakMemoryBreakdown(
        qubitCount: Int,
        method: QuantumSimulationMethod,
        device: SimulationDeviceHint,
        noise: NoiseModel?,
        policy: SimulationPolicy,
        shots: Int?
    ) -> (stateBytes: Int, peakBytes: Int, trajShots: Int?) {
        // CPU engines always store Double, even when policy.precision is the Metal default .float32.
        let complexBytes = device == .cpu
            ? 2 * MemoryLayout<Double>.stride
            : complexElementBytes(precision: policy.precision)
        let dim = 1 << qubitCount

        switch method {
        case .statevector:
            let stateBytes = dim * complexBytes
            return (stateBytes, stateBytes * 2, nil)
        case .densityMatrix:
            let stateBytes = dim * dim * complexBytes
            return (stateBytes, stateBytes * 2, nil)
        case .trajectory:
            let ensemble = shots ?? policy.defaultTrajectoryShots
            let stateBytes = dim * complexBytes
            // Noisy evolution forces BatchSampleExecutor batchSize = 1; only charge a Metal
            // StateVectorBatch pool when shots can actually be batched.
            let forcesSerialShots =
                (noise?.hasGateNoise == true)
                || (noise?.hasPreparationNoise == true)
                || (noise?.hasMeasurementChannelNoise == true)
            let metalBatch: Int
            if device == .metal && !forcesSerialShots {
                metalBatch = min(ensemble, SampleCountOptions().batchSize)
            } else {
                metalBatch = 1
            }
            let peakBytes = stateBytes * (2 + max(metalBatch - 1, 0))
            return (stateBytes, peakBytes, ensemble)
        }
    }

    static func resolveDeviceHint(policy: SimulationPolicy) throws -> SimulationDeviceHint {
        if policy.precision == .float64 {
            if policy.devicePreference == .metal {
                throw SimulationPrecisionError.metalFloat64Unsupported
            }
            return .cpu
        }
        switch policy.devicePreference {
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

    static func complexElementBytes(precision: SimulationPrecision) -> Int {
        switch precision {
        case .float32:
            return 2 * MemoryLayout<Float32>.stride
        case .float64:
            return 2 * MemoryLayout<Double>.stride
        }
    }

    /// Order-of-magnitude heuristic only — not a performance guarantee.
    static func roughRuntimeHintNanoseconds(
        method: QuantumSimulationMethod,
        qubitCount: Int,
        gateCount: Int,
        trajectoryShots: Int,
        device: SimulationDeviceHint
    ) -> UInt64 {
        let dim = Double(1 << min(qubitCount, 30))
        let gateFactor = Double(max(gateCount, 1))
        let deviceScale: Double = device == .metal ? 1.0 : 4.0
        let baseNS: Double
        switch method {
        case .statevector:
            // ~O(gate · 2ⁿ) amplitude touches.
            baseNS = gateFactor * dim * 20.0
        case .densityMatrix:
            // ~O(gate · 4ⁿ) element touches.
            baseNS = gateFactor * dim * dim * 40.0
        case .trajectory:
            baseNS = gateFactor * dim * 20.0 * Double(max(trajectoryShots, 1))
        }
        let clamped = min(max(baseNS * deviceScale, 1.0), Double(UInt64.max) * 0.5)
        return UInt64(clamped)
    }
}
