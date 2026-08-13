import Foundation

public enum SimulationPolicyError: Error, Equatable {
    case qubitCountExceedsAllLimits(requested: Int, statevectorMax: Int, densityMatrixMax: Int)
    case densityMatrixRequiredButTooWide(requested: Int, max: Int)
    case trajectoryRequiredButTooWide(requested: Int, max: Int)
    case estimatedMemoryExceedsBudget(estimated: Int, budget: Int)
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
    /// When noise has channels, prefer density-matrix if width fits.
    public var preferDensityMatrixWhenNoisy: Bool
    /// When noisy width exceeds the DM limit (or DM is disabled) and noise is
    /// trajectory-compatible, recommend ``QuantumSimulationMethod/trajectory``.
    public var preferTrajectoryWhenDensityMatrixTooWide: Bool
    /// Metal vs CPU engine selection.
    public var devicePreference: SimulationDevicePreference
    /// Soft width cap for CPU statevector backends.
    public var cpuStatevectorQubitLimit: Int
    /// Soft width cap for CPU density-matrix backends.
    public var cpuDensityMatrixQubitLimit: Int
    /// Float32 (default / Metal) vs Float64 (CPU Double). See ``SimulationPrecision``.
    public var precision: SimulationPrecision
    /// Optional hard cap on estimated peak memory; ``estimateResources`` / ``makeRecommended``
    /// fail early when the footprint would exceed this budget.
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

        let noisy = noise?.hasAnyChannel == true
        let trajectoryCompatible = noise?.supportsTrajectorySimulation ?? true
        let dmFits = qubitCount <= policy.densityMatrixQubitLimit
        let svFits = qubitCount <= policy.statevectorQubitLimit

        if noisy {
            if policy.preferDensityMatrixWhenNoisy && dmFits {
                return .densityMatrix
            }
            if trajectoryCompatible
                && svFits
                && (policy.preferTrajectoryWhenDensityMatrixTooWide || !dmFits)
            {
                return .trajectory
            }
            if dmFits {
                return .densityMatrix
            }
            if !trajectoryCompatible {
                throw SimulationPolicyError.densityMatrixRequiredButTooWide(
                    requested: qubitCount,
                    max: policy.densityMatrixQubitLimit
                )
            }
            throw SimulationPolicyError.trajectoryRequiredButTooWide(
                requested: qubitCount,
                max: policy.statevectorQubitLimit
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
            statevectorMax: policy.statevectorQubitLimit,
            densityMatrixMax: policy.densityMatrixQubitLimit
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
                policy: policy
            )
        }
    }

    /// Builds a trajectory (SV Monte-Carlo ensemble) backend for the requested device preference.
    public static func makeTrajectory(
        renormalizationInterval: Int = 50,
        devicePreference: SimulationDevicePreference = .automatic,
        qubitCount: Int? = nil,
        policy: SimulationPolicy = .default
    ) throws -> any QuantumBackend {
        let underlying = try makeStatevector(
            renormalizationInterval: renormalizationInterval,
            devicePreference: devicePreference,
            qubitCount: qubitCount,
            policy: policy
        )
        return TrajectoryBackend(wrapping: underlying)
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
        let device = resolveDeviceHint(policy: policy)
        let complexBytes = complexElementBytes(precision: policy.precision)
        let dim = 1 << qubitCount

        let stateBytes: Int
        let peakBytes: Int
        let trajShots: Int?
        switch method {
        case .statevector:
            stateBytes = dim * complexBytes
            // Scratch for gates / measurement CDF ≈ one extra statevector.
            peakBytes = stateBytes * 2
            trajShots = nil
        case .densityMatrix:
            stateBytes = dim * dim * complexBytes
            // Real/imag scratch pair ≈ another full ρ.
            peakBytes = stateBytes * 2
            trajShots = nil
        case .trajectory:
            let ensemble = shots ?? policy.defaultTrajectoryShots
            stateBytes = dim * complexBytes
            // Peak is one working SV (+scratch); ensemble is sequential by default.
            peakBytes = stateBytes * 2
            trajShots = ensemble
        }

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
            statevectorLimit: policy.statevectorQubitLimit,
            densityMatrixLimit: policy.densityMatrixQubitLimit,
            assumedTrajectoryShots: trajShots
        )
    }

    static func resolveDeviceHint(policy: SimulationPolicy) -> SimulationDeviceHint {
        if policy.precision == .float64 {
            return .cpu
        }
        switch policy.devicePreference {
        case .cpu:
            return .cpu
        case .metal:
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
