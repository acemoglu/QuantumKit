import Foundation

public enum SimulationPolicyError: Error, Equatable {
    case qubitCountExceedsAllLimits(requested: Int, statevectorMax: Int, densityMatrixMax: Int)
    case densityMatrixRequiredButTooWide(requested: Int, max: Int)
}

/// Tiering policy for automatic backend / method selection (B6 / F1).
public struct SimulationPolicy: Sendable, Equatable {
    public var statevectorQubitLimit: Int
    public var densityMatrixQubitLimit: Int
    /// When noise has channels, prefer density-matrix if width fits.
    public var preferDensityMatrixWhenNoisy: Bool

    public init(
        statevectorQubitLimit: Int = StateVector.maxQubitCount,
        densityMatrixQubitLimit: Int = DensityMatrix.maxQubitCount,
        preferDensityMatrixWhenNoisy: Bool = true
    ) {
        self.statevectorQubitLimit = statevectorQubitLimit
        self.densityMatrixQubitLimit = densityMatrixQubitLimit
        self.preferDensityMatrixWhenNoisy = preferDensityMatrixWhenNoisy
    }

    public static let `default` = SimulationPolicy()
}

/// Lightweight pre-execution resource estimate (supports F1 / I13-lite).
public struct ResourceEstimate: Sendable, Equatable {
    public let qubitCount: Int
    public let recommendedMethod: QuantumSimulationMethod
    public let estimatedStateBytes: Int
    public let statevectorLimit: Int
    public let densityMatrixLimit: Int

    public init(
        qubitCount: Int,
        recommendedMethod: QuantumSimulationMethod,
        estimatedStateBytes: Int,
        statevectorLimit: Int,
        densityMatrixLimit: Int
    ) {
        self.qubitCount = qubitCount
        self.recommendedMethod = recommendedMethod
        self.estimatedStateBytes = estimatedStateBytes
        self.statevectorLimit = statevectorLimit
        self.densityMatrixLimit = densityMatrixLimit
    }
}

extension QuantumBackendFactory {

    /// Recommends SV vs DM from width and optional noise (no backend construction).
    public static func recommendMethod(
        qubitCount: Int,
        noise: NoiseModel? = nil,
        policy: SimulationPolicy = .default
    ) throws -> QuantumSimulationMethod {
        guard qubitCount > 0 else {
            throw StateVectorError.invalidQubitCount(qubitCount)
        }

        let noisy = policy.preferDensityMatrixWhenNoisy && (noise?.hasAnyChannel == true)
        if noisy {
            guard qubitCount <= policy.densityMatrixQubitLimit else {
                throw SimulationPolicyError.densityMatrixRequiredButTooWide(
                    requested: qubitCount,
                    max: policy.densityMatrixQubitLimit
                )
            }
            return .densityMatrix
        }

        if qubitCount <= policy.statevectorQubitLimit {
            return .statevector
        }
        if qubitCount <= policy.densityMatrixQubitLimit {
            return .densityMatrix
        }

        throw SimulationPolicyError.qubitCountExceedsAllLimits(
            requested: qubitCount,
            statevectorMax: policy.statevectorQubitLimit,
            densityMatrixMax: policy.densityMatrixQubitLimit
        )
    }

    /// Builds the backend recommended by ``recommendMethod``.
    public static func makeRecommended(
        qubitCount: Int,
        noise: NoiseModel? = nil,
        policy: SimulationPolicy = .default,
        renormalizationInterval: Int = 50
    ) throws -> any QuantumBackend {
        switch try recommendMethod(qubitCount: qubitCount, noise: noise, policy: policy) {
        case .statevector:
            return try makeStatevector(renormalizationInterval: renormalizationInterval)
        case .densityMatrix:
            return try makeDensityMatrix(renormalizationInterval: renormalizationInterval)
        }
    }

    /// Estimates memory footprint and recommended method without allocating GPU buffers.
    public static func estimateResources(
        qubitCount: Int,
        noise: NoiseModel? = nil,
        policy: SimulationPolicy = .default
    ) throws -> ResourceEstimate {
        let method = try recommendMethod(qubitCount: qubitCount, noise: noise, policy: policy)
        let complexBytes = 2 * MemoryLayout<QFloat>.stride
        let estimatedBytes: Int
        switch method {
        case .statevector:
            estimatedBytes = (1 << qubitCount) * complexBytes
        case .densityMatrix:
            let dim = 1 << qubitCount
            estimatedBytes = dim * dim * complexBytes
        }
        return ResourceEstimate(
            qubitCount: qubitCount,
            recommendedMethod: method,
            estimatedStateBytes: estimatedBytes,
            statevectorLimit: policy.statevectorQubitLimit,
            densityMatrixLimit: policy.densityMatrixQubitLimit
        )
    }
}
