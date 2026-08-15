import Foundation

/// Shared shot-budget hint used by primitives (and documented for ``Estimator`` / ``Sampler``).
///
/// ## Precedence (Estimator)
/// When resolving how many shots to take, ``Estimator`` uses:
/// 1. ``shots`` if set (explicit budget)
/// 2. else ``precision`` → `shots ≈ max(1, ⌈1/ε²⌉)`
/// 3. else ``QuantumRunOptions/shots``
/// 4. else exact analytic expectation (no sampling)
///
/// ``Sampler`` ignores `precision` and uses only ``QuantumRunOptions/shots`` (or exact Born
/// probabilities when `shots` is `nil`).
///
/// This type does **not** change defaults: an empty budget means “no sampling hint”.
public struct ShotBudget: Sendable, Equatable {
    /// Explicit shot count. Wins over ``precision`` when both are set.
    public var shots: Int?
    /// Target absolute standard-error scale for Estimator sampling.
    public var precision: QFloat?

    public init(shots: Int? = nil, precision: QFloat? = nil) {
        self.shots = shots
        self.precision = precision
    }

    /// No sampling hint (exact path when used alone).
    public static let exact = ShotBudget()

    /// Resolves an Estimator-style shot count, or `nil` for exact evaluation.
    public func resolvedShots() throws -> Int? {
        if let shots {
            guard shots > 0 else { throw EstimatorError.invalidShotCount(shots) }
            return shots
        }
        if let precision {
            guard precision > 0 else { throw EstimatorError.invalidPrecision(precision) }
            let estimate = ceil(1.0 / (Double(precision) * Double(precision)))
            return max(1, Int(estimate))
        }
        return nil
    }
}

/// Opt-in resilience / error-mitigation knobs for primitive jobs.
///
/// Default ``disabled`` leaves Sampler / Estimator bit-identical to pre-resilience behavior.
///
/// ## Shipped
/// - ``readoutMitigation``: host-side inverse of a ``ReadoutConfusionMatrix`` applied to
///   shot histograms on ``Sampler`` (via ``QuantumRunOptions/resilience``) and on
///   shot ``Estimator`` ensembles (via ``EstimatorOptions/resilience``, falling back to
///   run-options resilience when Estimator resilience is disabled).
///
/// Physics-affecting resilience is included in ``PipelineFingerprint`` (via
/// ``QuantumRunOptions/resilience``, or Estimator-resolved resilience when hashing an
/// Estimator job). Telemetry-only ``QuantumRunOptions/profiling`` remains excluded.
///
/// ## Deferred (not implemented)
/// Pauli twirling ensembles, zero-noise extrapolation (ZNE), probabilistic error cancellation
/// (PEC), Choi-based process tomography, and full mitigation stacks.
public struct ResilienceOptions: Sendable, Equatable {
    /// When non-`nil`, correct terminal shot histograms with this assignment matrix
    /// (`P(measured|prepared)`). Must match the sampled qubit width.
    public var readoutMitigation: ReadoutConfusionMatrix?

    public init(readoutMitigation: ReadoutConfusionMatrix? = nil) {
        self.readoutMitigation = readoutMitigation
    }

    public static let disabled = ResilienceOptions()

    public var isEnabled: Bool { readoutMitigation != nil }
}
