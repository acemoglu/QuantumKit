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
///   run-options resilience when Estimator resilience is disabled). When Estimator resilience
///   enables ZNE/PEC without its own matrix, run-options ``readoutMitigation`` is still inherited.
/// - ``zne``: zero-noise extrapolation on **shot** ``Estimator`` only (global depolarizing
///   stretch + linear fit to `λ = 0`; see ``ZNEOptions``). Only **active** options
///   (`ZNEOptions/isActive`, ≥ 2 scale factors) enable ZNE; inactive `zne` values are ignored.
///   Ignored on exact Estimator and on ``Sampler``. When active, per-scale runs still honor
///   ``readoutMitigation``. Requires a ``NoiseModel`` with ``NoiseModel/appliesDepolarizing``.
/// - ``pec``: PEC lite on **shot** ``Estimator`` only — inverse quasiprobability for 1Q
///   global depolarizing / equal-rate Pauli channel (see ``PECOptions``). Incompatible with
///   **active** ``zne`` in this MVP. Ignored on exact Estimator and ``Sampler``.
///
/// Physics-affecting resilience that a primitive actually applies is included in
/// ``PipelineFingerprint`` (Estimator hashes resolved active ZNE/PEC + readout; Sampler hashes
/// readout only). Telemetry-only ``QuantumRunOptions/profiling`` remains excluded.
///
/// ## Deferred (not implemented)
/// Pauli twirling ensembles, unitary folding / identity-insertion ZNE, full gate-set PEC
/// (2Q / unequal Pauli / localized), Choi-based process tomography, and full mitigation stacks.
public struct ResilienceOptions: Sendable, Equatable {
    /// When non-`nil`, correct terminal shot histograms with this assignment matrix
    /// (`P(measured|prepared)`). Must match the sampled qubit width.
    public var readoutMitigation: ReadoutConfusionMatrix?

    /// When non-`nil` and ``ZNEOptions/isActive``, shot ``Estimator`` amplifies global
    /// depolarizing by each scale factor and linearly extrapolates to zero noise (C13).
    /// Inactive values (`scaleFactors.count < 2`) are ignored.
    public var zne: ZNEOptions?

    /// When non-`nil`, shot ``Estimator`` runs PEC lite (C13).
    public var pec: PECOptions?

    public init(
        readoutMitigation: ReadoutConfusionMatrix? = nil,
        zne: ZNEOptions? = nil,
        pec: PECOptions? = nil
    ) {
        self.readoutMitigation = readoutMitigation
        self.zne = zne
        self.pec = pec
    }

    public static let disabled = ResilienceOptions()

    public var isEnabled: Bool {
        readoutMitigation != nil || activeZNE != nil || pec != nil
    }

    /// ``zne`` when present and ``ZNEOptions/isActive``; otherwise `nil`.
    public var activeZNE: ZNEOptions? {
        guard let zne, zne.isActive else { return nil }
        return zne
    }

    /// Readout-only view (ZNE/PEC stripped). Used for nested scale/sample evaluation and
    /// Sampler fingerprinting — those paths do not apply ZNE/PEC.
    public var readoutOnly: ResilienceOptions {
        ResilienceOptions(readoutMitigation: readoutMitigation)
    }

    /// Same options with ``zne`` cleared; ``readoutMitigation`` and ``pec`` preserved.
    public func withoutZNE() -> ResilienceOptions {
        ResilienceOptions(readoutMitigation: readoutMitigation, zne: nil, pec: pec)
    }

    /// Same options with ``pec`` cleared; ``readoutMitigation`` and ``zne`` preserved.
    public func withoutPEC() -> ResilienceOptions {
        ResilienceOptions(readoutMitigation: readoutMitigation, zne: zne, pec: nil)
    }
}
