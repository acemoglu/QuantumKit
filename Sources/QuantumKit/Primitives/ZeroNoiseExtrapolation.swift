import Foundation

/// Zero-noise extrapolation (C13) — shot Estimator only.
///
/// ## Scaling method (this MVP)
/// **Global depolarizing stretch:** for each scale factor `λ`, run the shot Estimator with
/// `NoiseModel.depolarizingProbability` replaced by `clamp(p · λ)` (other noise channels
/// unchanged). `λ = 1` is the physical / nominal model; `λ > 1` amplifies global depolarizing
/// only. This is a **simulator** noise-amplification ZNE — not unitary folding / identity
/// insertion (those are out of scope for this MVP).
///
/// Requires ``NoiseModel/appliesDepolarizing`` (`p > 0`). Scaling with `p = 0` / missing noise
/// is rejected — otherwise per-scale seed offsets would extrapolate shot noise alone.
///
/// **Clamp saturation:** when `p · λ > 1` the stretched probability is clipped to 1, so large
/// `λ` no longer follow a linear noise model and bias the `E(0)` fit. Prefer scales with
/// `p · λ ≤ 1`, or keep `p` small enough for the chosen factors.
///
/// ## Extrapolation
/// **Linear least-squares:** fit `E(λ) ≈ a + b λ` through the measured `(λ, E(λ))` points and
/// return `a = E(0)`. With two distinct scales this is exact linear interpolation to zero
/// (equivalent to first-order Richardson). Higher-order Richardson / polynomial fits are not
/// implemented. Trace preservation / model correctness of the stretch are not proven.
///
/// Default ``scaleFactors`` are `[1, 3, 5]`. Requires at least two distinct finite
/// **non-negative** scales (`λ ≥ 0`).
public struct ZNEOptions: Sendable, Equatable {
    /// Noise scale factors `λ` (must contain ≥ 2 distinct values). Default `[1, 3, 5]`.
    public var scaleFactors: [QFloat]
    /// Fit used to extrapolate to `λ = 0`. Only ``linear`` is available in this MVP.
    public var extrapolator: ZNEExtrapolator

    public init(
        scaleFactors: [QFloat] = [1, 3, 5],
        extrapolator: ZNEExtrapolator = .linear
    ) {
        self.scaleFactors = scaleFactors
        self.extrapolator = extrapolator
    }

    /// Canonical default scales for global-depolarizing ZNE.
    public static let `default` = ZNEOptions()

    /// Whether these options request an active ZNE run (≥ 2 scale factors).
    public var isActive: Bool { scaleFactors.count >= 2 }
}

/// Extrapolator choice for ``ZNEOptions`` (C13).
public enum ZNEExtrapolator: String, Sendable, Equatable, Codable, CaseIterable {
    /// Least-squares fit `E(λ) = a + bλ` → report `a`.
    case linear
}

/// Fixed scaling method token hashed into ``PipelineFingerprint`` (only one MVP method).
public enum ZNEScalingMethod: String, Sendable, Equatable, Codable, CaseIterable {
    /// Stretch ``NoiseModel/depolarizingProbability`` by `λ` (clamped to `[0, 1]`).
    case globalDepolarizing
}

/// Per-scale measurements and extrapolated zero-noise value returned by shot ``Estimator``.
public struct ZNEExtrapolationMetadata: Sendable, Equatable {
    public let scalingMethod: ZNEScalingMethod
    public let extrapolator: ZNEExtrapolator
    /// Scale factors in evaluation order.
    public let scaleFactors: [QFloat]
    /// Shot expectation at each scale (same order as ``scaleFactors``).
    public let valuesAtScale: [QFloat]
    /// Extrapolated `E(0)`.
    public let extrapolatedValue: QFloat

    public init(
        scalingMethod: ZNEScalingMethod = .globalDepolarizing,
        extrapolator: ZNEExtrapolator,
        scaleFactors: [QFloat],
        valuesAtScale: [QFloat],
        extrapolatedValue: QFloat
    ) {
        self.scalingMethod = scalingMethod
        self.extrapolator = extrapolator
        self.scaleFactors = scaleFactors
        self.valuesAtScale = valuesAtScale
        self.extrapolatedValue = extrapolatedValue
    }
}

public enum ZNEError: Error, Equatable {
    case insufficientScaleFactors(Int)
    case nonDistinctScaleFactors
    case nonFiniteScaleFactor
    /// Scale factors must be `≥ 0` (`scalingGlobalDepolarizing` clamps negatives to zero noise).
    case negativeScaleFactor
    case scaleValueCountMismatch(scales: Int, values: Int)
    case singularLinearFit
    /// Global-depolarizing ZNE needs ``NoiseModel/depolarizingProbability`` `> 0`.
    case missingGlobalDepolarizing
}

/// Host-side zero-noise fit helpers (C13).
public enum ZeroNoiseExtrapolation {
    /// Linear least-squares `E(λ) = a + bλ`; returns `a` (value at `λ = 0`).
    public static func extrapolateLinear(
        scaleFactors: [QFloat],
        values: [QFloat]
    ) throws -> QFloat {
        guard scaleFactors.count >= 2 else {
            throw ZNEError.insufficientScaleFactors(scaleFactors.count)
        }
        guard scaleFactors.count == values.count else {
            throw ZNEError.scaleValueCountMismatch(scales: scaleFactors.count, values: values.count)
        }
        let n = scaleFactors.count
        var sx: Double = 0
        var sy: Double = 0
        var sxx: Double = 0
        var sxy: Double = 0
        for index in 0..<n {
            let x = Double(scaleFactors[index])
            let y = Double(values[index])
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
        }
        let denom = Double(n) * sxx - sx * sx
        guard abs(denom) > 1e-18 else { throw ZNEError.singularLinearFit }
        let slope = (Double(n) * sxy - sx * sy) / denom
        let intercept = (sy - slope * sx) / Double(n)
        return QFloat(intercept)
    }

    /// Validates and applies ``ZNEOptions/extrapolator``.
    public static func extrapolate(
        options: ZNEOptions,
        valuesAtScale: [QFloat]
    ) throws -> QFloat {
        try validate(options)
        switch options.extrapolator {
        case .linear:
            return try extrapolateLinear(
                scaleFactors: options.scaleFactors,
                values: valuesAtScale
            )
        }
    }

    public static func validate(_ options: ZNEOptions) throws {
        guard options.scaleFactors.count >= 2 else {
            throw ZNEError.insufficientScaleFactors(options.scaleFactors.count)
        }
        var seen = Set<QFloat>()
        for scale in options.scaleFactors {
            guard scale.isFinite else { throw ZNEError.nonFiniteScaleFactor }
            guard scale >= 0 else { throw ZNEError.negativeScaleFactor }
            guard seen.insert(scale).inserted else {
                throw ZNEError.nonDistinctScaleFactors
            }
        }
    }
}
