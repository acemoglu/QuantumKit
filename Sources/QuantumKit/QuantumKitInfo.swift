import Foundation

/// Package identity surfaced in reproducibility metadata and fingerprints.
///
/// The ``version`` string must stay aligned with the published package / git tag when a
/// release is cut. Stability rules for what that version guarantees are documented on
/// ``QuantumKitAPIPolicy``.
public enum QuantumKitInfo {
    /// Library version (SemVer).
    ///
    /// Bumped to `0.2.0` for **breaking** public-API removals / visibility changes per
    /// ``QuantumKitAPIPolicy``: H6c (device: inits, `DensityMatrix.device`), H7b (MTLBuffer
    /// accessors, `outputBuffer:` shim), and package-internal demotion of `Pipelines` /
    /// `TRNGCollapse`. Pre-1.0: MINOR bump for breaking changes until 1.0.0.
    public static let version = "0.2.0"
}
