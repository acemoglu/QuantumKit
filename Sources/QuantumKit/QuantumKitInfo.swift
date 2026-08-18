import Foundation

/// Package identity surfaced in reproducibility metadata and fingerprints.
///
/// The ``version`` string must stay aligned with the published package / git tag when a
/// release is cut.
public enum QuantumKitInfo {
    /// Library version (SemVer). Breaking public-API changes bump MAJOR.
    public static let version = "1.0.0"
}
