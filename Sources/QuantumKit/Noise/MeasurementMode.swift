import Foundation

/// How a computational-basis ``Gate/measure`` updates the quantum state (C10).
public enum MeasurementMode: String, Sendable, Equatable, Codable, CaseIterable {
    /// Born-rule sample + projective collapse (default).
    case projective
    /// Born-rule sample for classical bits, but replace collapse with full Z-dephasing
    /// on each measured qubit (populations kept, coherences cleared). Density-matrix only.
    case dephasingOnly
}
