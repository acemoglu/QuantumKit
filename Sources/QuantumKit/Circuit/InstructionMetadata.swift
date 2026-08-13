import Foundation

/// Non-semantic annotation attached to a circuit instruction.
///
/// Labels and names do **not** affect unitary action or execution. Classical conditioning
/// remains the ``Gate/c_if`` mechanism; `conditionNote` is documentation only.
///
/// **Transpile policy:** ``PassManager`` rebuilds gate lists and **strips** instruction
/// metadata by default. Set ``TranspileOptions/preserveInstructionMetadata`` to retain
/// metadata on identity-preserving remaps and non-expanding keep-paths (layout / route
/// remaps of original gates, basis gates already in the target set). Expanding passes
/// (unroll, basis expansion, schedule delay insertion) still drop metadata because
/// instruction indices no longer align 1:1. Inserted SWAPs always get `nil` metadata.
public struct InstructionMetadata: Sendable, Equatable, Codable, Hashable {
    /// Optional canonical / display name (e.g. `"mcx"`, `"oracle"`).
    public var name: String?
    /// User-facing label that survives Codable round-trips.
    public var label: String?
    /// Human-readable note about classical conditioning; not an executable predicate.
    public var conditionNote: String?

    public init(name: String? = nil, label: String? = nil, conditionNote: String? = nil) {
        self.name = name
        self.label = label
        self.conditionNote = conditionNote
    }

    public var isEmpty: Bool {
        (name == nil || name?.isEmpty == true)
            && (label == nil || label?.isEmpty == true)
            && (conditionNote == nil || conditionNote?.isEmpty == true)
    }
}
