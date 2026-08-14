import Foundation

/// Versioned serialization contract for persisted circuit IR
/// (``QuantumCircuit``, ``GateSequence`` / ``Subcircuit``).
///
/// **Current version:** ``current`` = **1**.
///
/// **Migration rule:** payloads that omit `schemaVersion` (the pre-contract /
/// unversioned Codable shape) are accepted and decoded as version 1 — field layout
/// is identical, so migration is a no-op upgrade that only establishes the version
/// tag on the next encode. Explicit `schemaVersion` values other than ``current``
/// (including future versions) throw ``CircuitIRError/unsupportedSchemaVersion``.
///
/// This contract does not change in-memory engine bit ordering; see ``QubitBitOrdering``.
public enum CircuitIRSchema: Sendable {
    /// Schema version written by current encoders and required (or implied) on decode.
    public static let current: Int = 1

    /// Resolves a decoded optional version: `nil` → legacy unversioned → ``current``.
    /// Throws when `version` is present and not ``current``.
    public static func resolve(_ version: Int?) throws -> Int {
        let resolved = version ?? current
        guard resolved == current else {
            throw CircuitIRError.unsupportedSchemaVersion(
                found: resolved,
                supported: current
            )
        }
        return resolved
    }
}

/// Errors from versioned circuit IR encode/decode.
public enum CircuitIRError: Error, Equatable, Sendable {
    /// Payload declared a `schemaVersion` this library cannot read.
    case unsupportedSchemaVersion(found: Int, supported: Int)
}
