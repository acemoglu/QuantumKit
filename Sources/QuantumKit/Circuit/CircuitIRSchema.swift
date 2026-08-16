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
/// **Validation:** decode rebuilds the circuit via ``QuantumCircuit/apply(_:metadata:)``.
/// Out-of-range qubits, undeclared classical registers, and an `instructionMetadata`
/// array whose length does not match `gates` throw ``CircuitIRError`` (no silent
/// truncate / pad of a present metadata payload). Omitting `instructionMetadata`
/// remains valid and means all-`nil` metadata.
///
/// **G10 lite / ``Gate/while_c``:** bounded classical loops exist in the in-memory IR
/// and on CPU / Metal **host** execution, but are **not serialized yet**.
/// **Encode** of a circuit (or gate) containing ``Gate/while_c`` throws
/// ``CircuitIRError/controlFlowNotSerialized``. Hand-crafted JSON with an unknown
/// gate `type` (including a future `while_c` tag) fails ordinary `Gate`/`Decodable`
/// type decoding — it does **not** round-trip under schema v1. Schema stays at **1**;
/// a future bump will persist control-flow ops. Static circuits without `while_c`
/// are unchanged.
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
    /// Present `instructionMetadata` length does not match `gates`.
    case metadataLengthMismatch(metadataCount: Int, gateCount: Int)
    /// Gate list failed validation (qubit OOB, undeclared classical register, …).
    case invalidCircuit(reason: String)
    /// Dynamic control-flow op (``Gate/while_c``) cannot be encoded under schema v1.
    case controlFlowNotSerialized(op: String)
}
