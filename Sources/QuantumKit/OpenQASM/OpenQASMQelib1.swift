import Foundation

/// Builtin OpenQASM 2 `qelib1.inc` catalog for QuantumKit.
///
/// `include "qelib1.inc"` is accepted without reading a file from disk. Gate names
/// listed here lower to ``Gate`` in the importer (parameter-free and numeric-angle
/// parametric forms). Some names (`ch`, `cy`) expand to known decompositions into
/// existing ``Gate`` cases rather than a dedicated enum case.
public enum OpenQASMQelib1: Sendable {
    /// Canonical include path recognized as the builtin standard library.
    public static let includeFileName = "qelib1.inc"

    /// Gate names that map to ``Gate`` (parameter-free and parametric).
    ///
    /// `ch` / `cy` lower via standard decompositions (see importer). `mcx` / `mcz`
    /// map to ``Gate/mcx`` / ``Gate/mcz`` (2-control `mcx` uses ``Gate/ccx``).
    /// `cu1` aliases ``Gate/cp``.
    public static let mappedGateNames: Set<String> = [
        // Parameter-free
        "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
        "cx", "cz", "cy", "ch", "swap", "ccx", "cswap",
        "mcx", "mcz",
        // Parametric (numeric angle expressions)
        "u", "u1", "u2", "u3", "p",
        "rx", "ry", "rz",
        "crx", "cry", "crz", "cp", "cu1",
    ]

    /// Returns `true` when `path` refers to the builtin qelib1 include (basename match).
    public static func isBuiltinInclude(_ path: String) -> Bool {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == includeFileName { return true }
        // Allow path-style includes that end with the standard filename.
        if trimmed.hasSuffix("/" + includeFileName) { return true }
        return false
    }
}
