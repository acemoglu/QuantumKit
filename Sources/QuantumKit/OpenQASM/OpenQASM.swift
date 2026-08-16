/// Public OpenQASM front-end for QuantumKit: detect, parse, import, and export.
///
/// ## Bit / qubit order
/// QuantumKit uses **LSB = qubit 0**. An OpenQASM subscript `q[0]` maps to engine
/// qubit `0` (same index). Multiple `qreg` / `qubit` declarations are linearized
/// in declaration order into one contiguous address space — see
/// ``OpenQASM2Importer`` for details.
///
/// Classical registers stay separate ``ClassicalRegisterSpec`` entries in
/// declaration order (not bit-linearized across registers).
public enum OpenQASM {
    /// Detects the OpenQASM language version from `source`.
    ///
    /// Missing `OPENQASM` header defaults to ``OpenQASMVersion/v2``.
    public static func detectVersion(from source: String) throws -> OpenQASMVersion {
        try OpenQASMVersion.detect(from: source)
    }

    /// Lexes and parses `source` into an ``OpenQASMProgram`` AST.
    public static func parse(_ source: String) throws -> OpenQASMProgram {
        var parser = try OpenQASMParser(source: source)
        return try parser.parse()
    }

    /// Parses `source` and lowers it to a ``QuantumCircuit`` (version-aware).
    ///
    /// Dispatches via ``OpenQASMImporter``. For OpenQASM 3 `while`, supply a bound
    /// through ``OpenQASMImporterOptions/v3`` (``OpenQASM3ImporterOptions/defaultWhileMaxIterations``)
    /// or a `// @quantumkit.max_while_iterations N` pragma in the source.
    public static func importCircuit(
        _ source: String,
        options: OpenQASMImporterOptions = OpenQASMImporterOptions()
    ) throws -> QuantumCircuit {
        try OpenQASMImporter(options: options).`import`(source: source)
    }

    /// Convenience: import with OpenQASM 3 while-bound options only.
    ///
    /// Equivalent to ``importCircuit(_:options:)`` with
    /// `OpenQASMImporterOptions(v3: qasm3)`.
    public static func importCircuit(
        _ source: String,
        qasm3: OpenQASM3ImporterOptions
    ) throws -> QuantumCircuit {
        try importCircuit(source, options: OpenQASMImporterOptions(v3: qasm3))
    }

    /// Exports `circuit` as OpenQASM source.
    ///
    /// Default ``OpenQASMExportOptions/version`` is ``OpenQASMVersion/v3``.
    public static func export(
        _ circuit: QuantumCircuit,
        options: OpenQASMExportOptions = OpenQASMExportOptions()
    ) throws -> String {
        try OpenQASMExporter(options: options).export(circuit)
    }

    /// Convenience: export the static OpenQASM 2 subset (`OPENQASM 2.0` + qelib1).
    public static func exportQASM2(_ circuit: QuantumCircuit) throws -> String {
        try OpenQASM2Exporter().export(circuit)
    }
}
