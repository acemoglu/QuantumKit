/// Thin OpenQASM entry points on ``QuantumCircuit``.
///
/// These hooks delegate entirely to ``OpenQASM`` / the OpenQASM importers and
/// exporters. They do not change circuit validation, simulation defaults, or IR.
///
/// ## Bit / qubit order
/// QuantumKit LSB = qubit 0. OpenQASM `q[0]` maps to engine qubit `0`.
extension QuantumCircuit {
    /// Builds a circuit by parsing and lowering OpenQASM 2/3 source.
    ///
    /// Version is detected from the `OPENQASM` header (missing header → QASM2).
    /// For OpenQASM 3 `while`, supply a bound via `options.v3` or a
    /// `// @quantumkit.max_while_iterations N` pragma in `source`.
    public init(
        openQASM source: String,
        options: OpenQASMImporterOptions = OpenQASMImporterOptions()
    ) throws {
        self = try OpenQASM.importCircuit(source, options: options)
    }

    /// Exports this circuit as OpenQASM text.
    ///
    /// Default ``OpenQASMExportOptions/version`` is OpenQASM 3.
    public func openQASM(options: OpenQASMExportOptions = OpenQASMExportOptions()) throws -> String {
        try OpenQASM.export(self, options: options)
    }

    /// Exports this circuit as the static OpenQASM 2.0 + qelib1 subset.
    public func openQASM2() throws -> String {
        try OpenQASM.exportQASM2(self)
    }
}
