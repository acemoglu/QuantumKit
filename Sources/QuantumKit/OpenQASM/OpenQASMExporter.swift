import Foundation

/// Options for ``QuantumCircuit`` → OpenQASM export.
public struct OpenQASMExportOptions: Equatable, Sendable {
    /// Target language version. Defaults to OpenQASM 3.
    public var version: OpenQASMVersion

    public init(version: OpenQASMVersion = .v3) {
        self.version = version
    }
}

/// Serializes a ``QuantumCircuit`` to OpenQASM, branching on ``OpenQASMExportOptions/version``.
///
/// Default export is **OpenQASM 3.0** (`qubit` / `bit`, no `include`).
/// Pass ``OpenQASMExportOptions/init(version:)`` with `.v2`, or use ``OpenQASM2Exporter``,
/// for OpenQASM 2.0 + `include "qelib1.inc"`.
///
/// ## Register naming
/// - Quantum register is always a single `q` of size ``QuantumCircuit/qubitCount``
///   (`qubit[N] q;` or `qreg q[N];`).
/// - Classical registers: one register → `c`; multiple → `c0`, `c1`, … in
///   ``QuantumCircuit/classicalRegisters`` order.
///
/// ## Angle policy
/// Only ``QFloatExpr/literal`` angles are exported. Symbolic / parameter /
/// structured expressions throw ``OpenQASMError/unsupported``.
///
/// ## Classical if
/// OpenQASM 3: consecutive ``Gate/c_if`` with the same condition are coalesced into a
/// braced `if (c==imm) { … }` block when there is more than one statement; a single
/// conditioned statement stays brace-less. OpenQASM 2 always emits brace-less
/// `if(c==imm) <stmt>;` lines (including one line per gate in a multi-gate chain).
///
/// ## Bounded while (OpenQASM 3 only)
/// ``Gate/while_c`` exports as a QASM3 `while` plus a comment pragma
/// `// @quantumkit.max_while_iterations N` so re-import can recover the bound
/// (see ``OpenQASMUnsupported/whileMaxIterationsPragmaPrefix``). OpenQASM 2
/// export still rejects `while_c`.
///
/// ## Unsupported
/// `unitary1`, `customUnitary`, `initialize`, `delay`, `mcx`/`mcz`,
/// Ising / ECR / iSWAP / DCX, and other non-qelib1 gates; `while_c` under QASM2.
public struct OpenQASMExporter: Sendable {
    public var options: OpenQASMExportOptions

    public init(options: OpenQASMExportOptions = OpenQASMExportOptions()) {
        self.options = options
    }

    /// Exports `circuit` as OpenQASM source for ``OpenQASMExportOptions/version``.
    public func export(_ circuit: QuantumCircuit) throws -> String {
        switch options.version {
        case .v2:
            return try exportV2(circuit)
        case .v3:
            return try exportV3(circuit)
        }
    }

    private func exportV2(_ circuit: QuantumCircuit) throws -> String {
        var lines: [String] = []
        lines.append("OPENQASM 2.0;")
        lines.append("include \"\(OpenQASMQelib1.includeFileName)\";")
        lines.append("qreg q[\(circuit.qubitCount)];")

        let cregNames = Self.classicalRegisterNames(count: circuit.classicalRegisters.count)
        for (index, spec) in circuit.classicalRegisters.enumerated() {
            lines.append("creg \(cregNames[index])[\(spec.bitCount)];")
        }

        try appendGates(of: circuit, cregNames: cregNames, dialectLabel: "OpenQASM 2", to: &lines)
        return lines.joined(separator: "\n") + "\n"
    }

    private func exportV3(_ circuit: QuantumCircuit) throws -> String {
        var lines: [String] = []
        lines.append("OPENQASM 3.0;")
        lines.append("qubit[\(circuit.qubitCount)] q;")

        let cregNames = Self.classicalRegisterNames(count: circuit.classicalRegisters.count)
        for (index, spec) in circuit.classicalRegisters.enumerated() {
            lines.append("bit[\(spec.bitCount)] \(cregNames[index]);")
        }

        try appendGates(of: circuit, cregNames: cregNames, dialectLabel: "OpenQASM 3", to: &lines)
        return lines.joined(separator: "\n") + "\n"
    }

    private func appendGates(
        of circuit: QuantumCircuit,
        cregNames: [String],
        dialectLabel: String,
        to lines: inout [String]
    ) throws {
        let ctx = ExportContext(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters,
            cregNames: cregNames,
            dialectLabel: dialectLabel
        )
        try lines.append(contentsOf: ctx.emitGates(circuit.gates))
    }

    /// `c` for a single classical register; `c0`, `c1`, … when there are several.
    public static func classicalRegisterNames(count: Int) -> [String] {
        guard count > 0 else { return [] }
        if count == 1 { return ["c"] }
        return (0..<count).map { "c\($0)" }
    }
}

/// Convenience exporter that always emits OpenQASM 2.0 + qelib1.
///
/// Existing round-trip tests depend on this type remaining QASM2-only.
public struct OpenQASM2Exporter: Sendable {
    public var options: OpenQASMExportOptions

    public init(options: OpenQASMExportOptions = OpenQASMExportOptions(version: .v2)) {
        var forced = options
        forced.version = .v2
        self.options = forced
    }

    /// Exports `circuit` as OpenQASM 2.0 source.
    public func export(_ circuit: QuantumCircuit) throws -> String {
        try OpenQASMExporter(options: options).export(circuit)
    }

    /// `c` for a single classical register; `c0`, `c1`, … when there are several.
    public static func classicalRegisterNames(count: Int) -> [String] {
        OpenQASMExporter.classicalRegisterNames(count: count)
    }
}

// MARK: - Emission context

private struct ExportContext {
    let qubitCount: Int
    let classicalRegisters: [ClassicalRegisterSpec]
    let cregNames: [String]
    let dialectLabel: String

    /// Emits the full gate list, coalescing consecutive same-condition `c_if` into
    /// braced OpenQASM 3 `if` blocks when helpful.
    func emitGates(_ gates: [Gate]) throws -> [String] {
        var lines: [String] = []
        var index = 0
        let coalesceBracedIf = dialectLabel.hasPrefix("OpenQASM 3")
        while index < gates.count {
            if coalesceBracedIf,
               case .c_if(let register, let value, _) = gates[index] {
                var runEnd = index + 1
                while runEnd < gates.count,
                      case .c_if(let r, let v, _) = gates[runEnd],
                      r == register,
                      v == value {
                    runEnd += 1
                }
                if runEnd - index > 1 {
                    let name = try classicalName(register)
                    lines.append("if(\(name)==\(value)) {")
                    for gateIndex in index..<runEnd {
                        guard case .c_if(_, _, let inner) = gates[gateIndex] else {
                            continue
                        }
                        for bodyLine in try emitGate(inner, conditioned: nil) {
                            lines.append("  \(bodyLine)")
                        }
                    }
                    lines.append("}")
                    index = runEnd
                    continue
                }
            }
            lines.append(contentsOf: try emitGate(gates[index], conditioned: nil))
            index += 1
        }
        return lines
    }

    /// Emits one or more OpenQASM statements for `gate`.
    /// When `conditioned` is set, each statement is wrapped as `if(creg==imm) <stmt>`.
    func emitGate(
        _ gate: Gate,
        conditioned: (register: Int, value: Int)?
    ) throws -> [String] {
        switch gate {
        case .c_if(let classicalRegister, let expectedValue, let inner):
            let lines = try emitGate(inner, conditioned: (classicalRegister, expectedValue))
            if let conditioned {
                return try wrapAll(lines, conditioned: conditioned)
            }
            return lines

        case .while_c(let classicalRegister, let expectedValue, let body, let maxIterations):
            if dialectLabel.hasPrefix("OpenQASM 2") {
                throw unsupported(
                    feature: "while_c",
                    message: "OpenQASM 2 export does not support while_c"
                )
            }
            if conditioned != nil {
                throw unsupported(
                    feature: "while_c",
                    message: "OpenQASM 3 export does not support while_c nested under if"
                )
            }
            guard maxIterations > 0 else {
                throw OpenQASMError.semanticError(
                    line: 1,
                    column: 1,
                    message: "while_c maxIterations must be > 0"
                )
            }
            let name = try classicalName(classicalRegister)
            var lines: [String] = []
            // Pragma consumed by ``OpenQASMWhilePragmaScanner`` on re-import.
            lines.append(
                "// \(OpenQASMUnsupported.whileMaxIterationsPragmaPrefix) \(maxIterations)"
            )
            lines.append("while (\(name)==\(expectedValue)) {")
            for inner in body {
                let bodyLines = try emitGate(inner, conditioned: nil)
                for bodyLine in bodyLines {
                    lines.append("  \(bodyLine)")
                }
            }
            lines.append("}")
            return lines

        case .h(let target):
            return try wrap("h \(q(target));", conditioned: conditioned)
        case .x(let target):
            return try wrap("x \(q(target));", conditioned: conditioned)
        case .y(let target):
            return try wrap("y \(q(target));", conditioned: conditioned)
        case .z(let target):
            return try wrap("z \(q(target));", conditioned: conditioned)
        case .s(let target):
            return try wrap("s \(q(target));", conditioned: conditioned)
        case .sdg(let target):
            return try wrap("sdg \(q(target));", conditioned: conditioned)
        case .t(let target):
            return try wrap("t \(q(target));", conditioned: conditioned)
        case .tdg(let target):
            return try wrap("tdg \(q(target));", conditioned: conditioned)
        case .sx(let target):
            return try wrap("sx \(q(target));", conditioned: conditioned)
        case .sxdg(let target):
            return try wrap("sxdg \(q(target));", conditioned: conditioned)
        case .id(let target):
            return try wrap("id \(q(target));", conditioned: conditioned)

        case .cx(let control, let target):
            return try wrap("cx \(q(control)),\(q(target));", conditioned: conditioned)
        case .cz(let control, let target):
            return try wrap("cz \(q(control)),\(q(target));", conditioned: conditioned)
        case .swap(let q1, let q2):
            return try wrap("swap \(q(q1)),\(q(q2));", conditioned: conditioned)
        case .ccx(let control1, let control2, let target):
            return try wrap(
                "ccx \(q(control1)),\(q(control2)),\(q(target));",
                conditioned: conditioned
            )
        case .cswap(let control, let q1, let q2):
            return try wrap(
                "cswap \(q(control)),\(q(q1)),\(q(q2));",
                conditioned: conditioned
            )

        case .p(let theta, let target):
            let angle = try requireLiteralAngle(theta, gate: "p")
            // qelib1 flavor: Gate.p ↔ u1 (accepted by both importers)
            return try wrap("u1(\(angle)) \(q(target));", conditioned: conditioned)

        case .u(let theta, let phi, let lambda, let target):
            let t = try requireLiteralAngle(theta, gate: "u")
            let p = try requireLiteralAngle(phi, gate: "u")
            let l = try requireLiteralAngle(lambda, gate: "u")
            return try wrap("u3(\(t),\(p),\(l)) \(q(target));", conditioned: conditioned)

        case .rx(let theta, let target):
            let angle = try requireLiteralAngle(theta, gate: "rx")
            return try wrap("rx(\(angle)) \(q(target));", conditioned: conditioned)
        case .ry(let theta, let target):
            let angle = try requireLiteralAngle(theta, gate: "ry")
            return try wrap("ry(\(angle)) \(q(target));", conditioned: conditioned)
        case .rz(let theta, let target):
            let angle = try requireLiteralAngle(theta, gate: "rz")
            return try wrap("rz(\(angle)) \(q(target));", conditioned: conditioned)

        case .crx(let theta, let control, let target):
            let angle = try requireLiteralAngle(theta, gate: "crx")
            return try wrap(
                "crx(\(angle)) \(q(control)),\(q(target));",
                conditioned: conditioned
            )
        case .cry(let theta, let control, let target):
            let angle = try requireLiteralAngle(theta, gate: "cry")
            return try wrap(
                "cry(\(angle)) \(q(control)),\(q(target));",
                conditioned: conditioned
            )
        case .crz(let theta, let control, let target):
            let angle = try requireLiteralAngle(theta, gate: "crz")
            return try wrap(
                "crz(\(angle)) \(q(control)),\(q(target));",
                conditioned: conditioned
            )
        case .cp(let theta, let control, let target):
            let angle = try requireLiteralAngle(theta, gate: "cp")
            return try wrap(
                "cp(\(angle)) \(q(control)),\(q(target));",
                conditioned: conditioned
            )

        case .measure(let spec):
            return try emitMeasure(spec, conditioned: conditioned)

        case .reset(let qubit):
            return try wrap("reset \(q(qubit));", conditioned: conditioned)

        case .barrier(let qubits):
            if qubits.isEmpty {
                return try wrap("barrier;", conditioned: conditioned)
            }
            if qubits.count == qubitCount,
               qubits.sorted() == Array(0..<qubitCount) {
                return try wrap("barrier q;", conditioned: conditioned)
            }
            let args = qubits.map { q($0) }.joined(separator: ",")
            return try wrap("barrier \(args);", conditioned: conditioned)

        case .delay:
            throw unsupported(
                feature: "delay",
                message: "\(dialectLabel) export does not support delay"
            )
        case .iswap:
            throw unsupported(
                feature: "iswap",
                message: "\(dialectLabel) / qelib1 export does not support iswap"
            )
        case .ecr:
            throw unsupported(
                feature: "ecr",
                message: "\(dialectLabel) / qelib1 export does not support ecr"
            )
        case .rxx:
            throw unsupported(
                feature: "rxx",
                message: "\(dialectLabel) / qelib1 export does not support rxx"
            )
        case .ryy:
            throw unsupported(
                feature: "ryy",
                message: "\(dialectLabel) / qelib1 export does not support ryy"
            )
        case .rzz:
            throw unsupported(
                feature: "rzz",
                message: "\(dialectLabel) / qelib1 export does not support rzz"
            )
        case .dcx:
            throw unsupported(
                feature: "dcx",
                message: "\(dialectLabel) / qelib1 export does not support dcx"
            )
        case .mcx:
            throw unsupported(
                feature: "mcx",
                message: "\(dialectLabel) / qelib1 export does not support mcx (use ccx when applicable)"
            )
        case .mcz:
            throw unsupported(
                feature: "mcz",
                message: "\(dialectLabel) / qelib1 export does not support mcz"
            )
        case .unitary1:
            throw unsupported(
                feature: "unitary1",
                message: "\(dialectLabel) export does not support unitary1"
            )
        case .initialize:
            throw unsupported(
                feature: "initialize",
                message: "\(dialectLabel) export does not support initialize"
            )
        case .customUnitary:
            throw unsupported(
                feature: "customUnitary",
                message: "\(dialectLabel) export does not support customUnitary"
            )
        }
    }

    private func emitMeasure(
        _ spec: MeasureSpec,
        conditioned: (register: Int, value: Int)?
    ) throws -> [String] {
        let cregName = try classicalName(spec.classicalRegister)
        let width = try classicalWidth(spec.classicalRegister)

        if isWholeRegisterMeasure(spec, cregWidth: width) {
            return try wrap("measure q -> \(cregName);", conditioned: conditioned)
        }

        var lines: [String] = []
        for (offset, qubit) in spec.qubits.enumerated() {
            let bit = spec.classicalBitOffset + offset
            guard bit >= 0, bit < width else {
                throw OpenQASMError.semanticError(
                    line: 1,
                    column: 1,
                    message: "measure classical bit \(bit) out of range for \(cregName)[\(width)]"
                )
            }
            let stmt = "measure \(q(qubit)) -> \(cregName)[\(bit)];"
            lines.append(contentsOf: try wrap(stmt, conditioned: conditioned))
        }
        return lines
    }

    /// Whole-register measure when qubits cover the full `q` and map onto the full creg from bit 0.
    private func isWholeRegisterMeasure(_ spec: MeasureSpec, cregWidth: Int) -> Bool {
        guard spec.classicalBitOffset == 0,
              spec.qubits.count == cregWidth,
              spec.qubits.count == qubitCount,
              spec.qubits == Array(0..<qubitCount)
        else {
            return false
        }
        return true
    }

    private func wrap(
        _ statement: String,
        conditioned: (register: Int, value: Int)?
    ) throws -> [String] {
        guard let conditioned else { return [statement] }
        let name = try classicalName(conditioned.register)
        // Single-statement form stays brace-less (QASM2-compatible / compact QASM3).
        return ["if(\(name)==\(conditioned.value)) \(statement)"]
    }

    private func wrapAll(
        _ statements: [String],
        conditioned: (register: Int, value: Int)
    ) throws -> [String] {
        let name = try classicalName(conditioned.register)
        if statements.isEmpty { return [] }
        if statements.count == 1 || dialectLabel.hasPrefix("OpenQASM 2") {
            return statements.map { stmt in
                "if(\(name)==\(conditioned.value)) \(stmt)"
            }
        }
        // OpenQASM 3 multi-statement conditioned region → braced block.
        var lines: [String] = ["if(\(name)==\(conditioned.value)) {"]
        for stmt in statements {
            lines.append("  \(stmt)")
        }
        lines.append("}")
        return lines
    }

    private func q(_ index: Int) -> String {
        "q[\(index)]"
    }

    private func classicalName(_ index: Int) throws -> String {
        guard index >= 0, index < cregNames.count else {
            throw OpenQASMError.semanticError(
                line: 1,
                column: 1,
                message: "Classical register index \(index) out of range"
            )
        }
        return cregNames[index]
    }

    private func classicalWidth(_ index: Int) throws -> Int {
        guard index >= 0, index < classicalRegisters.count else {
            throw OpenQASMError.semanticError(
                line: 1,
                column: 1,
                message: "Classical register index \(index) out of range"
            )
        }
        return classicalRegisters[index].bitCount
    }

    private func requireLiteralAngle(_ expr: QFloatExpr, gate: String) throws -> String {
        switch expr {
        case .literal(let value):
            return formatAngle(value)
        case .parameter(let parameter):
            throw unsupported(
                feature: "parameter",
                message: "Gate '\(gate)' angle is symbolic ('\(parameter.name)'); static \(dialectLabel) export requires literal angles"
            )
        case .negated, .scaled:
            throw unsupported(
                feature: "angle expression",
                message: "Gate '\(gate)' angle is not a literal; static \(dialectLabel) export requires QFloatExpr.literal"
            )
        }
    }

    private func unsupported(feature: String, message: String) -> OpenQASMError {
        OpenQASMError.unsupported(line: 1, column: 1, feature: feature, message: message)
    }
}

// MARK: - Angle formatting

/// Formats a ``QFloat`` for OpenQASM, preferring small multiples of `pi` when exact under Float32.
private func formatAngle(_ value: QFloat) -> String {
    let d = Double(value)
    if let piForm = formatAsPiMultiple(d) {
        return piForm
    }
    return formatDecimal(d)
}

private func formatAsPiMultiple(_ d: Double) -> String? {
    if abs(d) < 1e-12 {
        return "0"
    }
    let pi = Double.pi
    // Match k * pi / n for small integers (Float32-friendly tolerance).
    for denom in 1...64 {
        let scaled = d * Double(denom) / pi
        let nearest = scaled.rounded()
        if abs(scaled - nearest) <= 1e-5 {
            let num = Int(nearest)
            if num == 0 { return "0" }
            return formatPiFraction(numerator: num, denominator: denom)
        }
    }
    return nil
}

private func formatPiFraction(numerator: Int, denominator: Int) -> String {
    if denominator == 1 {
        if numerator == 1 { return "pi" }
        if numerator == -1 { return "-pi" }
        return "\(numerator)*pi"
    }
    if numerator == 1 { return "pi/\(denominator)" }
    if numerator == -1 { return "-pi/\(denominator)" }
    return "\(numerator)*pi/\(denominator)"
}

private func formatDecimal(_ d: Double) -> String {
    // Enough digits for Float32 round-trip; strip trailing zeros / decimal point.
    var s = String(format: "%.9g", d)
    if s.contains("e") || s.contains("E") {
        s = String(format: "%.9f", d)
    }
    if s.contains(".") {
        while s.hasSuffix("0") {
            s.removeLast()
        }
        if s.hasSuffix(".") {
            s.removeLast()
        }
    }
    if s.isEmpty || s == "-" {
        return "0"
    }
    return s
}
