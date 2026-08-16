import Foundation

/// Options for ``QuantumCircuit`` → OpenQASM export.
///
/// This slice always emits OpenQASM 2.0 with `include "qelib1.inc"`.
/// A version preference is reserved for a future QASM3 exporter slice.
public struct OpenQASMExportOptions: Equatable, Sendable {
    public init() {}
}

/// Serializes a ``QuantumCircuit`` to a static OpenQASM 2.0 string (qelib1 subset).
///
/// ## Register naming
/// - Quantum register is always a single `qreg q[N];` (qubit `i` ↔ `q[i]`).
/// - Classical registers: one register → `creg c[W];`; multiple → `creg c0[W0];`,
///   `creg c1[W1];`, … in ``QuantumCircuit/classicalRegisters`` order.
///
/// ## Angle policy
/// Only ``QFloatExpr/literal`` angles are exported. Symbolic / parameter /
/// structured expressions throw ``OpenQASMError/unsupported``.
///
/// ## Unsupported for QASM2
/// `while_c`, `unitary1`, `customUnitary`, `initialize`, `delay`, `mcx`/`mcz`,
/// Ising / ECR / iSWAP / DCX, and other non-qelib1 gates.
public struct OpenQASM2Exporter: Sendable {
    public var options: OpenQASMExportOptions

    public init(options: OpenQASMExportOptions = OpenQASMExportOptions()) {
        self.options = options
    }

    /// Exports `circuit` as OpenQASM 2.0 source.
    public func export(_ circuit: QuantumCircuit) throws -> String {
        var lines: [String] = []
        lines.append("OPENQASM 2.0;")
        lines.append("include \"\(OpenQASMQelib1.includeFileName)\";")
        lines.append("qreg q[\(circuit.qubitCount)];")

        let cregNames = Self.classicalRegisterNames(count: circuit.classicalRegisters.count)
        for (index, spec) in circuit.classicalRegisters.enumerated() {
            lines.append("creg \(cregNames[index])[\(spec.bitCount)];")
        }

        let ctx = ExportContext(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters,
            cregNames: cregNames
        )
        for gate in circuit.gates {
            try lines.append(contentsOf: ctx.emitGate(gate, conditioned: nil))
        }

        return lines.joined(separator: "\n") + "\n"
    }

    /// `c` for a single classical register; `c0`, `c1`, … when there are several.
    public static func classicalRegisterNames(count: Int) -> [String] {
        guard count > 0 else { return [] }
        if count == 1 { return ["c"] }
        return (0..<count).map { "c\($0)" }
    }
}

/// Alias matching the generic “exporter” naming used in slice docs.
public typealias OpenQASMExporter = OpenQASM2Exporter

// MARK: - Emission context

private struct ExportContext {
    let qubitCount: Int
    let classicalRegisters: [ClassicalRegisterSpec]
    let cregNames: [String]

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

        case .while_c:
            throw unsupported(
                feature: "while_c",
                message: "OpenQASM 2 export does not support while_c (QASM3 slice)"
            )

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
            // qelib1 flavor: Gate.p ↔ u1
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
                message: "OpenQASM 2 export does not support delay"
            )
        case .iswap:
            throw unsupported(
                feature: "iswap",
                message: "OpenQASM 2 / qelib1 export does not support iswap"
            )
        case .ecr:
            throw unsupported(
                feature: "ecr",
                message: "OpenQASM 2 / qelib1 export does not support ecr"
            )
        case .rxx:
            throw unsupported(
                feature: "rxx",
                message: "OpenQASM 2 / qelib1 export does not support rxx"
            )
        case .ryy:
            throw unsupported(
                feature: "ryy",
                message: "OpenQASM 2 / qelib1 export does not support ryy"
            )
        case .rzz:
            throw unsupported(
                feature: "rzz",
                message: "OpenQASM 2 / qelib1 export does not support rzz"
            )
        case .dcx:
            throw unsupported(
                feature: "dcx",
                message: "OpenQASM 2 / qelib1 export does not support dcx"
            )
        case .mcx:
            throw unsupported(
                feature: "mcx",
                message: "OpenQASM 2 / qelib1 export does not support mcx (use ccx when applicable)"
            )
        case .mcz:
            throw unsupported(
                feature: "mcz",
                message: "OpenQASM 2 / qelib1 export does not support mcz"
            )
        case .unitary1:
            throw unsupported(
                feature: "unitary1",
                message: "OpenQASM 2 export does not support unitary1"
            )
        case .initialize:
            throw unsupported(
                feature: "initialize",
                message: "OpenQASM 2 export does not support initialize"
            )
        case .customUnitary:
            throw unsupported(
                feature: "customUnitary",
                message: "OpenQASM 2 export does not support customUnitary"
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

    /// Whole-register measure when qubits cover the full `qreg` and map onto the full creg from bit 0.
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
        return ["if(\(name)==\(conditioned.value)) \(statement)"]
    }

    private func wrapAll(
        _ statements: [String],
        conditioned: (register: Int, value: Int)
    ) throws -> [String] {
        let name = try classicalName(conditioned.register)
        return statements.map { stmt in
            "if(\(name)==\(conditioned.value)) \(stmt)"
        }
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
                message: "Gate '\(gate)' angle is symbolic ('\(parameter.name)'); static OpenQASM 2 export requires literal angles"
            )
        case .negated, .scaled:
            throw unsupported(
                feature: "angle expression",
                message: "Gate '\(gate)' angle is not a literal; static OpenQASM 2 export requires QFloatExpr.literal"
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
