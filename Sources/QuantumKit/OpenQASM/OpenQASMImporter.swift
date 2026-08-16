import Foundation

/// Options for OpenQASM 2 → ``QuantumCircuit`` import.
///
/// Reserved for future knobs (dialect strictness, include search paths, etc.).
public struct OpenQASM2ImporterOptions: Equatable, Sendable {
    public init() {}
}

/// Lowers an OpenQASM 2 program into a ``QuantumCircuit``.
///
/// ## Bit / qubit order
/// QuantumKit uses LSB = qubit 0. An OpenQASM subscript `q[0]` maps directly to
/// engine qubit `0` (same for classical bits). Multiple `qreg` declarations are
/// linearized in declaration order: the first register’s `r[0]` is global qubit 0,
/// then contiguous indices for subsequent registers.
///
/// ## Slice 03 coverage
/// qreg / creg, builtin `include "qelib1.inc"`, measure / reset / barrier, and the
/// parameter-free qelib1 gate map in ``OpenQASMQelib1``. Parametric gates, `if`,
/// `opaque`, user gate expansion, and non-qelib1 includes are rejected with
/// ``OpenQASMError``.
public struct OpenQASM2Importer: Sendable {
    public var options: OpenQASM2ImporterOptions

    public init(options: OpenQASM2ImporterOptions = OpenQASM2ImporterOptions()) {
        self.options = options
    }

    /// Parses `source` and lowers it to a circuit.
    public func `import`(source: String) throws -> QuantumCircuit {
        var parser = try OpenQASMParser(source: source)
        let program = try parser.parse()
        return try `import`(program: program)
    }

    /// Lowers an already-parsed program to a circuit.
    public func `import`(program: OpenQASMProgram) throws -> QuantumCircuit {
        try LoweringContext(program: program).lower()
    }
}

/// Alias matching the generic “importer” naming used in slice docs.
public typealias OpenQASMImporter = OpenQASM2Importer

// MARK: - Lowering

private struct RegisterBinding: Equatable {
    var name: String
    var size: Int
    /// Global base index (qubit or classical-register-local bit base for qubits;
    /// for cregs this is the declaration-order register index).
    var base: Int
    var isClassical: Bool
}

private final class LoweringContext {
    let program: OpenQASMProgram
    var qelib1Available = false
    var qubitRegs: [String: RegisterBinding] = [:]
    var classicalRegs: [String: RegisterBinding] = [:]
    var classicalSpecs: [ClassicalRegisterSpec] = []
    var qubitCount = 0
    /// Names from `gate` declarations (bodies not expanded in this slice).
    var declaredUserGates: Set<String> = []

    init(program: OpenQASMProgram) {
        self.program = program
    }

    func lower() throws -> QuantumCircuit {
        // Pass 1: collect declarations / includes / unsupported top-level forms.
        for statement in program.statements {
            try collectDeclarations(statement)
        }

        guard qubitCount > 0 else {
            throw OpenQASMError.semanticError(
                line: 1,
                column: 1,
                message: "Program must declare at least one qubit (qreg)"
            )
        }

        var circuit = try QuantumCircuit(
            qubitCount: qubitCount,
            classicalRegisters: classicalSpecs
        )

        // Pass 2: lower executable statements.
        for statement in program.statements {
            try applyExecutable(statement, to: &circuit)
        }

        return circuit
    }

    // MARK: Declarations

    private func collectDeclarations(_ statement: OpenQASMStatement) throws {
        switch statement {
        case .version, .empty, .gateCall, .measure, .reset, .barrier:
            return

        case .include(let path, let location):
            if OpenQASMQelib1.isBuiltinInclude(path) {
                qelib1Available = true
            } else {
                throw OpenQASMError.unsupported(
                    line: location.line,
                    column: location.column,
                    feature: "include",
                    message: "Only builtin include \"\(OpenQASMQelib1.includeFileName)\" is supported; got \"\(path)\""
                )
            }

        case .qreg(let name, let size, let location):
            try declareQreg(name: name, size: size, location: location)

        case .creg(let name, let size, let location):
            try declareCreg(name: name, size: size, location: location)

        case .qubitDecl(_, _, let location), .bitDecl(_, _, let location):
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: "OpenQASM 3 declarations",
                message: "OpenQASM 2 importer does not accept qubit/bit declarations yet"
            )

        case .gateDecl(let name, _, _, _, let location):
            if OpenQASMQelib1.mappedGateNames.contains(name) {
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "Cannot redefine builtin gate '\(name)'"
                )
            }
            // Record for later expansion (slice 06); calls still error in this slice.
            declaredUserGates.insert(name)

        case .opaqueDecl(_, _, _, let location):
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: "opaque",
                message: "opaque declarations are not supported yet"
            )

        case .ifStatement(_, _, let location):
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: "if",
                message: "classical if statements are not supported in this import path yet"
            )

        case .whileStatement(_, _, let location):
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: "while",
                message: "while statements are not supported yet"
            )
        }
    }

    private func declareQreg(name: String, size: Int, location: SourceLocation) throws {
        guard size > 0 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "qreg '\(name)' size must be positive"
            )
        }
        guard qubitRegs[name] == nil, classicalRegs[name] == nil else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Register '\(name)' is already declared"
            )
        }
        qubitRegs[name] = RegisterBinding(
            name: name,
            size: size,
            base: qubitCount,
            isClassical: false
        )
        qubitCount += size
    }

    private func declareCreg(name: String, size: Int, location: SourceLocation) throws {
        guard size > 0 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "creg '\(name)' size must be positive"
            )
        }
        guard qubitRegs[name] == nil, classicalRegs[name] == nil else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Register '\(name)' is already declared"
            )
        }
        let index = classicalSpecs.count
        classicalRegs[name] = RegisterBinding(
            name: name,
            size: size,
            base: index,
            isClassical: true
        )
        do {
            classicalSpecs.append(try ClassicalRegisterSpec(bitCount: size))
        } catch {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Invalid creg '\(name)' size \(size)"
            )
        }
    }

    // MARK: Executable statements

    private func applyExecutable(_ statement: OpenQASMStatement, to circuit: inout QuantumCircuit) throws {
        switch statement {
        case .version, .empty, .include, .qreg, .creg, .qubitDecl, .bitDecl, .gateDecl:
            return

        case .opaqueDecl, .ifStatement, .whileStatement:
            // Already rejected in collectDeclarations.
            return

        case .gateCall(let name, let params, let qubits, let location):
            try applyGateCall(name: name, params: params, qubits: qubits, location: location, to: &circuit)

        case .measure(let qubits, let classical, let location):
            try applyMeasure(qubits: qubits, classical: classical, location: location, to: &circuit)

        case .reset(let qubits, let location):
            try applyReset(qubits: qubits, location: location, to: &circuit)

        case .barrier(let qubits, let location):
            try applyBarrier(qubits: qubits, location: location, to: &circuit)
        }
    }

    private func applyGateCall(
        name: String,
        params: [OpenQASMExpr],
        qubits: [OpenQASMArgument],
        location: SourceLocation,
        to circuit: inout QuantumCircuit
    ) throws {
        if !params.isEmpty {
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: name,
                message: "Parametric / angle gates are not supported in this import path yet"
            )
        }

        if declaredUserGates.contains(name) {
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: name,
                message: "User-defined gate '\(name)' is declared but not expanded yet"
            )
        }

        guard OpenQASMQelib1.mappedGateNames.contains(name) else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Unknown or unmapped gate '\(name)'"
            )
        }

        // qelib1 builtins may be used without an explicit include (common in tests);
        // include still marks availability for tooling.
        _ = qelib1Available

        let gate = try mapBuiltinGate(name: name, qubits: qubits, location: location)
        do {
            try circuit.apply(gate)
        } catch {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Failed to apply gate '\(name)': \(error)"
            )
        }
    }

    private func mapBuiltinGate(
        name: String,
        qubits: [OpenQASMArgument],
        location: SourceLocation
    ) throws -> Gate {
        switch name {
        case "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg":
            let target = try requireSingleQubit(qubits, gate: name, location: location)
            switch name {
            case "id": return .id(target: target)
            case "x": return .x(target: target)
            case "y": return .y(target: target)
            case "z": return .z(target: target)
            case "h": return .h(target: target)
            case "s": return .s(target: target)
            case "sdg": return .sdg(target: target)
            case "t": return .t(target: target)
            case "tdg": return .tdg(target: target)
            case "sx": return .sx(target: target)
            case "sxdg": return .sxdg(target: target)
            default:
                preconditionFailure("unreachable single-qubit map")
            }

        case "cx":
            let pair = try requireTwoQubits(qubits, gate: name, location: location)
            return .cx(control: pair.0, target: pair.1)

        case "cz":
            let pair = try requireTwoQubits(qubits, gate: name, location: location)
            return .cz(control: pair.0, target: pair.1)

        case "swap":
            let pair = try requireTwoQubits(qubits, gate: name, location: location)
            return .swap(q1: pair.0, q2: pair.1)

        case "ccx":
            let triple = try requireThreeQubits(qubits, gate: name, location: location)
            return .ccx(control1: triple.0, control2: triple.1, target: triple.2)

        default:
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Unknown or unmapped gate '\(name)'"
            )
        }
    }

    // MARK: Measure / reset / barrier

    private func applyMeasure(
        qubits: [OpenQASMArgument],
        classical: [OpenQASMArgument],
        location: SourceLocation,
        to circuit: inout QuantumCircuit
    ) throws {
        guard qubits.count == 1, classical.count == 1 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "measure expects a single qubit argument and a single classical argument"
            )
        }

        let qArg = qubits[0]
        let cArg = classical[0]
        let qIndices = try resolveQubitIndices(qArg, location: location)
        let (cregIndex, bitOffset, cWidth) = try resolveClassicalTarget(cArg, location: location)

        if qArg.index == nil && cArg.index == nil {
            // Whole-register measure: widths must match; classicalBitOffset = 0.
            guard qIndices.count == cWidth else {
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "measure register widths must match (\(qIndices.count) qubits vs \(cWidth) bits)"
                )
            }
            let spec = MeasureSpec(
                qubits: qIndices,
                classicalRegister: cregIndex,
                classicalBitOffset: 0
            )
            try applyGate(.measure(spec), location: location, to: &circuit)
            return
        }

        if qArg.index != nil && cArg.index != nil {
            guard qIndices.count == 1 else {
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "Indexed measure expects a single qubit"
                )
            }
            let spec = MeasureSpec(
                qubits: qIndices,
                classicalRegister: cregIndex,
                classicalBitOffset: bitOffset
            )
            try applyGate(.measure(spec), location: location, to: &circuit)
            return
        }

        throw OpenQASMError.semanticError(
            line: location.line,
            column: location.column,
            message: "measure must use both whole registers or both indexed bits"
        )
    }

    private func applyReset(
        qubits: [OpenQASMArgument],
        location: SourceLocation,
        to circuit: inout QuantumCircuit
    ) throws {
        var indices: [Int] = []
        for arg in qubits {
            indices.append(contentsOf: try resolveQubitIndices(arg, location: location))
        }
        guard !indices.isEmpty else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "reset requires at least one qubit"
            )
        }
        for index in indices {
            try applyGate(.reset(qubit: index), location: location, to: &circuit)
        }
    }

    private func applyBarrier(
        qubits: [OpenQASMArgument],
        location: SourceLocation,
        to circuit: inout QuantumCircuit
    ) throws {
        if qubits.isEmpty {
            // Empty barrier means all circuit qubits.
            try applyGate(.barrier(qubits: []), location: location, to: &circuit)
            return
        }
        var indices: [Int] = []
        for arg in qubits {
            indices.append(contentsOf: try resolveQubitIndices(arg, location: location))
        }
        try applyGate(.barrier(qubits: indices), location: location, to: &circuit)
    }

    private func applyGate(
        _ gate: Gate,
        location: SourceLocation,
        to circuit: inout QuantumCircuit
    ) throws {
        do {
            try circuit.apply(gate)
        } catch {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Failed to apply instruction: \(error)"
            )
        }
    }

    // MARK: Argument resolution

    /// Resolves a quantum argument to one or more global qubit indices.
    ///
    /// `q[0]` → global index `base + 0` (QuantumKit LSB = qubit 0).
    private func resolveQubitIndices(
        _ arg: OpenQASMArgument,
        location: SourceLocation
    ) throws -> [Int] {
        guard let binding = qubitRegs[arg.name] else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Undeclared quantum register '\(arg.name)'"
            )
        }
        if let index = arg.index {
            guard index >= 0, index < binding.size else {
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "Qubit index \(index) out of range for '\(arg.name)[\(binding.size)]'"
                )
            }
            return [binding.base + index]
        }
        return Array(binding.base..<(binding.base + binding.size))
    }

    /// Resolves a classical argument to (registerIndex, bitOffset, registerWidth).
    private func resolveClassicalTarget(
        _ arg: OpenQASMArgument,
        location: SourceLocation
    ) throws -> (Int, Int, Int) {
        guard let binding = classicalRegs[arg.name] else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Undeclared classical register '\(arg.name)'"
            )
        }
        if let index = arg.index {
            guard index >= 0, index < binding.size else {
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "Classical bit index \(index) out of range for '\(arg.name)[\(binding.size)]'"
                )
            }
            return (binding.base, index, binding.size)
        }
        return (binding.base, 0, binding.size)
    }

    private func requireSingleQubit(
        _ qubits: [OpenQASMArgument],
        gate: String,
        location: SourceLocation
    ) throws -> Int {
        guard qubits.count == 1 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Gate '\(gate)' expects 1 qubit argument, got \(qubits.count)"
            )
        }
        let indices = try resolveQubitIndices(qubits[0], location: location)
        guard indices.count == 1 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Gate '\(gate)' expects a single qubit, not a whole register"
            )
        }
        return indices[0]
    }

    private func requireTwoQubits(
        _ qubits: [OpenQASMArgument],
        gate: String,
        location: SourceLocation
    ) throws -> (Int, Int) {
        guard qubits.count == 2 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Gate '\(gate)' expects 2 qubit arguments, got \(qubits.count)"
            )
        }
        let a = try resolveQubitIndices(qubits[0], location: location)
        let b = try resolveQubitIndices(qubits[1], location: location)
        guard a.count == 1, b.count == 1 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Gate '\(gate)' expects indexed qubits, not whole registers"
            )
        }
        return (a[0], b[0])
    }

    private func requireThreeQubits(
        _ qubits: [OpenQASMArgument],
        gate: String,
        location: SourceLocation
    ) throws -> (Int, Int, Int) {
        guard qubits.count == 3 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Gate '\(gate)' expects 3 qubit arguments, got \(qubits.count)"
            )
        }
        let a = try resolveQubitIndices(qubits[0], location: location)
        let b = try resolveQubitIndices(qubits[1], location: location)
        let c = try resolveQubitIndices(qubits[2], location: location)
        guard a.count == 1, b.count == 1, c.count == 1 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Gate '\(gate)' expects indexed qubits, not whole registers"
            )
        }
        return (a[0], b[0], c[0])
    }
}
