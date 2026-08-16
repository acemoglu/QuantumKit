import Foundation

/// Options for OpenQASM 2 → ``QuantumCircuit`` import.
///
/// Reserved for future knobs (dialect strictness, include search paths, etc.).
public struct OpenQASM2ImporterOptions: Equatable, Sendable {
    public init() {}
}

/// Options for OpenQASM 3 → ``QuantumCircuit`` import.
public struct OpenQASM3ImporterOptions: Equatable, Sendable {
    public init() {}
}

/// Options for version-dispatching ``OpenQASMImporter``.
public struct OpenQASMImporterOptions: Equatable, Sendable {
    public init() {}
}

/// Lowers an OpenQASM 2 program into a ``QuantumCircuit``.
///
/// ## Bit / qubit order (linear addressing)
/// QuantumKit uses LSB = qubit 0. An OpenQASM subscript `q[0]` maps directly to
/// engine qubit `0` (same for classical bit offsets within a creg).
///
/// Multiple `qreg` declarations are linearized in declaration order into one
/// contiguous global qubit address space:
/// `qreg a[2]; qreg b[3];` → `a[0]=0`, `a[1]=1`, `b[0]=2`, `b[1]=3`, `b[2]=4`.
/// So `cx a[1], b[0]` lowers to `cx(control: 1, target: 2)`.
///
/// Multiple `creg` declarations become separate ``ClassicalRegisterSpec`` entries
/// in declaration order (not bit-linearized across registers):
/// `creg c[2]; creg d[1];` → `classicalRegisters[0].bitCount == 2`,
/// `classicalRegisters[1].bitCount == 1`. Measure targets the creg index plus
/// bit offset; `if (d == 1)` uses `classicalRegister` index `1`.
///
/// ## Coverage
/// qreg / creg, builtin `include "qelib1.inc"`, measure / reset / barrier, the
/// qelib1 gate map in ``OpenQASMQelib1`` including numeric-angle parametric gates
/// (`u`/`u1`/`u2`/`u3`, `p`, `rx`/`ry`/`rz`, `crx`/`cry`/`crz`/`cp`, `cswap`),
/// OpenQASM 2 `if (creg == imm) <stmt>` lowered via ``Gate/c_if``, and
/// user-defined `gate` expand/inline (numeric params, recursive nesting).
/// Angle expressions evaluate `pi` and arithmetic; formal gate parameters are
/// substituted during expansion. `opaque`, `while`, and non-qelib1 includes
/// are rejected with ``OpenQASMError``. OpenQASM 3 `qubit`/`bit` declarations
/// are rejected here (use ``OpenQASM3Importer`` / ``OpenQASMImporter``).
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
        try LoweringContext(program: program, dialect: .v2).lower()
    }
}

/// Lowers an OpenQASM 3 core-subset program into a ``QuantumCircuit``.
///
/// ## Slice 08 coverage
/// `OPENQASM 3` / `3.0`, `qubit` / `bit` (size omitted → 1), plus compatibility
/// `qreg` / `creg`. Optional qelib1-style `include` only; gate calls reuse the
/// same map as OpenQASM 2. `measure` / `reset` / `barrier` / `if` → ``Gate`` /
/// ``Gate/c_if``. User `gate` expand is shared with the v2 path. `while`,
/// `opaque`, `defcal`, and non-builtin includes are rejected.
public struct OpenQASM3Importer: Sendable {
    public var options: OpenQASM3ImporterOptions

    public init(options: OpenQASM3ImporterOptions = OpenQASM3ImporterOptions()) {
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
        try LoweringContext(program: program, dialect: .v3).lower()
    }
}

/// Version-dispatching OpenQASM → ``QuantumCircuit`` importer.
///
/// Detects ``OpenQASMVersion`` from source (or uses ``OpenQASMProgram/version``)
/// and lowers via ``OpenQASM2Importer`` or ``OpenQASM3Importer``.
public struct OpenQASMImporter: Sendable {
    public var options: OpenQASMImporterOptions

    public init(options: OpenQASMImporterOptions = OpenQASMImporterOptions()) {
        self.options = options
    }

    /// Parses `source`, detects version, and lowers to a circuit.
    public func `import`(source: String) throws -> QuantumCircuit {
        let version = try OpenQASMVersion.detect(from: source)
        switch version {
        case .v2:
            return try OpenQASM2Importer().`import`(source: source)
        case .v3:
            return try OpenQASM3Importer().`import`(source: source)
        }
    }

    /// Lowers an already-parsed program using ``OpenQASMProgram/version`` (defaults to `.v2`).
    public func `import`(program: OpenQASMProgram) throws -> QuantumCircuit {
        switch program.version ?? .v2 {
        case .v2:
            return try OpenQASM2Importer().`import`(program: program)
        case .v3:
            return try OpenQASM3Importer().`import`(program: program)
        }
    }
}

// MARK: - Lowering

private enum LoweringDialect: Equatable, Sendable {
    case v2
    case v3
}

private struct RegisterBinding: Equatable {
    var name: String
    var size: Int
    /// Global base index (qubit or classical-register-local bit base for qubits;
    /// for cregs this is the declaration-order register index).
    var base: Int
    var isClassical: Bool
}

/// Stored user-defined OpenQASM gate (`gate name(params) qubits { body }`).
private struct UserGateDefinition: Equatable {
    var name: String
    var params: [String]
    var qubits: [String]
    var body: [OpenQASMStatement]
    var location: SourceLocation
}

private final class LoweringContext {
    let program: OpenQASMProgram
    let dialect: LoweringDialect
    var qelib1Available = false
    var qubitRegs: [String: RegisterBinding] = [:]
    var classicalRegs: [String: RegisterBinding] = [:]
    var classicalSpecs: [ClassicalRegisterSpec] = []
    var qubitCount = 0
    /// User-defined gates collected in the declaration pass (expanded on call).
    var userGates: [String: UserGateDefinition] = [:]
    /// Formal parameter → numeric value while expanding a user gate body.
    var activeParamBindings: [String: Double] = [:]
    /// Gate names currently being expanded (cycle detection).
    var expansionStack: Set<String> = []

    init(program: OpenQASMProgram, dialect: LoweringDialect) {
        self.program = program
        self.dialect = dialect
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
                message: dialect == .v3
                    ? "Program must declare at least one qubit (qubit or qreg)"
                    : "Program must declare at least one qubit (qreg)"
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

        case .qubitDecl(let name, let size, let location):
            guard dialect == .v3 else {
                throw OpenQASMError.unsupported(
                    line: location.line,
                    column: location.column,
                    feature: "OpenQASM 3 declarations",
                    message: "OpenQASM 2 importer does not accept qubit/bit declarations"
                )
            }
            try declareQubit(name: name, size: size, location: location)

        case .bitDecl(let name, let size, let location):
            guard dialect == .v3 else {
                throw OpenQASMError.unsupported(
                    line: location.line,
                    column: location.column,
                    feature: "OpenQASM 3 declarations",
                    message: "OpenQASM 2 importer does not accept qubit/bit declarations"
                )
            }
            try declareBit(name: name, size: size, location: location)

        case .gateDecl(let name, let params, let qubits, let body, let location):
            if OpenQASMQelib1.mappedGateNames.contains(name) {
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "Cannot redefine builtin gate '\(name)'"
                )
            }
            if userGates[name] != nil {
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "Gate '\(name)' is already declared"
                )
            }
            try validateUserGateBody(body, gateName: name)
            userGates[name] = UserGateDefinition(
                name: name,
                params: params,
                qubits: qubits,
                body: body,
                location: location
            )

        case .opaqueDecl(_, _, _, let location):
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: "opaque",
                message: "opaque gates are not supported"
            )

        case .ifStatement:
            // Executable; lowered in pass 2. Nested declarations inside `if` are not
            // OpenQASM 2 style and are rejected when lowering the body.
            return

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
        try declareQubitRegister(
            name: name,
            size: size,
            location: location,
            kindLabel: "qreg"
        )
    }

    /// OpenQASM 3 `qubit[n] q;` / `qubit q;` — `size == nil` means a single qubit.
    private func declareQubit(name: String, size: Int?, location: SourceLocation) throws {
        try declareQubitRegister(
            name: name,
            size: size ?? 1,
            location: location,
            kindLabel: "qubit"
        )
    }

    private func declareQubitRegister(
        name: String,
        size: Int,
        location: SourceLocation,
        kindLabel: String
    ) throws {
        guard size > 0 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "\(kindLabel) '\(name)' size must be positive"
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
        try declareClassicalRegister(
            name: name,
            size: size,
            location: location,
            kindLabel: "creg"
        )
    }

    /// OpenQASM 3 `bit[n] c;` / `bit c;` — `size == nil` means a single bit.
    private func declareBit(name: String, size: Int?, location: SourceLocation) throws {
        try declareClassicalRegister(
            name: name,
            size: size ?? 1,
            location: location,
            kindLabel: "bit"
        )
    }

    private func declareClassicalRegister(
        name: String,
        size: Int,
        location: SourceLocation,
        kindLabel: String
    ) throws {
        guard size > 0 else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "\(kindLabel) '\(name)' size must be positive"
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
                message: "Invalid \(kindLabel) '\(name)' size \(size)"
            )
        }
    }

    // MARK: Executable statements

    private func applyExecutable(_ statement: OpenQASMStatement, to circuit: inout QuantumCircuit) throws {
        switch statement {
        case .version, .empty, .include, .qreg, .creg, .qubitDecl, .bitDecl, .gateDecl:
            return

        case .opaqueDecl, .whileStatement:
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

        case .ifStatement(let condition, let body, let location):
            try applyIfStatement(condition: condition, body: body, location: location, to: &circuit)
        }
    }

    // MARK: Classical if → Gate.c_if

    /// Lowers `if (creg == imm) <statement>` to one or more ``Gate/c_if`` wrappers.
    ///
    /// The condition register name resolves to the declaration-order classical
    /// register index. The body must lower to gate(s); each resulting gate is
    /// wrapped as `.c_if(classicalRegister:idx, expectedValue:imm, gate:)`.
    /// Whole-register `reset` expands to one `c_if` per qubit. Nested `if`
    /// wraps an inner `c_if` gate.
    private func applyIfStatement(
        condition: OpenQASMCondition,
        body: OpenQASMStatement,
        location: SourceLocation,
        to circuit: inout QuantumCircuit
    ) throws {
        let (cregIndex, expectedValue) = try resolveCondition(condition, location: location)
        let bodyGates = try lowerStatementToGates(body, wrappingLocation: location)
        guard !bodyGates.isEmpty else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "if body produced no gates"
            )
        }
        for gate in bodyGates {
            try applyGate(
                .c_if(
                    classicalRegister: cregIndex,
                    expectedValue: expectedValue,
                    gate: gate
                ),
                location: location,
                to: &circuit
            )
        }
    }

    private func resolveCondition(
        _ condition: OpenQASMCondition,
        location: SourceLocation
    ) throws -> (Int, Int) {
        switch condition {
        case .equals(let register, let value):
            guard let binding = classicalRegs[register] else {
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "Undeclared classical register '\(register)' in if condition"
                )
            }
            return (binding.base, value)
        }
    }

    /// Lowers a single statement that may appear as an `if` body into one or more gates
    /// (without applying them to the circuit).
    private func lowerStatementToGates(
        _ statement: OpenQASMStatement,
        wrappingLocation: SourceLocation
    ) throws -> [Gate] {
        switch statement {
        case .gateCall(let name, let params, let qubits, let location):
            return try lowerGateCallToGates(
                name: name,
                params: params,
                qubits: qubits,
                location: location
            )

        case .measure(let qubits, let classical, let location):
            return [try buildMeasureGate(qubits: qubits, classical: classical, location: location)]

        case .reset(let qubits, let location):
            return try buildResetGates(qubits: qubits, location: location)

        case .barrier(let qubits, let location):
            return [try buildBarrierGate(qubits: qubits, location: location)]

        case .ifStatement(let condition, let body, let location):
            // Nested if: lower inner body, wrap each gate, return as gates for outer wrap.
            let (cregIndex, expectedValue) = try resolveCondition(condition, location: location)
            let innerGates = try lowerStatementToGates(body, wrappingLocation: location)
            guard !innerGates.isEmpty else {
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "if body produced no gates"
                )
            }
            return innerGates.map { gate in
                .c_if(
                    classicalRegister: cregIndex,
                    expectedValue: expectedValue,
                    gate: gate
                )
            }

        case .empty:
            return []

        case .version, .include, .qreg, .creg, .qubitDecl, .bitDecl, .gateDecl, .opaqueDecl:
            throw OpenQASMError.unsupported(
                line: wrappingLocation.line,
                column: wrappingLocation.column,
                feature: "if",
                message: "if body must be a gate, measure, reset, barrier, or nested if"
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

    private func applyGateCall(
        name: String,
        params: [OpenQASMExpr],
        qubits: [OpenQASMArgument],
        location: SourceLocation,
        to circuit: inout QuantumCircuit
    ) throws {
        // qelib1 builtins may be used without an explicit include (common in tests);
        // include still marks availability for tooling.
        _ = qelib1Available

        let gates = try lowerGateCallToGates(
            name: name,
            params: params,
            qubits: qubits,
            location: location
        )
        for gate in gates {
            try applyGate(gate, location: location, to: &circuit)
        }
    }

    // MARK: - Gate call lowering / user-gate expand

    /// Lowers a gate call to one or more ``Gate`` values (builtin map or user expand).
    private func lowerGateCallToGates(
        name: String,
        params: [OpenQASMExpr],
        qubits: [OpenQASMArgument],
        location: SourceLocation
    ) throws -> [Gate] {
        if OpenQASMQelib1.mappedGateNames.contains(name) {
            return [
                try mapBuiltinGate(
                    name: name,
                    params: params,
                    qubits: qubits,
                    location: location
                )
            ]
        }
        if userGates[name] != nil {
            return try expandUserGate(
                name: name,
                params: params,
                qubits: qubits,
                location: location
            )
        }
        throw OpenQASMError.semanticError(
            line: location.line,
            column: location.column,
            message: "Unknown or unmapped gate '\(name)'"
        )
    }

    /// Inlines a user-defined gate: bind numeric params and formal qubits, then
    /// recursively expand the body (wrapping under `if` yields one gate per body gate).
    private func expandUserGate(
        name: String,
        params: [OpenQASMExpr],
        qubits: [OpenQASMArgument],
        location: SourceLocation
    ) throws -> [Gate] {
        guard let definition = userGates[name] else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Unknown or unmapped gate '\(name)'"
            )
        }

        if expansionStack.contains(name) {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Recursive expansion of gate '\(name)'"
            )
        }

        guard params.count == definition.params.count else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Gate '\(name)' expects \(definition.params.count) parameter(s), got \(params.count)"
            )
        }
        guard qubits.count == definition.qubits.count else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Gate '\(name)' expects \(definition.qubits.count) qubit argument(s), got \(qubits.count)"
            )
        }

        var paramBindings: [String: Double] = [:]
        for (formal, expr) in zip(definition.params, params) {
            paramBindings[formal] = try evaluateNumeric(expr, location: location)
        }

        var qubitBindings: [String: Int] = [:]
        for (formal, arg) in zip(definition.qubits, qubits) {
            let indices = try resolveQubitIndices(arg, location: location)
            guard indices.count == 1 else {
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "Gate '\(name)' expects indexed qubits, not whole registers"
                )
            }
            qubitBindings[formal] = indices[0]
        }

        expansionStack.insert(name)
        defer { expansionStack.remove(name) }

        let savedParams = activeParamBindings
        activeParamBindings = paramBindings
        defer { activeParamBindings = savedParams }

        return try withTemporaryQubitBindings(qubitBindings) {
            var gates: [Gate] = []
            for statement in definition.body {
                gates.append(contentsOf: try lowerUserGateBodyStatement(
                    statement,
                    gateName: name
                ))
            }
            return gates
        }
    }

    private func lowerUserGateBodyStatement(
        _ statement: OpenQASMStatement,
        gateName: String
    ) throws -> [Gate] {
        switch statement {
        case .gateCall(let name, let params, let qubits, let location):
            return try lowerGateCallToGates(
                name: name,
                params: params,
                qubits: qubits,
                location: location
            )

        case .empty:
            return []

        case .measure(_, _, let location):
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: "measure",
                message: "measure is not allowed inside gate '\(gateName)' body"
            )

        case .reset(_, let location):
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: "reset",
                message: "reset is not allowed inside gate '\(gateName)' body"
            )

        case .barrier(_, let location):
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: "barrier",
                message: "barrier is not allowed inside gate '\(gateName)' body"
            )

        case .ifStatement(_, _, let location):
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: "if",
                message: "if is not allowed inside gate '\(gateName)' body"
            )

        case .whileStatement(_, _, let location):
            throw OpenQASMError.unsupported(
                line: location.line,
                column: location.column,
                feature: "while",
                message: "while is not allowed inside gate '\(gateName)' body"
            )

        case .version, .include, .qreg, .creg, .qubitDecl, .bitDecl, .gateDecl, .opaqueDecl:
            throw OpenQASMError.unsupported(
                line: statementLocation(statement).line,
                column: statementLocation(statement).column,
                feature: "gate body",
                message: "Only unitary gate calls are allowed inside gate '\(gateName)' body"
            )
        }
    }

    private func validateUserGateBody(_ body: [OpenQASMStatement], gateName: String) throws {
        for statement in body {
            switch statement {
            case .gateCall, .empty:
                continue
            case .measure(_, _, let location):
                throw OpenQASMError.unsupported(
                    line: location.line,
                    column: location.column,
                    feature: "measure",
                    message: "measure is not allowed inside gate '\(gateName)' body"
                )
            case .reset(_, let location):
                throw OpenQASMError.unsupported(
                    line: location.line,
                    column: location.column,
                    feature: "reset",
                    message: "reset is not allowed inside gate '\(gateName)' body"
                )
            case .barrier(_, let location):
                throw OpenQASMError.unsupported(
                    line: location.line,
                    column: location.column,
                    feature: "barrier",
                    message: "barrier is not allowed inside gate '\(gateName)' body"
                )
            case .ifStatement(_, _, let location):
                throw OpenQASMError.unsupported(
                    line: location.line,
                    column: location.column,
                    feature: "if",
                    message: "if is not allowed inside gate '\(gateName)' body"
                )
            case .whileStatement(_, _, let location):
                throw OpenQASMError.unsupported(
                    line: location.line,
                    column: location.column,
                    feature: "while",
                    message: "while is not allowed inside gate '\(gateName)' body"
                )
            case .version, .include, .qreg, .creg, .qubitDecl, .bitDecl, .gateDecl, .opaqueDecl:
                let loc = statementLocation(statement)
                throw OpenQASMError.unsupported(
                    line: loc.line,
                    column: loc.column,
                    feature: "gate body",
                    message: "Only unitary gate calls are allowed inside gate '\(gateName)' body"
                )
            }
        }
    }

    /// Temporarily maps formal qubit names to single-wire bindings at bound global indices.
    private func withTemporaryQubitBindings<T>(
        _ bindings: [String: Int],
        _ body: () throws -> T
    ) rethrows -> T {
        var saved: [String: RegisterBinding?] = [:]
        for (name, index) in bindings {
            saved[name] = qubitRegs[name]
            qubitRegs[name] = RegisterBinding(
                name: name,
                size: 1,
                base: index,
                isClassical: false
            )
        }
        defer {
            for (name, previous) in saved {
                if let previous {
                    qubitRegs[name] = previous
                } else {
                    qubitRegs.removeValue(forKey: name)
                }
            }
        }
        return try body()
    }

    private func statementLocation(_ statement: OpenQASMStatement) -> SourceLocation {
        switch statement {
        case .version:
            return SourceLocation(line: 1, column: 1)
        case .include(_, let location),
             .qreg(_, _, let location),
             .creg(_, _, let location),
             .qubitDecl(_, _, let location),
             .bitDecl(_, _, let location),
             .gateDecl(_, _, _, _, let location),
             .opaqueDecl(_, _, _, let location),
             .gateCall(_, _, _, let location),
             .measure(_, _, let location),
             .reset(_, let location),
             .barrier(_, let location),
             .ifStatement(_, _, let location),
             .whileStatement(_, _, let location),
             .empty(let location):
            return location
        }
    }

    private func mapBuiltinGate(
        name: String,
        params: [OpenQASMExpr],
        qubits: [OpenQASMArgument],
        location: SourceLocation
    ) throws -> Gate {
        switch name {
        case "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg":
            try requireParamCount(params, expected: 0, gate: name, location: location)
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
            try requireParamCount(params, expected: 0, gate: name, location: location)
            let pair = try requireTwoQubits(qubits, gate: name, location: location)
            return .cx(control: pair.0, target: pair.1)

        case "cz":
            try requireParamCount(params, expected: 0, gate: name, location: location)
            let pair = try requireTwoQubits(qubits, gate: name, location: location)
            return .cz(control: pair.0, target: pair.1)

        case "swap":
            try requireParamCount(params, expected: 0, gate: name, location: location)
            let pair = try requireTwoQubits(qubits, gate: name, location: location)
            return .swap(q1: pair.0, q2: pair.1)

        case "ccx":
            try requireParamCount(params, expected: 0, gate: name, location: location)
            let triple = try requireThreeQubits(qubits, gate: name, location: location)
            return .ccx(control1: triple.0, control2: triple.1, target: triple.2)

        case "cswap":
            try requireParamCount(params, expected: 0, gate: name, location: location)
            let triple = try requireThreeQubits(qubits, gate: name, location: location)
            return .cswap(control: triple.0, q1: triple.1, q2: triple.2)

        case "u", "u3":
            try requireParamCount(params, expected: 3, gate: name, location: location)
            let target = try requireSingleQubit(qubits, gate: name, location: location)
            let theta = try evaluateAngle(params[0], location: location)
            let phi = try evaluateAngle(params[1], location: location)
            let lambda = try evaluateAngle(params[2], location: location)
            return .u(theta: theta, phi: phi, lambda: lambda, target: target)

        case "u2":
            try requireParamCount(params, expected: 2, gate: name, location: location)
            let target = try requireSingleQubit(qubits, gate: name, location: location)
            let phi = try evaluateAngle(params[0], location: location)
            let lambda = try evaluateAngle(params[1], location: location)
            let halfPi = QFloatExpr.literal(QFloat(Double.pi / 2))
            return .u(theta: halfPi, phi: phi, lambda: lambda, target: target)

        case "u1":
            try requireParamCount(params, expected: 1, gate: name, location: location)
            let target = try requireSingleQubit(qubits, gate: name, location: location)
            let lambda = try evaluateAngle(params[0], location: location)
            return .p(theta: lambda, target: target)

        case "p":
            try requireParamCount(params, expected: 1, gate: name, location: location)
            let target = try requireSingleQubit(qubits, gate: name, location: location)
            let theta = try evaluateAngle(params[0], location: location)
            return .p(theta: theta, target: target)

        case "rx":
            try requireParamCount(params, expected: 1, gate: name, location: location)
            let target = try requireSingleQubit(qubits, gate: name, location: location)
            let theta = try evaluateAngle(params[0], location: location)
            return .rx(theta: theta, target: target)

        case "ry":
            try requireParamCount(params, expected: 1, gate: name, location: location)
            let target = try requireSingleQubit(qubits, gate: name, location: location)
            let theta = try evaluateAngle(params[0], location: location)
            return .ry(theta: theta, target: target)

        case "rz":
            try requireParamCount(params, expected: 1, gate: name, location: location)
            let target = try requireSingleQubit(qubits, gate: name, location: location)
            let theta = try evaluateAngle(params[0], location: location)
            return .rz(theta: theta, target: target)

        case "crx":
            try requireParamCount(params, expected: 1, gate: name, location: location)
            let pair = try requireTwoQubits(qubits, gate: name, location: location)
            let theta = try evaluateAngle(params[0], location: location)
            return .crx(theta: theta, control: pair.0, target: pair.1)

        case "cry":
            try requireParamCount(params, expected: 1, gate: name, location: location)
            let pair = try requireTwoQubits(qubits, gate: name, location: location)
            let theta = try evaluateAngle(params[0], location: location)
            return .cry(theta: theta, control: pair.0, target: pair.1)

        case "crz":
            try requireParamCount(params, expected: 1, gate: name, location: location)
            let pair = try requireTwoQubits(qubits, gate: name, location: location)
            let theta = try evaluateAngle(params[0], location: location)
            return .crz(theta: theta, control: pair.0, target: pair.1)

        case "cp":
            try requireParamCount(params, expected: 1, gate: name, location: location)
            let pair = try requireTwoQubits(qubits, gate: name, location: location)
            let theta = try evaluateAngle(params[0], location: location)
            return .cp(theta: theta, control: pair.0, target: pair.1)

        default:
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Unknown or unmapped gate '\(name)'"
            )
        }
    }

    // MARK: Angle expression evaluation

    /// Evaluates a numeric (non-parameterized) angle expression to ``QFloatExpr/literal``.
    private func evaluateAngle(_ expr: OpenQASMExpr, location: SourceLocation) throws -> QFloatExpr {
        let value = try evaluateNumeric(expr, location: location)
        return .literal(QFloat(value))
    }

    private func evaluateNumeric(_ expr: OpenQASMExpr, location: SourceLocation) throws -> Double {
        switch expr {
        case .integer(let value):
            return Double(value)
        case .float(let value):
            return value
        case .identifier(let name):
            if let bound = activeParamBindings[name] {
                return bound
            }
            switch name {
            case "pi", "π":
                return Double.pi
            case "tau":
                return 2 * Double.pi
            case "euler", "e":
                return Foundation.exp(1.0)
            default:
                throw OpenQASMError.semanticError(
                    line: location.line,
                    column: location.column,
                    message: "Unknown identifier '\(name)' in angle expression"
                )
            }
        case .unaryMinus(let inner):
            return -(try evaluateNumeric(inner, location: location))
        case .paren(let inner):
            return try evaluateNumeric(inner, location: location)
        case .binary(let op, let lhs, let rhs):
            let left = try evaluateNumeric(lhs, location: location)
            let right = try evaluateNumeric(rhs, location: location)
            switch op {
            case .add:
                return left + right
            case .subtract:
                return left - right
            case .multiply:
                return left * right
            case .divide:
                guard right != 0 else {
                    throw OpenQASMError.semanticError(
                        line: location.line,
                        column: location.column,
                        message: "Division by zero in angle expression"
                    )
                }
                return left / right
            }
        }
    }

    private func requireParamCount(
        _ params: [OpenQASMExpr],
        expected: Int,
        gate: String,
        location: SourceLocation
    ) throws {
        guard params.count == expected else {
            throw OpenQASMError.semanticError(
                line: location.line,
                column: location.column,
                message: "Gate '\(gate)' expects \(expected) parameter(s), got \(params.count)"
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
        try applyGate(
            try buildMeasureGate(qubits: qubits, classical: classical, location: location),
            location: location,
            to: &circuit
        )
    }

    private func buildMeasureGate(
        qubits: [OpenQASMArgument],
        classical: [OpenQASMArgument],
        location: SourceLocation
    ) throws -> Gate {
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
            return .measure(spec)
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
            return .measure(spec)
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
        for gate in try buildResetGates(qubits: qubits, location: location) {
            try applyGate(gate, location: location, to: &circuit)
        }
    }

    private func buildResetGates(
        qubits: [OpenQASMArgument],
        location: SourceLocation
    ) throws -> [Gate] {
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
        return indices.map { .reset(qubit: $0) }
    }

    private func applyBarrier(
        qubits: [OpenQASMArgument],
        location: SourceLocation,
        to circuit: inout QuantumCircuit
    ) throws {
        try applyGate(
            try buildBarrierGate(qubits: qubits, location: location),
            location: location,
            to: &circuit
        )
    }

    private func buildBarrierGate(
        qubits: [OpenQASMArgument],
        location: SourceLocation
    ) throws -> Gate {
        if qubits.isEmpty {
            // Empty barrier means all circuit qubits.
            return .barrier(qubits: [])
        }
        var indices: [Int] = []
        for arg in qubits {
            indices.append(contentsOf: try resolveQubitIndices(arg, location: location))
        }
        return .barrier(qubits: indices)
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
