/// OpenQASM abstract syntax tree for QuantumKit front-ends.
///
/// Indexing note: QuantumKit uses LSB = qubit 0. An OpenQASM subscript `q[0]`
/// maps directly to engine qubit 0 (same for classical bits).

// MARK: - Program

/// A parsed OpenQASM program (statements plus optional detected version).
public struct OpenQASMProgram: Equatable, Sendable {
    /// Language version from an `OPENQASM` header, if present.
    public var version: OpenQASMVersion?
    /// Top-level statements in source order.
    public var statements: [OpenQASMStatement]

    public init(version: OpenQASMVersion? = nil, statements: [OpenQASMStatement] = []) {
        self.version = version
        self.statements = statements
    }
}

// MARK: - Statements

/// A single OpenQASM statement or declaration.
public indirect enum OpenQASMStatement: Equatable, Sendable {
    /// `OPENQASM 2.0;` / `OPENQASM 3;` header when present as a statement.
    case version(OpenQASMVersion)
    /// `include "path";`
    case include(path: String, location: SourceLocation)
    /// OpenQASM 2 `qreg name[size];`
    case qreg(name: String, size: Int, location: SourceLocation)
    /// OpenQASM 2 `creg name[size];`
    case creg(name: String, size: Int, location: SourceLocation)
    /// OpenQASM 3 `qubit name;` or `qubit[size] name;`
    case qubitDecl(name: String, size: Int?, location: SourceLocation)
    /// OpenQASM 3 `bit name;` or `bit[size] name;`
    case bitDecl(name: String, size: Int?, location: SourceLocation)
    /// `gate name(params) qubits { body }`
    case gateDecl(
        name: String,
        params: [String],
        qubits: [String],
        body: [OpenQASMStatement],
        location: SourceLocation
    )
    /// `opaque name(params) qubits;` — parsed; lowering may reject later.
    case opaqueDecl(
        name: String,
        params: [String],
        qubits: [String],
        location: SourceLocation
    )
    /// Gate / unitary application, e.g. `h q[0];` or `u3(pi/2,0,pi) q[0];`
    case gateCall(
        name: String,
        params: [OpenQASMExpr],
        qubits: [OpenQASMArgument],
        location: SourceLocation
    )
    /// `measure qubits -> classical;`
    case measure(
        qubits: [OpenQASMArgument],
        classical: [OpenQASMArgument],
        location: SourceLocation
    )
    /// `reset qubits;`
    case reset(qubits: [OpenQASMArgument], location: SourceLocation)
    /// `barrier;` or `barrier qubits;` — empty qubit list means all qubits.
    case barrier(qubits: [OpenQASMArgument], location: SourceLocation)
    /// `if (c == imm) <statement>;` or `if (c == imm) { … }` (one or more body statements).
    case ifStatement(
        condition: OpenQASMCondition,
        body: [OpenQASMStatement],
        location: SourceLocation
    )
    /// OpenQASM 3 `while (cond) { ... }`
    case whileStatement(
        condition: OpenQASMCondition,
        body: [OpenQASMStatement],
        location: SourceLocation
    )
    /// Lone `;`
    case empty(location: SourceLocation)
}

// MARK: - Arguments / conditions / expressions

/// A register reference with optional integer index (`q` or `q[0]`).
public struct OpenQASMArgument: Equatable, Sendable, Hashable {
    public var name: String
    /// When `nil`, the whole register; when set, a single wire/bit.
    public var index: Int?

    public init(name: String, index: Int? = nil) {
        self.name = name
        self.index = index
    }
}

/// Condition used by `if` / `while`.
///
/// OpenQASM 2 uses `creg == integer`. Additional forms may be added later.
public enum OpenQASMCondition: Equatable, Sendable {
    /// `classicalRegister == immediate`
    case equals(register: String, value: Int)
}

/// Angle / parameter expression tree.
public indirect enum OpenQASMExpr: Equatable, Sendable {
    case integer(Int)
    case float(Double)
    /// Identifier such as `pi`, `tau`, or a gate parameter name.
    case identifier(String)
    case unaryMinus(OpenQASMExpr)
    case paren(OpenQASMExpr)
    case binary(OpenQASMBinaryOp, OpenQASMExpr, OpenQASMExpr)
}

/// Binary operators for angle expressions.
public enum OpenQASMBinaryOp: Equatable, Sendable, Hashable {
    case add
    case subtract
    case multiply
    case divide
}
