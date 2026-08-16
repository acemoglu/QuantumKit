/// Lexical token kind for OpenQASM 2 / 3 front-ends.
public enum OpenQASMTokenKind: Equatable, Sendable, Hashable {
    // MARK: Keywords
    case keywordOPENQASM
    case keywordInclude
    case keywordQreg
    case keywordCreg
    case keywordQubit
    case keywordBit
    case keywordGate
    case keywordOpaque
    case keywordMeasure
    case keywordReset
    case keywordBarrier
    case keywordIf
    case keywordElse
    case keywordWhile
    case keywordFor
    case keywordIn
    case keywordReturn
    case keywordDef
    case keywordConst
    case keywordInput
    case keywordOutput
    case keywordLet
    case keywordSwitch
    case keywordCase
    case keywordDefault
    case keywordBreak
    case keywordContinue
    case keywordEnd
    case keywordBox
    case keywordDelay
    case keywordCtrl
    case keywordNegctrl
    case keywordInv
    case keywordPow
    case keywordGphase
    case keywordDefcal
    case keywordCal
    case keywordExtern
    case keywordArray
    case keywordDuration
    case keywordStretch
    case keywordAngle
    case keywordComplex
    case keywordBool
    case keywordInt
    case keywordUint
    case keywordFloat
    case keywordDurationof
    case keywordSizeof
    case keywordLengthof

    // MARK: Punctuation / operators
    case semicolon
    case comma
    case leftParen
    case rightParen
    case leftBracket
    case rightBracket
    case leftBrace
    case rightBrace
    case colon
    case equals
    case equalEqual
    case notEqual
    case less
    case lessEqual
    case greater
    case greaterEqual
    case plus
    case minus
    case star
    case slash
    case percent
    case starStar
    case arrow
    case at
    case hash
    case bang
    case tilde
    case ampersand
    case pipe
    case caret
    case leftShift
    case rightShift
    case ampAmp
    case pipePipe

    // MARK: Literals / names
    case identifier
    case integer
    case float
    case string

    /// End of input.
    case eof
}

/// A single OpenQASM token with source location (1-based line/column).
public struct OpenQASMToken: Equatable, Sendable, Hashable {
    public let kind: OpenQASMTokenKind
    public let lexeme: String
    public let line: Int
    public let column: Int

    public init(kind: OpenQASMTokenKind, lexeme: String, line: Int, column: Int) {
        self.kind = kind
        self.lexeme = lexeme
        self.line = line
        self.column = column
    }

    public var location: SourceLocation {
        SourceLocation(line: line, column: column)
    }
}

extension OpenQASMTokenKind {
    /// Maps an identifier lexeme to a keyword kind, or `nil` if not a keyword.
    ///
    /// Keywords are case-sensitive per OpenQASM (`OPENQASM`, `include`, `qreg`, …).
    /// Constants such as `pi` / `tau` / `euler` remain ordinary identifiers.
    public static func keyword(for lexeme: String) -> OpenQASMTokenKind? {
        switch lexeme {
        case "OPENQASM": return .keywordOPENQASM
        case "include": return .keywordInclude
        case "qreg": return .keywordQreg
        case "creg": return .keywordCreg
        case "qubit": return .keywordQubit
        case "bit": return .keywordBit
        case "gate": return .keywordGate
        case "opaque": return .keywordOpaque
        case "measure": return .keywordMeasure
        case "reset": return .keywordReset
        case "barrier": return .keywordBarrier
        case "if": return .keywordIf
        case "else": return .keywordElse
        case "while": return .keywordWhile
        case "for": return .keywordFor
        case "in": return .keywordIn
        case "return": return .keywordReturn
        case "def": return .keywordDef
        case "const": return .keywordConst
        case "input": return .keywordInput
        case "output": return .keywordOutput
        case "let": return .keywordLet
        case "switch": return .keywordSwitch
        case "case": return .keywordCase
        case "default": return .keywordDefault
        case "break": return .keywordBreak
        case "continue": return .keywordContinue
        case "end": return .keywordEnd
        case "box": return .keywordBox
        case "delay": return .keywordDelay
        case "ctrl": return .keywordCtrl
        case "negctrl": return .keywordNegctrl
        case "inv": return .keywordInv
        case "pow": return .keywordPow
        case "gphase": return .keywordGphase
        case "defcal": return .keywordDefcal
        case "cal": return .keywordCal
        case "extern": return .keywordExtern
        case "array": return .keywordArray
        case "duration": return .keywordDuration
        case "stretch": return .keywordStretch
        case "angle": return .keywordAngle
        case "complex": return .keywordComplex
        case "bool": return .keywordBool
        case "int": return .keywordInt
        case "uint": return .keywordUint
        case "float": return .keywordFloat
        case "durationof": return .keywordDurationof
        case "sizeof": return .keywordSizeof
        case "lengthof": return .keywordLengthof
        default: return nil
        }
    }
}
