import Foundation

/// 1-based source position in OpenQASM text.
public struct SourceLocation: Equatable, Sendable, Hashable {
    public let line: Int
    public let column: Int

    public init(line: Int, column: Int) {
        self.line = line
        self.column = column
    }
}

/// Public errors for OpenQASM lexing, parsing, and lowering.
public enum OpenQASMError: Error, Equatable, Sendable {
    /// Lexer could not tokenize at the given location.
    case lexError(line: Int, column: Int, message: String)
    /// Parser rejected syntax at the given location.
    case parseError(line: Int, column: Int, message: String)
    /// Recognized but unsupported OpenQASM feature (version / construct).
    case unsupported(line: Int, column: Int, feature: String, message: String)
    /// Semantically invalid program (undeclared register, type mismatch, etc.).
    case semanticError(line: Int, column: Int, message: String)

    public var location: SourceLocation {
        switch self {
        case .lexError(let line, let column, _),
             .parseError(let line, let column, _),
             .unsupported(let line, let column, _, _),
             .semanticError(let line, let column, _):
            return SourceLocation(line: line, column: column)
        }
    }
}

extension OpenQASMError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .lexError(let line, let column, let message):
            return "OpenQASM lex error at \(line):\(column): \(message)"
        case .parseError(let line, let column, let message):
            return "OpenQASM parse error at \(line):\(column): \(message)"
        case .unsupported(let line, let column, let feature, let message):
            return "OpenQASM unsupported feature '\(feature)' at \(line):\(column): \(message)"
        case .semanticError(let line, let column, let message):
            return "OpenQASM semantic error at \(line):\(column): \(message)"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .lexError:
            return "Fix the invalid character, unterminated string/comment, or malformed number near the reported location."
        case .parseError:
            return "Check OpenQASM syntax (version header, statements, punctuation) around the reported location."
        case .unsupported(_, _, let feature, _):
            return "Remove or rewrite '\(feature)', or use an OpenQASM dialect / version QuantumKit supports."
        case .semanticError:
            return "Ensure registers, gates, and types are declared and used consistently before the reported location."
        }
    }
}
