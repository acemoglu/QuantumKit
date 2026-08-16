/// Hand-written OpenQASM 2 / 3 lexer.
///
/// Skips whitespace and comments (`//`, `/* */`). Emits keywords, punctuation,
/// identifiers, integers, floats, strings, and EOF. Does not reject QASM3-only
/// constructs — the parser / lowerer decide support later.
public struct OpenQASMLexer: Sendable {
    private let source: String
    private var index: String.Index
    private var line: Int
    private var column: Int

    public init(source: String) {
        self.source = source
        self.index = source.startIndex
        self.line = 1
        self.column = 1
    }

    /// Tokenizes the entire source, including a trailing `.eof` token.
    public mutating func tokenize() throws -> [OpenQASMToken] {
        var tokens: [OpenQASMToken] = []
        while true {
            let token = try nextToken()
            tokens.append(token)
            if token.kind == .eof {
                break
            }
        }
        return tokens
    }

    /// Returns the next token, advancing the lexer.
    public mutating func nextToken() throws -> OpenQASMToken {
        try skipWhitespaceAndComments()

        let startLine = line
        let startColumn = column

        guard let ch = peek() else {
            return OpenQASMToken(kind: .eof, lexeme: "", line: startLine, column: startColumn)
        }

        // Identifier / keyword
        if isIdentStart(ch) {
            return scanIdentifierOrKeyword(startLine: startLine, startColumn: startColumn)
        }

        // Number: digits, or leading '.' followed by digit
        if ch.isASCII && ch.isNumber {
            return try scanNumber(startLine: startLine, startColumn: startColumn)
        }
        if ch == "." {
            let next = peek(offset: 1)
            if let next, next.isASCII && next.isNumber {
                return try scanNumber(startLine: startLine, startColumn: startColumn)
            }
        }

        // String
        if ch == "\"" {
            return try scanString(startLine: startLine, startColumn: startColumn)
        }

        // Multi-character / single-character punctuation
        return try scanPunctuation(startLine: startLine, startColumn: startColumn)
    }

    // MARK: - Scanning

    private mutating func scanIdentifierOrKeyword(startLine: Int, startColumn: Int) -> OpenQASMToken {
        var lexeme = ""
        while let ch = peek(), isIdentContinue(ch) {
            lexeme.append(ch)
            advance()
        }
        let kind = OpenQASMTokenKind.keyword(for: lexeme) ?? .identifier
        return OpenQASMToken(kind: kind, lexeme: lexeme, line: startLine, column: startColumn)
    }

    private mutating func scanNumber(startLine: Int, startColumn: Int) throws -> OpenQASMToken {
        var lexeme = ""
        var isFloat = false

        // Leading digits (optional if we started with '.')
        if peek() == "." {
            isFloat = true
            lexeme.append(".")
            advance()
            while let ch = peek(), ch.isASCII && ch.isNumber {
                lexeme.append(ch)
                advance()
            }
        } else {
            while let ch = peek(), ch.isASCII && ch.isNumber {
                lexeme.append(ch)
                advance()
            }
            if peek() == "." {
                // Float when `.` is followed by a digit, or a bare trailing `.` (not `1.ident`).
                let afterDot = peek(offset: 1)
                if let afterDot, afterDot.isASCII && afterDot.isNumber {
                    isFloat = true
                    lexeme.append(".")
                    advance()
                    while let ch = peek(), ch.isASCII && ch.isNumber {
                        lexeme.append(ch)
                        advance()
                    }
                } else if afterDot == nil || (afterDot.map { !isIdentStart($0) && $0 != "." } ?? false) {
                    isFloat = true
                    lexeme.append(".")
                    advance()
                }
            }
        }

        // Exponent: [eE][+-]?digits
        if let e = peek(), e == "e" || e == "E" {
            let signOrDigit = peek(offset: 1)
            let digitOffset: Int
            if let signOrDigit, signOrDigit == "+" || signOrDigit == "-" {
                digitOffset = 2
            } else {
                digitOffset = 1
            }
            if let dig = peek(offset: digitOffset), dig.isASCII && dig.isNumber {
                isFloat = true
                lexeme.append(e)
                advance()
                if let sign = peek(), sign == "+" || sign == "-" {
                    lexeme.append(sign)
                    advance()
                }
                while let ch = peek(), ch.isASCII && ch.isNumber {
                    lexeme.append(ch)
                    advance()
                }
            }
        }

        let kind: OpenQASMTokenKind = isFloat ? .float : .integer
        return OpenQASMToken(kind: kind, lexeme: lexeme, line: startLine, column: startColumn)
    }

    private mutating func scanString(startLine: Int, startColumn: Int) throws -> OpenQASMToken {
        // Opening quote; lexeme keeps quotes and raw escape text.
        advance()
        var raw = "\""

        while let ch = peek() {
            if ch == "\"" {
                raw.append(ch)
                advance()
                return OpenQASMToken(kind: .string, lexeme: raw, line: startLine, column: startColumn)
            }
            if ch == "\n" {
                throw OpenQASMError.lexError(
                    line: startLine,
                    column: startColumn,
                    message: "Unterminated string literal"
                )
            }
            if ch == "\\" {
                raw.append(ch)
                advance()
                guard let esc = peek() else {
                    throw OpenQASMError.lexError(
                        line: startLine,
                        column: startColumn,
                        message: "Unterminated string escape"
                    )
                }
                // Basic escapes: keep escape text in lexeme; parser may decode later.
                raw.append(esc)
                advance()
                continue
            }
            raw.append(ch)
            advance()
        }

        throw OpenQASMError.lexError(
            line: startLine,
            column: startColumn,
            message: "Unterminated string literal"
        )
    }

    private mutating func scanPunctuation(startLine: Int, startColumn: Int) throws -> OpenQASMToken {
        guard let ch = peek() else {
            return OpenQASMToken(kind: .eof, lexeme: "", line: startLine, column: startColumn)
        }

        // Two / three character operators first
        if ch == "-", peek(offset: 1) == ">" {
            advance(); advance()
            return OpenQASMToken(kind: .arrow, lexeme: "->", line: startLine, column: startColumn)
        }
        if ch == "=", peek(offset: 1) == "=" {
            advance(); advance()
            return OpenQASMToken(kind: .equalEqual, lexeme: "==", line: startLine, column: startColumn)
        }
        if ch == "!", peek(offset: 1) == "=" {
            advance(); advance()
            return OpenQASMToken(kind: .notEqual, lexeme: "!=", line: startLine, column: startColumn)
        }
        if ch == "<", peek(offset: 1) == "=" {
            advance(); advance()
            return OpenQASMToken(kind: .lessEqual, lexeme: "<=", line: startLine, column: startColumn)
        }
        if ch == "<", peek(offset: 1) == "<" {
            advance(); advance()
            return OpenQASMToken(kind: .leftShift, lexeme: "<<", line: startLine, column: startColumn)
        }
        if ch == ">", peek(offset: 1) == "=" {
            advance(); advance()
            return OpenQASMToken(kind: .greaterEqual, lexeme: ">=", line: startLine, column: startColumn)
        }
        if ch == ">", peek(offset: 1) == ">" {
            advance(); advance()
            return OpenQASMToken(kind: .rightShift, lexeme: ">>", line: startLine, column: startColumn)
        }
        if ch == "*", peek(offset: 1) == "*" {
            advance(); advance()
            return OpenQASMToken(kind: .starStar, lexeme: "**", line: startLine, column: startColumn)
        }
        if ch == "&", peek(offset: 1) == "&" {
            advance(); advance()
            return OpenQASMToken(kind: .ampAmp, lexeme: "&&", line: startLine, column: startColumn)
        }
        if ch == "|", peek(offset: 1) == "|" {
            advance(); advance()
            return OpenQASMToken(kind: .pipePipe, lexeme: "||", line: startLine, column: startColumn)
        }

        let kind: OpenQASMTokenKind
        switch ch {
        case ";": kind = .semicolon
        case ",": kind = .comma
        case "(": kind = .leftParen
        case ")": kind = .rightParen
        case "[": kind = .leftBracket
        case "]": kind = .rightBracket
        case "{": kind = .leftBrace
        case "}": kind = .rightBrace
        case ":": kind = .colon
        case "=": kind = .equals
        case "<": kind = .less
        case ">": kind = .greater
        case "+": kind = .plus
        case "-": kind = .minus
        case "*": kind = .star
        case "/": kind = .slash
        case "%": kind = .percent
        case "@": kind = .at
        case "#": kind = .hash
        case "!": kind = .bang
        case "~": kind = .tilde
        case "&": kind = .ampersand
        case "|": kind = .pipe
        case "^": kind = .caret
        case ".":
            // Lone '.' that is not part of a float (handled earlier) — treat as lex error.
            throw OpenQASMError.lexError(
                line: startLine,
                column: startColumn,
                message: "Unexpected character '.'"
            )
        default:
            throw OpenQASMError.lexError(
                line: startLine,
                column: startColumn,
                message: "Unexpected character '\(ch)'"
            )
        }

        advance()
        return OpenQASMToken(kind: kind, lexeme: String(ch), line: startLine, column: startColumn)
    }

    // MARK: - Whitespace / comments

    private mutating func skipWhitespaceAndComments() throws {
        while let ch = peek() {
            if ch == " " || ch == "\t" || ch == "\r" || ch == "\n" {
                advance()
                continue
            }
            if ch == "/" {
                if peek(offset: 1) == "/" {
                    advance(); advance()
                    while let c = peek(), c != "\n" {
                        advance()
                    }
                    continue
                }
                if peek(offset: 1) == "*" {
                    let startLine = line
                    let startColumn = column
                    advance(); advance()
                    var closed = false
                    while peek() != nil {
                        if peek() == "*", peek(offset: 1) == "/" {
                            advance(); advance()
                            closed = true
                            break
                        }
                        advance()
                    }
                    if !closed {
                        throw OpenQASMError.lexError(
                            line: startLine,
                            column: startColumn,
                            message: "Unterminated block comment"
                        )
                    }
                    continue
                }
            }
            break
        }
    }

    // MARK: - Cursor

    private func peek(offset: Int = 0) -> Character? {
        var i = index
        var remaining = offset
        while remaining > 0 {
            guard i < source.endIndex else { return nil }
            i = source.index(after: i)
            remaining -= 1
        }
        guard i < source.endIndex else { return nil }
        return source[i]
    }

    private mutating func advance() {
        guard let ch = peek() else { return }
        if ch == "\n" {
            line += 1
            column = 1
        } else {
            column += 1
        }
        index = source.index(after: index)
    }
}

private func isIdentStart(_ ch: Character) -> Bool {
    ch.isASCII && (ch.isLetter || ch == "_")
}

private func isIdentContinue(_ ch: Character) -> Bool {
    ch.isASCII && (ch.isLetter || ch.isNumber || ch == "_")
}
