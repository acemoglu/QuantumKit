/// Recursive-descent OpenQASM parser producing `OpenQASMProgram` ASTs.
///
/// Targets OpenQASM 2-style circuits first (qreg/creg, gates, measure, if).
/// Recognized but unsupported constructs (defcal, cal, pulse-oriented keywords,
/// etc.) throw `OpenQASMError.unsupported` with a source location.
public struct OpenQASMParser: Sendable {
    private let tokens: [OpenQASMToken]
    private var current: Int = 0

    /// Parses a pre-tokenized stream (including a trailing `.eof`).
    public init(tokens: [OpenQASMToken]) {
        self.tokens = tokens
    }

    /// Lexes `source` then parses.
    public init(source: String) throws {
        var lexer = OpenQASMLexer(source: source)
        self.tokens = try lexer.tokenize()
    }

    /// Parses tokens into an `OpenQASMProgram`.
    public mutating func parse() throws -> OpenQASMProgram {
        current = 0
        var statements: [OpenQASMStatement] = []
        var version: OpenQASMVersion?

        while !isAtEnd {
            if check(.semicolon) {
                let loc = peek().location
                advance()
                statements.append(.empty(location: loc))
                continue
            }
            let stmt = try parseStatement()
            if case .version(let v) = stmt {
                version = v
            }
            statements.append(stmt)
        }

        return OpenQASMProgram(version: version, statements: statements)
    }

    // MARK: - Statements

    private mutating func parseStatement() throws -> OpenQASMStatement {
        let token = peek()
        switch token.kind {
        case .keywordOPENQASM:
            return try parseVersionStatement()
        case .keywordInclude:
            return try parseInclude()
        case .keywordQreg:
            return try parseQreg()
        case .keywordCreg:
            return try parseCreg()
        case .keywordQubit:
            return try parseQubitDecl()
        case .keywordBit:
            return try parseBitDecl()
        case .keywordGate:
            return try parseGateDecl()
        case .keywordOpaque:
            return try parseOpaqueDecl()
        case .keywordMeasure:
            return try parseMeasure()
        case .keywordReset:
            return try parseReset()
        case .keywordBarrier:
            return try parseBarrier()
        case .keywordIf:
            return try parseIf()
        case .keywordWhile:
            return try parseWhile()
        case .identifier:
            return try parseGateCall()
        case .keywordDefcal, .keywordCal, .keywordDelay, .keywordBox,
             .keywordExtern, .keywordDef, .keywordSwitch, .keywordFor,
             .keywordInput, .keywordOutput, .keywordLet, .keywordConst,
             .keywordArray, .keywordDuration, .keywordStretch, .keywordGphase:
            throw OpenQASMError.unsupported(
                line: token.line,
                column: token.column,
                feature: token.lexeme,
                message: "'\(token.lexeme)' is not supported by QuantumKit's OpenQASM front-end"
            )
        case .eof:
            throw parseError(token, "Unexpected end of input")
        default:
            throw parseError(token, "Unexpected token '\(token.lexeme)'")
        }
    }

    /// Statement allowed inside a gate body (no nested decls / measure / if).
    private mutating func parseGateBodyStatement() throws -> OpenQASMStatement {
        let token = peek()
        switch token.kind {
        case .keywordBarrier:
            return try parseBarrier()
        case .keywordReset:
            return try parseReset()
        case .identifier:
            return try parseGateCall()
        case .semicolon:
            let loc = peek().location
            advance()
            return .empty(location: loc)
        default:
            throw parseError(token, "Unexpected token '\(token.lexeme)' in gate body")
        }
    }

    // MARK: - Declarations

    private mutating func parseVersionStatement() throws -> OpenQASMStatement {
        let start = advance() // OPENQASM
        let versionToken = peek()

        let version: OpenQASMVersion
        switch versionToken.kind {
        case .float:
            let text = versionToken.lexeme
            advance()
            if text.hasPrefix("2") {
                version = .v2
            } else if text.hasPrefix("3") {
                version = .v3
            } else {
                throw OpenQASMError.unsupported(
                    line: versionToken.line,
                    column: versionToken.column,
                    feature: "OPENQASM \(text)",
                    message: "Unknown OpenQASM version \(text); QuantumKit supports 2 and 3"
                )
            }
        case .integer:
            let major = try intValue(versionToken)
            advance()
            switch major {
            case 2: version = .v2
            case 3: version = .v3
            default:
                throw OpenQASMError.unsupported(
                    line: versionToken.line,
                    column: versionToken.column,
                    feature: "OPENQASM \(major)",
                    message: "Unknown OpenQASM major version \(major); QuantumKit supports 2 and 3"
                )
            }
        default:
            throw parseError(versionToken, "Expected version number after OPENQASM")
        }

        try expectSemicolon(after: start)
        return .version(version)
    }

    private mutating func parseInclude() throws -> OpenQASMStatement {
        let start = advance() // include
        let pathToken = try expect(.string, message: "Expected string path after include")
        let path = stripStringLiteral(pathToken.lexeme)
        try expectSemicolon(after: start)
        return .include(path: path, location: start.location)
    }

    private mutating func parseQreg() throws -> OpenQASMStatement {
        let start = advance() // qreg
        let name = try expectIdentifier(message: "Expected register name after qreg")
        try expect(.leftBracket, message: "Expected '[' after qreg name")
        let sizeTok = try expect(.integer, message: "Expected integer size in qreg declaration")
        let size = try intValue(sizeTok)
        try expect(.rightBracket, message: "Expected ']' after qreg size")
        try expectSemicolon(after: start)
        return .qreg(name: name.lexeme, size: size, location: start.location)
    }

    private mutating func parseCreg() throws -> OpenQASMStatement {
        let start = advance() // creg
        let name = try expectIdentifier(message: "Expected register name after creg")
        try expect(.leftBracket, message: "Expected '[' after creg name")
        let sizeTok = try expect(.integer, message: "Expected integer size in creg declaration")
        let size = try intValue(sizeTok)
        try expect(.rightBracket, message: "Expected ']' after creg size")
        try expectSemicolon(after: start)
        return .creg(name: name.lexeme, size: size, location: start.location)
    }

    private mutating func parseQubitDecl() throws -> OpenQASMStatement {
        let start = advance() // qubit
        var size: Int?
        if match(.leftBracket) {
            let sizeTok = try expect(.integer, message: "Expected integer size in qubit declaration")
            size = try intValue(sizeTok)
            try expect(.rightBracket, message: "Expected ']' after qubit size")
        }
        let name = try expectIdentifier(message: "Expected name in qubit declaration")
        try expectSemicolon(after: start)
        return .qubitDecl(name: name.lexeme, size: size, location: start.location)
    }

    private mutating func parseBitDecl() throws -> OpenQASMStatement {
        let start = advance() // bit
        var size: Int?
        if match(.leftBracket) {
            let sizeTok = try expect(.integer, message: "Expected integer size in bit declaration")
            size = try intValue(sizeTok)
            try expect(.rightBracket, message: "Expected ']' after bit size")
        }
        let name = try expectIdentifier(message: "Expected name in bit declaration")
        try expectSemicolon(after: start)
        return .bitDecl(name: name.lexeme, size: size, location: start.location)
    }

    private mutating func parseGateDecl() throws -> OpenQASMStatement {
        let start = advance() // gate
        let name = try expectIdentifier(message: "Expected gate name")
        var params: [String] = []
        if match(.leftParen) {
            if !check(.rightParen) {
                repeat {
                    let p = try expectIdentifier(message: "Expected parameter name")
                    params.append(p.lexeme)
                } while match(.comma)
            }
            try expect(.rightParen, message: "Expected ')' after gate parameters")
        }
        var qubits: [String] = []
        repeat {
            let q = try expectIdentifier(message: "Expected qubit argument in gate declaration")
            qubits.append(q.lexeme)
        } while match(.comma)

        try expect(.leftBrace, message: "Expected '{' to start gate body")
        var body: [OpenQASMStatement] = []
        while !check(.rightBrace) && !isAtEnd {
            body.append(try parseGateBodyStatement())
        }
        try expect(.rightBrace, message: "Expected '}' to close gate body")
        return .gateDecl(
            name: name.lexeme,
            params: params,
            qubits: qubits,
            body: body,
            location: start.location
        )
    }

    private mutating func parseOpaqueDecl() throws -> OpenQASMStatement {
        let start = advance() // opaque
        let name = try expectIdentifier(message: "Expected opaque gate name")
        var params: [String] = []
        if match(.leftParen) {
            if !check(.rightParen) {
                repeat {
                    let p = try expectIdentifier(message: "Expected parameter name")
                    params.append(p.lexeme)
                } while match(.comma)
            }
            try expect(.rightParen, message: "Expected ')' after opaque parameters")
        }
        var qubits: [String] = []
        repeat {
            let q = try expectIdentifier(message: "Expected qubit argument in opaque declaration")
            qubits.append(q.lexeme)
        } while match(.comma)
        try expectSemicolon(after: start)
        return .opaqueDecl(
            name: name.lexeme,
            params: params,
            qubits: qubits,
            location: start.location
        )
    }

    // MARK: - Executable statements

    private mutating func parseGateCall() throws -> OpenQASMStatement {
        let nameTok = try expectIdentifier(message: "Expected gate name")
        var params: [OpenQASMExpr] = []
        if match(.leftParen) {
            if !check(.rightParen) {
                repeat {
                    params.append(try parseExpression())
                } while match(.comma)
            }
            try expect(.rightParen, message: "Expected ')' after gate parameters")
        }

        var qubits: [OpenQASMArgument] = []
        // At least one qubit argument required for a gate call.
        qubits.append(try parseArgument())
        while match(.comma) {
            qubits.append(try parseArgument())
        }
        try expect(.semicolon, message: "Expected ';' after gate call")
        return .gateCall(
            name: nameTok.lexeme,
            params: params,
            qubits: qubits,
            location: nameTok.location
        )
    }

    private mutating func parseMeasure() throws -> OpenQASMStatement {
        let start = advance() // measure
        let qubits = try parseArgumentList()
        try expect(.arrow, message: "Expected '->' in measure statement")
        let classical = try parseArgumentList()
        try expectSemicolon(after: start)
        return .measure(qubits: qubits, classical: classical, location: start.location)
    }

    private mutating func parseReset() throws -> OpenQASMStatement {
        let start = advance() // reset
        let qubits = try parseArgumentList()
        try expectSemicolon(after: start)
        return .reset(qubits: qubits, location: start.location)
    }

    private mutating func parseBarrier() throws -> OpenQASMStatement {
        let start = advance() // barrier
        var qubits: [OpenQASMArgument] = []
        if !check(.semicolon) {
            qubits = try parseArgumentList()
        }
        try expectSemicolon(after: start)
        return .barrier(qubits: qubits, location: start.location)
    }

    private mutating func parseIf() throws -> OpenQASMStatement {
        let start = advance() // if
        try expect(.leftParen, message: "Expected '(' after if")
        let condition = try parseCondition()
        try expect(.rightParen, message: "Expected ')' after if condition")
        let body = try parseStatement()
        return .ifStatement(condition: condition, body: body, location: start.location)
    }

    private mutating func parseWhile() throws -> OpenQASMStatement {
        let start = advance() // while
        try expect(.leftParen, message: "Expected '(' after while")
        let condition = try parseCondition()
        try expect(.rightParen, message: "Expected ')' after while condition")
        try expect(.leftBrace, message: "Expected '{' after while condition")
        var body: [OpenQASMStatement] = []
        while !check(.rightBrace) && !isAtEnd {
            if check(.semicolon) {
                let loc = peek().location
                advance()
                body.append(.empty(location: loc))
                continue
            }
            body.append(try parseStatement())
        }
        try expect(.rightBrace, message: "Expected '}' to close while body")
        return .whileStatement(condition: condition, body: body, location: start.location)
    }

    private mutating func parseCondition() throws -> OpenQASMCondition {
        let reg = try expectIdentifier(message: "Expected classical register in condition")
        try expect(.equalEqual, message: "Expected '==' in condition")
        let valueTok = try expect(.integer, message: "Expected integer in condition")
        let value = try intValue(valueTok)
        return .equals(register: reg.lexeme, value: value)
    }

    // MARK: - Arguments

    private mutating func parseArgumentList() throws -> [OpenQASMArgument] {
        var args: [OpenQASMArgument] = []
        args.append(try parseArgument())
        while match(.comma) {
            args.append(try parseArgument())
        }
        return args
    }

    private mutating func parseArgument() throws -> OpenQASMArgument {
        let name = try expectIdentifier(message: "Expected register or qubit name")
        var index: Int?
        if match(.leftBracket) {
            let idxTok = try expect(.integer, message: "Expected integer index")
            index = try intValue(idxTok)
            try expect(.rightBracket, message: "Expected ']' after index")
        }
        return OpenQASMArgument(name: name.lexeme, index: index)
    }

    // MARK: - Expressions (angle / parameters)

    /// expr := term (("+" | "-") term)*
    private mutating func parseExpression() throws -> OpenQASMExpr {
        var left = try parseTerm()
        while true {
            if match(.plus) {
                let right = try parseTerm()
                left = .binary(.add, left, right)
            } else if match(.minus) {
                let right = try parseTerm()
                left = .binary(.subtract, left, right)
            } else {
                break
            }
        }
        return left
    }

    /// term := unary (("*" | "/") unary)*
    private mutating func parseTerm() throws -> OpenQASMExpr {
        var left = try parseUnary()
        while true {
            if match(.star) {
                let right = try parseUnary()
                left = .binary(.multiply, left, right)
            } else if match(.slash) {
                let right = try parseUnary()
                left = .binary(.divide, left, right)
            } else {
                break
            }
        }
        return left
    }

    /// unary := "-" unary | primary
    private mutating func parseUnary() throws -> OpenQASMExpr {
        if match(.minus) {
            return .unaryMinus(try parseUnary())
        }
        return try parsePrimary()
    }

    private mutating func parsePrimary() throws -> OpenQASMExpr {
        let token = peek()
        switch token.kind {
        case .integer:
            advance()
            return .integer(try intValue(token))
        case .float:
            advance()
            guard let value = Double(token.lexeme) else {
                throw parseError(token, "Invalid float literal '\(token.lexeme)'")
            }
            return .float(value)
        case .identifier:
            advance()
            return .identifier(token.lexeme)
        case .leftParen:
            advance()
            let inner = try parseExpression()
            try expect(.rightParen, message: "Expected ')' after expression")
            return .paren(inner)
        default:
            throw parseError(token, "Expected expression, found '\(token.lexeme)'")
        }
    }

    // MARK: - Token helpers

    private var isAtEnd: Bool {
        peek().kind == .eof
    }

    private func peek() -> OpenQASMToken {
        tokens[current]
    }

    @discardableResult
    private mutating func advance() -> OpenQASMToken {
        let token = tokens[current]
        if !isAtEnd {
            current += 1
        }
        return token
    }

    private func check(_ kind: OpenQASMTokenKind) -> Bool {
        !isAtEnd && peek().kind == kind
    }

    @discardableResult
    private mutating func match(_ kind: OpenQASMTokenKind) -> Bool {
        if check(kind) {
            advance()
            return true
        }
        return false
    }

    @discardableResult
    private mutating func expect(
        _ kind: OpenQASMTokenKind,
        message: String
    ) throws -> OpenQASMToken {
        if check(kind) {
            return advance()
        }
        throw parseError(peek(), message)
    }

    @discardableResult
    private mutating func expectIdentifier(message: String) throws -> OpenQASMToken {
        try expect(.identifier, message: message)
    }

    private mutating func expectSemicolon(after _: OpenQASMToken) throws {
        try expect(.semicolon, message: "Expected ';'")
    }

    private func parseError(_ token: OpenQASMToken, _ message: String) -> OpenQASMError {
        .parseError(line: token.line, column: token.column, message: message)
    }

    private func intValue(_ token: OpenQASMToken) throws -> Int {
        guard let value = Int(token.lexeme) else {
            throw parseError(token, "Invalid integer literal '\(token.lexeme)'")
        }
        return value
    }

    private func stripStringLiteral(_ lexeme: String) -> String {
        var s = lexeme
        if s.hasPrefix("\"") { s.removeFirst() }
        if s.hasSuffix("\"") { s.removeLast() }
        return s
    }
}
