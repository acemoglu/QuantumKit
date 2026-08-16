/// Supported OpenQASM language major versions for QuantumKit front-ends.
public enum OpenQASMVersion: Equatable, Sendable, Hashable {
    /// OpenQASM 2.x (including qelib1-style circuits).
    case v2
    /// OpenQASM 3.x.
    case v3
}

/// Detects `OPENQASM` version headers in source text.
///
/// Policy:
/// - `OPENQASM 2.0;` → `.v2`
/// - `OPENQASM 3.0;` or `OPENQASM 3;` → `.v3`
/// - **Missing header defaults to `.v2`** (qelib1-style circuits without an explicit version).
/// - Unknown major versions throw `OpenQASMError.unsupported` with a 1-based location.
public enum OpenQASMVersionDetector: Sendable {
    /// Detects the OpenQASM version from `source`.
    ///
    /// - Parameter source: Full OpenQASM program text.
    /// - Returns: Detected version, or `.v2` when no `OPENQASM` statement is present.
    /// - Throws: `OpenQASMError` for malformed / unknown version headers.
    public static func detect(from source: String) throws -> OpenQASMVersion {
        try OpenQASMVersion.detect(from: source)
    }
}

extension OpenQASMVersion {
    /// Detects the OpenQASM version from `source`.
    ///
    /// Missing `OPENQASM` header defaults to `.v2`. See `OpenQASMVersionDetector`.
    public static func detect(from source: String) throws -> OpenQASMVersion {
        var index = source.startIndex
        var line = 1
        var column = 1

        func peek() -> Character? {
            index < source.endIndex ? source[index] : nil
        }

        func advance() {
            guard let ch = peek() else { return }
            if ch == "\n" {
                line += 1
                column = 1
            } else {
                column += 1
            }
            index = source.index(after: index)
        }

        func skipWhitespaceAndComments() throws {
            while let ch = peek() {
                if ch == " " || ch == "\t" || ch == "\r" || ch == "\n" {
                    advance()
                    continue
                }
                if ch == "/", index < source.endIndex {
                    let next = source.index(after: index)
                    if next < source.endIndex, source[next] == "/" {
                        advance() // /
                        advance() // /
                        while let c = peek(), c != "\n" {
                            advance()
                        }
                        continue
                    }
                    if next < source.endIndex, source[next] == "*" {
                        let startLine = line
                        let startColumn = column
                        advance() // /
                        advance() // *
                        var closed = false
                        while let c = peek() {
                            if c == "*" {
                                advance()
                                if peek() == "/" {
                                    advance()
                                    closed = true
                                    break
                                }
                                continue
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

        func matchIdentifier(_ expected: String) -> Bool {
            let savedIndex = index
            let savedLine = line
            let savedColumn = column
            for expectedChar in expected {
                guard let ch = peek(), ch == expectedChar else {
                    index = savedIndex
                    line = savedLine
                    column = savedColumn
                    return false
                }
                advance()
            }
            if let ch = peek(), isIdentContinue(ch) {
                index = savedIndex
                line = savedLine
                column = savedColumn
                return false
            }
            return true
        }

        try skipWhitespaceAndComments()

        // No OPENQASM header → default to OpenQASM 2 (qelib1-style).
        guard matchIdentifier("OPENQASM") else {
            return .v2
        }

        let versionLine = line
        let versionColumn = column
        try skipWhitespaceAndComments()

        let majorStartLine = line
        let majorStartColumn = column
        guard let first = peek(), first.isASCII && first.isNumber else {
            throw OpenQASMError.parseError(
                line: majorStartLine,
                column: majorStartColumn,
                message: "Expected version number after OPENQASM"
            )
        }

        var majorText = ""
        while let ch = peek(), ch.isASCII && ch.isNumber {
            majorText.append(ch)
            advance()
        }

        // Optional ".minor"
        if peek() == "." {
            advance()
            guard let dig = peek(), dig.isASCII && dig.isNumber else {
                throw OpenQASMError.parseError(
                    line: line,
                    column: column,
                    message: "Expected minor version digits after '.'"
                )
            }
            while let ch = peek(), ch.isASCII && ch.isNumber {
                advance()
            }
        }

        guard let major = Int(majorText) else {
            throw OpenQASMError.parseError(
                line: majorStartLine,
                column: majorStartColumn,
                message: "Invalid OPENQASM version number '\(majorText)'"
            )
        }

        switch major {
        case 2:
            return .v2
        case 3:
            return .v3
        default:
            throw OpenQASMError.unsupported(
                line: versionLine,
                column: versionColumn,
                feature: "OPENQASM \(major)",
                message: "Unknown OpenQASM major version \(major); QuantumKit supports 2 and 3"
            )
        }
    }
}

private func isIdentContinue(_ ch: Character) -> Bool {
    ch.isASCII && (ch.isLetter || ch.isNumber || ch == "_")
}
