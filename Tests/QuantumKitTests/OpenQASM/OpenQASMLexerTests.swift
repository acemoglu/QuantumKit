import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Version detect

    func testOpenQASMVersionDetectV2() throws {
        XCTAssertEqual(try OpenQASMVersion.detect(from: "OPENQASM 2.0;\nqreg q[1];"), .v2)
        XCTAssertEqual(try OpenQASMVersionDetector.detect(from: "OPENQASM 2.0;"), .v2)
    }

    func testOpenQASMVersionDetectV3() throws {
        XCTAssertEqual(try OpenQASMVersion.detect(from: "OPENQASM 3.0;\nqubit q;"), .v3)
        XCTAssertEqual(try OpenQASMVersion.detect(from: "OPENQASM 3;\nqubit[2] q;"), .v3)
    }

    func testOpenQASMVersionDetectMissingDefaultsToV2() throws {
        // Missing header defaults to v2 for qelib1-style circuits.
        XCTAssertEqual(try OpenQASMVersion.detect(from: "include \"qelib1.inc\";\nqreg q[2];"), .v2)
        XCTAssertEqual(try OpenQASMVersion.detect(from: "qreg q[1];"), .v2)
    }

    func testOpenQASMVersionDetectUnknownThrows() {
        XCTAssertThrowsError(try OpenQASMVersion.detect(from: "OPENQASM 4.0;")) { error in
            guard let e = error as? OpenQASMError else {
                XCTFail("Expected OpenQASMError, got \(error)")
                return
            }
            if case .unsupported(let line, let column, let feature, _) = e {
                XCTAssertEqual(line, 1)
                XCTAssertGreaterThanOrEqual(column, 1)
                XCTAssertTrue(feature.contains("4"), feature)
            } else {
                XCTFail("Expected .unsupported, got \(e)")
            }
        }
    }

    func testOpenQASMVersionDetectSkipsComments() throws {
        let source = """
        // header comment
        /* block */
        OPENQASM 3.0;
        """
        XCTAssertEqual(try OpenQASMVersion.detect(from: source), .v3)
    }

    // MARK: - Lex basic QASM2

    func testOpenQASMLexerBasicCircuit() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
        """
        var lexer = OpenQASMLexer(source: source)
        let tokens = try lexer.tokenize()

        XCTAssertEqual(tokens.first?.kind, .keywordOPENQASM)
        XCTAssertEqual(tokens.last?.kind, .eof)

        let kinds = tokens.map(\.kind)
        XCTAssertTrue(kinds.contains(.keywordInclude))
        XCTAssertTrue(kinds.contains(.keywordQreg))
        XCTAssertTrue(kinds.contains(.keywordCreg))
        XCTAssertTrue(kinds.contains(.keywordMeasure))
        XCTAssertTrue(kinds.contains(.arrow))
        XCTAssertTrue(kinds.contains(.string))

        let stringToken = tokens.first { $0.kind == .string }
        XCTAssertEqual(stringToken?.lexeme, "\"qelib1.inc\"")

        // Gate names are identifiers (not keywords)
        let idLexemes = tokens.filter { $0.kind == .identifier }.map(\.lexeme)
        XCTAssertTrue(idLexemes.contains("h"))
        XCTAssertTrue(idLexemes.contains("cx"))
        XCTAssertTrue(idLexemes.contains("q"))
        XCTAssertTrue(idLexemes.contains("c"))

        // Version float / integers present
        XCTAssertTrue(tokens.contains { $0.kind == .float && $0.lexeme == "2.0" })
        XCTAssertTrue(tokens.contains { $0.kind == .integer && $0.lexeme == "2" })
        XCTAssertTrue(tokens.contains { $0.kind == .integer && $0.lexeme == "0" })
        XCTAssertTrue(tokens.contains { $0.kind == .integer && $0.lexeme == "1" })
    }

    // MARK: - Line / column on error

    func testOpenQASMLexerUnterminatedStringReportsLocation() {
        let source = "include \"qelib1.inc"
        var lexer = OpenQASMLexer(source: source)
        XCTAssertThrowsError(try lexer.tokenize()) { error in
            guard let e = error as? OpenQASMError else {
                XCTFail("Expected OpenQASMError, got \(error)")
                return
            }
            if case .lexError(let line, let column, let message) = e {
                XCTAssertEqual(line, 1)
                XCTAssertEqual(column, 9) // opening quote of the string
                XCTAssertTrue(message.lowercased().contains("unterminated"), message)
            } else {
                XCTFail("Expected .lexError, got \(e)")
            }
            XCTAssertEqual(e.location.line, 1)
            XCTAssertEqual(e.location.column, 9)
        }
    }

    func testOpenQASMLexerBadCharacterReportsLocation() {
        let source = "qreg q[1];\n$"
        var lexer = OpenQASMLexer(source: source)
        XCTAssertThrowsError(try lexer.tokenize()) { error in
            guard let e = error as? OpenQASMError else {
                XCTFail("Expected OpenQASMError, got \(error)")
                return
            }
            if case .lexError(let line, let column, let message) = e {
                XCTAssertEqual(line, 2)
                XCTAssertEqual(column, 1)
                XCTAssertTrue(message.contains("$"), message)
            } else {
                XCTFail("Expected .lexError, got \(e)")
            }
        }
    }

    // MARK: - Comments skipped

    func testOpenQASMLexerSkipsComments() throws {
        let source = """
        OPENQASM 2.0; // version
        /* multi
           line */
        qreg q[1];
        """
        var lexer = OpenQASMLexer(source: source)
        let tokens = try lexer.tokenize()
        let lexemes = tokens.map(\.lexeme)
        XCTAssertFalse(lexemes.contains(where: { $0.contains("version") }))
        XCTAssertFalse(lexemes.contains(where: { $0.contains("multi") }))
        XCTAssertEqual(tokens.map(\.kind).filter { $0 == .keywordQreg }.count, 1)
    }

    // MARK: - pi and floats

    func testOpenQASMLexerPiAndFloats() throws {
        let source = "rz(pi/2) q[0];\nu3(0.5,1e-3,.25) q[0];"
        var lexer = OpenQASMLexer(source: source)
        let tokens = try lexer.tokenize()

        let pi = tokens.first { $0.kind == .identifier && $0.lexeme == "pi" }
        XCTAssertNotNil(pi)

        XCTAssertTrue(tokens.contains { $0.kind == .float && $0.lexeme == "0.5" })
        XCTAssertTrue(tokens.contains { $0.kind == .float && $0.lexeme == "1e-3" })
        XCTAssertTrue(tokens.contains { $0.kind == .float && $0.lexeme == ".25" })
        XCTAssertTrue(tokens.contains { $0.kind == .integer && $0.lexeme == "2" })
    }

    func testOpenQASMErrorLocalizedDescription() {
        let err = OpenQASMError.lexError(line: 3, column: 4, message: "bad")
        XCTAssertEqual(err.errorDescription, "OpenQASM lex error at 3:4: bad")
        XCTAssertNotNil(err.recoverySuggestion)
    }
}
