import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Bell-like program

    func testOpenQASMParserBellProgramAST() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
        """
        var parser = try OpenQASMParser(source: source)
        let program = try parser.parse()

        XCTAssertEqual(program.version, .v2)
        XCTAssertGreaterThanOrEqual(program.statements.count, 7)

        guard case .version(.v2) = program.statements[0] else {
            return XCTFail("Expected version statement")
        }
        guard case .include(let path, _) = program.statements[1] else {
            return XCTFail("Expected include")
        }
        XCTAssertEqual(path, "qelib1.inc")

        guard case .qreg(let qName, let qSize, _) = program.statements[2] else {
            return XCTFail("Expected qreg")
        }
        XCTAssertEqual(qName, "q")
        XCTAssertEqual(qSize, 2)

        guard case .creg(let cName, let cSize, _) = program.statements[3] else {
            return XCTFail("Expected creg")
        }
        XCTAssertEqual(cName, "c")
        XCTAssertEqual(cSize, 2)

        guard case .gateCall(let hName, let hParams, let hQubits, _) = program.statements[4] else {
            return XCTFail("Expected h gate call")
        }
        XCTAssertEqual(hName, "h")
        XCTAssertTrue(hParams.isEmpty)
        XCTAssertEqual(hQubits, [OpenQASMArgument(name: "q", index: 0)])

        guard case .gateCall(let cxName, _, let cxQubits, _) = program.statements[5] else {
            return XCTFail("Expected cx gate call")
        }
        XCTAssertEqual(cxName, "cx")
        XCTAssertEqual(
            cxQubits,
            [OpenQASMArgument(name: "q", index: 0), OpenQASMArgument(name: "q", index: 1)]
        )

        guard case .measure(let mQ, let mC, _) = program.statements[6] else {
            return XCTFail("Expected measure")
        }
        XCTAssertEqual(mQ, [OpenQASMArgument(name: "q", index: nil)])
        XCTAssertEqual(mC, [OpenQASMArgument(name: "c", index: nil)])
    }

    // MARK: - Measure forms

    func testOpenQASMParserMeasureArrowForms() throws {
        let indexed = """
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        measure q[0] -> c[0];
        """
        var p1 = try OpenQASMParser(source: indexed)
        let prog1 = try p1.parse()
        guard case .measure(let q1, let c1, _) = prog1.statements.last else {
            return XCTFail("Expected indexed measure")
        }
        XCTAssertEqual(q1, [OpenQASMArgument(name: "q", index: 0)])
        XCTAssertEqual(c1, [OpenQASMArgument(name: "c", index: 0)])

        let whole = """
        qreg q[2];
        creg c[2];
        measure q -> c;
        """
        var p2 = try OpenQASMParser(source: whole)
        let prog2 = try p2.parse()
        guard case .measure(let q2, let c2, _) = prog2.statements.last else {
            return XCTFail("Expected whole-register measure")
        }
        XCTAssertEqual(q2, [OpenQASMArgument(name: "q")])
        XCTAssertEqual(c2, [OpenQASMArgument(name: "c")])
    }

    // MARK: - if(c==1)

    func testOpenQASMParserIfStatement() throws {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        if(c==1) x q[0];
        """
        var parser = try OpenQASMParser(source: source)
        let program = try parser.parse()

        guard case .ifStatement(let cond, let body, _) = program.statements.last else {
            return XCTFail("Expected if statement")
        }
        XCTAssertEqual(cond, .equals(register: "c", value: 1))
        guard case .gateCall(let name, _, let qubits, _) = body else {
            return XCTFail("Expected gate call body")
        }
        XCTAssertEqual(name, "x")
        XCTAssertEqual(qubits, [OpenQASMArgument(name: "q", index: 0)])
    }

    // MARK: - Gate decl

    func testOpenQASMParserGateDeclWithBody() throws {
        let source = """
        OPENQASM 2.0;
        gate foo(a) q {
          h q;
        }
        """
        var parser = try OpenQASMParser(source: source)
        let program = try parser.parse()

        guard case .gateDecl(let name, let params, let qubits, let body, _) = program.statements.last else {
            return XCTFail("Expected gate decl")
        }
        XCTAssertEqual(name, "foo")
        XCTAssertEqual(params, ["a"])
        XCTAssertEqual(qubits, ["q"])
        XCTAssertEqual(body.count, 1)
        guard case .gateCall(let callName, _, let callQubits, _) = body[0] else {
            return XCTFail("Expected h in gate body")
        }
        XCTAssertEqual(callName, "h")
        XCTAssertEqual(callQubits, [OpenQASMArgument(name: "q")])
    }

    // MARK: - Angle expression

    func testOpenQASMParserU3AngleExpression() throws {
        let source = "u3(pi/2,0,pi) q[0];"
        var parser = try OpenQASMParser(source: source)
        let program = try parser.parse()

        guard case .gateCall(let name, let params, let qubits, _) = program.statements.first else {
            return XCTFail("Expected u3 gate call")
        }
        XCTAssertEqual(name, "u3")
        XCTAssertEqual(qubits, [OpenQASMArgument(name: "q", index: 0)])
        XCTAssertEqual(params.count, 3)

        // pi/2
        guard case .binary(.divide, let lhs, let rhs) = params[0] else {
            return XCTFail("Expected pi/2 binary divide")
        }
        XCTAssertEqual(lhs, .identifier("pi"))
        XCTAssertEqual(rhs, .integer(2))

        XCTAssertEqual(params[1], .integer(0))
        XCTAssertEqual(params[2], .identifier("pi"))
    }

    // MARK: - Version sets program.version

    func testOpenQASMParserVersionSetsProgramVersion() throws {
        var p2 = try OpenQASMParser(source: "OPENQASM 2.0;\nqreg q[1];")
        XCTAssertEqual(try p2.parse().version, .v2)

        var p3 = try OpenQASMParser(source: "OPENQASM 3.0;\nqubit q;")
        let prog3 = try p3.parse()
        XCTAssertEqual(prog3.version, .v3)
        guard case .qubitDecl(let name, let size, _) = prog3.statements.last else {
            return XCTFail("Expected qubit decl")
        }
        XCTAssertEqual(name, "q")
        XCTAssertNil(size)

        var p3int = try OpenQASMParser(source: "OPENQASM 3;\nbit[2] b;")
        let prog3i = try p3int.parse()
        XCTAssertEqual(prog3i.version, .v3)
        guard case .bitDecl(let bName, let bSize, _) = prog3i.statements.last else {
            return XCTFail("Expected bit decl")
        }
        XCTAssertEqual(bName, "b")
        XCTAssertEqual(bSize, 2)
    }

    // MARK: - Syntax error location

    func testOpenQASMParserSyntaxErrorReportsLineColumn() {
        let source = """
        OPENQASM 2.0;
        qreg q[2]
        h q[0];
        """
        // Missing ';' after qreg — error at `h` or at end of qreg line.
        do {
            var parser = try OpenQASMParser(source: source)
            _ = try parser.parse()
            XCTFail("Expected parse error")
        } catch let error as OpenQASMError {
            if case .parseError(let line, let column, let message) = error {
                XCTAssertEqual(line, 3)
                XCTAssertGreaterThanOrEqual(column, 1)
                XCTAssertFalse(message.isEmpty)
                XCTAssertEqual(error.location.line, 3)
            } else {
                XCTFail("Expected .parseError, got \(error)")
            }
        } catch {
            XCTFail("Expected OpenQASMError, got \(error)")
        }
    }

    func testOpenQASMParserResetAndBarrier() throws {
        let source = """
        qreg q[2];
        reset q[0];
        barrier q;
        barrier;
        """
        var parser = try OpenQASMParser(source: source)
        let program = try parser.parse()

        guard case .reset(let rQ, _) = program.statements[1] else {
            return XCTFail("Expected reset")
        }
        XCTAssertEqual(rQ, [OpenQASMArgument(name: "q", index: 0)])

        guard case .barrier(let b1, _) = program.statements[2] else {
            return XCTFail("Expected barrier q")
        }
        XCTAssertEqual(b1, [OpenQASMArgument(name: "q")])

        guard case .barrier(let b2, _) = program.statements[3] else {
            return XCTFail("Expected empty barrier")
        }
        XCTAssertTrue(b2.isEmpty)
    }

    func testOpenQASMParserUnsupportedDefcal() {
        do {
            var parser = try OpenQASMParser(source: "defcal foo() q {}")
            _ = try parser.parse()
            XCTFail("Expected unsupported")
        } catch let error as OpenQASMError {
            if case .unsupported(_, _, let feature, _) = error {
                XCTAssertEqual(feature, "defcal")
            } else {
                XCTFail("Expected .unsupported, got \(error)")
            }
        } catch {
            XCTFail("Expected OpenQASMError, got \(error)")
        }
    }
}
