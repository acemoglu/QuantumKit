import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Simple expand

    func testOpenQASM2UserGateMyX() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        gate my_x a { x a; }
        my_x q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [.x(target: 0)])
    }

    func testOpenQASM2UserGateParametricRz() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        gate rz2(a) q { rz(a) q; }
        rz2(pi/2) q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 1)
        guard case .rz(let theta, let target) = circuit.gates[0] else {
            return XCTFail("Expected rz")
        }
        XCTAssertEqual(target, 0)
        assertUserGateLiteralApprox(theta, Double.pi / 2)
    }

    // MARK: - Nested expand

    func testOpenQASM2UserGateNestedBell() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        gate bell a,b { h a; cx a,b; }
        bell q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .h(target: 0),
            .cx(control: 0, target: 1),
        ])
    }

    func testOpenQASM2UserGateNestedCall() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        gate bell a,b { h a; cx a,b; }
        gate bell2 c,d { bell c,d; }
        bell2 q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .h(target: 0),
            .cx(control: 0, target: 1),
        ])
    }

    func testOpenQASM2UserGateEmptyBodyIsNoOp() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        gate noop a { }
        noop q[0];
        x q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [.x(target: 0)])
    }

    // MARK: - Classical if + multi-gate expand

    func testOpenQASM2UserGateUnderClassicalIfExpandsPerGate() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[1];
        gate bell a,b { h a; cx a,b; }
        if(c==1) bell q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .c_if(classicalRegister: 0, expectedValue: 1, gate: .h(target: 0)),
            .c_if(
                classicalRegister: 0,
                expectedValue: 1,
                gate: .cx(control: 0, target: 1)
            ),
        ])
    }

    // MARK: - Errors

    func testOpenQASM2UserGateOpaqueFails() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        opaque foo q;
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            switch e {
            case .unsupported(let line, let column, let feature, let message):
                XCTAssertEqual(feature, "opaque")
                XCTAssertTrue(message.lowercased().contains("opaque"), message)
                XCTAssertGreaterThan(line, 0)
                XCTAssertGreaterThan(column, 0)
            case .semanticError(let line, let column, let message):
                XCTAssertTrue(message.lowercased().contains("opaque"), message)
                XCTAssertGreaterThan(line, 0)
                XCTAssertGreaterThan(column, 0)
            default:
                XCTFail("Expected unsupported or semanticError, got \(e)")
            }
        }
    }

    func testOpenQASM2UserGateRedefineBuiltinFails() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        gate h a { x a; }
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            guard case .semanticError(let line, let column, let message) = e else {
                return XCTFail("Expected semanticError, got \(e)")
            }
            XCTAssertTrue(
                message.lowercased().contains("redefine") || message.contains("h"),
                message
            )
            XCTAssertGreaterThan(line, 0)
            XCTAssertGreaterThan(column, 0)
        }
    }

    func testOpenQASM2UserGateRecursiveFails() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        gate foo a { foo a; }
        foo q[0];
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            guard case .semanticError(let line, let column, let message) = e else {
                return XCTFail("Expected semanticError, got \(e)")
            }
            XCTAssertTrue(
                message.lowercased().contains("recursive") || message.lowercased().contains("foo"),
                message
            )
            XCTAssertGreaterThan(line, 0)
            XCTAssertGreaterThan(column, 0)
        }
    }

    func testOpenQASM2UserGateMutualRecursiveFails() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        gate a q { b q; }
        gate b q { a q; }
        a q[0];
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            guard case .semanticError(_, _, let message) = e else {
                return XCTFail("Expected semanticError, got \(e)")
            }
            XCTAssertTrue(message.lowercased().contains("recursive"), message)
        }
    }

    func testOpenQASM2UserGateMeasureInBodyFails() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        gate bad a { measure a -> c; }
        """
        // Parser may reject measure in gate body; either parse or lower error is OK.
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            XCTAssertTrue(error is OpenQASMError, "Expected OpenQASMError, got \(error)")
        }
    }

    // MARK: - Helpers

    private func assertUserGateLiteralApprox(
        _ expr: QFloatExpr,
        _ expected: Double,
        accuracy: Double = 1e-5,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard case .literal(let value) = expr else {
            XCTFail("Expected QFloatExpr.literal, got \(expr)", file: file, line: line)
            return
        }
        XCTAssertEqual(Double(value), expected, accuracy: accuracy, file: file, line: line)
    }
}
