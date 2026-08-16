import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - u / u1 / u2 / u3

    func testOpenQASM2ImporterU3MapsToU() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u3(pi/2,0,pi) q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 1)
        guard case .u(let theta, let phi, let lambda, let target) = circuit.gates[0] else {
            return XCTFail("Expected Gate.u")
        }
        XCTAssertEqual(target, 0)
        assertLiteralApprox(theta, Double.pi / 2)
        assertLiteralApprox(phi, 0)
        assertLiteralApprox(lambda, Double.pi)
    }

    func testOpenQASM2ImporterU1MapsToP() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u1(pi/4) q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        guard case .p(let theta, let target) = circuit.gates[0] else {
            return XCTFail("Expected Gate.p")
        }
        XCTAssertEqual(target, 0)
        assertLiteralApprox(theta, Double.pi / 4)
    }

    func testOpenQASM2ImporterU2ThetaIsHalfPi() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u2(0,pi) q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        guard case .u(let theta, let phi, let lambda, let target) = circuit.gates[0] else {
            return XCTFail("Expected Gate.u")
        }
        XCTAssertEqual(target, 0)
        assertLiteralApprox(theta, Double.pi / 2)
        assertLiteralApprox(phi, 0)
        assertLiteralApprox(lambda, Double.pi)
    }

    // MARK: - rx / ry / rz

    func testOpenQASM2ImporterRxRyRz() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(pi/2) q[0];
        ry(pi/4) q[0];
        rz(pi) q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 3)
        guard case .rx(let rxTheta, let rxTarget) = circuit.gates[0] else {
            return XCTFail("Expected rx")
        }
        XCTAssertEqual(rxTarget, 0)
        assertLiteralApprox(rxTheta, Double.pi / 2)

        guard case .ry(let ryTheta, let ryTarget) = circuit.gates[1] else {
            return XCTFail("Expected ry")
        }
        XCTAssertEqual(ryTarget, 0)
        assertLiteralApprox(ryTheta, Double.pi / 4)

        guard case .rz(let rzTheta, let rzTarget) = circuit.gates[2] else {
            return XCTFail("Expected rz")
        }
        XCTAssertEqual(rzTarget, 0)
        assertLiteralApprox(rzTheta, Double.pi)
    }

    // MARK: - Controlled parametric + cswap

    func testOpenQASM2ImporterCrxCryCrzCp() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        crx(pi/2) q[0],q[1];
        cry(pi/4) q[0],q[1];
        crz(pi) q[0],q[1];
        cp(pi/8) q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 4)

        guard case .crx(let crxTheta, let crxC, let crxT) = circuit.gates[0] else {
            return XCTFail("Expected crx")
        }
        XCTAssertEqual(crxC, 0)
        XCTAssertEqual(crxT, 1)
        assertLiteralApprox(crxTheta, Double.pi / 2)

        guard case .cry(let cryTheta, let cryC, let cryT) = circuit.gates[1] else {
            return XCTFail("Expected cry")
        }
        XCTAssertEqual(cryC, 0)
        XCTAssertEqual(cryT, 1)
        assertLiteralApprox(cryTheta, Double.pi / 4)

        guard case .crz(let crzTheta, let crzC, let crzT) = circuit.gates[2] else {
            return XCTFail("Expected crz")
        }
        XCTAssertEqual(crzC, 0)
        XCTAssertEqual(crzT, 1)
        assertLiteralApprox(crzTheta, Double.pi)

        guard case .cp(let cpTheta, let cpC, let cpT) = circuit.gates[3] else {
            return XCTFail("Expected cp")
        }
        XCTAssertEqual(cpC, 0)
        XCTAssertEqual(cpT, 1)
        assertLiteralApprox(cpTheta, Double.pi / 8)
    }

    func testOpenQASM2ImporterCswap() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        cswap q[0],q[1],q[2];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [.cswap(control: 0, q1: 1, q2: 2)])
    }

    // MARK: - Angle expressions

    func testOpenQASM2ImporterAngleExprArithmetic() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx((pi/2)*(1+1)) q[0];
        ry(pi/2+pi/4) q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 2)
        guard case .rx(let rxTheta, _) = circuit.gates[0] else {
            return XCTFail("Expected rx")
        }
        assertLiteralApprox(rxTheta, Double.pi)
        guard case .ry(let ryTheta, _) = circuit.gates[1] else {
            return XCTFail("Expected ry")
        }
        assertLiteralApprox(ryTheta, Double.pi / 2 + Double.pi / 4)
    }

    func testOpenQASM2ImporterUnknownIdentifierInAngleFails() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(theta) q[0];
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            guard case .semanticError(let line, let column, let message) = e else {
                return XCTFail("Expected semanticError, got \(e)")
            }
            XCTAssertTrue(message.lowercased().contains("theta") || message.lowercased().contains("unknown"), message)
            XCTAssertGreaterThan(line, 0)
            XCTAssertGreaterThan(column, 0)
        }
    }

    // MARK: - Helpers

    private func assertLiteralApprox(
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
