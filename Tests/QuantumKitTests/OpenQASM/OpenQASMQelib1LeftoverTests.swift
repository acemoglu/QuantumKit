import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - ch / cy decompositions

    func testOpenQASM2ImporterCHDecomposition() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        ch q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        // s t; h t; t t; cx c,t; tdg t; h t; sdg t;
        XCTAssertEqual(circuit.gates, [
            .s(target: 1),
            .h(target: 1),
            .t(target: 1),
            .cx(control: 0, target: 1),
            .tdg(target: 1),
            .h(target: 1),
            .sdg(target: 1),
        ])
    }

    func testOpenQASM2ImporterCYDecomposition() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cy q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .sdg(target: 1),
            .cx(control: 0, target: 1),
            .s(target: 1),
        ])
    }

    // MARK: - mcx / mcz

    func testOpenQASM2ImporterMCXTwoControlsUsesCCX() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        mcx q[0],q[1],q[2];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .ccx(control1: 0, control2: 1, target: 2),
        ])
    }

    func testOpenQASM2ImporterMCXOneControlUsesCX() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        mcx q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .cx(control: 0, target: 1),
        ])
    }

    func testOpenQASM2ImporterMCXThreeControlsUsesMCX() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        mcx q[0],q[1],q[2],q[3];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .mcx(controls: [0, 1, 2], target: 3),
        ])
    }

    func testOpenQASM2ImporterMCZMapsToGateMCZ() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        mcz q[0],q[1],q[2];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .mcz(controls: [0, 1], target: 2),
        ])
    }

    func testOpenQASM2ImporterMCZOneControlUsesCZ() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        mcz q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .cz(control: 0, target: 1),
        ])
    }

    // MARK: - cu1 → cp

    func testOpenQASM2ImporterCU1MapsToCP() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cu1(pi/2) q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 1)
        guard case .cp(let theta, let control, let target) = circuit.gates[0] else {
            return XCTFail("Expected cp from cu1")
        }
        XCTAssertEqual(control, 0)
        XCTAssertEqual(target, 1)
        guard case .literal(let value) = theta else {
            return XCTFail("Expected literal angle")
        }
        XCTAssertEqual(Double(value), Double.pi / 2, accuracy: 1e-5)
    }

    // MARK: - Still rejected leftovers

    func testOpenQASM2ImporterUnknownQelib1StyleGateStillFails() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rzz(0.1) q[0],q[0];
        """
        // rzz is not in the mapped set; importer should hard-error (even if arity is wrong).
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard case OpenQASMError.semanticError(_, _, let message) = error else {
                return XCTFail("Expected semanticError, got \(error)")
            }
            XCTAssertTrue(
                message.contains("Unknown or unmapped") || message.contains("expects"),
                message
            )
        }
    }

    func testOpenQASM2ImporterMCXTooFewQubitsFails() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        mcx q[0];
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard case OpenQASMError.semanticError(_, _, let message) = error else {
                return XCTFail("Expected semanticError, got \(error)")
            }
            XCTAssertTrue(message.contains("mcx"), message)
        }
    }

    func testOpenQASM3ImporterCHCYMCX() throws {
        let source = """
        OPENQASM 3.0;
        qubit[4] q;
        ch q[0], q[1];
        cy q[1], q[2];
        mcx q[0], q[1], q[2], q[3];
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 7 + 3 + 1)
        XCTAssertEqual(circuit.gates.last, .mcx(controls: [0, 1, 2], target: 3))
    }
}
