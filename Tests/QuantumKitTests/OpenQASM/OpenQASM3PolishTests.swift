import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - stdgates.inc alias

    func testOpenQASM3StdgatesIncludeAliasesQelib1() throws {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
        """
        let circuit = try OpenQASM.importCircuit(source)
        XCTAssertEqual(circuit.gates, [
            .h(target: 0),
            .cx(control: 0, target: 1),
        ])
    }

    // MARK: - Whole-register broadcast

    func testOpenQASM3BroadcastSingleQubitGate() throws {
        let source = """
        OPENQASM 3.0;
        qubit[3] q;
        h q;
        """
        let circuit = try OpenQASM.importCircuit(source)
        XCTAssertEqual(circuit.gates, [
            .h(target: 0),
            .h(target: 1),
            .h(target: 2),
        ])
    }

    func testOpenQASM2BroadcastSingleQubitGate() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        x q;
        """
        let circuit = try OpenQASM.importCircuit(source)
        XCTAssertEqual(circuit.gates, [
            .x(target: 0),
            .x(target: 1),
        ])
    }

    func testOpenQASM3BroadcastPairwiseCX() throws {
        let source = """
        OPENQASM 3.0;
        qubit[2] a;
        qubit[2] b;
        cx a, b;
        """
        let circuit = try OpenQASM.importCircuit(source)
        XCTAssertEqual(circuit.gates, [
            .cx(control: 0, target: 2),
            .cx(control: 1, target: 3),
        ])
    }

    // MARK: - Modifiers

    func testOpenQASM3CtrlXLowersToCX() throws {
        let source = """
        OPENQASM 3.0;
        qubit[2] q;
        ctrl @ x q[0], q[1];
        """
        let circuit = try OpenQASM.importCircuit(source)
        XCTAssertEqual(circuit.gates, [.cx(control: 0, target: 1)])
    }

    func testOpenQASM3CtrlCtrlXLowersToCCX() throws {
        let source = """
        OPENQASM 3.0;
        qubit[3] q;
        ctrl @ ctrl @ x q[0], q[1], q[2];
        """
        let circuit = try OpenQASM.importCircuit(source)
        XCTAssertEqual(circuit.gates, [
            .ccx(control1: 0, control2: 1, target: 2),
        ])
    }

    func testOpenQASM3InvSLowersToSdg() throws {
        let source = """
        OPENQASM 3.0;
        qubit q;
        inv @ s q;
        """
        let circuit = try OpenQASM.importCircuit(source)
        XCTAssertEqual(circuit.gates, [.sdg(target: 0)])
    }

    func testOpenQASM3PowRxScalesAngle() throws {
        let source = """
        OPENQASM 3.0;
        qubit q;
        pow(2) @ rx(pi/4) q;
        """
        let circuit = try OpenQASM.importCircuit(source)
        guard case .rx(let theta, let target) = circuit.gates[0] else {
            return XCTFail("Expected rx, got \(circuit.gates)")
        }
        XCTAssertEqual(target, 0)
        XCTAssertEqual(theta, .scaled(.literal(QFloat(Double.pi / 4)), 2))
    }

    func testOpenQASM3CtrlInvXEqualsCX() throws {
        let source = """
        OPENQASM 3.0;
        qubit[2] q;
        ctrl @ inv @ x q[0], q[1];
        """
        let circuit = try OpenQASM.importCircuit(source)
        XCTAssertEqual(circuit.gates, [.cx(control: 0, target: 1)])
    }

    func testOpenQASM3NegctrlStillUnsupported() {
        let source = """
        OPENQASM 3.0;
        qubit[2] q;
        negctrl @ x q[0], q[1];
        """
        XCTAssertThrowsError(try OpenQASM.importCircuit(source)) { error in
            guard case OpenQASMError.unsupported(_, _, let feature, _) = error else {
                return XCTFail("Expected unsupported, got \(error)")
            }
            XCTAssertEqual(feature, "negctrl")
        }
    }
}
