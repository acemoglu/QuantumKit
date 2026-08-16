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

    // MARK: - Expanded / mapped qelib1 leftovers

    func testOpenQASM2ImporterRZZMapsToGateRZZ() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        rzz(0.1) q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 1)
        guard case .rzz(let theta, let q1, let q2) = circuit.gates[0] else {
            return XCTFail("Expected Gate.rzz fast-path")
        }
        XCTAssertEqual(q1, 0)
        XCTAssertEqual(q2, 1)
        guard case .literal(let value) = theta else {
            return XCTFail("Expected literal angle")
        }
        XCTAssertEqual(Double(value), 0.1, accuracy: 1e-5)
    }

    func testOpenQASM2ImporterRXXMapsToGateRXX() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        rxx(pi/4) q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 1)
        guard case .rxx(let theta, let q1, let q2) = circuit.gates[0] else {
            return XCTFail("Expected Gate.rxx fast-path")
        }
        XCTAssertEqual(q1, 0)
        XCTAssertEqual(q2, 1)
        guard case .literal(let value) = theta else {
            return XCTFail("Expected literal angle")
        }
        XCTAssertEqual(Double(value), Double.pi / 4, accuracy: 1e-5)
    }

    func testOpenQASM2MappedBuiltinWinsOverEmbeddedExpand() throws {
        // `x` is both mapped and defined in qelib1 as u3(pi,0,pi); fast-path must win.
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        x q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [.x(target: 0)])
    }

    func testOpenQASM2CannotRedefineMappedBuiltinAfterInclude() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        gate h a { x a; }
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard case OpenQASMError.semanticError(_, _, let message) = error else {
                return XCTFail("Expected semanticError, got \(error)")
            }
            XCTAssertTrue(message.contains("Cannot redefine builtin"), message)
        }
    }

    func testOpenQASM2CannotRedefineEmbeddedQelib1Gate() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        gate cu3(a,b,c) x,y { cx x,y; }
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard case OpenQASMError.semanticError(_, _, let message) = error else {
                return XCTFail("Expected semanticError, got \(error)")
            }
            XCTAssertTrue(message.contains("already declared"), message)
        }
    }

    func testOpenQASM2ImporterCU3ExpandsFromEmbeddedQelib1() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cu3(pi/2,0,0) q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertFalse(circuit.gates.isEmpty)
        // Expanded body uses mapped u1/cx/u3 — not a single opaque gate.
        XCTAssertGreaterThan(circuit.gates.count, 1)
    }

    func testOpenQASM2EmbeddedQelib1DeclCount() {
        // Sanity: every gate in the embedded source parsed successfully.
        XCTAssertEqual(
            OpenQASMQelib1.embeddedGateDeclarations.count,
            OpenQASMQelib1.embeddedGateNames.count
        )
        XCTAssertTrue(OpenQASMQelib1.embeddedGateNames.contains("cu3"))
        XCTAssertTrue(OpenQASMQelib1.embeddedGateNames.contains("c4x"))
        XCTAssertTrue(OpenQASMQelib1.embeddedGateNames.contains("rxx"))
    }

    func testOpenQASM2ImporterPrimitiveUAndCX() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        U(pi,0,pi) q[0];
        CX q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 2)
        guard case .u(let theta, let phi, let lambda, let target) = circuit.gates[0] else {
            return XCTFail("Expected U → Gate.u")
        }
        XCTAssertEqual(target, 0)
        guard case .literal(let t) = theta,
              case .literal(let p) = phi,
              case .literal(let l) = lambda else {
            return XCTFail("Expected literal U angles")
        }
        XCTAssertEqual(Double(t), Double.pi, accuracy: 1e-5)
        XCTAssertEqual(Double(p), 0, accuracy: 1e-5)
        XCTAssertEqual(Double(l), Double.pi, accuracy: 1e-5)
        XCTAssertEqual(circuit.gates[1], .cx(control: 0, target: 1))
    }

    func testOpenQASM2ImporterU0MapsToId() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u0(3.5) q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [.id(target: 0)])
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
