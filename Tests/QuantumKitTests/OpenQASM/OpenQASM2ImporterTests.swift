import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Bell

    func testOpenQASM2ImporterBellCircuit() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)

        XCTAssertEqual(circuit.qubitCount, 2)
        XCTAssertEqual(circuit.classicalRegisters.count, 1)
        XCTAssertEqual(circuit.classicalRegisters[0].bitCount, 2)
        XCTAssertEqual(circuit.gates.count, 3)
        XCTAssertEqual(circuit.gates[0], .h(target: 0))
        XCTAssertEqual(circuit.gates[1], .cx(control: 0, target: 1))
        guard case .measure(let spec) = circuit.gates[2] else {
            return XCTFail("Expected measure gate")
        }
        XCTAssertEqual(spec.qubits, [0, 1])
        XCTAssertEqual(spec.classicalRegister, 0)
        XCTAssertEqual(spec.classicalBitOffset, 0)
    }

    // MARK: - Bit order

    func testOpenQASM2ImporterBitOrderQ0IsLSB() throws {
        // QuantumKit LSB = qubit 0; OpenQASM q[0] maps to engine qubit 0.
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        x q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [.x(target: 0)])
    }

    // MARK: - Reset + barrier

    func testOpenQASM2ImporterResetAndBarrier() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        reset q[0];
        barrier q[0],q[1];
        barrier;
        reset q;
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 5)
        XCTAssertEqual(circuit.gates[0], .reset(qubit: 0))
        XCTAssertEqual(circuit.gates[1], .barrier(qubits: [0, 1]))
        XCTAssertEqual(circuit.gates[2], .barrier(qubits: []))
        XCTAssertEqual(circuit.gates[3], .reset(qubit: 0))
        XCTAssertEqual(circuit.gates[4], .reset(qubit: 1))
    }

    // MARK: - Includes

    func testOpenQASM2ImporterQelib1IncludeWithoutFileWorks() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [.h(target: 0)])
    }

    func testOpenQASM2ImporterOtherIncludeFails() {
        let source = """
        OPENQASM 2.0;
        include "other.inc";
        qreg q[1];
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            guard case .unsupported(let line, let column, let feature, _) = e else {
                return XCTFail("Expected unsupported, got \(e)")
            }
            XCTAssertEqual(feature, "include")
            XCTAssertGreaterThan(line, 0)
            XCTAssertGreaterThan(column, 0)
        }
    }

    // MARK: - Unknown gate

    func testOpenQASM2ImporterUnknownGateFailsWithLocation() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        foo q[0];
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            guard case .semanticError(let line, let column, let message) = e else {
                return XCTFail("Expected semanticError, got \(e)")
            }
            XCTAssertTrue(message.contains("foo"), message)
            XCTAssertGreaterThan(line, 0)
            XCTAssertGreaterThan(column, 0)
            XCTAssertEqual(e.location.line, line)
            XCTAssertEqual(e.location.column, column)
        }
    }

    // MARK: - Measure mapping

    func testOpenQASM2ImporterMeasureIndexedMapping() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        measure q[0] -> c[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        guard case .measure(let spec) = circuit.gates.first else {
            return XCTFail("Expected measure")
        }
        XCTAssertEqual(spec.qubits, [0])
        XCTAssertEqual(spec.classicalRegister, 0)
        XCTAssertEqual(spec.classicalBitOffset, 0)
    }

    func testOpenQASM2ImporterMeasureIndexedBit1() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        measure q[1] -> c[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        guard case .measure(let spec) = circuit.gates.first else {
            return XCTFail("Expected measure")
        }
        XCTAssertEqual(spec.qubits, [1])
        XCTAssertEqual(spec.classicalRegister, 0)
        XCTAssertEqual(spec.classicalBitOffset, 1)
    }

    // MARK: - Parameter-free multi-qubit (cz / swap / ccx)

    func testOpenQASM2ImporterCZSwapCCX() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        cz q[0],q[1];
        swap q[1],q[2];
        ccx q[0],q[1],q[2];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .cz(control: 0, target: 1),
            .swap(q1: 1, q2: 2),
            .ccx(control1: 0, control2: 1, target: 2),
        ])
    }

    // MARK: - Multi-qreg linearization

    func testOpenQASM2ImporterMultiQregLinearAddressing() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg a[2];
        qreg b[1];
        x a[1];
        h b[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.qubitCount, 3)
        XCTAssertEqual(circuit.gates, [.x(target: 1), .h(target: 2)])
    }

    // MARK: - Program import entry point

    func testOpenQASM2ImporterImportProgram() throws {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        id q[0];
        """
        var parser = try OpenQASMParser(source: source)
        let program = try parser.parse()
        let circuit = try OpenQASM2Importer().`import`(program: program)
        XCTAssertEqual(circuit.gates, [.id(target: 0)])
    }
}
