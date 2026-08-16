import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - if(c==imm) → Gate.c_if

    func testOpenQASM2ImporterClassicalIfX() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        if(c==1) x q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 0)),
        ])
    }

    func testOpenQASM2ImporterClassicalIfCX() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[1];
        if(c==1) cx q[0],q[1];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .c_if(
                classicalRegister: 0,
                expectedValue: 1,
                gate: .cx(control: 0, target: 1)
            ),
        ])
    }

    func testOpenQASM2ImporterClassicalIfMeasure() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        if(c==1) measure q[0] -> c[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 1)
        guard case .c_if(let reg, let expected, let gate) = circuit.gates[0] else {
            return XCTFail("Expected c_if")
        }
        XCTAssertEqual(reg, 0)
        XCTAssertEqual(expected, 1)
        guard case .measure(let spec) = gate else {
            return XCTFail("Expected measure body")
        }
        XCTAssertEqual(spec.qubits, [0])
        XCTAssertEqual(spec.classicalRegister, 0)
        XCTAssertEqual(spec.classicalBitOffset, 0)
    }

    func testOpenQASM2ImporterClassicalIfResetWholeRegister() throws {
        // Prefer multiple c_if gates (one per reset qubit) for whole-register reset.
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[1];
        if(c==1) reset q;
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .c_if(classicalRegister: 0, expectedValue: 1, gate: .reset(qubit: 0)),
            .c_if(classicalRegister: 0, expectedValue: 1, gate: .reset(qubit: 1)),
        ])
    }

    func testOpenQASM2ImporterNestedClassicalIf() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        creg d[1];
        if(c==1) if(d==1) x q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .c_if(
                classicalRegister: 0,
                expectedValue: 1,
                gate: .c_if(
                    classicalRegister: 1,
                    expectedValue: 1,
                    gate: .x(target: 0)
                )
            ),
        ])
    }

    func testOpenQASM2ImporterClassicalIfUnknownCregFailsWithLocation() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        if(missing==1) x q[0];
        """
        XCTAssertThrowsError(try OpenQASM2Importer().`import`(source: source)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            guard case .semanticError(let line, let column, let message) = e else {
                return XCTFail("Expected semanticError, got \(e)")
            }
            XCTAssertTrue(message.contains("missing"), message)
            XCTAssertGreaterThan(line, 0)
            XCTAssertGreaterThan(column, 0)
            XCTAssertEqual(e.location.line, line)
            XCTAssertEqual(e.location.column, column)
        }
    }

    // MARK: - Multi-qreg linear addressing

    func testOpenQASM2ImporterMultiQregCXLinearAddressing() throws {
        // qreg a[2]; qreg b[3]; → a[0]=0, a[1]=1, b[0]=2, b[1]=3, b[2]=4
        // cx a[1], b[0] → cx(control:1, target:2)
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg a[2];
        qreg b[3];
        cx a[1],b[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.qubitCount, 5)
        XCTAssertEqual(circuit.gates, [.cx(control: 1, target: 2)])
    }

    func testOpenQASM2ImporterMultiQregFullIndexMap() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg a[2];
        qreg b[3];
        x a[0];
        x a[1];
        x b[0];
        x b[1];
        x b[2];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.qubitCount, 5)
        XCTAssertEqual(circuit.gates, [
            .x(target: 0),
            .x(target: 1),
            .x(target: 2),
            .x(target: 3),
            .x(target: 4),
        ])
    }

    // MARK: - Multi-creg indexing

    func testOpenQASM2ImporterMultiCregClassicalIfUsesDeclarationIndex() throws {
        // creg c[2]; creg d[1]; → classicalRegisters[0]/2, [1]/1; if(d==1) → index 1
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[2];
        creg d[1];
        if(d==1) x q[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.classicalRegisters.count, 2)
        XCTAssertEqual(circuit.classicalRegisters[0].bitCount, 2)
        XCTAssertEqual(circuit.classicalRegisters[1].bitCount, 1)
        XCTAssertEqual(circuit.gates, [
            .c_if(classicalRegister: 1, expectedValue: 1, gate: .x(target: 0)),
        ])
    }

    func testOpenQASM2ImporterMultiCregMeasureTargetsCorrectRegister() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        creg d[1];
        measure q[0] -> c[1];
        measure q[1] -> d[0];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 2)

        guard case .measure(let spec0) = circuit.gates[0] else {
            return XCTFail("Expected first measure")
        }
        XCTAssertEqual(spec0.qubits, [0])
        XCTAssertEqual(spec0.classicalRegister, 0)
        XCTAssertEqual(spec0.classicalBitOffset, 1)

        guard case .measure(let spec1) = circuit.gates[1] else {
            return XCTFail("Expected second measure")
        }
        XCTAssertEqual(spec1.qubits, [1])
        XCTAssertEqual(spec1.classicalRegister, 1)
        XCTAssertEqual(spec1.classicalBitOffset, 0)
    }

    func testOpenQASM2ImporterMultiQregAndCregTogether() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg a[2];
        qreg b[3];
        creg c[2];
        creg d[1];
        h a[1];
        cx a[1],b[0];
        measure b[0] -> d[0];
        if(d==1) x b[2];
        """
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.qubitCount, 5)
        XCTAssertEqual(circuit.classicalRegisters.map(\.bitCount), [2, 1])
        XCTAssertEqual(circuit.gates.count, 4)
        XCTAssertEqual(circuit.gates[0], .h(target: 1))
        XCTAssertEqual(circuit.gates[1], .cx(control: 1, target: 2))
        guard case .measure(let spec) = circuit.gates[2] else {
            return XCTFail("Expected measure")
        }
        XCTAssertEqual(spec.qubits, [2])
        XCTAssertEqual(spec.classicalRegister, 1)
        XCTAssertEqual(spec.classicalBitOffset, 0)
        XCTAssertEqual(
            circuit.gates[3],
            .c_if(classicalRegister: 1, expectedValue: 1, gate: .x(target: 4))
        )
    }
}
