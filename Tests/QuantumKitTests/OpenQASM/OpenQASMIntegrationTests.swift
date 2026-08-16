import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - QuantumCircuit+OpenQASM thin hooks

    func testQuantumCircuitInitFromOpenQASM2Bell() throws {
        let source = OpenQASMGoldenFixtures.bell_qasm2
        let circuit = try QuantumCircuit(openQASM: source)
        XCTAssertEqual(circuit.qubitCount, 2)
        XCTAssertEqual(circuit.gates.count, 3)
        guard case .h(let t) = circuit.gates[0] else {
            return XCTFail("expected h")
        }
        XCTAssertEqual(t, 0)
        guard case .cx(let c, let tgt) = circuit.gates[1] else {
            return XCTFail("expected cx")
        }
        XCTAssertEqual(c, 0)
        XCTAssertEqual(tgt, 1)
    }

    func testQuantumCircuitInitFromOpenQASM3Bell() throws {
        let circuit = try QuantumCircuit(openQASM: OpenQASMGoldenFixtures.bell_qasm3)
        XCTAssertEqual(circuit.qubitCount, 2)
        let qasm3 = try circuit.openQASM()
        XCTAssertTrue(qasm3.contains("OPENQASM 3"))
        XCTAssertTrue(qasm3.contains("qubit"))
    }

    func testQuantumCircuitOpenQASM2ExportRoundTrip() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 2,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 2)]
        )
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.measure(qubits: [0, 1], classicalRegister: 0, classicalBitOffset: 0)

        let text = try circuit.openQASM2()
        XCTAssertTrue(text.contains("OPENQASM 2.0"))
        XCTAssertTrue(text.contains("include \"qelib1.inc\""))

        let again = try QuantumCircuit(openQASM: text)
        XCTAssertEqual(again.qubitCount, circuit.qubitCount)
        XCTAssertEqual(again.classicalRegisters, circuit.classicalRegisters)
        XCTAssertEqual(again.gates, circuit.gates)
    }

    func testQuantumCircuitOpenQASM3DefaultExportHasNoInclude() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let text = try circuit.openQASM()
        XCTAssertTrue(text.hasPrefix("OPENQASM 3"))
        XCTAssertFalse(text.contains("include"))
        XCTAssertTrue(text.contains("x q[0];"))
    }

    func testQuantumCircuitOpenQASMBitOrderQ0IsLSB() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        x q[0];
        """
        let circuit = try QuantumCircuit(openQASM: source)
        XCTAssertEqual(circuit.gates, [.x(target: 0)])
    }
}
