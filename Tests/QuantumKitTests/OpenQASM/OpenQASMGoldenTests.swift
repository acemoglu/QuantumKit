import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Import fixtures

    func testOpenQASMGoldenImportBellQASM2() throws {
        let circuit = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.bell_qasm2)
        XCTAssertEqual(circuit.qubitCount, 2)
        XCTAssertEqual(circuit.gates[0], .h(target: 0))
        XCTAssertEqual(circuit.gates[1], .cx(control: 0, target: 1))
    }

    func testOpenQASMGoldenImportBellQASM3() throws {
        let circuit = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.bell_qasm3)
        XCTAssertEqual(circuit.qubitCount, 2)
        XCTAssertEqual(circuit.gates[0], .h(target: 0))
        XCTAssertEqual(circuit.gates[1], .cx(control: 0, target: 1))
    }

    func testOpenQASMGoldenImportToffoliQASM2() throws {
        let circuit = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.toffoli_qasm2)
        XCTAssertEqual(
            circuit.gates,
            [.ccx(control1: 0, control2: 1, target: 2)]
        )
    }

    func testOpenQASMGoldenImportTeleportIshQASM2() throws {
        let circuit = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.teleport_ish_qasm2)
        XCTAssertEqual(circuit.qubitCount, 3)
        XCTAssertEqual(circuit.gates[0], .h(target: 1))
        XCTAssertTrue(
            circuit.gates.contains {
                if case .c_if = $0 { return true }
                return false
            }
        )
    }

    func testOpenQASMGoldenImportParametricAnglesQASM2() throws {
        let circuit = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.parametric_angles_qasm2)
        XCTAssertEqual(circuit.gates.count, 2)
        guard case .rx = circuit.gates[0] else {
            return XCTFail("Expected rx, got \(circuit.gates[0])")
        }
        guard case .u = circuit.gates[1] else {
            return XCTFail("Expected u, got \(circuit.gates[1])")
        }
    }

    func testOpenQASMGoldenImportWhileBoundedViaPragma() throws {
        let circuit = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.while_bounded_qasm3)
        guard case .while_c(_, _, let body, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c, got \(circuit.gates[0])")
        }
        XCTAssertEqual(maxIterations, 8)
        XCTAssertEqual(body, [.x(target: 0)])
    }

    func testOpenQASMGoldenImportBracedIfQASM3() throws {
        let circuit = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.braced_if_qasm3)
        XCTAssertEqual(circuit.gates, [
            .c_if(classicalRegister: 0, expectedValue: 1, gate: .h(target: 0)),
            .c_if(
                classicalRegister: 0,
                expectedValue: 1,
                gate: .cx(control: 0, target: 1)
            ),
        ])
    }

    func testOpenQASMGoldenImportWhileUnderIfQASM3() throws {
        let circuit = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.while_under_if_qasm3)
        XCTAssertEqual(circuit.gates.count, 1)
        guard case .c_if(let reg, let expected, let inner) = circuit.gates[0] else {
            return XCTFail("Expected c_if")
        }
        XCTAssertEqual(reg, 0)
        XCTAssertEqual(expected, 1)
        guard case .while_c(_, _, let body, let maxIterations) = inner else {
            return XCTFail("Expected while_c under if")
        }
        XCTAssertEqual(maxIterations, 5)
        XCTAssertEqual(body, [.x(target: 0)])
    }

    func testOpenQASMGoldenImportChCyMcxQASM2() throws {
        let circuit = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.ch_cy_mcx_qasm2)
        // ch → 7 gates, cy → 3, mcx(3 controls) → 1
        XCTAssertEqual(circuit.gates.count, 11)
        XCTAssertEqual(circuit.gates[0], .s(target: 1))
        XCTAssertEqual(circuit.gates[7], .sdg(target: 2))
        XCTAssertEqual(circuit.gates[10], .mcx(controls: [0, 1, 2], target: 3))
    }

    func testOpenQASMGoldenImportWhileBoundedViaOptions() throws {
        // Same body without relying on pragma — options path.
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        while (c == 1) { x q[0]; }
        """
        let circuit = try OpenQASM.importCircuit(
            source,
            qasm3: OpenQASM3ImporterOptions(defaultWhileMaxIterations: 8)
        )
        guard case .while_c(_, _, _, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c")
        }
        XCTAssertEqual(maxIterations, 8)
    }

    // MARK: - Round-trip (static fixtures)

    func testOpenQASMGoldenRoundTripBellQASM2() throws {
        try assertGoldenQASM2RoundTrip(OpenQASMGoldenFixtures.bell_qasm2)
    }

    func testOpenQASMGoldenRoundTripToffoliQASM2() throws {
        try assertGoldenQASM2RoundTrip(OpenQASMGoldenFixtures.toffoli_qasm2)
    }

    func testOpenQASMGoldenRoundTripTeleportIshQASM2() throws {
        try assertGoldenQASM2RoundTrip(OpenQASMGoldenFixtures.teleport_ish_qasm2)
    }

    func testOpenQASMGoldenRoundTripParametricAnglesQASM2() throws {
        try assertGoldenQASM2RoundTrip(OpenQASMGoldenFixtures.parametric_angles_qasm2)
    }

    func testOpenQASMGoldenRoundTripBellQASM3() throws {
        try assertGoldenQASM3RoundTrip(OpenQASMGoldenFixtures.bell_qasm3)
    }

    func testOpenQASMGoldenRoundTripWhileBoundedQASM3() throws {
        let first = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.while_bounded_qasm3)
        let exported = try OpenQASM.export(first)
        let second = try OpenQASM.importCircuit(exported)
        XCTAssertEqual(second.gates, first.gates)
        XCTAssertTrue(
            exported.contains("// \(OpenQASMUnsupported.whileMaxIterationsPragmaPrefix) 8"),
            exported
        )
    }

    func testOpenQASMGoldenRoundTripBracedIfQASM3() throws {
        try assertGoldenQASM3RoundTrip(OpenQASMGoldenFixtures.braced_if_qasm3)
        let first = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.braced_if_qasm3)
        let exported = try OpenQASM.export(first)
        XCTAssertTrue(exported.contains("if(c==1) {"), exported)
    }

    func testOpenQASMGoldenRoundTripWhileUnderIfQASM3() throws {
        let first = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.while_under_if_qasm3)
        let exported = try OpenQASM.export(first)
        let second = try OpenQASM.importCircuit(exported)
        XCTAssertEqual(second.gates, first.gates)
        XCTAssertTrue(exported.contains("if(c==1) {"), exported)
        XCTAssertTrue(exported.contains("while (c==1)"), exported)
        XCTAssertTrue(
            exported.contains("// \(OpenQASMUnsupported.whileMaxIterationsPragmaPrefix) 5"),
            exported
        )
    }

    // MARK: - Bit-order lock

    func testOpenQASMGoldenBitOrderXQ0IsGateX0() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        x q[0];
        """
        let circuit = try OpenQASM.importCircuit(source)
        XCTAssertEqual(circuit.gates, [.x(target: 0)], "q[0] must map to qubit 0 (LSB)")
    }

    // MARK: - detectVersion on fixtures

    func testOpenQASMGoldenDetectVersion() throws {
        XCTAssertEqual(try OpenQASM.detectVersion(from: OpenQASMGoldenFixtures.bell_qasm2), .v2)
        XCTAssertEqual(try OpenQASM.detectVersion(from: OpenQASMGoldenFixtures.bell_qasm3), .v3)
        XCTAssertEqual(try OpenQASM.detectVersion(from: OpenQASMGoldenFixtures.toffoli_qasm2), .v2)
        XCTAssertEqual(try OpenQASM.detectVersion(from: OpenQASMGoldenFixtures.teleport_ish_qasm2), .v2)
        XCTAssertEqual(try OpenQASM.detectVersion(from: OpenQASMGoldenFixtures.parametric_angles_qasm2), .v2)
        XCTAssertEqual(try OpenQASM.detectVersion(from: OpenQASMGoldenFixtures.while_bounded_qasm3), .v3)
        XCTAssertEqual(try OpenQASM.detectVersion(from: OpenQASMGoldenFixtures.braced_if_qasm3), .v3)
        XCTAssertEqual(try OpenQASM.detectVersion(from: OpenQASMGoldenFixtures.while_under_if_qasm3), .v3)
        XCTAssertEqual(try OpenQASM.detectVersion(from: OpenQASMGoldenFixtures.ch_cy_mcx_qasm2), .v2)
    }

    // MARK: - Facade smoke

    func testOpenQASMGoldenFacadeParseImportExportSmoke() throws {
        let program = try OpenQASM.parse(OpenQASMGoldenFixtures.bell_qasm2)
        XCTAssertEqual(program.version, .v2)
        XCTAssertFalse(program.statements.isEmpty)

        let circuit = try OpenQASM.importCircuit(OpenQASMGoldenFixtures.bell_qasm2)
        let qasm3 = try OpenQASM.export(circuit)
        XCTAssertTrue(qasm3.contains("OPENQASM 3.0"), qasm3)
        let qasm2 = try OpenQASM.exportQASM2(circuit)
        XCTAssertTrue(qasm2.contains("OPENQASM 2.0"), qasm2)
        XCTAssertTrue(qasm2.contains("include \"qelib1.inc\""), qasm2)
    }

    // MARK: - Helpers

    private func assertGoldenQASM2RoundTrip(
        _ source: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let first = try OpenQASM.importCircuit(source)
        let exported = try OpenQASM.exportQASM2(first)
        let second = try OpenQASM.importCircuit(exported)
        XCTAssertEqual(second.qubitCount, first.qubitCount, file: file, line: line)
        XCTAssertEqual(second.classicalRegisters, first.classicalRegisters, file: file, line: line)
        XCTAssertEqual(
            second.gates,
            first.gates,
            "round-trip mismatch.\nexported:\n\(exported)",
            file: file,
            line: line
        )
    }

    private func assertGoldenQASM3RoundTrip(
        _ source: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let first = try OpenQASM.importCircuit(source)
        let exported = try OpenQASM.export(first, options: OpenQASMExportOptions(version: .v3))
        let second = try OpenQASM.importCircuit(exported)
        XCTAssertTrue(exported.contains("OPENQASM 3"), "exported:\n\(exported)", file: file, line: line)
        XCTAssertEqual(second.gates, first.gates, "round-trip mismatch.\nexported:\n\(exported)", file: file, line: line)
    }
}
