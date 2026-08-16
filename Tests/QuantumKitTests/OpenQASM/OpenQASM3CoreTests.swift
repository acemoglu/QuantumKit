import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Import: qubit / bit Bell

    func testOpenQASM3ImporterBellWithQubitBit() throws {
        let source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0],q[1];
        measure q -> c;
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)

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

    func testOpenQASM3ImporterScalarQubitBit() throws {
        let source = """
        OPENQASM 3;
        qubit q;
        bit c;
        x q;
        measure q -> c;
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        XCTAssertEqual(circuit.qubitCount, 1)
        XCTAssertEqual(circuit.classicalRegisters[0].bitCount, 1)
        XCTAssertEqual(circuit.gates[0], .x(target: 0))
    }

    // MARK: - Import: qreg / creg compatibility in v3

    func testOpenQASM3ImporterAcceptsQregCreg() throws {
        let source = """
        OPENQASM 3.0;
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        XCTAssertEqual(circuit.qubitCount, 2)
        XCTAssertEqual(circuit.gates[0], .h(target: 0))
        XCTAssertEqual(circuit.gates[1], .cx(control: 0, target: 1))
    }

    // MARK: - if → c_if

    func testOpenQASM3ImporterClassicalIfMapsToCIf() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if(c==1) x q[0];
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        XCTAssertEqual(
            circuit.gates,
            [.c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 0))]
        )
    }

    func testOpenQASM3ImporterBracedIfMultiGate() throws {
        let source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[1] c;
        if (c == 1) {
          h q[0];
          cx q[0], q[1];
        }
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .c_if(classicalRegister: 0, expectedValue: 1, gate: .h(target: 0)),
            .c_if(
                classicalRegister: 0,
                expectedValue: 1,
                gate: .cx(control: 0, target: 1)
            ),
        ])
    }

    func testOpenQASM3ImporterNestedBracedIf() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        bit[1] d;
        if (c == 1) {
          if (d == 0) {
            x q[0];
            h q[0];
          }
        }
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates, [
            .c_if(
                classicalRegister: 0,
                expectedValue: 1,
                gate: .c_if(
                    classicalRegister: 1,
                    expectedValue: 0,
                    gate: .x(target: 0)
                )
            ),
            .c_if(
                classicalRegister: 0,
                expectedValue: 1,
                gate: .c_if(
                    classicalRegister: 1,
                    expectedValue: 0,
                    gate: .h(target: 0)
                )
            ),
        ])
    }

    func testOpenQASM3ImporterBracedIfMeasureAndReset() throws {
        let source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[1] c;
        if (c == 1) {
          measure q[0] -> c[0];
          reset q[1];
        }
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 2)
        guard case .c_if(let mReg, let mExpected, let measureGate) = circuit.gates[0] else {
            return XCTFail("Expected c_if(measure)")
        }
        XCTAssertEqual(mReg, 0)
        XCTAssertEqual(mExpected, 1)
        guard case .measure(let spec) = measureGate else {
            return XCTFail("Expected measure body")
        }
        XCTAssertEqual(spec.qubits, [0])
        XCTAssertEqual(spec.classicalRegister, 0)
        XCTAssertEqual(spec.classicalBitOffset, 0)
        XCTAssertEqual(
            circuit.gates[1],
            .c_if(classicalRegister: 0, expectedValue: 1, gate: .reset(qubit: 1))
        )
    }

    func testOpenQASM3ImporterBracedIfContainingWhile() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c == 1) {
          // @quantumkit.max_while_iterations 4
          while (c == 1) { x q[0]; }
        }
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 1)
        guard case .c_if(let reg, let expected, let inner) = circuit.gates[0] else {
            return XCTFail("Expected outer c_if")
        }
        XCTAssertEqual(reg, 0)
        XCTAssertEqual(expected, 1)
        guard case .while_c(let wReg, let wExpected, let body, let maxIterations) = inner else {
            return XCTFail("Expected while_c under c_if")
        }
        XCTAssertEqual(wReg, 0)
        XCTAssertEqual(wExpected, 1)
        XCTAssertEqual(maxIterations, 4)
        XCTAssertEqual(body, [.x(target: 0)])
    }

    // MARK: - Unified importer dispatches on version

    func testOpenQASMImporterDispatchesV3() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        h q[0];
        """
        let circuit = try OpenQASMImporter().`import`(source: source)
        XCTAssertEqual(circuit.gates, [.h(target: 0)])
    }

    func testOpenQASMImporterDispatchesV2() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
        """
        let circuit = try OpenQASMImporter().`import`(source: source)
        XCTAssertEqual(circuit.gates, [.h(target: 0)])
    }

    // MARK: - Default exporter is QASM3

    func testOpenQASMExporterDefaultIsQASM3() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.apply(.h(target: 0))
        let source = try OpenQASMExporter().export(circuit)
        XCTAssertTrue(source.contains("OPENQASM 3.0"), source)
        XCTAssertTrue(source.contains("qubit[1] q;"), source)
        XCTAssertFalse(source.contains("include"), source)
        XCTAssertFalse(source.contains("qreg"), source)
        XCTAssertTrue(source.contains("h q[0];"), source)
    }

    func testOpenQASM2ExporterStillEmitsQASM2() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.apply(.h(target: 0))
        let source = try OpenQASM2Exporter().export(circuit)
        XCTAssertTrue(source.contains("OPENQASM 2.0"), source)
        XCTAssertTrue(source.contains("include \"qelib1.inc\""), source)
        XCTAssertTrue(source.contains("qreg q[1];"), source)
    }

    // MARK: - Round-trip QASM3

    func testOpenQASM3RoundTripBell() throws {
        let source = """
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0],q[1];
        measure q -> c;
        """
        try assertOpenQASM3RoundTrip(source)
    }

    func testOpenQASM3RoundTripClassicalIf() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if(c==1) x q[0];
        """
        try assertOpenQASM3RoundTrip(source)

        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.apply(.c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 0)))
        let exported = try OpenQASMExporter().export(circuit)
        XCTAssertTrue(exported.contains("OPENQASM 3.0"), exported)
        XCTAssertTrue(exported.contains("if(c==1) x q[0];"), exported)
        let again = try OpenQASM3Importer().`import`(source: exported)
        XCTAssertEqual(again.gates, circuit.gates)
    }

    // MARK: - while requires maxIterations (options or pragma)

    func testOpenQASM3ImporterWhileRequiresBound() {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        while (c == 1) { x q[0]; }
        """
        XCTAssertThrowsError(try OpenQASM3Importer().`import`(source: source)) { error in
            guard case OpenQASMError.unsupported(_, _, let feature, let message) = error else {
                return XCTFail("Expected unsupported, got \(error)")
            }
            XCTAssertEqual(feature, "while")
            XCTAssertTrue(message.contains("maxIterations"), message)
        }
    }

    func testOpenQASM3ImporterWhileWithOptionsSucceeds() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        while (c == 1) { x q[0]; }
        """
        let options = OpenQASM3ImporterOptions(defaultWhileMaxIterations: 10)
        let circuit = try OpenQASM3Importer(options: options).`import`(source: source)
        guard case .while_c(_, _, _, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c")
        }
        XCTAssertEqual(maxIterations, 10)
    }

    // MARK: - Helpers

    private func assertOpenQASM3RoundTrip(
        _ source: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let first = try OpenQASM3Importer().`import`(source: source)
        let exported = try OpenQASMExporter(options: OpenQASMExportOptions(version: .v3)).export(first)
        let second = try OpenQASM3Importer().`import`(source: exported)

        XCTAssertTrue(exported.contains("OPENQASM 3"), "exported:\n\(exported)", file: file, line: line)
        XCTAssertEqual(second.qubitCount, first.qubitCount, file: file, line: line)
        XCTAssertEqual(
            second.classicalRegisters,
            first.classicalRegisters,
            file: file,
            line: line
        )
        XCTAssertEqual(
            second.gates.count,
            first.gates.count,
            "gate count mismatch.\nexported:\n\(exported)",
            file: file,
            line: line
        )
        for (index, (lhs, rhs)) in zip(first.gates, second.gates).enumerated() {
            XCTAssertEqual(
                lhs,
                rhs,
                "gate[\(index)] mismatch.\nexported:\n\(exported)",
                file: file,
                line: line
            )
        }
    }
}
