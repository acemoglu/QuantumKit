import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Export smoke

    func testOpenQASM2ExporterEmitsVersionAndQelib1() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.apply(.h(target: 0))
        let source = try OpenQASM2Exporter().export(circuit)
        XCTAssertTrue(source.contains("OPENQASM 2.0"), source)
        XCTAssertTrue(source.contains("include \"qelib1.inc\""), source)
        XCTAssertTrue(source.contains("qreg q[1];"), source)
        XCTAssertTrue(source.contains("h q[0];"), source)
    }

    // MARK: - Round-trip: Bell

    func testOpenQASM2RoundTripBell() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
        """
        try assertOpenQASM2RoundTrip(source)
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.gates[0], .h(target: 0), "q[0] remains qubit 0")
    }

    // MARK: - Round-trip: Toffoli

    func testOpenQASM2RoundTripToffoli() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        ccx q[0],q[1],q[2];
        """
        try assertOpenQASM2RoundTrip(source)
    }

    // MARK: - Round-trip: expr angles

    func testOpenQASM2RoundTripExprAngles() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(pi/2) q[0];
        u3(pi/2,0,pi) q[0];
        """
        try assertOpenQASM2RoundTrip(source)

        // Export from constructed circuit also preserves structure.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.apply(.rx(theta: .literal(QFloat(Double.pi / 2)), target: 0))
        try circuit.apply(
            .u(
                theta: .literal(QFloat(Double.pi / 2)),
                phi: .literal(QFloat(0)),
                lambda: .literal(QFloat(Double.pi)),
                target: 0
            )
        )
        let exported = try OpenQASM2Exporter().export(circuit)
        XCTAssertTrue(exported.contains("rx("), exported)
        XCTAssertTrue(exported.contains("u3("), exported)
        try assertOpenQASM2RoundTrip(exported)
    }

    // MARK: - Round-trip: classical if

    func testOpenQASM2RoundTripClassicalIf() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        if(c==1) x q[0];
        """
        try assertOpenQASM2RoundTrip(source)
    }

    // MARK: - Round-trip: teleport-ish

    func testOpenQASM2RoundTripTeleportIsh() throws {
        // Static structure: Bell pair + measure + classically conditioned X/Z corrections.
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[2];
        h q[1];
        cx q[1],q[2];
        cx q[0],q[1];
        h q[0];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        if(c==1) z q[2];
        if(c==2) x q[2];
        if(c==3) z q[2];
        if(c==3) x q[2];
        """
        try assertOpenQASM2RoundTrip(source)
        let circuit = try OpenQASM2Importer().`import`(source: source)
        XCTAssertEqual(circuit.qubitCount, 3)
        XCTAssertEqual(circuit.gates[0], .h(target: 1))
        guard case .measure(let m0) = circuit.gates[4] else {
            return XCTFail("Expected measure")
        }
        XCTAssertEqual(m0.qubits, [0])
        XCTAssertEqual(m0.classicalBitOffset, 0)
    }

    // MARK: - Measure mapping preserved

    func testOpenQASM2RoundTripMeasureMapping() throws {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """
        try assertOpenQASM2RoundTrip(source)

        let circuit = try OpenQASM2Importer().`import`(source: source)
        guard case .measure(let a) = circuit.gates[0],
              case .measure(let b) = circuit.gates[1]
        else {
            return XCTFail("Expected two measure gates")
        }
        XCTAssertEqual(a.qubits, [0])
        XCTAssertEqual(a.classicalRegister, 0)
        XCTAssertEqual(a.classicalBitOffset, 0)
        XCTAssertEqual(b.qubits, [1])
        XCTAssertEqual(b.classicalRegister, 0)
        XCTAssertEqual(b.classicalBitOffset, 1)
    }

    // MARK: - Multiple cregs naming

    func testOpenQASM2ExporterMultipleCregsUseC0C1() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [
                try ClassicalRegisterSpec(bitCount: 1),
                try ClassicalRegisterSpec(bitCount: 2),
            ]
        )
        try circuit.apply(
            .measure(MeasureSpec(qubits: [0], classicalRegister: 1, classicalBitOffset: 1))
        )
        let source = try OpenQASM2Exporter().export(circuit)
        XCTAssertTrue(source.contains("creg c0[1];"), source)
        XCTAssertTrue(source.contains("creg c1[2];"), source)
        XCTAssertTrue(source.contains("measure q[0] -> c1[1];"), source)
        try assertOpenQASM2RoundTrip(source)
    }

    // MARK: - Unsupported export

    func testOpenQASM2ExporterUnsupportedGateThrows() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.apply(.delay(duration: 1.0, qubit: 0))
        XCTAssertThrowsError(try OpenQASM2Exporter().export(circuit)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            guard case .unsupported(_, _, let feature, _) = e else {
                return XCTFail("Expected unsupported, got \(e)")
            }
            XCTAssertEqual(feature, "delay")
        }
    }

    func testOpenQASM2ExporterSymbolicAngleThrows() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.apply(.rx(theta: Parameter("theta"), target: 0))
        XCTAssertThrowsError(try OpenQASM2Exporter().export(circuit)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            guard case .unsupported(_, _, let feature, _) = e else {
                return XCTFail("Expected unsupported, got \(e)")
            }
            XCTAssertEqual(feature, "parameter")
        }
    }

    func testOpenQASM2ExporterWhileCThrows() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.apply(
            .while_c(
                classicalRegister: 0,
                expectedValue: 1,
                body: [.x(target: 0)],
                maxIterations: 4
            )
        )
        XCTAssertThrowsError(try OpenQASM2Exporter().export(circuit)) { error in
            guard let e = error as? OpenQASMError else {
                return XCTFail("Expected OpenQASMError, got \(error)")
            }
            guard case .unsupported(_, _, let feature, _) = e else {
                return XCTFail("Expected unsupported, got \(e)")
            }
            XCTAssertEqual(feature, "while_c")
        }
    }

    // MARK: - Bit order

    func testOpenQASM2RoundTripBitOrderQ0IsQubit0() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.apply(.x(target: 0))
        let exported = try OpenQASM2Exporter().export(circuit)
        XCTAssertTrue(exported.contains("x q[0];"), exported)
        let again = try OpenQASM2Importer().`import`(source: exported)
        XCTAssertEqual(again.gates, [.x(target: 0)])
    }

    // MARK: - Helpers

    private func assertOpenQASM2RoundTrip(
        _ source: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let first = try OpenQASM2Importer().`import`(source: source)
        let exported = try OpenQASM2Exporter().export(first)
        let second = try OpenQASM2Importer().`import`(source: exported)

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
