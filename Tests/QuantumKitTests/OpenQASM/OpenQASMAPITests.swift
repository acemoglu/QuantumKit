import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - detectVersion

    func testOpenQASMFacadeDetectVersionV2() throws {
        XCTAssertEqual(
            try OpenQASM.detectVersion(from: "OPENQASM 2.0;\nqreg q[1];"),
            .v2
        )
    }

    func testOpenQASMFacadeDetectVersionV3() throws {
        XCTAssertEqual(
            try OpenQASM.detectVersion(from: "OPENQASM 3.0;\nqubit[1] q;"),
            .v3
        )
        XCTAssertEqual(
            try OpenQASM.detectVersion(from: "OPENQASM 3;\nqubit q;"),
            .v3
        )
    }

    func testOpenQASMFacadeDetectVersionMissingHeaderDefaultsV2() throws {
        XCTAssertEqual(
            try OpenQASM.detectVersion(from: "qreg q[1];\nh q[0];"),
            .v2
        )
    }

    func testOpenQASMFacadeDetectVersionUnknownThrows() {
        XCTAssertThrowsError(try OpenQASM.detectVersion(from: "OPENQASM 4.0;")) { error in
            guard case OpenQASMError.unsupported(_, _, let feature, _) = error else {
                return XCTFail("Expected unsupported, got \(error)")
            }
            XCTAssertTrue(feature.contains("OPENQASM 4"), feature)
        }
    }

    // MARK: - parse

    func testOpenQASMFacadeParseReturnsProgram() throws {
        let program = try OpenQASM.parse("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
        """)
        XCTAssertEqual(program.version, .v2)
        XCTAssertGreaterThanOrEqual(program.statements.count, 3)
    }

    // MARK: - importCircuit

    func testOpenQASMFacadeImportCircuitDispatchesV2() throws {
        let circuit = try OpenQASM.importCircuit("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        x q[0];
        """)
        XCTAssertEqual(circuit.gates, [.x(target: 0)])
    }

    func testOpenQASMFacadeImportCircuitDispatchesV3() throws {
        let circuit = try OpenQASM.importCircuit("""
        OPENQASM 3.0;
        qubit[1] q;
        h q[0];
        """)
        XCTAssertEqual(circuit.gates, [.h(target: 0)])
    }

    func testOpenQASMFacadeImportCircuitQasm3OptionsWhile() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        while (c == 0) { z q[0]; }
        """
        let circuit = try OpenQASM.importCircuit(
            source,
            options: OpenQASMImporterOptions(
                v3: OpenQASM3ImporterOptions(defaultWhileMaxIterations: 4)
            )
        )
        guard case .while_c(_, let expected, let body, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c")
        }
        XCTAssertEqual(expected, 0)
        XCTAssertEqual(maxIterations, 4)
        XCTAssertEqual(body, [.z(target: 0)])
    }

    func testOpenQASMFacadeImportCircuitQasm3ConvenienceOverload() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        while (c == 1) { x q[0]; }
        """
        let circuit = try OpenQASM.importCircuit(
            source,
            qasm3: OpenQASM3ImporterOptions(defaultWhileMaxIterations: 2)
        )
        guard case .while_c(_, _, _, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c")
        }
        XCTAssertEqual(maxIterations, 2)
    }

    // MARK: - export / exportQASM2

    func testOpenQASMFacadeExportDefaultsToQASM3() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.apply(.h(target: 0))
        let source = try OpenQASM.export(circuit)
        XCTAssertTrue(source.contains("OPENQASM 3.0"), source)
        XCTAssertTrue(source.contains("qubit[1] q;"), source)
        XCTAssertFalse(source.contains("include"), source)
    }

    func testOpenQASMFacadeExportQASM2() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.apply(.h(target: 0))
        let source = try OpenQASM.exportQASM2(circuit)
        XCTAssertTrue(source.contains("OPENQASM 2.0"), source)
        XCTAssertTrue(source.contains("include \"qelib1.inc\""), source)
        XCTAssertTrue(source.contains("qreg q[1];"), source)
        XCTAssertTrue(source.contains("h q[0];"), source)
    }

    func testOpenQASMFacadeExportExplicitV2Options() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.apply(.x(target: 0))
        let source = try OpenQASM.export(circuit, options: OpenQASMExportOptions(version: .v2))
        XCTAssertTrue(source.contains("OPENQASM 2.0"), source)
        XCTAssertTrue(source.contains("x q[0];"), source)
    }

    // MARK: - Bit order documented in facade path

    func testOpenQASMFacadeBitOrderQ0MapsToQubit0() throws {
        let circuit = try OpenQASM.importCircuit("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        x q[0];
        """)
        XCTAssertEqual(circuit.gates, [.x(target: 0)])
        let exported = try OpenQASM.exportQASM2(circuit)
        XCTAssertTrue(exported.contains("x q[0];"), exported)
    }
}
