import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - while → while_c with options

    func testOpenQASM3WhileWithOptionsMapsToWhileC() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        while (c == 1) { x q[0]; }
        """
        let options = OpenQASM3ImporterOptions(defaultWhileMaxIterations: 16)
        let circuit = try OpenQASM3Importer(options: options).`import`(source: source)
        XCTAssertEqual(circuit.gates.count, 1)
        guard case .while_c(let reg, let expected, let body, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c, got \(circuit.gates[0])")
        }
        XCTAssertEqual(reg, 0)
        XCTAssertEqual(expected, 1)
        XCTAssertEqual(maxIterations, 16)
        XCTAssertEqual(body, [.x(target: 0)])
    }

    func testOpenQASM3WhileWithoutBoundErrors() {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        while (c == 1) { x q[0]; }
        """
        XCTAssertThrowsError(try OpenQASM3Importer().`import`(source: source)) { error in
            guard case OpenQASMError.unsupported(let line, _, let feature, let message) = error else {
                return XCTFail("Expected unsupported, got \(error)")
            }
            XCTAssertEqual(feature, OpenQASMUnsupportedFeature.whileUnbounded.rawValue)
            XCTAssertTrue(message.contains("maxIterations"), message)
            XCTAssertGreaterThan(line, 0)
        }
    }

    func testOpenQASM3WhilePragmaProvidesBound() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        // @quantumkit.max_while_iterations 7
        while (c == 1) { x q[0]; }
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        guard case .while_c(_, _, let body, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c")
        }
        XCTAssertEqual(maxIterations, 7)
        XCTAssertEqual(body, [.x(target: 0)])
    }

    func testOpenQASM3WhilePragmaOverridesOptions() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        // @quantumkit.max_while_iterations 3
        while (c == 0) { h q[0]; }
        """
        let options = OpenQASM3ImporterOptions(defaultWhileMaxIterations: 99)
        let circuit = try OpenQASM3Importer(options: options).`import`(source: source)
        guard case .while_c(_, let expected, _, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c")
        }
        XCTAssertEqual(expected, 0)
        XCTAssertEqual(maxIterations, 3)
    }

    func testOpenQASM3ImportProgramRequiresOptionsWithoutPragmas() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        while (c == 1) { x q[0]; }
        """
        let program = try OpenQASM.parse(source)
        XCTAssertThrowsError(try OpenQASM3Importer().`import`(program: program)) { error in
            guard case OpenQASMError.unsupported(_, _, let feature, let message) = error else {
                return XCTFail("Expected unsupported, got \(error)")
            }
            XCTAssertEqual(feature, OpenQASMUnsupportedFeature.whileUnbounded.rawValue)
            XCTAssertTrue(message.contains("maxIterations"), message)
        }
    }

    func testOpenQASM3ImportProgramWithOptionsSucceeds() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        while (c == 1) { x q[0]; }
        """
        let program = try OpenQASM.parse(source)
        let options = OpenQASM3ImporterOptions(defaultWhileMaxIterations: 12)
        let circuit = try OpenQASM3Importer(options: options).`import`(program: program)
        guard case .while_c(_, _, let body, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c")
        }
        XCTAssertEqual(maxIterations, 12)
        XCTAssertEqual(body, [.x(target: 0)])
    }

    func testOpenQASM3ImportProgramWithScannedPragmaBounds() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        // @quantumkit.max_while_iterations 9
        while (c == 1) { z q[0]; }
        """
        let program = try OpenQASM.parse(source)
        let bounds = OpenQASMWhilePragmaScanner.scan(source)
        // No options default — bounds map alone must suffice.
        let circuit = try OpenQASM3Importer().`import`(
            program: program,
            whilePragmaBounds: bounds
        )
        guard case .while_c(_, _, let body, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c")
        }
        XCTAssertEqual(maxIterations, 9)
        XCTAssertEqual(body, [.z(target: 0)])
    }

    func testOpenQASM3NestedWhileWithBounds() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        // @quantumkit.max_while_iterations 5
        while (c == 1) {
          // @quantumkit.max_while_iterations 2
          while (c == 1) { x q[0]; }
        }
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        guard case .while_c(_, _, let outerBody, let outerMax) = circuit.gates[0] else {
            return XCTFail("Expected outer while_c")
        }
        XCTAssertEqual(outerMax, 5)
        XCTAssertEqual(outerBody.count, 1)
        guard case .while_c(_, _, let innerBody, let innerMax) = outerBody[0] else {
            return XCTFail("Expected nested while_c")
        }
        XCTAssertEqual(innerMax, 2)
        XCTAssertEqual(innerBody, [.x(target: 0)])
    }

    // MARK: - Export / round-trip

    func testOpenQASM3ExportWhileCIncludesPragma() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.apply(
            .while_c(
                classicalRegister: 0,
                expectedValue: 1,
                body: [.x(target: 0)],
                maxIterations: 11
            )
        )
        let exported = try OpenQASMExporter().export(circuit)
        XCTAssertTrue(
            exported.contains("// \(OpenQASMUnsupported.whileMaxIterationsPragmaPrefix) 11"),
            exported
        )
        XCTAssertTrue(exported.contains("while (c==1)"), exported)
        XCTAssertTrue(exported.contains("x q[0];"), exported)
    }

    func testOpenQASM3WhileRoundTripViaPragma() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.apply(
            .while_c(
                classicalRegister: 0,
                expectedValue: 1,
                body: [.h(target: 0), .x(target: 0)],
                maxIterations: 4
            )
        )
        let exported = try OpenQASMExporter().export(circuit)
        let again = try OpenQASM3Importer().`import`(source: exported)
        XCTAssertEqual(again.gates, circuit.gates)
    }

    func testOpenQASM3ExportWhileNestedUnderIf() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.apply(
            .c_if(
                classicalRegister: 0,
                expectedValue: 1,
                gate: .while_c(
                    classicalRegister: 0,
                    expectedValue: 1,
                    body: [.x(target: 0)],
                    maxIterations: 5
                )
            )
        )
        let exported = try OpenQASMExporter().export(circuit)
        XCTAssertTrue(exported.contains("if(c==1) {"), exported)
        XCTAssertTrue(
            exported.contains("// \(OpenQASMUnsupported.whileMaxIterationsPragmaPrefix) 5"),
            exported
        )
        XCTAssertTrue(exported.contains("while (c==1)"), exported)
        let again = try OpenQASM3Importer().`import`(source: exported)
        XCTAssertEqual(again.gates, circuit.gates)
    }

    func testOpenQASM3RoundTripWhileNestedUnderBracedIf() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c == 1) {
          // @quantumkit.max_while_iterations 6
          while (c == 1) { h q[0]; }
        }
        """
        let first = try OpenQASM3Importer().`import`(source: source)
        guard case .c_if(_, _, let inner) = first.gates[0],
              case .while_c(_, _, let body, let maxIterations) = inner else {
            return XCTFail("Expected c_if(while_c)")
        }
        XCTAssertEqual(maxIterations, 6)
        XCTAssertEqual(body, [.h(target: 0)])

        let exported = try OpenQASMExporter().export(first)
        let second = try OpenQASM3Importer().`import`(source: exported)
        XCTAssertEqual(second.gates, first.gates)
    }

    func testOpenQASM2ExportRejectsWhileC() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.apply(
            .while_c(
                classicalRegister: 0,
                expectedValue: 1,
                body: [.x(target: 0)],
                maxIterations: 2
            )
        )
        XCTAssertThrowsError(try OpenQASM2Exporter().export(circuit)) { error in
            guard case OpenQASMError.unsupported(_, _, let feature, _) = error else {
                return XCTFail("Expected unsupported, got \(error)")
            }
            XCTAssertEqual(feature, "while_c")
        }
    }

    func testOpenQASM3WhileBodyMeasureResetBarrier() throws {
        let source = """
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        // @quantumkit.max_while_iterations 8
        while (c == 1) {
          reset q[0];
          barrier q;
          measure q[0] -> c[0];
        }
        """
        let circuit = try OpenQASM3Importer().`import`(source: source)
        guard case .while_c(_, _, let body, let maxIterations) = circuit.gates[0] else {
            return XCTFail("Expected while_c")
        }
        XCTAssertEqual(maxIterations, 8)
        XCTAssertEqual(body.count, 3)
        XCTAssertEqual(body[0], .reset(qubit: 0))
        guard case .barrier = body[1] else { return XCTFail("Expected barrier") }
        guard case .measure = body[2] else { return XCTFail("Expected measure") }
    }
}
