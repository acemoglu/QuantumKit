import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - G5 / A11 / A12 new gates

    func testIdentityAndBarrierAreStructuralNoOps() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var withMeta = try QuantumCircuit(qubitCount: 1)
        try withMeta.h(0)
        try withMeta.id(0)
        try withMeta.barrier([0])
        try withMeta.delay(duration: 1.0, 0)

        var plain = try QuantumCircuit(qubitCount: 1)
        try plain.h(0)

        let engine = try QuantumEngine()
        let a = try StateVector(qubitCount: 1)
        let b = try StateVector(qubitCount: 1)
        _ = try engine.execute(withMeta, on: a)
        _ = try engine.execute(plain, on: b)

        XCTAssertEqual(
            try QuantumMeasurement.amplitudes(state: a),
            try QuantumMeasurement.amplitudes(state: b)
        )
    }

    func testRZZDecompositionPreservesUnitary() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let theta = QFloat(Double.pi / 3)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.rzz(theta: theta, 0, 1)

        let expanded = try GateDecomposition.expand(.rzz(theta: QFloatExpr(theta), q1: 0, q2: 1))
        var decomposed = try QuantumCircuit(qubitCount: 2)
        for gate in expanded {
            try decomposed.apply(gate)
        }

        XCTAssertTrue(try CircuitUnitary.areEquivalent(circuit, decomposed, tolerance: 1e-4))
    }

    func testCSWAPAndDCXBuildersRoundTripThroughExecution() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.x(0)
        try circuit.x(1)
        try circuit.cswap(control: 0, 1, 2)
        try circuit.dcx(1, 2)

        let backend = try StatevectorBackend()
        let result = try backend.run(circuit: circuit, options: QuantumRunOptions(seed: 1))
        XCTAssertEqual(result.metadata.gateCount, 4)
    }

    func testISWAPDecomposesToBasis() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.iswap(0, 1)

        let transpiled = try Transpiler.transpile(circuit, targetBasis: .ibmEagle)
        for gate in transpiled.gates {
            XCTAssertNotNil(BasisGateKind(gate: gate), "Unexpected gate \(gate)")
        }
    }

    // MARK: - A6 algebraic pass

    func testAlgebraicOptimizationPassCancelsHH() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.h(0)

        let optimized = try AlgebraicOptimizationPass().run(on: circuit)
        XCTAssertTrue(optimized.gates.isEmpty)
    }

    // MARK: - A8 MCX 3+

    func testMCXThreeControlsExpandsAndRuns() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.x(0)
        try circuit.x(1)
        try circuit.x(2)
        try circuit.mcx(controls: [0, 1, 2], target: 3)

        let transpiled = try Transpiler.transpile(circuit, targetBasis: .ibmEagle)
        XCTAssertFalse(transpiled.gates.isEmpty)
        for gate in transpiled.gates {
            XCTAssertNotNil(BasisGateKind(gate: gate))
        }

        let backend = try StatevectorBackend()
        _ = try backend.run(circuit: transpiled, options: QuantumRunOptions(seed: 2))
    }

    // MARK: - A9 pre-validation

    func testPreTranspileValidationRejectsWideCircuit() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)

        let map = try CouplingMap.linear(2)
        XCTAssertThrowsError(
            try PreTranspileValidationPass(couplingMap: map).run(on: circuit)
        ) { error in
            guard case TranspilerError.circuitWiderThanDevice = error else {
                return XCTFail("Expected circuitWiderThanDevice, got \(error)")
            }
        }
    }

    func testPreTranspileValidationRejectsUnboundParameters() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)

        XCTAssertThrowsError(
            try PreTranspileValidationPass().run(on: circuit)
        ) { error in
            guard case TranspilerError.unboundParameters = error else {
                return XCTFail("Expected unboundParameters, got \(error)")
            }
        }
    }

    func testPreTranspileValidationRejectsMaxDepth() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.x(0)
        try circuit.z(0)

        XCTAssertThrowsError(
            try PreTranspileValidationPass(maxDepth: 2).run(on: circuit)
        ) { error in
            guard case TranspilerError.circuitExceedsMaxDepth = error else {
                return XCTFail("Expected circuitExceedsMaxDepth, got \(error)")
            }
        }
    }
}
