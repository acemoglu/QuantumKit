import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - DAG circuit IR

    func testDAGCircuitRoundTripPreservesExecutableSemantics() throws {
        var original = try QuantumCircuit(qubitCount: 2)
        try original.h(0)
        try original.apply(.id(target: 1), metadata: InstructionMetadata(label: "idle"))
        try original.cx(0, 1)
        try original.apply(.x(target: 1), metadata: InstructionMetadata(label: "flip"))

        let dag = try DAGCircuit(circuit: original)
        XCTAssertEqual(dag.nodeCount, original.gates.count)

        let restored = try dag.toQuantumCircuit()
        XCTAssertEqual(restored.gates, original.gates)
        XCTAssertEqual(restored.metadata(at: 1)?.label, "idle")
        XCTAssertEqual(restored.metadata(at: 3)?.label, "flip")

        let engine = CPUStatevectorEngine()
        let a = try CPUStateVector(qubitCount: 2)
        let b = try CPUStateVector(qubitCount: 2)
        _ = try engine.execute(original, on: a)
        _ = try engine.execute(restored, on: b)
        let pa = a.probabilities()
        let pb = b.probabilities()
        for index in 0..<4 {
            XCTAssertEqual(pa[index], pb[index], accuracy: 1e-6)
        }
    }

    func testIdleIdentityRemovalPassRemovesIdKeepsEmptyBarrier() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.apply(.id(target: 0))
        try circuit.apply(.barrier(qubits: []))
        try circuit.cx(0, 1)
        try circuit.apply(.id(target: 1))

        let manager = PassManager.dag([IdleIdentityRemovalPass()])
        let optimized = try manager.run(on: circuit)

        XCTAssertEqual(
            optimized.gates,
            [.h(target: 0), .barrier(qubits: []), .cx(control: 0, target: 1)]
        )

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 2)
        _ = try engine.execute(optimized, on: state)
        let probs = state.probabilities()
        XCTAssertEqual(probs[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(probs[3], 0.5, accuracy: 1e-5)
    }

    func testEmptyBarrierIsFullWidthDependencyNotIdle() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.apply(.barrier(qubits: []))
        try circuit.x(1)

        let dag = try DAGCircuit(circuit: circuit)
        let ordered = dag.nodes.values.sorted { $0.sequence < $1.sequence }
        XCTAssertEqual(ordered.count, 3)
        XCTAssertFalse(DAGCircuit.isIdleIdentity(.barrier(qubits: [])))
        XCTAssertEqual(
            DAGCircuit.dependencyQubits(for: .barrier(qubits: []), qubitCount: 2),
            [0, 1]
        )
        XCTAssertTrue(dag.hasEdge(from: ordered[0].id, to: ordered[1].id))
        XCTAssertTrue(dag.hasEdge(from: ordered[1].id, to: ordered[2].id))

        let out = try IdleIdentityRemovalPass().run(on: circuit)
        XCTAssertEqual(out.gates, circuit.gates)
    }

    func testGateListPassManagerSmokeStillWorks() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.h(0)

        // Existing gate-list path (AlgebraicOptimizationPass), not DAG-native.
        let manager = PassManager(passes: [AlgebraicOptimizationPass()])
        let out = try manager.run(on: circuit)
        XCTAssertLessThan(out.gates.count, circuit.gates.count)

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 1)
        _ = try engine.execute(out, on: state)
        XCTAssertEqual(state.probabilities()[0], 1.0, accuracy: 1e-5)
    }

    func testNonEmptyBarrierSurvivesIdleIdentityRemoval() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.barrier([0])
        try circuit.x(0)

        let out = try IdleIdentityRemovalPass().run(on: circuit)
        XCTAssertEqual(out.gates.count, 3)
        XCTAssertEqual(out.gates[1], .barrier(qubits: [0]))
    }

    func testDAGMeasureWAWSameClassicalRegisterDifferentQubits() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.measure(qubits: [0], classicalRegister: 0, classicalBitOffset: 0)
        try circuit.measure(qubits: [1], classicalRegister: 0, classicalBitOffset: 0)

        let dag = try DAGCircuit(circuit: circuit)
        let ordered = dag.nodes.values.sorted { $0.sequence < $1.sequence }
        XCTAssertEqual(ordered.count, 2)
        XCTAssertEqual(DAGCircuit.classicalWrites(for: ordered[0].gate), [0])
        XCTAssertEqual(DAGCircuit.classicalWrites(for: ordered[1].gate), [0])
        XCTAssertTrue(
            dag.hasEdge(from: ordered[0].id, to: ordered[1].id),
            "WAW: two measures into the same creg must be ordered even on disjoint qubits"
        )
    }

    func testDAGCIfThenMeasureWARDisjointQubits() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.apply(.c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 0)))
        try circuit.measure(qubits: [1], classicalRegister: 0, classicalBitOffset: 0)

        let dag = try DAGCircuit(circuit: circuit)
        let ordered = dag.nodes.values.sorted { $0.sequence < $1.sequence }
        XCTAssertEqual(DAGCircuit.classicalReads(for: ordered[0].gate), [0])
        XCTAssertEqual(DAGCircuit.classicalWrites(for: ordered[1].gate), [0])
        XCTAssertTrue(
            dag.hasEdge(from: ordered[0].id, to: ordered[1].id),
            "WAR: measure write cannot float before a prior c_if read of the same creg"
        )
    }

    func testDAGNestedCIfMeasureWriteVisibleToLaterReader() throws {
        let cregs = [try ClassicalRegisterSpec(bitCount: 1), try ClassicalRegisterSpec(bitCount: 1)]
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: cregs)
        try circuit.apply(
            .c_if(
                classicalRegister: 0,
                expectedValue: 1,
                gate: .measure(MeasureSpec(qubits: [0], classicalRegister: 1, classicalBitOffset: 0))
            )
        )
        try circuit.apply(.c_if(classicalRegister: 1, expectedValue: 1, gate: .x(target: 1)))

        XCTAssertEqual(
            DAGCircuit.classicalWrites(for: circuit.gates[0]),
            [1],
            "inner measure write must surface on the outer c_if node"
        )
        XCTAssertEqual(DAGCircuit.classicalReads(for: circuit.gates[0]), [0])
        XCTAssertEqual(DAGCircuit.classicalReads(for: circuit.gates[1]), [1])

        let dag = try DAGCircuit(circuit: circuit)
        let ordered = dag.nodes.values.sorted { $0.sequence < $1.sequence }
        XCTAssertTrue(
            dag.hasEdge(from: ordered[0].id, to: ordered[1].id),
            "RAW: later c_if on creg 1 must depend on the nested measure write"
        )
    }
}
