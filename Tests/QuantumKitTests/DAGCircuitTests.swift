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

    func testIdleIdentityRemovalPassRemovesIdAndEmptyBarrier() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.apply(.id(target: 0))
        try circuit.apply(.barrier(qubits: []))
        try circuit.cx(0, 1)
        try circuit.apply(.id(target: 1))

        let beforeCount = circuit.gates.count
        XCTAssertEqual(beforeCount, 5)

        let manager = PassManager.dag([IdleIdentityRemovalPass()])
        let optimized = try manager.run(on: circuit)

        XCTAssertEqual(optimized.gates, [.h(target: 0), .cx(control: 0, target: 1)])
        XCTAssertLessThan(optimized.gates.count, beforeCount)

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 2)
        _ = try engine.execute(optimized, on: state)
        let probs = state.probabilities()
        XCTAssertEqual(probs[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(probs[3], 0.5, accuracy: 1e-5)
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
}
