import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - GateSequence / Subcircuit

    func testGateSequenceBellAppendedTwiceMatchesHandBuilt() throws {
        var bell = try GateSequence(name: "bell", qubitCount: 2)
        try bell.apply(.h(target: 0))
        try bell.apply(.cx(control: 0, target: 1))
        XCTAssertEqual(bell.name, "bell")

        // Same type under the Subcircuit alias.
        let asSubcircuit: Subcircuit = bell
        XCTAssertEqual(asSubcircuit.gates.count, 2)

        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.append(bell, qubitMap: [0, 1])
        try circuit.append(bell, qubitMap: [2, 3])

        var handBuilt = try QuantumCircuit(qubitCount: 4)
        try handBuilt.h(0)
        try handBuilt.cx(0, 1)
        try handBuilt.h(2)
        try handBuilt.cx(2, 3)

        XCTAssertEqual(circuit.gates, handBuilt.gates)

        let engine = CPUStatevectorEngine()
        let fromSequence = try CPUStateVector(qubitCount: 4)
        let fromHand = try CPUStateVector(qubitCount: 4)
        _ = try engine.execute(circuit, on: fromSequence)
        _ = try engine.execute(handBuilt, on: fromHand)

        let pSeq = fromSequence.probabilities()
        let pHand = fromHand.probabilities()
        XCTAssertEqual(pSeq.count, 16)
        for index in 0..<16 {
            XCTAssertEqual(pSeq[index], pHand[index], accuracy: 1e-6)
        }
        // Two independent Bell pairs on (0,1) and (2,3).
        // Qubit 0 = LSB → equal weight on |0000⟩, |0011⟩, |1100⟩, |1111⟩ → indices 0, 3, 12, 15.
        for index in [0, 3, 12, 15] {
            XCTAssertEqual(pSeq[index], 0.25, accuracy: 1e-5)
        }
    }

    func testGateSequenceAppendPreservesMetadata() throws {
        var seq = try GateSequence(name: "labeled", qubitCount: 1)
        try seq.apply(.x(target: 0), metadata: InstructionMetadata(label: "flip"))

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.append(seq)
        XCTAssertEqual(circuit.metadata(at: 0)?.label, "flip")
    }

    func testGateSequenceQubitMapOverlapThrows() throws {
        let seq = try GateSequence(name: "x", qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 4)
        XCTAssertThrowsError(try circuit.append(seq, qubitMap: [1, 1])) { error in
            guard case QuantumCircuitError.invalidComposition(let reason) = error else {
                return XCTFail("expected invalidComposition, got \(error)")
            }
            XCTAssertTrue(reason.contains("distinct") || reason.contains("overlap"))
        }
    }

    func testGateSequenceQubitMapOutOfBoundsThrows() throws {
        var seq = try GateSequence(qubitCount: 2)
        try seq.apply(.h(target: 0))

        var circuit = try QuantumCircuit(qubitCount: 3)
        XCTAssertThrowsError(try circuit.append(seq, qubitMap: [0, 3])) { error in
            guard case QuantumCircuitError.invalidComposition(let reason) = error else {
                return XCTFail("expected invalidComposition, got \(error)")
            }
            XCTAssertTrue(reason.contains("out of bounds") || reason.contains("bounds"))
        }

        // Wider sequence without a map must not silently grow the destination.
        let wide = try GateSequence(qubitCount: 4)
        XCTAssertThrowsError(try circuit.append(wide)) { error in
            guard case QuantumCircuitError.invalidComposition = error else {
                return XCTFail("expected invalidComposition, got \(error)")
            }
        }
    }

    func testGateSequenceComposeEqualsAppendFlattening() throws {
        var seq = try GateSequence(name: "rx", qubitCount: 1)
        try seq.apply(.x(target: 0))

        let base = try QuantumCircuit(qubitCount: 2)
        let composed = try base.compose(seq, qubitMap: [1])

        var appended = try QuantumCircuit(qubitCount: 2)
        try appended.append(seq, qubitMap: [1])

        XCTAssertEqual(composed.gates, appended.gates)
        XCTAssertEqual(composed.gates, [.x(target: 1)])
    }
}
