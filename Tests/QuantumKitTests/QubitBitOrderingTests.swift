import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - QubitBitOrdering

    func testEngineLSBQubit0ExcitationFlipsIndexBit0() throws {
        // |q0⟩ excitation → index 1 under engineLSB.
        XCTAssertEqual(
            try QubitBitOrdering.engineLSB.index(fromBitstring: "10", qubitCount: 2),
            1
        )
        XCTAssertEqual(
            try QubitBitOrdering.engineLSB.bitstring(forIndex: 1, qubitCount: 2),
            "10"
        )

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 2)
        _ = try engine.execute(circuit, on: state)
        let probs = state.probabilities()
        XCTAssertEqual(probs[1], 1.0, accuracy: 1e-6)
        XCTAssertEqual(probs[0], 0, accuracy: 1e-6)
        XCTAssertEqual(probs[2], 0, accuracy: 1e-6)
        XCTAssertEqual(probs[3], 0, accuracy: 1e-6)
    }

    func testBitstringMSBTenMapsToIndexTwoAndInverse() throws {
        // MSB-first "10" on 2 qubits: left=q1=1, right=q0=0 → index 2.
        XCTAssertEqual(
            try QubitBitOrdering.bitstringMSB.index(fromBitstring: "10", qubitCount: 2),
            2
        )
        XCTAssertEqual(
            try QubitBitOrdering.bitstringMSB.bitstring(forIndex: 2, qubitCount: 2),
            "10"
        )
        XCTAssertEqual(
            try QubitBitOrdering.bitstringMSB.index(fromBitstring: "01", qubitCount: 2),
            1
        )

        let converted = try QubitBitOrdering.convertBitstring(
            "10",
            from: .bitstringMSB,
            to: .engineLSB
        )
        XCTAssertEqual(converted, "01") // q0=0, q1=1 in LSB-first writing
    }

    func testShotCountsBitstringDefaultsToMSBAndOptionalLSB() throws {
        let counts = ShotCounts(shots: 10, counts: [1: 4, 2: 6])
        let msb = counts.bitstringCounts(qubitCount: 2)
        XCTAssertEqual(msb["01"], 4) // index 1
        XCTAssertEqual(msb["10"], 6) // index 2

        let lsb = counts.bitstringCounts(qubitCount: 2, ordering: .engineLSB)
        XCTAssertEqual(lsb["10"], 4)
        XCTAssertEqual(lsb["01"], 6)
    }

    func testReadoutConfusionProductUsesEngineLSB() throws {
        // Distinct flip rates: qubit 0 (LSB) vs qubit 1.
        let matrix = try ReadoutConfusionMatrix.product(of: [
            (p01: 0.0, p10: 0.0), // qubit 0: identity
            (p01: 1.0, p10: 0.0), // qubit 1: always flip 0→1
        ])
        XCTAssertEqual(matrix.qubitCount, 2)
        // Prepared |00⟩ (index 0): measured should be |10⟩ (index 2) with probability 1.
        XCTAssertEqual(matrix.probabilities[0][2], 1.0, accuracy: 1e-5)
        // Prepared |01⟩ (index 1, qubit0=1): still flips only q1 → |11⟩ index 3.
        XCTAssertEqual(matrix.probabilities[1][3], 1.0, accuracy: 1e-5)
    }
}
