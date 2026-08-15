import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - B20 Trotterized Hamiltonian evolution

    func testTrotterEmptyHamiltonianIsIdentity() throws {
        let circuit = try TrotterEvolution.circuit(
            hamiltonian: Hamiltonian(terms: []),
            time: 1.25,
            steps: 4,
            qubitCount: 2
        )
        XCTAssertEqual(circuit.qubitCount, 2)
        XCTAssertTrue(circuit.gates.isEmpty)
    }

    func testTrotterZeroTimeIsIdentity() throws {
        let h = try Hamiltonian(PauliTerm(coefficient: 1, label: "X0"))
        let circuit = try TrotterEvolution.circuit(
            hamiltonian: h,
            time: 0,
            steps: 3,
            qubitCount: 1
        )
        XCTAssertTrue(circuit.gates.isEmpty)
    }

    func testTrotterInvalidStepsThrows() throws {
        let h = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        XCTAssertThrowsError(
            try TrotterEvolution.circuit(
                hamiltonian: h,
                time: 1,
                steps: 0,
                qubitCount: 1
            )
        ) { error in
            guard case TrotterError.invalidStepCount(0) = error else {
                return XCTFail("expected invalidStepCount, got \(error)")
            }
        }
        XCTAssertThrowsError(
            try TrotterEvolution.circuit(
                hamiltonian: h,
                time: 1,
                steps: -2,
                qubitCount: 1
            )
        ) { error in
            guard case TrotterError.invalidStepCount = error else {
                return XCTFail("expected invalidStepCount, got \(error)")
            }
        }
    }

    func testTrotterSingleXExactExpectation() throws {
        // H = X: exp(-i t X)|0⟩ ⇒ ⟨Z⟩ = cos(2t). Single term ⇒ exact for any r.
        let t = QFloat(0.37)
        let h = try Hamiltonian(PauliTerm(coefficient: 1, label: "X0"))
        let observable = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let backend = CPUStatevectorBackend()
        let expected = cos(2 * t)

        for steps in [1, 2, 7] {
            let result = try TrotterEvolution.expectation(
                hamiltonian: h,
                time: t,
                steps: steps,
                observable: observable,
                backend: backend,
                order: .first
            )
            XCTAssertEqual(result.value, expected, accuracy: 1e-5)
        }
    }

    func testTrotterSingleXXExactExpectation() throws {
        // H = XX: exp(-i t XX)|00⟩ = cos(t)|00⟩ − i sin(t)|11⟩ ⇒ ⟨Z0⟩ = cos(2t).
        let t = QFloat(0.41)
        let h = try Hamiltonian(PauliTerm(coefficient: 1, label: "X0 X1"))
        let observable = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let backend = CPUStatevectorBackend()
        let expected = cos(2 * t)

        let first = try TrotterEvolution.expectation(
            hamiltonian: h,
            time: t,
            steps: 1,
            observable: observable,
            backend: backend,
            order: .first
        )
        let second = try TrotterEvolution.expectation(
            hamiltonian: h,
            time: t,
            steps: 3,
            observable: observable,
            backend: backend,
            order: .second
        )
        XCTAssertEqual(first.value, expected, accuracy: 1e-5)
        XCTAssertEqual(second.value, expected, accuracy: 1e-5)
    }

    func testTrotterXZPlusNoncommutingConvergesWithSteps() throws {
        // H = X + Z on one qubit. Exact ⟨Z⟩(t) = 1/2 + 1/2 cos(2√2 t) from |0⟩.
        let t = QFloat(0.55)
        let exact = QFloat(0.5) + QFloat(0.5) * cos(2 * sqrt(2) * t)
        let h = try Hamiltonian(
            PauliTerm(coefficient: 1, label: "X0"),
            PauliTerm(coefficient: 1, label: "Z0")
        )
        let observable = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let backend = CPUStatevectorBackend()

        let coarse = try TrotterEvolution.expectation(
            hamiltonian: h,
            time: t,
            steps: 2,
            observable: observable,
            backend: backend,
            order: .first
        )
        let fine = try TrotterEvolution.expectation(
            hamiltonian: h,
            time: t,
            steps: 64,
            observable: observable,
            backend: backend,
            order: .first
        )
        let coarseErr = abs(coarse.value - exact)
        let fineErr = abs(fine.value - exact)
        XCTAssertLessThan(fineErr, coarseErr)
        XCTAssertEqual(fine.value, exact, accuracy: 2e-3)
    }

    func testTrotterOrder2ImprovesOverOrder1ForFixedSteps() throws {
        // Longer time so the O(t³/r²) vs O(t²/r) asymptotics show up at modest r.
        let t = QFloat(1.4)
        let exact = QFloat(0.5) + QFloat(0.5) * cos(2 * sqrt(2) * t)
        let h = try Hamiltonian(
            PauliTerm(coefficient: 1, label: "X0"),
            PauliTerm(coefficient: 1, label: "Z0")
        )
        let observable = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let backend = CPUStatevectorBackend()
        let steps = 20

        let o1 = try TrotterEvolution.expectation(
            hamiltonian: h,
            time: t,
            steps: steps,
            observable: observable,
            backend: backend,
            order: .first
        )
        let o2 = try TrotterEvolution.expectation(
            hamiltonian: h,
            time: t,
            steps: steps,
            observable: observable,
            backend: backend,
            order: .second
        )
        XCTAssertLessThan(abs(o2.value - exact), abs(o1.value - exact))
        XCTAssertEqual(o2.value, exact, accuracy: 5e-3)
    }

    func testTrotterThreeBodyLadderSynthesizes() throws {
        // H = Z0 Z1 Z2 (weight-3): single term ⇒ exact; ⟨ZZZ⟩ on |000⟩ stays 1 under exp(-i t ZZZ).
        let h = try Hamiltonian(PauliTerm(coefficient: 0.3, label: "Z0 Z1 Z2"))
        let circuit = try TrotterEvolution.circuit(
            hamiltonian: h,
            time: 0.5,
            steps: 1,
            qubitCount: 3
        )
        XCTAssertFalse(circuit.gates.isEmpty)
        // Ladder uses CX + RZ (no native 3-body gate).
        XCTAssertTrue(circuit.gates.contains { if case .cx = $0 { return true }; return false })
        XCTAssertTrue(circuit.gates.contains { if case .rz = $0 { return true }; return false })

        let observable = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0 Z1 Z2"))
        let result = try TrotterEvolution.expectation(
            hamiltonian: h,
            time: 0.5,
            steps: 1,
            observable: observable,
            backend: CPUStatevectorBackend()
        )
        XCTAssertEqual(result.value, 1, accuracy: 1e-10)
    }

    func testTrotterAppendOntoProductState() throws {
        // Prepare |+⟩ then evolve under H=Z: ⟨X⟩ = cos(2t).
        let t = QFloat(0.22)
        var initial = try QuantumCircuit(qubitCount: 1)
        try initial.h(0)
        let h = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let observable = try Hamiltonian(PauliTerm(coefficient: 1, label: "X0"))

        let result = try TrotterEvolution.expectation(
            hamiltonian: h,
            time: t,
            steps: 1,
            observable: observable,
            backend: CPUStatevectorBackend(),
            initial: initial
        )
        XCTAssertEqual(result.value, cos(2 * t), accuracy: 1e-5)
    }

    func testTrotterGateSequenceWrapsCircuit() throws {
        let h = try Hamiltonian(PauliTerm(coefficient: 1, label: "Y0"))
        let sequence = try TrotterEvolution.gateSequence(
            hamiltonian: h,
            time: 0.1,
            steps: 2,
            qubitCount: 1,
            order: .second
        )
        XCTAssertEqual(sequence.name, "trotter")
        XCTAssertEqual(sequence.qubitCount, 1)
        XCTAssertFalse(sequence.body.gates.isEmpty)
    }
}
