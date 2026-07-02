import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    func testEstimatorBellStateHamiltonianOnStatevectorBackend() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let hamiltonian = try Hamiltonian(
            terms: [
                PauliTerm(coefficient: 0.5, label: "Z0 Z1"),
                PauliTerm(coefficient: 1.0, label: "X0"),
            ]
        )

        let backend = try StatevectorBackend()
        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 42)
        )

        // Bell: ⟨Z0 Z1⟩ = 1, ⟨X0⟩ = 0 ⇒ ⟨H⟩ = 0.5
        XCTAssertEqual(result.value, 0.5, accuracy: 1e-5)
        XCTAssertEqual(result.metadata.method, .statevector)
        XCTAssertEqual(result.metadata.qubitCount, 2)
        XCTAssertEqual(result.metadata.gateCount, 2)
    }

    func testEstimatorBellStateHamiltonianOnDensityMatrixBackend() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let hamiltonian = try Hamiltonian(
            terms: [
                PauliTerm(coefficient: 0.5, label: "Z0 Z1"),
                PauliTerm(coefficient: 1.0, label: "X0"),
            ]
        )

        let backend = try DensityMatrixBackend()
        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 42)
        )

        XCTAssertEqual(result.value, 0.5, accuracy: 1e-4)
        XCTAssertEqual(result.metadata.method, .densityMatrix)
    }

    func testSamplerExactBellStateProbabilities() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let backend = try StatevectorBackend()
        let result = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(seed: 7)
        )

        XCTAssertEqual(result.metadata.method, .statevector)
        XCTAssertNil(result.shotCounts)
        XCTAssertEqual(result.quasiProbabilities["00"] ?? 0, 0.5, accuracy: 1e-5)
        XCTAssertEqual(result.quasiProbabilities["11"] ?? 0, 0.5, accuracy: 1e-5)
        XCTAssertEqual(result.quasiProbabilities["01"] ?? 0, 0, accuracy: 1e-5)
        XCTAssertEqual(result.quasiProbabilities["10"] ?? 0, 0, accuracy: 1e-5)
    }

    func testSamplerShotHistogramOnBellState() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let backend = try StatevectorBackend()
        let result = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(seed: 99, shots: 256)
        )

        XCTAssertEqual(result.shotCounts?.shots, 256)
        XCTAssertEqual(result.quasiProbabilities.keys.sorted(), ["00", "11"])
        XCTAssertEqual(
            result.quasiProbabilities.values.reduce(0, +),
            1,
            accuracy: 1e-5
        )
    }

    func testPauliTermSparseLabelParsing() throws {
        let term = try PauliTerm(coefficient: 1.0, label: "Z0 Z1")
        XCTAssertEqual(term.paulis, [0: .z, 1: .z])
    }
}
