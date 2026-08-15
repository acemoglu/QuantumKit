import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - QWC Pauli grouping (Estimator shot path)

    func testQWCPartitionMergesZBasisTerms() throws {
        let terms = try [
            PauliTerm(coefficient: 0.5, label: "Z0"),
            PauliTerm(coefficient: -0.25, label: "Z1"),
            PauliTerm(coefficient: 1.0, label: "Z0 Z1"),
        ]
        let partition = PauliCommutingGroups.partition(terms)

        XCTAssertEqual(partition.groups.count, 1)
        XCTAssertEqual(partition.groups[0].terms.count, 3)
        XCTAssertEqual(partition.groups[0].measurementAxes, [0: .z, 1: .z])
        XCTAssertEqual(partition.identityContribution, 0)
    }

    func testQWCPartitionDoesNotMergeNonCommutingXZ() throws {
        let x = try PauliTerm(coefficient: 1.0, label: "X0")
        let z = try PauliTerm(coefficient: 1.0, label: "Z0")
        XCTAssertFalse(PauliCommutingGroups.qubitWiseCommute(x, z))

        let partition = PauliCommutingGroups.partition([x, z])
        XCTAssertEqual(partition.groups.count, 2)
        XCTAssertEqual(partition.groups[0].terms, [x])
        XCTAssertEqual(partition.groups[1].terms, [z])
    }

    func testQWCPartitionSkipsIdentityAndEmptyTerms() throws {
        let identity = PauliTerm(coefficient: 2.5, paulis: [:])
        let z = try PauliTerm(coefficient: 1.0, label: "Z0")
        let emptyList = PauliCommutingGroups.partition([PauliTerm]())
        XCTAssertTrue(emptyList.groups.isEmpty)
        XCTAssertEqual(emptyList.identityContribution, 0)

        let partition = PauliCommutingGroups.partition([identity, z])
        XCTAssertEqual(partition.groups.count, 1)
        XCTAssertEqual(partition.groups[0].terms, [z])
        XCTAssertEqual(partition.identityContribution, 2.5)
    }

    func testGroupedShotEstimatorSharesOneEnsembleForCommutingZ() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let hamiltonian = try Hamiltonian(
            terms: [
                PauliTerm(coefficient: 0.5, label: "Z0"),
                PauliTerm(coefficient: 0.5, label: "Z1"),
            ]
        )

        PauliShotEstimator.resetSamplingEnsembleCountForTests()
        let sampled = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 21),
            estimatorOptions: EstimatorOptions(shots: 2048)
        )

        XCTAssertEqual(PauliShotEstimator.samplingEnsembleCountForTests, 1)
        XCTAssertLessThan(
            PauliShotEstimator.samplingEnsembleCountForTests,
            hamiltonian.terms.count
        )

        let exact = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 21)
        )
        XCTAssertNil(exact.shots)
        XCTAssertEqual(sampled.value, exact.value, accuracy: 0.08)
    }

    func testGroupedShotEstimatorNonCommutingUsesSeparateEnsembles() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)

        let hamiltonian = try Hamiltonian(
            terms: [
                PauliTerm(coefficient: 1.0, label: "X0"),
                PauliTerm(coefficient: 1.0, label: "Z0"),
            ]
        )

        PauliShotEstimator.resetSamplingEnsembleCountForTests()
        let sampled = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 3),
            estimatorOptions: EstimatorOptions(shots: 4096)
        )

        XCTAssertEqual(PauliShotEstimator.samplingEnsembleCountForTests, 2)

        // |+⟩: ⟨X⟩=1, ⟨Z⟩=0 ⇒ ⟨H⟩=1
        let exact = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend()
        )
        XCTAssertEqual(exact.value, 1.0, accuracy: 1e-10)
        XCTAssertEqual(sampled.value, exact.value, accuracy: 0.08)
    }

    func testGroupedShotEstimatorMatchesExactAnalyticBellZZPlusZ() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        // Bell: ⟨Z0 Z1⟩=1, ⟨Z0⟩=0, ⟨Z1⟩=0
        let hamiltonian = try Hamiltonian(
            terms: [
                PauliTerm(coefficient: 1.0, label: "Z0 Z1"),
                PauliTerm(coefficient: 0.3, label: "Z0"),
                PauliTerm(coefficient: -0.2, label: "Z1"),
            ]
        )

        let exact = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend()
        )
        XCTAssertEqual(exact.value, 1.0, accuracy: 1e-10)

        PauliShotEstimator.resetSamplingEnsembleCountForTests()
        let sampled = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 99),
            estimatorOptions: EstimatorOptions(shots: 4096)
        )

        XCTAssertEqual(PauliShotEstimator.samplingEnsembleCountForTests, 1)
        XCTAssertEqual(sampled.value, exact.value, accuracy: 0.06)
    }

    func testGroupedShotEstimatorYBasisMatchesAnalytic() throws {
        // RX(−π/2)|0⟩ = |+i⟩: ⟨Y⟩ = 1. Group {Y0} uses S†H basis change.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: QFloat(-Double.pi / 2), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Y0"))

        let exact = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend()
        )
        XCTAssertEqual(exact.value, 1.0, accuracy: 1e-10)

        PauliShotEstimator.resetSamplingEnsembleCountForTests()
        let sampled = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 41),
            estimatorOptions: EstimatorOptions(shots: 4096)
        )
        XCTAssertEqual(PauliShotEstimator.samplingEnsembleCountForTests, 1)
        XCTAssertEqual(sampled.value, exact.value, accuracy: 0.06)
    }

    func testGroupedShotStandardErrorIncludesCovarianceOnBell() throws {
        // Bell: H = Z0 + Z1. Shared shots ⇒ Var(H) = 4, not 1+1=2.
        // With enough shots, empirical stderr ≈ √(4/N).
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let hamiltonian = try Hamiltonian(
            terms: [
                PauliTerm(coefficient: 1.0, label: "Z0"),
                PauliTerm(coefficient: 1.0, label: "Z1"),
            ]
        )
        let shots = 8192
        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 7),
            estimatorOptions: EstimatorOptions(shots: shots)
        )

        XCTAssertEqual(result.value, 0, accuracy: 0.08)
        let stderr = try XCTUnwrap(result.standardError)
        let expected = sqrt(4.0 / QFloat(shots))
        // Old independent fold would report √(2/N) ≈ 0.707 × expected.
        XCTAssertEqual(stderr, expected, accuracy: expected * 0.25)
        XCTAssertGreaterThan(stderr, sqrt(2.0 / QFloat(shots)) * 1.1)
    }

    func testGroupedAndSingletonPartitionEnergiesAgreeNearExact() throws {
        // Commuting Z terms (one group) vs non-commuting X+Z (two groups) both near exact.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)

        let commuting = try Hamiltonian(
            terms: [
                PauliTerm(coefficient: 0.5, label: "Z0"),
                PauliTerm(coefficient: 0.5, label: "Z0"),
            ]
        )
        // Duplicate Z0 still one QWC group; value ⟨Z⟩=0 on |+⟩.
        let exactCommute = try Estimator().run(
            circuit: circuit,
            hamiltonian: commuting,
            backend: CPUStatevectorBackend()
        )
        let shotCommute = try Estimator().run(
            circuit: circuit,
            hamiltonian: commuting,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 13),
            estimatorOptions: EstimatorOptions(shots: 4096)
        )
        XCTAssertEqual(exactCommute.value, 0, accuracy: 1e-10)
        XCTAssertEqual(shotCommute.value, exactCommute.value, accuracy: 0.08)

        let nonCommute = try Hamiltonian(
            terms: [
                PauliTerm(coefficient: 1.0, label: "X0"),
                PauliTerm(coefficient: 1.0, label: "Z0"),
            ]
        )
        let exactNC = try Estimator().run(
            circuit: circuit,
            hamiltonian: nonCommute,
            backend: CPUStatevectorBackend()
        )
        let shotNC = try Estimator().run(
            circuit: circuit,
            hamiltonian: nonCommute,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 13),
            estimatorOptions: EstimatorOptions(shots: 4096)
        )
        XCTAssertEqual(exactNC.value, 1.0, accuracy: 1e-10)
        XCTAssertEqual(shotNC.value, exactNC.value, accuracy: 0.08)
    }

    func testGroupedAndUngroupedShotEnergiesAgreeNearExact() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let hamiltonian = try Hamiltonian(
            terms: [
                PauliTerm(coefficient: 0.5, label: "Z0"),
                PauliTerm(coefficient: 0.5, label: "Z1"),
                PauliTerm(coefficient: 1.0, label: "Z0 Z1"),
            ]
        )
        let backend = CPUStatevectorBackend()
        let exact = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend
        )
        XCTAssertEqual(exact.value, 1.0, accuracy: 1e-10)

        PauliShotEstimator.resetSamplingEnsembleCountForTests()
        let grouped = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 31),
            estimatorOptions: EstimatorOptions(shots: 4096, groupCommutingPaulis: true)
        )
        XCTAssertEqual(PauliShotEstimator.samplingEnsembleCountForTests, 1)

        PauliShotEstimator.resetSamplingEnsembleCountForTests()
        let ungrouped = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 47),
            estimatorOptions: EstimatorOptions(shots: 4096, groupCommutingPaulis: false)
        )
        XCTAssertEqual(PauliShotEstimator.samplingEnsembleCountForTests, 3)

        XCTAssertEqual(grouped.value, exact.value, accuracy: 0.08)
        XCTAssertEqual(ungrouped.value, exact.value, accuracy: 0.08)
    }

    func testPartitionSingletonsKeepsIdentityOutOfGroups() throws {
        let identity = PauliTerm(coefficient: 1.5, paulis: [:])
        let z = try PauliTerm(coefficient: 1.0, label: "Z0")
        let x = try PauliTerm(coefficient: -0.5, label: "X0")
        let partition = PauliCommutingGroups.partitionSingletons([identity, z, x])
        XCTAssertEqual(partition.identityContribution, 1.5)
        XCTAssertEqual(partition.groups.count, 2)
        XCTAssertEqual(partition.groups[0].terms, [z])
        XCTAssertEqual(partition.groups[1].terms, [x])
    }
}
