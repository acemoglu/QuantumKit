import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Post-selection / conditioning

    func testPostSelectionSyntheticKeepsBit0ZeroAndRenormalizes() throws {
        // engineLSB: bit0 = qubit 0. Outcomes 0 (00) and 2 (10) have qubit0 = 0.
        let counts = ShotCounts(
            shots: 10,
            counts: [
                0: 3, // 00
                1: 2, // 01
                2: 4, // 10
                3: 1, // 11
            ]
        )

        let filtered = try PostSelection.filter(
            counts,
            qubitCount: 2,
            where: .bit(0, equals: 0)
        )

        XCTAssertEqual(filtered.acceptedShots, 7)
        XCTAssertEqual(filtered.discardedShots, 3)
        XCTAssertEqual(filtered.acceptanceFraction, 0.7, accuracy: 1e-12)
        XCTAssertEqual(filtered.shotCounts.shots, 7)
        XCTAssertEqual(filtered.shotCounts.counts, [0: 3, 2: 4])

        let probs = filtered.quasiProbabilities
        XCTAssertEqual(probs["00"] ?? 0, QFloat(3) / 7, accuracy: 1e-12)
        XCTAssertEqual(probs["10"] ?? 0, QFloat(4) / 7, accuracy: 1e-12)
        XCTAssertNil(probs["01"])
        XCTAssertNil(probs["11"])
        XCTAssertEqual(probs.values.reduce(0, +), 1, accuracy: 1e-12)
    }

    func testPostSelectionImpossiblePredicateAndEmptyKeepSet() throws {
        let counts = ShotCounts(shots: 4, counts: [0: 2, 1: 2])

        XCTAssertThrowsError(
            try PostSelection.filter(
                counts,
                qubitCount: 2,
                where: .bit(5, equals: 0)
            )
        ) { error in
            XCTAssertEqual(
                error as? PostSelectionError,
                .qubitOutOfRange(qubit: 5, qubitCount: 2)
            )
        }

        XCTAssertThrowsError(
            try PostSelection.filter(
                counts,
                qubitCount: 1,
                where: .bit(0, equals: 2)
            )
        ) { error in
            XCTAssertEqual(error as? PostSelectionError, .invalidBitValue(2))
        }

        // All shots have qubit0 = 0 or 1 mixed; require impossible combo via and of opposites.
        XCTAssertThrowsError(
            try PostSelection.filter(
                counts,
                qubitCount: 1,
                where: .and([.bit(0, equals: 0), .bit(0, equals: 1)])
            )
        ) { error in
            XCTAssertEqual(
                error as? PostSelectionError,
                .emptyKeepSet(discardedShots: 4)
            )
        }

        let empty = try PostSelection.filter(
            counts,
            qubitCount: 1,
            where: .and([.bit(0, equals: 0), .bit(0, equals: 1)]),
            emptyKeepSet: .emptyResult
        )
        XCTAssertEqual(empty.acceptedShots, 0)
        XCTAssertEqual(empty.shotCounts.shots, 0)
        XCTAssertTrue(empty.quasiProbabilities.isEmpty)
        XCTAssertEqual(empty.acceptanceFraction, 0, accuracy: 1e-12)
    }

    func testPostSelectionMidCircuitCPUEndToEnd() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.cx(0, 1)
        // Projective mid-circuit measure — must remain serial under ShotExecutionPolicy.
        try circuit.measure(qubits: [0], classicalRegister: 0)
        XCTAssertTrue(ShotExecutionPolicy.mustSerial(circuit: circuit))
        XCTAssertFalse(ShotExecutionPolicy.canBatch(circuit: circuit))

        let sampled = try Sampler().run(
            circuit: circuit,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 17, shots: 2048)
        )
        let raw = try XCTUnwrap(sampled.shotCounts)

        let post = try PostSelection.filter(
            sampled,
            where: .bit(0, equals: 0)
        )

        // Hand filter on engineLSB histogram must match the API.
        var handKept: [Int: Int] = [:]
        var handAccepted = 0
        for (outcome, count) in raw.counts where (outcome & 1) == 0 {
            handKept[outcome, default: 0] += count
            handAccepted += count
        }
        XCTAssertEqual(post.acceptedShots, handAccepted)
        XCTAssertEqual(post.shotCounts.counts, handKept)
        XCTAssertEqual(post.acceptedShots + post.discardedShots, 2048)

        XCTAssertEqual(post.quasiProbabilities["00"] ?? 0, 1, accuracy: 0.02)
        XCTAssertEqual(post.quasiProbabilities["11"] ?? 0, 0, accuracy: 0.02)

        let zz = try Hamiltonian(
            PauliTerm(coefficient: 1, label: "Z0"),
            PauliTerm(coefficient: 1, label: "Z1")
        )
        let conditioned = try PostSelection.expectation(from: post, hamiltonian: zz)
        XCTAssertEqual(conditioned, 2, accuracy: 0.05)
    }

    func testPostSelectionFiltersFinalQubitNotClassicalRegister() throws {
        // measure(q0→creg) then X(0): creg stores the pre-X bit; Sampler histograms the
        // final qubit. Filtering `.bit(0, …)` follows the flipped qubit, not the creg.
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1, classicalRegisters: [creg])
        // Prepare |0⟩, measure into creg (stores 0), then flip the qubit to |1⟩.
        try circuit.measure(qubits: [0], classicalRegister: 0)
        try circuit.x(0)

        let sampled = try Sampler().run(
            circuit: circuit,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 5, shots: 256)
        )
        let raw = try XCTUnwrap(sampled.shotCounts)

        // Final qubit is always 1.
        XCTAssertEqual(raw.counts[1] ?? 0, 256)
        XCTAssertEqual(raw.counts[0] ?? 0, 0)

        let keepFinalOne = try PostSelection.filter(sampled, where: .bit(0, equals: 1))
        XCTAssertEqual(keepFinalOne.acceptedShots, 256)

        // Requiring qubit bit == 0 (the classical measurement value) yields an empty keep-set,
        // proving the API does not read the creg.
        XCTAssertThrowsError(
            try PostSelection.filter(sampled, where: .bit(0, equals: 0))
        ) { error in
            XCTAssertEqual(
                error as? PostSelectionError,
                .emptyKeepSet(discardedShots: 256)
            )
        }
    }
}
