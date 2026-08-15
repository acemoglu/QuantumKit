import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - B19 Stabilizer / Clifford tableau backend

    func testStabilizerBellExactMatchesCPUStatevector() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let stab = StabilizerBackend()
        let sv = CPUStatevectorBackend()

        // Exact path: sample many shots and compare empirical probs to SV amplitudes.
        let seed: UInt64 = 7
        let shots = 4000
        let stabResult = try stab.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: seed, shots: shots)
        )
        let svState = try CPUStateVector(qubitCount: 2)
        _ = try CPUStatevectorEngine().execute(circuit, on: svState)
        let exact = svState.probabilities()

        let counts = try XCTUnwrap(stabResult.shotCounts?.counts)
        XCTAssertEqual(stabResult.metadata.method, .stabilizer)
        XCTAssertEqual(stabResult.metadata.deviceName, "CPU")

        for index in 0..<4 {
            let empirical = QFloat(counts[index, default: 0]) / QFloat(shots)
            XCTAssertEqual(empirical, exact[index], accuracy: 0.04)
        }
        // Bell support: only |00⟩ and |11⟩.
        XCTAssertEqual(counts[1, default: 0], 0)
        XCTAssertEqual(counts[2, default: 0], 0)
    }

    func testStabilizerGHZMatchesCPUStatevectorShots() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.cx(0, 2)

        let stab = QuantumBackendFactory.makeStabilizer()
        let shots = 5000
        let options = QuantumRunOptions(seed: 99, shots: shots)

        let stabCounts = try XCTUnwrap(stab.run(circuit: circuit, options: options).shotCounts)

        let svEngine = CPUStatevectorEngine()
        let svState = try CPUStateVector(qubitCount: 3)
        _ = try svEngine.execute(circuit, on: svState)
        let exact = svState.probabilities()

        // GHZ support only |000⟩ and |111⟩; empirical probs match SV within tol.
        XCTAssertEqual(
            stabCounts.counts[0, default: 0] + stabCounts.counts[7, default: 0],
            shots
        )
        for index in 0..<8 {
            let empirical = QFloat(stabCounts.counts[index, default: 0]) / QFloat(shots)
            XCTAssertEqual(empirical, exact[index], accuracy: 0.04)
        }
    }

    func testStabilizerGraphStateParityVsCPU() throws {
        // 3-qubit linear cluster / graph state: H on all, CZ(0,1), CZ(1,2).
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.h(1)
        try circuit.h(2)
        try circuit.apply(.cz(control: 0, target: 1))
        try circuit.apply(.cz(control: 1, target: 2))

        let stab = StabilizerBackend()
        let svEngine = CPUStatevectorEngine()
        let svState = try CPUStateVector(qubitCount: 3)
        _ = try svEngine.execute(circuit, on: svState)
        let exact = svState.probabilities()

        let shots = 6000
        let result = try stab.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 123, shots: shots)
        )
        let counts = try XCTUnwrap(result.shotCounts?.counts)
        for index in 0..<8 {
            let empirical = QFloat(counts[index, default: 0]) / QFloat(shots)
            XCTAssertEqual(empirical, exact[index], accuracy: 0.05)
        }
    }

    func testStabilizerRejectsNonCliffordT() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.apply(.t(target: 0))

        let backend = StabilizerBackend()
        XCTAssertThrowsError(try backend.run(circuit: circuit)) { error in
            guard case StabilizerError.nonCliffordGate = error else {
                return XCTFail("expected nonCliffordGate, got \(error)")
            }
        }
    }

    func testStabilizerRejectsNonCliffordRX() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: QFloat(Double.pi / 4), 0)

        XCTAssertThrowsError(try StabilizerBackend().run(circuit: circuit)) { error in
            guard case StabilizerError.nonCliffordGate = error else {
                return XCTFail("expected nonCliffordGate, got \(error)")
            }
        }
    }

    func testStabilizerRejectsNoise() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let noise = NoiseModel(depolarizingProbability: 0.01)
        XCTAssertThrowsError(
            try StabilizerBackend().run(
                circuit: circuit,
                options: QuantumRunOptions(noise: noise, seed: 1, shots: 10)
            )
        ) { error in
            guard case StabilizerError.noiseNotSupported = error else {
                return XCTFail("expected noiseNotSupported, got \(error)")
            }
        }
    }

    func testStabilizerSeededShotsReproducible() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let backend = StabilizerBackend()
        let options = QuantumRunOptions(seed: 42, shots: 512)
        let a = try XCTUnwrap(backend.run(circuit: circuit, options: options).shotCounts)
        let b = try XCTUnwrap(backend.run(circuit: circuit, options: options).shotCounts)
        XCTAssertEqual(a, b)
    }

    func testRecommendMethodDefaultsUnchangedWithoutStabilizerOptIn() throws {
        let method = try QuantumBackendFactory.recommendMethod(qubitCount: 3, noise: nil)
        XCTAssertEqual(method, .statevector)

        var bell = try QuantumCircuit(qubitCount: 2)
        try bell.h(0)
        try bell.cx(0, 1)
        // Circuit-aware path still defaults to SV when preferStabilizerWhenClifford is false.
        let circuitMethod = try QuantumBackendFactory.recommendMethod(circuit: bell)
        XCTAssertEqual(circuitMethod, .statevector)
        XCTAssertFalse(SimulationPolicy.default.preferStabilizerWhenClifford)
    }

    func testRecommendMethodOptInStabilizerWhenClifford() throws {
        var bell = try QuantumCircuit(qubitCount: 2)
        try bell.h(0)
        try bell.cx(0, 1)

        let policy = SimulationPolicy(preferStabilizerWhenClifford: true)
        let method = try QuantumBackendFactory.recommendMethod(circuit: bell, policy: policy)
        XCTAssertEqual(method, .stabilizer)

        let backend = try QuantumBackendFactory.makeRecommended(circuit: bell, policy: policy)
        XCTAssertEqual(backend.method, .stabilizer)
        XCTAssertTrue(backend is StabilizerBackend)

        // Width-only recommender still ignores the flag (no gate inspection).
        let widthOnly = try QuantumBackendFactory.recommendMethod(
            qubitCount: 2,
            noise: nil,
            policy: policy
        )
        XCTAssertEqual(widthOnly, .statevector)
    }

    func testStabilizerFactoryMethodTag() throws {
        let backend = QuantumBackendFactory.makeStabilizer()
        XCTAssertEqual(backend.method, .stabilizer)
        XCTAssertTrue(backend is StabilizerBackend)
    }

    func testStabilizerSXAndMeasureClassical() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1, classicalRegisters: [creg])
        try circuit.apply(.sx(target: 0))
        try circuit.apply(.sx(target: 0)) // SX² = X → |1⟩
        try circuit.apply(.measure(MeasureSpec(qubits: [0], classicalRegister: 0)))

        let result = try StabilizerBackend().run(circuit: circuit)
        let execution = try XCTUnwrap(result.execution)
        XCTAssertEqual(execution.measurementOutcomes, [[1]])
        XCTAssertEqual(execution.classicalMemory.value(ofRegister: 0), 1)
    }
}
