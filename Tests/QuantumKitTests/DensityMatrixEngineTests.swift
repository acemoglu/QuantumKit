import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Helpers

    private func makeDensitySetup(qubitCount: Int) throws -> (DensityMatrixEngine, DensityMatrix)? {
        let engine = try DensityMatrixEngine()
        let density = try DensityMatrix(qubitCount: qubitCount)
        return (engine, density)
    }

    private func trace(of density: DensityMatrix, engine: DensityMatrixEngine) -> Double {
        engine.probabilities(of: density).reduce(0) { $0 + Double($1) }
    }

    // MARK: - Unitary baseline (Bug 3 path still correct after pooling)

    func testDensityMatrixBellStateProbabilities() throws {
        guard let (engine, density) = try makeDensitySetup(qubitCount: 2) else { return }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        try engine.execute(circuit, on: density)

        let p = engine.probabilities(of: density)
        XCTAssertEqual(p[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(p[1], 0.0, accuracy: 1e-5)
        XCTAssertEqual(p[2], 0.0, accuracy: 1e-5)
        XCTAssertEqual(p[3], 0.5, accuracy: 1e-5)
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-5)
    }

    func testDensityMatrixBareYGatePreservesTraceAndState() throws {
        // Regression for dm_right_multiply_single_qubit_dagger: applying Y to |0⟩ must yield
        // YρY† = |1⟩⟨1|. The Pauli-Y matrix is complex-asymmetric (Y = [[0,-i],[i,0]]), so a
        // right-multiply that conjugates without transposing computes ρȲ = -ρY instead of ρY†,
        // collapsing the state into a negative-trace, non-physical density matrix.
        guard let (engine, density) = try makeDensitySetup(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.y(0)

        try engine.execute(circuit, on: density)

        let p = engine.probabilities(of: density)
        // Population must be entirely in |1⟩.
        XCTAssertEqual(p[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(p[1], 1.0, accuracy: 1e-6)
        // Trace must remain exactly 1.0 (the pre-fix bug produced trace = -1.0).
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-6)
    }

    // MARK: - Depolarizing noise (Risk #1: cross-engine channel parity)

    func testDensityMatrixSingleQubitDepolarizingChannel() throws {
        // 1-qubit gate ⇒ single-qubit depolarizing: D(|1⟩⟨1|) = (1-2p/3)|1⟩⟨1| + (2p/3)|0⟩⟨0|.
        guard let (engine, density) = try makeDensitySetup(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let p: QFloat = 0.15
        var rng: QuantumRNG = .seeded(1)
        _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: NoiseModel(depolarizingProbability: p))

        let result = engine.probabilities(of: density)
        XCTAssertEqual(result[0], 2 * p / 3, accuracy: 1e-6)       // 0.10
        XCTAssertEqual(result[1], 1 - 2 * p / 3, accuracy: 1e-6)   // 0.90
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-6)
    }

    func testDensityMatrixTwoQubitDepolarizingIsCorrelated() throws {
        // 2-qubit gate ⇒ correlated 15-Pauli channel, NOT two independent single-qubit channels.
        // Starting from |00⟩, CX(0,1) leaves the state at |00⟩, so the channel acts on |00⟩⟨00|:
        //   ρ' = (1 - 4p/5)|00⟩⟨00| + (4p/15)(|01⟩⟨01| + |10⟩⟨10| + |11⟩⟨11|).
        // (Two *independent* single-qubit channels would instead give 0.81/0.09/0.09/0.01 at p=0.15,
        //  so this test fails under the old per-qubit behavior.)
        guard let (engine, density) = try makeDensitySetup(qubitCount: 2) else { return }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.cx(0, 1)

        let p: QFloat = 0.15
        var rng: QuantumRNG = .seeded(1)
        _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: NoiseModel(depolarizingProbability: p))

        let result = engine.probabilities(of: density)
        XCTAssertEqual(result[0], 1 - 4 * p / 5, accuracy: 1e-6)  // 0.88
        XCTAssertEqual(result[1], 4 * p / 15, accuracy: 1e-6)     // 0.04
        XCTAssertEqual(result[2], 4 * p / 15, accuracy: 1e-6)     // 0.04
        XCTAssertEqual(result[3], 4 * p / 15, accuracy: 1e-6)     // 0.04
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-6)
    }

    func testDensityMatrixDeepCircuitDoesNotLeakOrCrash() throws {
        // Exercises the pooled matrix-buffer path across many gates (Bug 3): a long run must stay
        // unitary (trace 1) and produce the analytic single-qubit-rotation populations.
        guard let (engine, density) = try makeDensitySetup(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        for _ in 0..<500 {
            try circuit.h(0)
            try circuit.h(0)
        }
        try circuit.x(0)

        try engine.execute(circuit, on: density)

        let p = engine.probabilities(of: density)
        XCTAssertEqual(p[0], 0.0, accuracy: 1e-4)
        XCTAssertEqual(p[1], 1.0, accuracy: 1e-4)
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-4)
    }

    // MARK: - SWAP (Bug 4a)

    func testDensityMatrixSwapMovesPopulation() throws {
        guard let (engine, density) = try makeDensitySetup(qubitCount: 2) else { return }

        // Prepare |q1=1, q0=0> (basis index 2), then SWAP → |q1=0, q0=1> (basis index 1).
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(1)
        try circuit.swap(0, 1)

        try engine.execute(circuit, on: density)

        let p = engine.probabilities(of: density)
        XCTAssertEqual(p[1], 1.0, accuracy: 1e-5)
        XCTAssertEqual(p[0] + p[2] + p[3], 0.0, accuracy: 1e-5)
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-5)
    }

    func testDensityMatrixSwapLeavesBellStateInvariant() throws {
        guard let (engine, density) = try makeDensitySetup(qubitCount: 2) else { return }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.swap(0, 1)

        try engine.execute(circuit, on: density)

        let p = engine.probabilities(of: density)
        XCTAssertEqual(p[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(p[3], 0.5, accuracy: 1e-5)
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-5)
    }

    // MARK: - reset (Bug 4b)

    func testDensityMatrixResetFromExcitedState() throws {
        guard let (engine, density) = try makeDensitySetup(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.reset(0)

        try engine.execute(circuit, on: density)

        let p = engine.probabilities(of: density)
        XCTAssertEqual(p[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(p[1], 0.0, accuracy: 1e-5)
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-5)
    }

    func testDensityMatrixResetFromSuperpositionIsTracePreserving() throws {
        // reset is a CPTP map: even from a coherent superposition it must drive the qubit to |0⟩
        // while preserving unit trace.
        guard let (engine, density) = try makeDensitySetup(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.reset(0)

        try engine.execute(circuit, on: density)

        let p = engine.probabilities(of: density)
        XCTAssertEqual(p[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(p[1], 0.0, accuracy: 1e-5)
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-5)
    }

    // MARK: - measure (Bug 4c)

    func testDensityMatrixMeasurementOfBasisStateIsDeterministic() throws {
        guard let (engine, density) = try makeDensitySetup(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.measure(0)

        var rng: QuantumRNG = .seeded(0xABCD_1234)
        let result = try engine.executeRNG(circuit, on: density, rng: &rng)

        XCTAssertEqual(result.measurementOutcomes, [[1]])
        let p = engine.probabilities(of: density)
        XCTAssertEqual(p[1], 1.0, accuracy: 1e-5)
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-5)
    }

    func testDensityMatrixMeasurementCollapsesSuperpositionToPureState() throws {
        guard let (engine, density) = try makeDensitySetup(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)

        var rng: QuantumRNG = .seeded(0x55AA_55AA)
        let result = try engine.executeRNG(circuit, on: density, rng: &rng)
        let outcome = result.measurementOutcomes[0][0]

        let p = engine.probabilities(of: density)
        // After collapse the qubit is in the pure basis state matching the recorded outcome.
        XCTAssertEqual(p[outcome], 1.0, accuracy: 1e-5)
        XCTAssertEqual(p[1 - outcome], 0.0, accuracy: 1e-5)
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-5)
    }

    func testDensityMatrixRepeatedMeasurementIsStable() throws {
        // Once collapsed, a second measurement must reproduce the first outcome (no further change).
        guard let (engine, density) = try makeDensitySetup(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)
        try circuit.measure(0)

        var rng: QuantumRNG = .seeded(0x1357_9BDF)
        let result = try engine.executeRNG(circuit, on: density, rng: &rng)

        XCTAssertEqual(result.measurementOutcomes.count, 2)
        XCTAssertEqual(result.measurementOutcomes[0], result.measurementOutcomes[1])
        XCTAssertEqual(trace(of: density, engine: engine), 1.0, accuracy: 1e-5)
    }

    func testDensityMatrixMeasurementBornStatistics() throws {
        // Sampling H|0⟩ many times must land near the Born 50/50 split.
        guard let (engine, density) = try makeDensitySetup(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)

        var rng: QuantumRNG = .seeded(0xDEAD_BEEF)
        let shots = 2000
        var ones = 0
        for _ in 0..<shots {
            density.resetToGroundState()
            let result = try engine.executeRNG(circuit, on: density, rng: &rng)
            if result.measurementOutcomes[0][0] == 1 { ones += 1 }
        }

        let frequency = Double(ones) / Double(shots)
        XCTAssertEqual(frequency, 0.5, accuracy: 0.05)
    }
}
