import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Helpers

    private func makeDensitySetup(qubitCount: Int) throws -> (DensityMatrixEngine, DensityMatrix)? {
        let engine = try DensityMatrixEngine()
        let density = try DensityMatrix(qubitCount: qubitCount, device: engine.device)
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
