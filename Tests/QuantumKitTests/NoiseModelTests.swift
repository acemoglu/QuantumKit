import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    func testZeroDepolarizingNoisePreservesBellState() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let idealState = try StateVector(qubitCount: 2)
        var idealCircuit = try QuantumCircuit(qubitCount: 2)
        try idealCircuit.applyBellState()
        try engine.execute(idealCircuit, on: idealState)
        let idealZZ = try QuantumMeasurement.expectationZZ(state: idealState, engine: engine, qubitA: 0, qubitB: 1)

        let noisyState = try StateVector(qubitCount: 2)
        var noisyCircuit = try QuantumCircuit(qubitCount: 2)
        try noisyCircuit.applyBellState()

        var rng: QuantumRNG = .seeded(42)
        let noise = NoiseModel(depolarizingProbability: 0)
        _ = try engine.executeRNG(noisyCircuit, on: noisyState, rng: &rng, noise: noise)

        let noisyZZ = try QuantumMeasurement.expectationZZ(state: noisyState, engine: engine, qubitA: 0, qubitB: 1)
        XCTAssertEqual(noisyZZ, idealZZ, accuracy: 1e-5)
    }

    func testDepolarizingNoiseCanFlipQubitWithPauliX() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        var rng: QuantumRNG = .seeded(11)
        let noise = NoiseModel(depolarizingProbability: 1)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)
    }

    func testAmplitudeDampingResetsExcitedQubit() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        var rng: QuantumRNG = .seeded(11)
        let noise = NoiseModel(amplitudeDampingProbability: 1)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)
    }

    func testPhaseDampingMatchesPhaseFlipChannel() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // Phase damping of strength λ is exactly the phase-flip channel, which decays the
        // coherence (⟨X⟩) of |+⟩ by a factor √(1 - λ) in the ensemble average.
        let lambda: QFloat = 0.5
        let expectedMeanX = (1 - lambda).squareRoot()

        let trajectories = 4000
        var rng: QuantumRNG = .seeded(123_456)
        let noise = NoiseModel(phaseDampingProbability: lambda)

        var accumulatedX: QFloat = 0
        for _ in 0..<trajectories {
            let state = try StateVector(qubitCount: 1)
            var circuit = try QuantumCircuit(qubitCount: 1)
            try circuit.h(0)
            _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
            accumulatedX += try QuantumMeasurement.expectationX(state: state, engine: engine, qubit: 0)
        }

        let meanX = accumulatedX / QFloat(trajectories)
        XCTAssertEqual(meanX, expectedMeanX, accuracy: 0.05)
    }

    func testPhaseDampingFullStrengthRemovesCoherence() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let trajectories = 4000
        var rng: QuantumRNG = .seeded(98_765)
        let noise = NoiseModel(phaseDampingProbability: 1)

        var accumulatedX: QFloat = 0
        for _ in 0..<trajectories {
            let state = try StateVector(qubitCount: 1)
            var circuit = try QuantumCircuit(qubitCount: 1)
            try circuit.h(0)
            _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
            accumulatedX += try QuantumMeasurement.expectationX(state: state, engine: engine, qubit: 0)
        }

        let meanX = accumulatedX / QFloat(trajectories)
        XCTAssertEqual(meanX, 0, accuracy: 0.05)
    }

    func testAmplitudeDampingPreservesClassicalCorrelationOfBellState() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // On the Bell state (|00⟩ + |11⟩)/√2, full amplitude damping on qubit 0 must always
        // drive qubit 0 to |0⟩ while leaving qubit 1 in a definite basis state (|0⟩ if no jump,
        // |1⟩ if a jump occurred). The old additive kernel incorrectly left qubit 1 in a
        // coherent superposition (⟨Z₁⟩ ≈ 0); the correct σ⁻ jump keeps |⟨Z₁⟩| = 1.
        let noise = NoiseModel(amplitudeDampingProbability: 1)

        for seed in UInt64(1)...UInt64(8) {
            let state = try StateVector(qubitCount: 2)
            var circuit = try QuantumCircuit(qubitCount: 2)
            try circuit.applyBellState(control: 0, target: 1)

            var rng: QuantumRNG = .seeded(seed)
            _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

            let z0 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
            let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)

            XCTAssertEqual(z0, 1, accuracy: 1e-5, "qubit 0 must relax to |0⟩ under full amplitude damping")
            XCTAssertEqual(abs(z1), 1, accuracy: 1e-5, "qubit 1 must remain in a definite basis state")
        }
    }

    func testT1GateTimeAmplitudeDampingProbability() {
        let gateTime = QFloat(0.69314718)
        let noise = NoiseModel(t1: 1, gateTime: gateTime)
        XCTAssertTrue(noise.usesT1TimeModel)
        XCTAssertEqual(noise.effectiveAmplitudeDampingProbability, 0.5, accuracy: 1e-5)
    }

    func testT1GateTimeResetsExcitedQubit() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        var rng: QuantumRNG = .seeded(11)
        let noise = NoiseModel(t1: 1, gateTime: 10)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)
    }

    func testAsymmetricReadoutErrorFlipsDirectionally() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let oneState = try StateVector(qubitCount: 1)
        var oneCircuit = try QuantumCircuit(qubitCount: 1)
        try oneCircuit.x(0)
        try engine.execute(oneCircuit, on: oneState)

        var rngOne: QuantumRNG = .seeded(42)
        let flipOneToZero = NoiseModel(readoutFlip0To1: 0, readoutFlip1To0: 1)
        let measuredOne = try QuantumMeasurement.measureRNG(
            state: oneState,
            engine: engine,
            rng: &rngOne,
            noise: flipOneToZero
        )
        XCTAssertEqual(measuredOne, [0])

        let zeroState = try StateVector(qubitCount: 1)
        var rngZero: QuantumRNG = .seeded(42)
        let flipZeroToOne = NoiseModel(readoutFlip0To1: 1, readoutFlip1To0: 0)
        let measuredZero = try QuantumMeasurement.measureRNG(
            state: zeroState,
            engine: engine,
            rng: &rngZero,
            noise: flipZeroToOne
        )
        XCTAssertEqual(measuredZero, [1])
    }

    func testMidCircuitMeasureWithReadoutFlipPreservesCollapsedState() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)

        var rng: QuantumRNG = .seeded(123)
        let noise = NoiseModel(readoutFlip0To1: 0, readoutFlip1To0: 1)
        let execution = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        XCTAssertEqual(execution.measurementOutcomes.count, 1)
        XCTAssertEqual(execution.measurementOutcomes[0][0], 0)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        let trueBit = expectation > 0 ? 0 : 1
        XCTAssertEqual(expectation, trueBit == 0 ? 1 : -1, accuracy: 1e-5)
        if trueBit == 1 {
            XCTAssertEqual(execution.measurementOutcomes[0][0], 0)
        }
    }

    func testRunSampleCountsWithReadoutNoiseOnly() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        var rng: QuantumRNG = .seeded(7)
        let noise = NoiseModel(readoutFlip0To1: 0, readoutFlip1To0: 1)
        let counts = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 1,
            rng: &rng,
            noise: noise
        )

        XCTAssertEqual(counts.shots, 1)
        XCTAssertEqual(counts.counts[0], 1)
    }

    func testReadoutErrorFlipsClassicalOutcomeNotState() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try engine.execute(circuit, on: state)

        // Deterministic 1→0 readout flip: the classical bit must flip while the |1⟩ state is left
        // untouched. (A symmetric readoutErrorProbability of 1 is a 50/50 coin per bit, so asserting
        // a specific measured value would depend on exact RNG draws rather than on the channel.)
        var rng: QuantumRNG = .seeded(99)
        let noise = NoiseModel(readoutFlip1To0: 1)
        let measured = try QuantumMeasurement.measureRNG(state: state, engine: engine, rng: &rng, noise: noise)

        let zAfter = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(zAfter, -1, accuracy: 1e-5)
        XCTAssertEqual(measured, [0])
    }

    func testZeroNoisePreservesBellStateWithAllChannelsDisabled() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let idealState = try StateVector(qubitCount: 2)
        var idealCircuit = try QuantumCircuit(qubitCount: 2)
        try idealCircuit.applyBellState()
        try engine.execute(idealCircuit, on: idealState)
        let idealZZ = try QuantumMeasurement.expectationZZ(state: idealState, engine: engine, qubitA: 0, qubitB: 1)

        let noisyState = try StateVector(qubitCount: 2)
        var noisyCircuit = try QuantumCircuit(qubitCount: 2)
        try noisyCircuit.applyBellState()

        var rng: QuantumRNG = .seeded(42)
        let noise = NoiseModel()
        _ = try engine.executeRNG(noisyCircuit, on: noisyState, rng: &rng, noise: noise)

        let noisyZZ = try QuantumMeasurement.expectationZZ(state: noisyState, engine: engine, qubitA: 0, qubitB: 1)
        XCTAssertEqual(noisyZZ, idealZZ, accuracy: 1e-5)
    }
}
