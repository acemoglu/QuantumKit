import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - C6 coherent over-rotation / unitary error

    func testCoherentOverRotationFlipsAfterX() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        // Ideal X maps |0⟩→|1⟩; coherent RX(π) after X returns |0⟩.
        let noise = NoiseModel().adding(
            .coherentOverRotation(axis: .x, angle: .pi),
            for: .gate(.x)
        )

        try engine.execute(circuit, on: density, noise: noise)
        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 0.0, accuracy: 1e-5)
    }

    func testCoherentUnitaryErrorAtCertaintyMatchesOverRotation() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let noise = NoiseModel().adding(
            .coherentUnitaryError(axis: .x, angle: .pi, probability: 1),
            for: .gateOnQubit(gate: .x, qubit: 0)
        )

        try engine.execute(circuit, on: density, noise: noise)
        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 1.0, accuracy: 1e-5)
    }

    func testCoherentUnitaryErrorMixtureOnGroundState() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        // Z leaves |0⟩ fixed; coherent X(π) with probability p → computational mixture.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.z(0)

        let p: QFloat = 0.25
        let noise = NoiseModel().adding(
            .coherentUnitaryError(axis: .x, angle: .pi, probability: p),
            for: .gate(.z)
        )

        try engine.execute(circuit, on: density, noise: noise)
        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 1 - p, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], p, accuracy: 1e-5)
    }

    // MARK: - C9 reset / preparation noise

    func testResetErrorProbabilityFlipsPreparedZero() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.reset(0)

        let noise = NoiseModel(resetErrorProbability: 1)
        try engine.execute(circuit, on: density, noise: noise)

        let probabilities = engine.probabilities(of: density)
        // Ideal reset → |0⟩; p=1 bit-flip → |1⟩.
        XCTAssertEqual(probabilities[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 1.0, accuracy: 1e-5)
    }

    func testLocalizedResetPreparationBitFlip() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.reset(0)

        let noise = NoiseModel().adding(
            .preparationBitFlip(probability: 1),
            for: .gate(.reset)
        )
        try engine.execute(circuit, on: density, noise: noise)

        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[1], 1.0, accuracy: 1e-5)
    }

    func testPreparationErrorOnInitialize() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.initialize(qubits: [0], amplitudes: [
            ComplexAmplitude(real: 1, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
        ])

        let noise = NoiseModel(preparationErrorProbability: 1)
        try engine.execute(circuit, on: density, noise: noise)

        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 1.0, accuracy: 1e-5)
    }

    func testStatevectorHonorsGlobalResetErrorWithoutLocalizedRules() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.reset(0)

        var rng: QuantumRNG = .seeded(1)
        let noise = NoiseModel(resetErrorProbability: 1)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5)
    }

    // MARK: - C7 noise at circuit points

    func testBarrierPointNoiseAppliesLocalizedChannel() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.barrier([0])

        let noise = NoiseModel().adding(
            .pauliXFlip(probability: 1),
            for: .gate(.barrier)
        )
        try engine.execute(circuit, on: density, noise: noise)

        // X → |1⟩; barrier X-flip → |0⟩.
        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 1.0, accuracy: 1e-5)
    }

    func testDelayPointNoiseAppliesLocalizedChannel() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.delay(duration: 1.0, 0)

        let noise = NoiseModel().adding(
            .pauliXFlip(probability: 1),
            for: .gateOnQubit(gate: .delay, qubit: 0)
        )
        try engine.execute(circuit, on: density, noise: noise)

        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 1.0, accuracy: 1e-5)
    }

    func testCircuitIndexNoiseTargetsSpecificInstruction() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0) // index 0 — attach X-flip here
        try circuit.z(0) // index 1

        let noise = NoiseModel().adding(
            .pauliXFlip(probability: 1),
            for: .circuitIndex(0)
        )
        try engine.execute(circuit, on: density, noise: noise)

        // Ideal X·Z |0⟩ → |1⟩; X-flip after index 0 restores |0⟩ before Z.
        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 0.0, accuracy: 1e-5)
    }

    func testCircuitIndexDoesNotMatchOtherInstructions() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.z(0)

        let noise = NoiseModel().adding(
            .pauliXFlip(probability: 1),
            for: .circuitIndex(1)
        )
        try engine.execute(circuit, on: density, noise: noise)

        // Flip after Z leaves |1⟩ unchanged under Z's channel (X after |1⟩ → |0⟩).
        // X|0⟩=|1⟩, Z|1⟩=−|1⟩, X-flip → |0⟩.
        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 1.0, accuracy: 1e-5)
    }

    func testHasPreparationNoiseSelectsDensityMatrixPolicy() throws {
        let noise = NoiseModel(resetErrorProbability: 0.01)
        XCTAssertTrue(noise.hasPreparationNoise)
        XCTAssertTrue(noise.hasAnyChannel)
        let method = try QuantumBackendFactory.recommendMethod(qubitCount: 2, noise: noise)
        XCTAssertEqual(method, .densityMatrix)
    }

    // MARK: - C8 idle / delay thermal relaxation

    func testDelayThermalRelaxationResetsExcitedState() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.delay(duration: 10, 0)

        let noise = NoiseModel(t1: 1, thermalRelaxationOnDelay: true)
        XCTAssertTrue(noise.hasIdleNoise)
        XCTAssertEqual(noise.amplitudeDampingProbability(forDuration: 10), 1 - exp(-10), accuracy: 1e-5)

        try engine.execute(circuit, on: density, noise: noise)
        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 1.0, accuracy: 1e-4)
        XCTAssertEqual(probabilities[1], 0.0, accuracy: 1e-4)
    }

    func testDelayWithoutIdleFlagIsNoOp() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.delay(duration: 10, 0)

        let noise = NoiseModel(t1: 1, thermalRelaxationOnDelay: false)
        XCTAssertFalse(noise.hasIdleNoise)

        try engine.execute(circuit, on: density, noise: noise)
        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[1], 1.0, accuracy: 1e-5)
    }

    func testLocalizedIdleThermalRelaxationUsesDelayDuration() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.delay(duration: 10, 0)

        let noise = NoiseModel().adding(
            .idleThermalRelaxation(t1: 1, t2: 0),
            for: .gate(.delay)
        )
        try engine.execute(circuit, on: density, noise: noise)

        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 1.0, accuracy: 1e-4)
    }

    func testLongerDelayDampsMoreThanShortDelay() throws {
        guard let (engine, shortDensity) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }
        let longDensity = try DensityMatrix(qubitCount: 1, device: engine.device)

        var shortCircuit = try QuantumCircuit(qubitCount: 1)
        try shortCircuit.x(0)
        try shortCircuit.delay(duration: 0.2, 0)

        var longCircuit = try QuantumCircuit(qubitCount: 1)
        try longCircuit.x(0)
        try longCircuit.delay(duration: 2.0, 0)

        let noise = NoiseModel(t1: 1, thermalRelaxationOnDelay: true)
        try engine.execute(shortCircuit, on: shortDensity, noise: noise)
        try engine.execute(longCircuit, on: longDensity, noise: noise)

        let shortP1 = engine.probabilities(of: shortDensity)[1]
        let longP1 = engine.probabilities(of: longDensity)[1]
        XCTAssertGreaterThan(shortP1, longP1)
        XCTAssertEqual(shortP1, exp(-0.2), accuracy: 1e-4)
        XCTAssertEqual(longP1, exp(-2.0), accuracy: 1e-4)
    }

    func testIdleNoiseSelectsDensityMatrixPolicy() throws {
        let noise = NoiseModel(t1: 50, t2: 70, thermalRelaxationOnDelay: true)
        XCTAssertTrue(noise.hasIdleNoise)
        let method = try QuantumBackendFactory.recommendMethod(qubitCount: 2, noise: noise)
        XCTAssertEqual(method, .densityMatrix)
    }

    // MARK: - C10 measurement-induced dephasing / NDM

    func testDephasingOnlyMeasureLeavesMixedState() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)

        let noise = NoiseModel(measurementMode: .dephasingOnly)
        var rng: QuantumRNG = .seeded(7)
        _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: noise)

        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[0], 0.5, accuracy: 1e-4)
        XCTAssertEqual(probabilities[1], 0.5, accuracy: 1e-4)
    }

    func testProjectiveMeasureCollapsesSuperposition() throws {
        guard let (engine, density) = try makeDensitySetupForPointNoise(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)

        var rng: QuantumRNG = .seeded(7)
        _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: nil)

        let probabilities = engine.probabilities(of: density)
        let collapsed =
            (abs(probabilities[0] - 1) < 1e-4 && abs(probabilities[1]) < 1e-4) ||
            (abs(probabilities[1] - 1) < 1e-4 && abs(probabilities[0]) < 1e-4)
        XCTAssertTrue(collapsed, "expected a computational-basis projector, got \(probabilities)")
    }

    func testMeasurementDephasingProbabilityIsGateChannel() {
        let noise = NoiseModel(measurementDephasingProbability: 0.3)
        XCTAssertTrue(noise.hasMeasurementChannelNoise)
        XCTAssertTrue(noise.hasAnyChannel)
    }

    func testStatevectorRejectsDephasingOnlyMeasure() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)

        var rng: QuantumRNG = .seeded(1)
        let noise = NoiseModel(measurementMode: .dephasingOnly)
        XCTAssertThrowsError(
            try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
        ) { error in
            guard case QuantumEngineError.nonProjectiveMeasurementRequiresDensityMatrixBackend = error else {
                return XCTFail("Expected nonProjectiveMeasurementRequiresDensityMatrixBackend, got \(error)")
            }
        }
    }

    func testDephasingOnlySelectsDensityMatrixPolicy() throws {
        let noise = NoiseModel(measurementMode: .dephasingOnly)
        let method = try QuantumBackendFactory.recommendMethod(qubitCount: 2, noise: noise)
        XCTAssertEqual(method, .densityMatrix)
    }

    private func makeDensitySetupForPointNoise(
        qubitCount: Int
    ) throws -> (DensityMatrixEngine, DensityMatrix)? {
        let engine = try DensityMatrixEngine()
        let density = try DensityMatrix(qubitCount: qubitCount, device: engine.device)
        return (engine, density)
    }
}
