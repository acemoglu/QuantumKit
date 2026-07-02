import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    private func makeDensitySetup(qubitCount: Int) throws -> (DensityMatrixEngine, DensityMatrix)? {
        let engine = try DensityMatrixEngine()
        let density = try DensityMatrix(qubitCount: qubitCount, device: engine.device)
        return (engine, density)
    }

    func testLocalizedPauliXFlipOnlyOnTargetQubit() throws {
        guard let (engine, density) = try makeDensitySetup(qubitCount: 2) else { return }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)

        let noise = NoiseModel().adding(
            .pauliXFlip(probability: 1),
            for: .gateOnQubit(gate: .x, qubit: 0)
        )

        try engine.execute(circuit, on: density, noise: noise)

        let probabilities = engine.probabilities(of: density)
        // X maps |00⟩ → |10⟩; a subsequent bit-flip on qubit 0 returns |00⟩.
        XCTAssertEqual(probabilities[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 0.0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[2], 0.0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[3], 0.0, accuracy: 1e-5)
    }

    func testLocalizedNoiseDoesNotAffectUnmatchedQubits() throws {
        guard let (engine, density) = try makeDensitySetup(qubitCount: 2) else { return }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)

        let noise = NoiseModel().adding(
            .pauliXFlip(probability: 1),
            for: .gateOnQubit(gate: .x, qubit: 0)
        )

        try engine.execute(circuit, on: density, noise: noise)

        // Qubit 1 never participated in a gate and should remain in |0⟩.
        let z1 = try marginalZExpectation(density: density, engine: engine, qubit: 1)
        XCTAssertEqual(z1, 1.0, accuracy: 1e-5)
    }

    func testLocalizedCXDepolarizingOnSpecificEdge() throws {
        guard let (engine, density) = try makeDensitySetup(qubitCount: 2) else { return }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.cx(0, 1)

        let p: QFloat = 0.15
        let noise = NoiseModel().adding(
            .depolarizing(probability: p),
            for: .gateOnQubits(gate: .cx, qubits: [0, 1])
        )

        try engine.execute(circuit, on: density, noise: noise)

        let result = engine.probabilities(of: density)
        // After X(0): |10⟩. CX(0,1) maps |10⟩ → |11⟩. Correlated 2Q depolarizing on |11⟩:
        // (1 - 4p/5)|11⟩⟨11| + (4p/15)(|01⟩⟨01| + |10⟩⟨10| + |00⟩⟨00|).
        XCTAssertEqual(result[0], 4 * p / 15, accuracy: 1e-5)
        XCTAssertEqual(result[1], 4 * p / 15, accuracy: 1e-5)
        XCTAssertEqual(result[2], 4 * p / 15, accuracy: 1e-5)
        XCTAssertEqual(result[3], 1 - 4 * p / 5, accuracy: 1e-5)
    }

    func testNoiseModelFromDeviceCalibration() {
        var calibration = DeviceCalibration(qubitCount: 2, gateTime: 0.1)
        calibration[qubit: 0] = QubitCalibration(t1: 50, t2: 70, readoutError0To1: 0.02, readoutError1To0: 0.03)
        calibration[qubit: 1] = QubitCalibration(t1: 60, t2: 80, readoutError0To1: 0.01, readoutError1To0: 0.04)
        calibration.gateErrors = [
            GateCalibration(gate: .sx, qubits: [0], errorRate: 0.001),
            GateCalibration(gate: .cx, qubits: [0, 1], errorRate: 0.01),
        ]

        let noise = NoiseModel.from(calibration: calibration)

        XCTAssertTrue(noise.hasLocalizedGateNoise)
        XCTAssertEqual(noise.readoutFlip0To1, 0.02, accuracy: 1e-6)
        XCTAssertEqual(noise.readoutFlip1To0, 0.04, accuracy: 1e-6)
        XCTAssertEqual(noise.localizedRules.count, 6)
    }

    func testStatevectorBackendRejectsLocalizedGateNoise() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let noise = NoiseModel().adding(
            .pauliXFlip(probability: 0.1),
            for: .gateOnQubit(gate: .x, qubit: 0)
        )

        let state = try StateVector(qubitCount: 1)
        var rng: QuantumRNG = .seeded(1)

        XCTAssertThrowsError(
            try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
        ) { error in
            guard case QuantumEngineError.localizedNoiseRequiresDensityMatrixBackend = error else {
                return XCTFail("Expected localizedNoiseRequiresDensityMatrixBackend, got \(error)")
            }
        }
    }

    func testGlobalNoiseRemainsBackwardCompatibleOnStatevector() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        var rng: QuantumRNG = .seeded(11)
        let noise = NoiseModel(depolarizingProbability: 1)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)
    }

    private func marginalZExpectation(
        density: DensityMatrix,
        engine: DensityMatrixEngine,
        qubit: Int
    ) throws -> QFloat {
        let probabilities = engine.probabilities(of: density)
        var expectation: QFloat = 0
        for (index, probability) in probabilities.enumerated() {
            let bit = (index >> qubit) & 1
            expectation += probability * (bit == 0 ? 1 : -1)
        }
        return expectation
    }
}
