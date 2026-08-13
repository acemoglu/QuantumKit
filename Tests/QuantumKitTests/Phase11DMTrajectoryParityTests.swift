import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - C11 DM ↔ trajectory parity (shared global channels)

    /// Global depolarizing / AD / PD (and a few shared point-noise extremes) must agree:
    /// density-matrix exact CPTP ↔ state-vector Monte-Carlo unraveling ensemble.
    /// Localized / coherent / crosstalk / `.dephasingOnly` remain DM-only (rejection covered elsewhere).

    func testParitySingleQubitDepolarizingDMVersusTrajectories() throws {
        guard let device = makeDevice() else {
            throw XCTSkip("Metal device unavailable")
        }

        let p: QFloat = 0.15
        let noise = NoiseModel(depolarizingProbability: p)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let dmProbs = try densityProbabilities(circuit: circuit, noise: noise, seed: 1)
        // Analytic: D(|1⟩⟨1|) = (1-2p/3)|1⟩⟨1| + (2p/3)|0⟩⟨0|
        XCTAssertEqual(dmProbs[0], 2 * p / 3, accuracy: 1e-5)
        XCTAssertEqual(dmProbs[1], 1 - 2 * p / 3, accuracy: 1e-5)

        let svProbs = try averageTrajectoryProbabilities(
            circuit: circuit,
            noise: noise,
            device: device,
            trajectories: 4000,
            seed: 42
        )
        XCTAssertEqual(svProbs[0], dmProbs[0], accuracy: 0.04)
        XCTAssertEqual(svProbs[1], dmProbs[1], accuracy: 0.04)
    }

    func testParityTwoQubitCorrelatedDepolarizingDMVersusTrajectories() throws {
        guard let device = makeDevice() else {
            throw XCTSkip("Metal device unavailable")
        }

        let p: QFloat = 0.15
        let noise = NoiseModel(depolarizingProbability: p)

        // CX on |00⟩ stays |00⟩; channel acts on |00⟩⟨00|.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.cx(0, 1)

        let dmProbs = try densityProbabilities(circuit: circuit, noise: noise, seed: 2)
        XCTAssertEqual(dmProbs[0], 1 - 4 * p / 5, accuracy: 1e-5)
        XCTAssertEqual(dmProbs[1], 4 * p / 15, accuracy: 1e-5)
        XCTAssertEqual(dmProbs[2], 4 * p / 15, accuracy: 1e-5)
        XCTAssertEqual(dmProbs[3], 4 * p / 15, accuracy: 1e-5)

        let svProbs = try averageTrajectoryProbabilities(
            circuit: circuit,
            noise: noise,
            device: device,
            trajectories: 6000,
            seed: 99
        )
        for index in 0..<4 {
            XCTAssertEqual(svProbs[index], dmProbs[index], accuracy: 0.05)
        }
    }

    func testParityAmplitudeDampingDMVersusTrajectories() throws {
        guard let device = makeDevice() else {
            throw XCTSkip("Metal device unavailable")
        }

        let gamma: QFloat = 0.3
        let noise = NoiseModel(amplitudeDampingProbability: gamma)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let dmProbs = try densityProbabilities(circuit: circuit, noise: noise, seed: 3)
        // AD on |1⟩: P(|0⟩)=γ, P(|1⟩)=1-γ
        XCTAssertEqual(dmProbs[0], gamma, accuracy: 1e-5)
        XCTAssertEqual(dmProbs[1], 1 - gamma, accuracy: 1e-5)

        let svProbs = try averageTrajectoryProbabilities(
            circuit: circuit,
            noise: noise,
            device: device,
            trajectories: 4000,
            seed: 7
        )
        XCTAssertEqual(svProbs[0], dmProbs[0], accuracy: 0.04)
        XCTAssertEqual(svProbs[1], dmProbs[1], accuracy: 0.04)
    }

    func testParityPhaseDampingDMVersusTrajectories() throws {
        guard let device = makeDevice() else {
            throw XCTSkip("Metal device unavailable")
        }

        let lambda: QFloat = 0.5
        let noise = NoiseModel(phaseDampingProbability: lambda)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)

        let dmEngine = try DensityMatrixEngine()
        let density = try DensityMatrix(qubitCount: 1, device: dmEngine.device)
        var dmRNG: QuantumRNG = .seeded(4)
        _ = try dmEngine.executeRNG(circuit, on: density, rng: &dmRNG, noise: noise)
        let dmX = try QuantumMeasurement.expectationPauli(
            density: density,
            paulis: [0: .x],
            engine: dmEngine
        )
        let expected = (1 - lambda).squareRoot()
        XCTAssertEqual(dmX, expected, accuracy: 1e-4)

        let svEngine = try QuantumEngine()
        var svRNG: QuantumRNG = .seeded(123_456)
        var accumulatedX: QFloat = 0
        let trajectories = 4000
        for _ in 0..<trajectories {
            let state = try StateVector(qubitCount: 1, device: device)
            _ = try svEngine.executeRNG(circuit, on: state, rng: &svRNG, noise: noise)
            accumulatedX += try QuantumMeasurement.expectationX(state: state, engine: svEngine, qubit: 0)
        }
        let meanX = accumulatedX / QFloat(trajectories)
        XCTAssertEqual(meanX, dmX, accuracy: 0.05)
    }

    func testParityFullAmplitudeDampingBothEngines() throws {
        guard let device = makeDevice() else {
            throw XCTSkip("Metal device unavailable")
        }

        let noise = NoiseModel(amplitudeDampingProbability: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let dmProbs = try densityProbabilities(circuit: circuit, noise: noise, seed: 5)
        XCTAssertEqual(dmProbs[0], 1, accuracy: 1e-5)

        let svEngine = try QuantumEngine()
        let state = try StateVector(qubitCount: 1, device: device)
        var rng: QuantumRNG = .seeded(11)
        _ = try svEngine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
        let z = try QuantumMeasurement.expectationZ(state: state, engine: svEngine, qubit: 0)
        XCTAssertEqual(z, 1, accuracy: 1e-5)
    }

    func testParityResetErrorBothEngines() throws {
        guard let device = makeDevice() else {
            throw XCTSkip("Metal device unavailable")
        }

        let noise = NoiseModel(resetErrorProbability: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.reset(0)

        let dmProbs = try densityProbabilities(circuit: circuit, noise: noise, seed: 6)
        XCTAssertEqual(dmProbs[1], 1, accuracy: 1e-5)

        let svEngine = try QuantumEngine()
        let state = try StateVector(qubitCount: 1, device: device)
        var rng: QuantumRNG = .seeded(8)
        _ = try svEngine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
        let z = try QuantumMeasurement.expectationZ(state: state, engine: svEngine, qubit: 0)
        XCTAssertEqual(z, -1, accuracy: 1e-5)
    }

    func testParityIdleDelayThermalBothEngines() throws {
        guard let device = makeDevice() else {
            throw XCTSkip("Metal device unavailable")
        }

        let noise = NoiseModel(t1: 1, thermalRelaxationOnDelay: true)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.delay(duration: 10, 0)

        let dmProbs = try densityProbabilities(circuit: circuit, noise: noise, seed: 9)
        XCTAssertEqual(dmProbs[0], 1, accuracy: 1e-4)

        let svEngine = try QuantumEngine()
        let state = try StateVector(qubitCount: 1, device: device)
        var rng: QuantumRNG = .seeded(10)
        _ = try svEngine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
        let z = try QuantumMeasurement.expectationZ(state: state, engine: svEngine, qubit: 0)
        XCTAssertEqual(z, 1, accuracy: 1e-4)
    }

    func testParityDocumentsLocalizedChannelsAsDensityMatrixOnly() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let noise = NoiseModel().adding(
            .pauliXFlip(probability: 0.1),
            for: .gate(.x)
        )

        var rng: QuantumRNG = .seeded(1)
        XCTAssertThrowsError(
            try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
        ) { error in
            guard case QuantumEngineError.localizedNoiseRequiresDensityMatrixBackend = error else {
                return XCTFail("Expected localizedNoiseRequiresDensityMatrixBackend, got \(error)")
            }
        }
    }

    // MARK: - C11 helpers

    private func densityProbabilities(
        circuit: QuantumCircuit,
        noise: NoiseModel,
        seed: UInt64
    ) throws -> [QFloat] {
        let engine = try DensityMatrixEngine()
        let density = try DensityMatrix(qubitCount: circuit.qubitCount, device: engine.device)
        var rng: QuantumRNG = .seeded(seed)
        _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: noise)
        return engine.probabilities(of: density)
    }

    private func averageTrajectoryProbabilities(
        circuit: QuantumCircuit,
        noise: NoiseModel,
        device: MTLDevice,
        trajectories: Int,
        seed: UInt64
    ) throws -> [QFloat] {
        let engine = try QuantumEngine()
        var rng: QuantumRNG = .seeded(seed)
        let dim = 1 << circuit.qubitCount
        var sums = Array(repeating: QFloat(0), count: dim)

        for _ in 0..<trajectories {
            let state = try StateVector(qubitCount: circuit.qubitCount, device: device)
            _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
            let probs = try QuantumMeasurement.probabilities(state: state, engine: engine)
            for index in 0..<dim {
                sums[index] += probs[index]
            }
        }

        let scale = QFloat(trajectories)
        return sums.map { $0 / scale }
    }
}
