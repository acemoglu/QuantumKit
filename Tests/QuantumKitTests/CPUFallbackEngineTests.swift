import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - CPU fallback engines (Dilim 1)

    func testCPUStatevectorBellMatchesAnalytic() throws {
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 2)

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        _ = try engine.execute(circuit, on: state)
        let probabilities = state.probabilities()
        XCTAssertEqual(probabilities[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 0.0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[2], 0.0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[3], 0.5, accuracy: 1e-5)
    }

    func testCPUDensityMatrixDepolarizingMatchesAnalytic() throws {
        let engine = CPUDensityMatrixEngine()
        let density = try CPUDensityMatrix(qubitCount: 1)
        let p: QFloat = 0.15

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let noise = NoiseModel(depolarizingProbability: p)
        _ = try engine.execute(circuit, on: density, noise: noise)

        let probabilities = density.probabilities()
        // D(|1⟩⟨1|) = (1-2p/3)|1⟩ + (2p/3)|0⟩
        XCTAssertEqual(probabilities[0], 2 * p / 3, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 1 - 2 * p / 3, accuracy: 1e-5)
    }

    func testCPUStatevectorRejectsLocalizedNoise() throws {
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let noise = NoiseModel().adding(.pauliXFlip(probability: 0.1), for: .gate(.x))

        XCTAssertThrowsError(try engine.execute(circuit, on: state, noise: noise)) { error in
            guard case CPUEngineError.localizedNoiseRequiresDensityMatrixBackend = error else {
                return XCTFail("expected localizedNoiseRequiresDensityMatrixBackend, got \(error)")
            }
        }
    }

    func testCPUDensityMatrixLocalizedPauliX() throws {
        let engine = CPUDensityMatrixEngine()
        let density = try CPUDensityMatrix(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.z(0) // leave |0⟩
        let noise = NoiseModel().adding(.pauliXFlip(probability: 1), for: .gate(.z))
        _ = try engine.execute(circuit, on: density, noise: noise)
        let probabilities = density.probabilities()
        XCTAssertEqual(probabilities[1], 1.0, accuracy: 1e-5)
    }

    func testCPUBackendFactoryForceCPU() throws {
        let backend = try QuantumBackendFactory.makeStatevector(devicePreference: .cpu)
        XCTAssertTrue(backend is CPUStatevectorBackend)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let result = try backend.run(circuit: circuit, options: QuantumRunOptions(seed: 1))
        XCTAssertEqual(result.metadata.deviceName, "CPU")
        XCTAssertEqual(result.metadata.method, .statevector)
    }

    func testMakeRecommendedHonorsCPUPreference() throws {
        let policy = SimulationPolicy(devicePreference: .cpu)
        let backend = try QuantumBackendFactory.makeRecommended(
            qubitCount: 2,
            noise: nil,
            policy: policy
        )
        XCTAssertTrue(backend is CPUStatevectorBackend)
    }

    func testMakeRecommendedNoisySelectsCPUDensityMatrix() throws {
        let policy = SimulationPolicy(devicePreference: .cpu)
        let noise = NoiseModel(depolarizingProbability: 0.01)
        let backend = try QuantumBackendFactory.makeRecommended(
            qubitCount: 2,
            noise: noise,
            policy: policy
        )
        XCTAssertTrue(backend is CPUDensityMatrixBackend)
        XCTAssertEqual(backend.method, .densityMatrix)
    }

    func testCPUVersusMetalStatevectorParityWhenMetalAvailable() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let cpuEngine = CPUStatevectorEngine()
        let cpuState = try CPUStateVector(qubitCount: 2)
        _ = try cpuEngine.execute(circuit, on: cpuState)
        let cpuProbs = cpuState.probabilities()

        let metalEngine = try QuantumEngine()
        let metalState = try StateVector(qubitCount: 2, device: metalEngine.device)
        _ = try metalEngine.execute(circuit, on: metalState)
        let metalProbs = try QuantumMeasurement.probabilities(state: metalState, engine: metalEngine)

        for index in 0..<4 {
            XCTAssertEqual(cpuProbs[index], metalProbs[index], accuracy: 1e-4)
        }
    }

    func testCPUVersusMetalDensityMatrixParityWhenMetalAvailable() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let p: QFloat = 0.2
        let noise = NoiseModel(amplitudeDampingProbability: p)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let cpuEngine = CPUDensityMatrixEngine()
        let cpuDensity = try CPUDensityMatrix(qubitCount: 1)
        _ = try cpuEngine.execute(circuit, on: cpuDensity, noise: noise)
        let cpuProbs = cpuDensity.probabilities()

        let metalEngine = try DensityMatrixEngine()
        let metalDensity = try DensityMatrix(qubitCount: 1, device: metalEngine.device)
        _ = try metalEngine.execute(circuit, on: metalDensity, noise: noise)
        let metalProbs = metalEngine.probabilities(of: metalDensity)

        XCTAssertEqual(cpuProbs[0], metalProbs[0], accuracy: 1e-4)
        XCTAssertEqual(cpuProbs[1], metalProbs[1], accuracy: 1e-4)
        XCTAssertEqual(cpuProbs[0], p, accuracy: 1e-4)
    }

    func testMetalRuntimeAvailabilityMatchesDeviceProbe() {
        XCTAssertEqual(MetalRuntime.isAvailable, makeDevice() != nil)
    }
}
