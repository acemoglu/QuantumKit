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
        let metalState = try StateVector(qubitCount: 2)
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
        let metalDensity = try DensityMatrix(qubitCount: 1)
        _ = try metalEngine.execute(circuit, on: metalDensity, noise: noise)
        let metalProbs = metalEngine.probabilities(of: metalDensity)

        XCTAssertEqual(cpuProbs[0], metalProbs[0], accuracy: 1e-4)
        XCTAssertEqual(cpuProbs[1], metalProbs[1], accuracy: 1e-4)
        XCTAssertEqual(cpuProbs[0], p, accuracy: 1e-4)
    }

    func testMetalRuntimeAvailabilityMatchesDeviceProbe() {
        XCTAssertEqual(MetalRuntime.isAvailable, makeDevice() != nil)
    }

    func testCPUStatevectorSixQubitParityVersusDensityMatrix() throws {
        var circuit = try QuantumCircuit(qubitCount: 6)
        for qubit in 0..<6 {
            try circuit.h(qubit)
            try circuit.rz(theta: QFloat(0.17 * Double(qubit + 1)), qubit)
        }
        for qubit in 0..<5 {
            try circuit.cx(qubit, qubit + 1)
        }
        try circuit.rz(theta: QFloat(0.31), 2)
        try circuit.cx(5, 0)
        try circuit.h(3)
        XCTAssertGreaterThanOrEqual(circuit.gates.count, 20)

        let svEngine = CPUStatevectorEngine()
        let sv = try CPUStateVector(qubitCount: 6)
        _ = try svEngine.execute(circuit, on: sv)
        let svProbs = sv.probabilitiesDouble()

        let dmEngine = CPUDensityMatrixEngine()
        let dm = try CPUDensityMatrix(qubitCount: 6)
        _ = try dmEngine.execute(circuit, on: dm)
        let dmProbs = dm.probabilitiesDouble()

        XCTAssertEqual(svProbs.count, dmProbs.count)
        for index in svProbs.indices {
            XCTAssertEqual(svProbs[index], dmProbs[index], accuracy: 1e-10)
        }

        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var measured = try QuantumCircuit(qubitCount: 6, classicalRegisters: [creg])
        for gate in circuit.gates {
            try measured.apply(gate)
        }
        try measured.measure(qubits: [0], classicalRegister: 0)
        let measuredState = try CPUStateVector(qubitCount: 6)
        var rng: QuantumRNG = .seeded(7)
        _ = try svEngine.executeRNG(measured, on: measuredState, rng: &rng)
        let measuredNorm = measuredState.probabilitiesDouble().reduce(0, +)
        XCTAssertEqual(measuredNorm, 1.0, accuracy: 1e-12)
    }

    func testCPUStatevectorEightQubitCompletesWithoutFullMatrix() throws {
        var circuit = try QuantumCircuit(qubitCount: 8)
        for qubit in 0..<8 {
            try circuit.h(qubit)
        }
        for qubit in 0..<7 {
            try circuit.cx(qubit, qubit + 1)
            try circuit.rz(theta: QFloat(0.05), qubit)
        }
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 8)
        let started = DispatchTime.now()
        _ = try engine.execute(circuit, on: state)
        let elapsedNS = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        // A dense 2^8×2^8 apply-per-gate would be far slower; subspace apply should finish quickly.
        XCTAssertLessThan(elapsedNS, 2_000_000_000)
        let total = state.probabilitiesDouble().reduce(0, +)
        XCTAssertEqual(total, 1.0, accuracy: 1e-12)
    }

    func testCPUDensityMatrixCRXMatchesStatevector() throws {
        // CRX(π) with control |0⟩ must leave the target; CircuitUnitary's RZ sandwich did not.
        var idle = try QuantumCircuit(qubitCount: 2)
        try idle.crx(theta: QFloat(Double.pi), control: 0, target: 1)
        let idleDM = try CPUDensityMatrix(qubitCount: 2)
        _ = try CPUDensityMatrixEngine().execute(idle, on: idleDM)
        XCTAssertEqual(idleDM.probabilitiesDouble()[0], 1.0, accuracy: 1e-10)

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.crx(theta: QFloat(Double.pi), control: 0, target: 1)

        let sv = try CPUStateVector(qubitCount: 2)
        _ = try CPUStatevectorEngine().execute(circuit, on: sv)
        let dm = try CPUDensityMatrix(qubitCount: 2)
        _ = try CPUDensityMatrixEngine().execute(circuit, on: dm)

        let svProbs = sv.probabilitiesDouble()
        let dmProbs = dm.probabilitiesDouble()
        for index in 0..<4 {
            XCTAssertEqual(svProbs[index], dmProbs[index], accuracy: 1e-10)
        }
        XCTAssertEqual(dmProbs[3], 1.0, accuracy: 1e-10)
    }

    func testCPUDensityMatrixCPPiMatchesCZ() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.h(1)
        try circuit.cp(theta: QFloat(Double.pi), control: 0, target: 1)
        try circuit.h(1)

        let sv = try CPUStateVector(qubitCount: 2)
        _ = try CPUStatevectorEngine().execute(circuit, on: sv)
        let dm = try CPUDensityMatrix(qubitCount: 2)
        _ = try CPUDensityMatrixEngine().execute(circuit, on: dm)

        let svProbs = sv.probabilitiesDouble()
        let dmProbs = dm.probabilitiesDouble()
        for index in 0..<4 {
            XCTAssertEqual(svProbs[index], dmProbs[index], accuracy: 1e-10)
        }
        // x(0)·H(1)·CP(π)·H(1) ≡ CZ on |1⟩|+⟩ → |11⟩
        XCTAssertEqual(dmProbs[3], 1.0, accuracy: 1e-10)
    }
}
