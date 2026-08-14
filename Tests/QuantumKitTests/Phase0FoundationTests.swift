import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    func testStatevectorBackendRunsUnitaryCircuit() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let backend = try StatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let result = try backend.run(circuit: circuit, options: QuantumRunOptions(seed: 42))

        XCTAssertEqual(result.metadata.method, .statevector)
        XCTAssertEqual(result.metadata.qubitCount, 2)
        XCTAssertEqual(result.metadata.gateCount, 2)
        XCTAssertEqual(result.metadata.seed, 42)
        XCTAssertEqual(result.metadata.quantumKitVersion, QuantumKitInfo.version)
        XCTAssertNotNil(result.metadata.deviceName)
        XCTAssertGreaterThan(result.metadata.wallClockNanoseconds, 0)
        XCTAssertNil(result.shotCounts)
        XCTAssertNotNil(result.execution)
    }

    func testStatevectorBackendRunsShots() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let backend = try StatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let result = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 99, shots: 256)
        )

        XCTAssertEqual(result.metadata.method, .statevector)
        XCTAssertEqual(result.shotCounts?.shots, 256)
        XCTAssertNil(result.execution)

        let bitstrings = result.shotCounts?.bitstringCounts(qubitCount: 2) ?? [:]
        XCTAssertEqual(bitstrings.keys.sorted(), ["00", "11"])
        XCTAssertEqual(bitstrings.values.reduce(0, +), 256)
        XCTAssertEqual(bitstrings["01"] ?? 0, 0)
        XCTAssertEqual(bitstrings["10"] ?? 0, 0)
    }

    func testDensityMatrixBackendRunsWithMetadata() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let backend = try DensityMatrixBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let noise = NoiseModel(depolarizingProbability: 0.01)
        let result = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(noise: noise, seed: 99)
        )

        XCTAssertEqual(result.metadata.method, .densityMatrix)
        XCTAssertEqual(result.metadata.noiseSnapshot, noise)
        XCTAssertEqual(result.metadata.seed, 99)
        XCTAssertNotNil(result.execution)
    }

    func testQuantumBackendFactoryCreatesBackends() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let statevector = try QuantumBackendFactory.makeStatevector()
        let densityMatrix = try QuantumBackendFactory.makeDensityMatrix()

        XCTAssertEqual(statevector.method, .statevector)
        XCTAssertEqual(densityMatrix.method, .densityMatrix)
    }

    func testRunSampleCountsWithoutExplicitDevice() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        var rng: QuantumRNG = .seeded(99)
        let counts = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 64,
            rng: &rng
        )

        XCTAssertEqual(counts.shots, 64)
        let bitstrings = counts.bitstringCounts(qubitCount: 2)
        XCTAssertEqual(bitstrings.keys.sorted(), ["00", "11"])
        XCTAssertEqual(bitstrings.values.reduce(0, +), 64)
    }

    func testStateVectorDefaultDeviceInitializer() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let state = try StateVector(qubitCount: 3)
        XCTAssertEqual(state.qubitCount, 3)
        XCTAssertEqual(state.stateCount, 8)
    }

    func testQuantumCircuitJSONRoundTrip() throws {
        var original = try QuantumCircuit(qubitCount: 3, classicalRegisters: [try ClassicalRegisterSpec(bitCount: 2)])
        try original.h(0)
        try original.cx(0, 1)
        try original.rz(theta: 0.25, 2)
        try original.measure(qubits: [0, 1])

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(QuantumCircuit.self, from: data)

        XCTAssertEqual(decoded.qubitCount, original.qubitCount)
        XCTAssertEqual(decoded.gates, original.gates)
    }

    func testConditionalGateJSONRoundTrip() throws {
        let gate = Gate.c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 2))
        let data = try JSONEncoder().encode(gate)
        let decoded = try JSONDecoder().decode(Gate.self, from: data)
        XCTAssertEqual(decoded, gate)
    }

    func testResultMetadataJSONRoundTrip() throws {
        let metadata = QuantumResultMetadata(
            method: .statevector,
            seed: 123,
            deviceName: "Test GPU",
            wallClockNanoseconds: 42,
            qubitCount: 4,
            gateCount: 8,
            noiseSnapshot: NoiseModel(depolarizingProbability: 0.05)
        )

        let data = try JSONEncoder().encode(metadata)
        let decoded = try JSONDecoder().decode(QuantumResultMetadata.self, from: data)
        XCTAssertEqual(decoded, metadata)
    }
}
