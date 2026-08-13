import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - D4 / D5

    func testPipelineHashIsStableForIdenticalInputs() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        let options = QuantumRunOptions(seed: 7, shots: 128)

        let a = PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: options)
        let b = PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: options)
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.count, 16)
    }

    func testBackendMetadataIncludesPipelineHash() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let backend = try StatevectorBackend()
        let result = try backend.run(circuit: circuit, options: QuantumRunOptions(seed: 1))
        XCTAssertNotNil(result.metadata.pipelineHash)
        XCTAssertEqual(result.metadata.pipelineHash?.count, 16)
    }

    func testShotResultExposesHexAndProbabilities() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let backend = try StatevectorBackend()
        let result = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 3, shots: 64)
        )

        XCTAssertEqual(result.hexCounts?["0x1"], 64)
        XCTAssertEqual(result.probabilities?["1"] ?? 0, 1, accuracy: 1e-6)
    }

    func testClassicalMemorySlotsAndHex() throws {
        var memory = ClassicalMemory(registerWidths: [4, 2])
        try memory.writeMeasuredBits([1, 0, 1], register: 0, bitOffset: 0)
        XCTAssertEqual(memory.memorySlots[0], 0b0101)
        XCTAssertEqual(memory.hexValue(ofRegister: 0), "0x5")
    }

    // MARK: - C4 readout confusion

    func testReadoutConfusionMatrixOverridesBitFlips() throws {
        // Always map prepared |0⟩ → measured |1⟩
        let matrix = try ReadoutConfusionMatrix(
            qubitCount: 1,
            probabilities: [
                [0, 1],
                [0, 1],
            ]
        )
        var noise = NoiseModel(readoutFlip0To1: 0, readoutFlip1To0: 0)
        noise.readoutConfusion = matrix

        var rng: QuantumRNG = .seeded(1)
        XCTAssertEqual(noise.flipReadoutOutcome(0, measuredQubitCount: 1, rng: &rng), 1)
        XCTAssertEqual(noise.flipReadoutBits([0], rng: &rng), [1])
    }

    func testSingleQubitConfusionFromAsymmetricFlips() throws {
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.2, p10: 0.1)
        XCTAssertEqual(matrix.probabilities[0][0], 0.8, accuracy: 1e-6)
        XCTAssertEqual(matrix.probabilities[0][1], 0.2, accuracy: 1e-6)
        XCTAssertEqual(matrix.probabilities[1][0], 0.1, accuracy: 1e-6)
        XCTAssertEqual(matrix.probabilities[1][1], 0.9, accuracy: 1e-6)
    }

    // MARK: - B5 init

    func testCircuitComputationalBasisInitialization() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.initializeComputationalBasis(0b10) // |10⟩ with qubit0=LSB → index 2

        let backend = try StatevectorBackend()
        let result = try backend.run(circuit: circuit, options: QuantumRunOptions(seed: 1, shots: 64))
        XCTAssertEqual(result.bitstringCounts?["10"], 64)
    }

    func testDensityMatrixComputationalBasisInit() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let density = try DensityMatrix(qubitCount: 1)
        try density.initializeComputationalBasis(index: 1)
        let probs = try DensityMatrixEngine().probabilities(of: density)
        XCTAssertEqual(probs[0], 0, accuracy: 1e-5)
        XCTAssertEqual(probs[1], 1, accuracy: 1e-5)
    }

    // MARK: - B6 / F1 method selection

    func testRecommendStatevectorForCleanNarrowCircuit() throws {
        let method = try QuantumBackendFactory.recommendMethod(qubitCount: 4, noise: nil)
        XCTAssertEqual(method, .statevector)
    }

    func testRecommendDensityMatrixWhenNoisy() throws {
        let noise = NoiseModel(depolarizingProbability: 0.01)
        let method = try QuantumBackendFactory.recommendMethod(qubitCount: 3, noise: noise)
        XCTAssertEqual(method, .densityMatrix)
    }

    func testResourceEstimateBytesScaleWithMethod() throws {
        let sv = try QuantumBackendFactory.estimateResources(qubitCount: 4, noise: nil)
        XCTAssertEqual(sv.recommendedMethod, .statevector)
        XCTAssertEqual(sv.estimatedStateBytes, (1 << 4) * 2 * MemoryLayout<QFloat>.stride)
        XCTAssertGreaterThanOrEqual(sv.estimatedPeakMemoryBytes, sv.estimatedStateBytes)
        XCTAssertGreaterThan(sv.estimatedRuntimeHintNanoseconds, 0)

        let noise = NoiseModel(depolarizingProbability: 0.01)
        let dm = try QuantumBackendFactory.estimateResources(qubitCount: 3, noise: noise)
        XCTAssertEqual(dm.recommendedMethod, .densityMatrix)
        let dim = 1 << 3
        XCTAssertEqual(dm.estimatedStateBytes, dim * dim * 2 * MemoryLayout<QFloat>.stride)
        XCTAssertGreaterThanOrEqual(dm.estimatedPeakMemoryBytes, dm.estimatedStateBytes)
    }

    func testMakeRecommendedReturnsMatchingBackend() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let backend = try QuantumBackendFactory.makeRecommended(qubitCount: 2)
        XCTAssertEqual(backend.method, .statevector)
    }
}
