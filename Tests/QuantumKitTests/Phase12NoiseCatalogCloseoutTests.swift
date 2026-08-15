import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - C close-out (C1 compose / C2 channels / C3 calibration)

    func testNoiseModelMergingConcatenatesRulesAndMaxesGlobals() {
        let a = NoiseModel(depolarizingProbability: 0.01)
            .adding(.pauliXFlip(probability: 0.1), for: .gate(.x))
        let b = NoiseModel(depolarizingProbability: 0.05, resetErrorProbability: 0.02)
            .adding(.pauliZFlip(probability: 0.2), for: .gate(.z))

        let merged = a.merging(b)
        XCTAssertEqual(merged.depolarizingProbability, 0.05, accuracy: 1e-6)
        XCTAssertEqual(merged.resetErrorProbability, 0.02, accuracy: 1e-6)
        XCTAssertEqual(merged.localizedRules.count, 2)
        XCTAssertTrue(merged.hasLocalizedGateNoise)
    }

    func testKraus1QBitFlipMatchesPauliX() throws {
        guard let (engine, density) = try makeDensitySetupForCCloseout(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.z(0) // leave |0⟩; apply X via Kraus after Z

        let xOp: [ComplexAmplitude] = [
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 1, imaginary: 0),
            ComplexAmplitude(real: 1, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
        ]
        let channel = try QuantumChannel.fromKraus1Q([xOp])
        let noise = NoiseModel().adding(channel, for: .gate(.z))
        try engine.execute(circuit, on: density, noise: noise)

        let probabilities = engine.probabilities(of: density)
        XCTAssertEqual(probabilities[1], 1.0, accuracy: 1e-5)
    }

    func testPauliChannelPopulationsOnGroundState() throws {
        guard let (engine, density) = try makeDensitySetupForCCloseout(qubitCount: 1) else { return }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.z(0)

        let px: QFloat = 0.1
        let py: QFloat = 0.05
        let pz: QFloat = 0.2
        let channel = try QuantumChannel.makePauliChannel(px: px, py: py, pz: pz)
        let noise = NoiseModel().adding(channel, for: .gate(.z))
        try engine.execute(circuit, on: density, noise: noise)

        let probabilities = engine.probabilities(of: density)
        // X and Y flip |0⟩→|1⟩; Z leaves |0⟩.
        XCTAssertEqual(probabilities[0], 1 - px - py, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], px + py, accuracy: 1e-5)
    }

    func testPauliLindbladFactoryMapsRatesToPauliChannel() throws {
        let channel = try QuantumChannel.fromPauliLindblad(lambdaZ: 1, duration: 0.5)
        guard case .pauliChannel(let px, let py, let pz) = channel else {
            return XCTFail("expected pauliChannel")
        }
        XCTAssertEqual(px, 0, accuracy: 1e-6)
        XCTAssertEqual(py, 0, accuracy: 1e-6)
        let expected = (1 - exp(-2 * QFloat(0.5))) / 2
        XCTAssertEqual(pz, expected, accuracy: 1e-5)
    }

    func testCalibrationPreservesPerQubitReadoutAsConfusionMatrix() throws {
        var calibration = DeviceCalibration(qubitCount: 2, gateTime: 0.1)
        calibration[qubit: 0] = QubitCalibration(readoutError0To1: 0.02, readoutError1To0: 0.03)
        calibration[qubit: 1] = QubitCalibration(readoutError0To1: 0.10, readoutError1To0: 0.01)

        let noise = NoiseModel.from(calibration: calibration)
        let matrix = try XCTUnwrap(noise.readoutConfusion)
        XCTAssertEqual(matrix.qubitCount, 2)

        // Exact matrix marginals (same 1e-6 discipline as the old global flip asserts).
        let row00 = matrix.probabilities[0]
        XCTAssertEqual(row00[1] + row00[3], 0.02, accuracy: 1e-6) // q0 p01
        XCTAssertEqual(row00[2] + row00[3], 0.10, accuracy: 1e-6) // q1 p01

        let row01 = matrix.probabilities[1]
        XCTAssertEqual(row01[0] + row01[2], 0.03, accuracy: 1e-6) // q0 p10

        let row10 = matrix.probabilities[2]
        XCTAssertEqual(row10[0] + row10[1], 0.01, accuracy: 1e-6) // q1 p10
    }

    func testCalibrationCouplingMapSeedsNearestNeighborCrosstalk() throws {
        let map = try CouplingMap.linear(3)
        var calibration = DeviceCalibration(
            qubitCount: 3,
            gateTime: 0.1,
            couplingMap: map,
            nearestNeighborCrosstalkProbability: 0.01
        )
        calibration[qubit: 0] = QubitCalibration(t1: 50, t2: 70)
        calibration[qubit: 1] = QubitCalibration(t1: 50, t2: 70)
        calibration[qubit: 2] = QubitCalibration(t1: 50, t2: 70)

        let noise = NoiseModel.from(calibration: calibration)
        // 3 qubits × (AD+PD) = 6, plus 4 directed crosstalk rules on 2 edges.
        XCTAssertEqual(noise.localizedRules.count, 10)
    }

    private func makeDensitySetupForCCloseout(
        qubitCount: Int
    ) throws -> (DensityMatrixEngine, DensityMatrix)? {
        let engine = try DensityMatrixEngine()
        let density = try DensityMatrix(qubitCount: qubitCount)
        return (engine, density)
    }
}
