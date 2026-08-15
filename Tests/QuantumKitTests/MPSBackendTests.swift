import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - B18 MPS backend MVP

    func testMPSFactoryMethodTagAndDefaultsUnchanged() throws {
        let backend = QuantumBackendFactory.makeMPS()
        XCTAssertEqual(backend.method, .mps)
        XCTAssertTrue(backend is MPSBackend)
        let recommended = try QuantumBackendFactory.recommendMethod(qubitCount: 3, noise: nil)
        XCTAssertEqual(recommended, .statevector)
    }

    func testMPSAlwaysUsesLocalTensorEvolution() throws {
        var mps = try MPSState(qubitCount: 3, configuration: .default)
        XCTAssertFalse(mps.usesDenseEvolution)
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        _ = try MPSEngine().execute(circuit, on: &mps)
        XCTAssertFalse(mps.usesDenseEvolution)
        XCTAssertGreaterThan(mps.bondDimensions.max() ?? 0, 0)
    }

    func testMPSProductStateAmplitudesMatchSV() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        var mps = try MPSState(qubitCount: 2, configuration: MPSConfiguration(maxBondDimension: 8))
        _ = try MPSEngine().execute(circuit, on: &mps)
        let amps = try mps.amplitudes()
        XCTAssertEqual(Double(amps[1].real), 1, accuracy: 1e-10)
    }

    func testMPSBellAdjacentCXParityVsCPUSV() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        let config = MPSConfiguration(maxBondDimension: 16, svdTruncationThreshold: 1e-14)
        var mps = try MPSState(qubitCount: 2, configuration: config)
        _ = try MPSEngine(configuration: config).execute(circuit, on: &mps)
        XCTAssertFalse(mps.usesDenseEvolution)
        let sv = try CPUStateVector(qubitCount: 2)
        _ = try CPUStatevectorEngine().execute(circuit, on: sv)
        let mpsProbs = try mps.probabilities()
        let svProbs = sv.probabilities()
        for i in 0..<4 {
            XCTAssertEqual(mpsProbs[i], svProbs[i], accuracy: 1e-5)
        }
    }

    func testMPSThreeQubitBellTensorZeroParity() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        var mps = try MPSState(qubitCount: 3, configuration: MPSConfiguration(maxBondDimension: 8))
        _ = try MPSEngine().execute(circuit, on: &mps)
        let sv = try CPUStateVector(qubitCount: 3)
        _ = try CPUStatevectorEngine().execute(circuit, on: sv)
        let mpsProbs = try mps.probabilities()
        let svProbs = sv.probabilities()
        for i in 0..<8 {
            XCTAssertEqual(mpsProbs[i], svProbs[i], accuracy: 1e-5)
        }
    }

    func testMPSManualGHZAmplitudes() throws {
        let s = 1.0 / sqrt(2.0)
        var state = try MPSState(qubitCount: 3, configuration: MPSConfiguration(maxBondDimension: 4))
        state.adoptTensorSites([
            MPSSite(
                leftDim: 1,
                rightDim: 2,
                data: [
                    MPSComplex(re: s, im: 0), .zero,
                    .zero, MPSComplex(re: s, im: 0),
                ]
            ),
            MPSSite(
                leftDim: 2,
                rightDim: 2,
                data: [
                    .one, .zero,
                    .zero, .zero,
                    .zero, .zero,
                    .zero, .one,
                ]
            ),
            MPSSite(
                leftDim: 2,
                rightDim: 1,
                data: [
                    .one, .zero,
                    .zero, .one,
                ]
            ),
        ])
        let probs = try state.probabilities()
        XCTAssertEqual(probs[0], 0.5, accuracy: 1e-6)
        XCTAssertEqual(probs[7], 0.5, accuracy: 1e-6)
    }

    func testMPSLowChiRunsWithoutCrash() throws {
        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.h(0)
        for q in 0..<3 { try circuit.cx(q, q + 1) }
        let config = MPSConfiguration(maxBondDimension: 1, svdTruncationThreshold: 1e-12)
        var mps = try MPSState(qubitCount: 4, configuration: config)
        _ = try MPSEngine(configuration: config).execute(circuit, on: &mps)
        try mps.compress()
        let norm = try mps.probabilities().reduce(QFloat(0), +)
        XCTAssertEqual(norm, 1, accuracy: 1e-3)
        XCTAssertGreaterThanOrEqual(mps.lastTruncationError, 0)
    }

    func testMPSAdjacentSwapMovesExcitation() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.apply(.swap(q1: 0, q2: 1))
        var mps = try MPSState(qubitCount: 2, configuration: MPSConfiguration(maxBondDimension: 4))
        _ = try MPSEngine().execute(circuit, on: &mps)
        XCTAssertEqual(try mps.probabilities()[2], 1, accuracy: 1e-6)
    }

    func testMPSGHZAdjacentChainParityVsCPUSV() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.cx(1, 2)
        let config = MPSConfiguration(maxBondDimension: 32, svdTruncationThreshold: 1e-14)
        var mps = try MPSState(qubitCount: 3, configuration: config)
        _ = try MPSEngine(configuration: config).execute(circuit, on: &mps)
        let sv = try CPUStateVector(qubitCount: 3)
        _ = try CPUStatevectorEngine().execute(circuit, on: sv)
        let mpsProbs = try mps.probabilities()
        let svProbs = sv.probabilities()
        for i in 0..<8 {
            XCTAssertEqual(mpsProbs[i], svProbs[i], accuracy: 1e-4)
        }
    }

    func testMPSNonAdjacentCXSwapChainParity() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 2)
        let config = MPSConfiguration(maxBondDimension: 32, svdTruncationThreshold: 1e-14)
        var mps = try MPSState(qubitCount: 3, configuration: config)
        _ = try MPSEngine(configuration: config).execute(circuit, on: &mps)
        let sv = try CPUStateVector(qubitCount: 3)
        _ = try CPUStatevectorEngine().execute(circuit, on: sv)
        let mpsProbs = try mps.probabilities()
        let svProbs = sv.probabilities()
        for i in 0..<8 {
            XCTAssertEqual(mpsProbs[i], svProbs[i], accuracy: 1e-4)
        }
    }

    /// Local tensor path with rotations + non-adjacent CX vs CPU SV (amplitude export).
    func testMPSLocalRotatedNonAdjacentParityVsCPUSV() throws {
        let config = MPSConfiguration(maxBondDimension: 64, svdTruncationThreshold: 1e-14)
        var mps = try MPSState(qubitCount: 4, configuration: config)
        XCTAssertFalse(mps.usesDenseEvolution)

        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.cx(1, 2)
        try circuit.cx(2, 3)
        try circuit.rz(theta: 0.37, 1)
        try circuit.ry(theta: 0.91, 2)
        try circuit.cx(0, 3)
        _ = try MPSEngine(configuration: config).execute(circuit, on: &mps)

        let sv = try CPUStateVector(qubitCount: 4)
        _ = try CPUStatevectorEngine().execute(circuit, on: sv)
        let mpsProbs = try mps.probabilities()
        let svProbs = sv.probabilities()
        XCTAssertEqual(mpsProbs.reduce(QFloat(0), +), 1, accuracy: 1e-6)
        for i in 0..<16 {
            XCTAssertEqual(mpsProbs[i], svProbs[i], accuracy: 1e-5)
        }
    }

    func testMPSRejectsMidcircuitMeasure() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.apply(.measure(MeasureSpec(qubits: [0], classicalRegister: 0)))
        XCTAssertThrowsError(try MPSBackend().run(circuit: circuit)) { error in
            guard case MPSError.unsupportedGate = error else {
                return XCTFail("expected unsupportedGate, got \(error)")
            }
        }
    }

    func testMPSRejectsResetAndInitialize() throws {
        var resetCircuit = try QuantumCircuit(qubitCount: 1)
        try resetCircuit.h(0)
        try resetCircuit.reset(0)
        XCTAssertThrowsError(try MPSBackend().run(circuit: resetCircuit)) { error in
            guard case MPSError.unsupportedGate = error else {
                return XCTFail("expected unsupportedGate for reset, got \(error)")
            }
        }

        var initCircuit = try QuantumCircuit(qubitCount: 1)
        try initCircuit.apply(
            .initialize(
                qubits: [0],
                amplitudes: [
                    ComplexAmplitude(real: 1, imaginary: 0),
                    ComplexAmplitude(real: 0, imaginary: 0),
                ]
            )
        )
        XCTAssertThrowsError(try MPSBackend().run(circuit: initCircuit)) { error in
            guard case MPSError.unsupportedGate = error else {
                return XCTFail("expected unsupportedGate for initialize, got \(error)")
            }
        }
    }

    func testMPSRejectsNoise() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let noise = NoiseModel(depolarizingProbability: 0.01)
        XCTAssertThrowsError(
            try MPSBackend().run(
                circuit: circuit,
                options: QuantumRunOptions(noise: noise, seed: 1, shots: 4)
            )
        ) { error in
            guard case MPSError.noiseNotSupported = error else {
                return XCTFail("expected noiseNotSupported, got \(error)")
            }
        }
    }

    func testMPSSeededShotsReproducible() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        let backend = MPSBackend(configuration: MPSConfiguration(maxBondDimension: 8))
        let options = QuantumRunOptions(seed: 11, shots: 256)
        let a = try XCTUnwrap(backend.run(circuit: circuit, options: options).shotCounts)
        let b = try XCTUnwrap(backend.run(circuit: circuit, options: options).shotCounts)
        XCTAssertEqual(a, b)
    }

    func testMPSShotHistogramMatchesBellSupport() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        let result = try MPSBackend(configuration: MPSConfiguration(maxBondDimension: 8)).run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 3, shots: 2000)
        )
        let counts = try XCTUnwrap(result.shotCounts?.counts)
        XCTAssertEqual(counts[1, default: 0], 0)
        XCTAssertEqual(counts[2, default: 0], 0)
        XCTAssertEqual(QFloat(counts[0, default: 0]) / 2000, 0.5, accuracy: 0.06)
        XCTAssertEqual(QFloat(counts[3, default: 0]) / 2000, 0.5, accuracy: 0.06)
    }

    /// Export cap only blocks amplitude materialization; evolution stays local and sampling works.
    func testMPSLocalPathForcedByExportCapParity() throws {
        let config = MPSConfiguration(
            maxBondDimension: 32,
            svdTruncationThreshold: 1e-14,
            maxAmplitudeExportQubits: 1
        )
        var mps = try MPSState(qubitCount: 3, configuration: config)
        XCTAssertFalse(mps.usesDenseEvolution)

        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.cx(1, 2)
        _ = try MPSEngine(configuration: config).execute(circuit, on: &mps)
        XCTAssertFalse(mps.usesDenseEvolution)

        XCTAssertThrowsError(try mps.amplitudes())
        var rng = QuantumRNG.seeded(7)
        var hist: [Int: Int] = [:]
        for _ in 0..<4000 {
            hist[try mps.sampleOutcome(rng: &rng), default: 0] += 1
        }
        XCTAssertEqual(hist[0, default: 0] + hist[7, default: 0], 4000)
        XCTAssertEqual(QFloat(hist[0, default: 0]) / 4000, 0.5, accuracy: 0.05)
        XCTAssertEqual(QFloat(hist[7, default: 0]) / 4000, 0.5, accuracy: 0.05)
    }

    /// Wide register: shallow Bell on qubits 0–1 must match analytic support under local MPS.
    func testMPSWideRegisterShotsMatchBellSupport() throws {
        var circuit = try QuantumCircuit(qubitCount: 17)
        try circuit.h(0)
        try circuit.cx(0, 1)
        let config = MPSConfiguration(maxBondDimension: 8, maxAmplitudeExportQubits: 16)
        let result = try MPSBackend(configuration: config).run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 9, shots: 2000)
        )
        let counts = try XCTUnwrap(result.shotCounts)
        XCTAssertEqual(counts.shots, 2000)
        XCTAssertEqual(counts.counts.values.reduce(0, +), 2000)
        // Only |0…0⟩ and |…0011⟩ (qubits 0 and 1 set) should appear.
        let unexpected = counts.counts.keys.filter { $0 != 0 && $0 != 3 }
        XCTAssertTrue(unexpected.isEmpty, "unexpected outcomes \(unexpected)")
        XCTAssertEqual(QFloat(counts.counts[0, default: 0]) / 2000, 0.5, accuracy: 0.06)
        XCTAssertEqual(QFloat(counts.counts[3, default: 0]) / 2000, 0.5, accuracy: 0.06)
    }

    func testMPSSamplerExactAndShots() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        let backend = MPSBackend(configuration: MPSConfiguration(maxBondDimension: 16))
        let exact = try Sampler().run(circuit: circuit, backend: backend)
        XCTAssertNil(exact.shotCounts)
        XCTAssertEqual(exact.metadata.method, .mps)
        XCTAssertEqual(exact.quasiProbabilities["00"] ?? 0, 0.5, accuracy: 1e-5)
        XCTAssertEqual(exact.quasiProbabilities["11"] ?? 0, 0.5, accuracy: 1e-5)

        let shots = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(seed: 5, shots: 1500)
        )
        XCTAssertEqual(shots.shotCounts?.shots, 1500)
        XCTAssertEqual(shots.metadata.method, .mps)
    }
}
