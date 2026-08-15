import XCTest
@testable import QuantumKit

/// Noise-channel stress: 1Q closed forms on CPU DM, localized Kraus/Pauli, one
/// global DM↔trajectory ensemble parity, and zero-rate NoiseModel ↔ noiseless SV.
///
/// Does **not** re-run the full Phase11 Metal C11 suite — CPU-first, non-flaky tols.
extension QuantumKitTests {

    // MARK: - 1Q depolarizing / AD / PD vs closed form (CPU DM exact)

    func testNoiseStress_1QDepolarizing_CPU_DM_matchesClosedForm() throws {
        // X → |1⟩; after one 1Q depolarizing hit: P(0)=2p/3, ⟨Z⟩=4p/3−1.
        let rates: [QFloat] = [0.05, 0.15, 0.3]
        for p in rates {
            var circuit = try QuantumCircuit(qubitCount: 1)
            try circuit.x(0)
            let density = try runCPUDensity(circuit: circuit, noise: NoiseModel(depolarizingProbability: p))
            let probs = density.probabilitiesDouble()
            let z = try QuantumMeasurement.expectation(density: density, paulis: [0: .z])

            XCTAssertEqual(probs[0], Double(2 * p / 3), accuracy: 1e-5, "dep P0 p=\(p)")
            XCTAssertEqual(probs[1], Double(1 - 2 * p / 3), accuracy: 1e-5, "dep P1 p=\(p)")
            XCTAssertEqual(Double(z), Double(4 * p / 3 - 1), accuracy: 1e-5, "dep ⟨Z⟩ p=\(p)")
        }
    }

    func testNoiseStress_1QAmplitudeDamping_CPU_DM_matchesClosedForm() throws {
        // X → |1⟩; AD(γ): P(0)=γ, ⟨Z⟩=2γ−1.
        let rates: [QFloat] = [0.1, 0.3, 0.7]
        for gamma in rates {
            var circuit = try QuantumCircuit(qubitCount: 1)
            try circuit.x(0)
            let density = try runCPUDensity(
                circuit: circuit,
                noise: NoiseModel(amplitudeDampingProbability: gamma)
            )
            let probs = density.probabilitiesDouble()
            let z = try QuantumMeasurement.expectation(density: density, paulis: [0: .z])

            XCTAssertEqual(probs[0], Double(gamma), accuracy: 1e-5, "AD P0 γ=\(gamma)")
            XCTAssertEqual(probs[1], Double(1 - gamma), accuracy: 1e-5, "AD P1 γ=\(gamma)")
            XCTAssertEqual(Double(z), Double(2 * gamma - 1), accuracy: 1e-5, "AD ⟨Z⟩ γ=\(gamma)")
        }
    }

    func testNoiseStress_1QPhaseDamping_CPU_DM_matchesClosedForm() throws {
        // H → |+⟩; PD(λ): ⟨X⟩=√(1−λ), ⟨Z⟩=0.
        let rates: [QFloat] = [0.0, 0.25, 0.64, 1.0]
        for lambda in rates {
            var circuit = try QuantumCircuit(qubitCount: 1)
            try circuit.h(0)
            let density = try runCPUDensity(
                circuit: circuit,
                noise: NoiseModel(phaseDampingProbability: lambda)
            )
            let x = try QuantumMeasurement.expectation(density: density, paulis: [0: .x])
            let z = try QuantumMeasurement.expectation(density: density, paulis: [0: .z])

            let expectedX = sqrt(max(0, 1 - Double(lambda)))
            XCTAssertEqual(Double(x), expectedX, accuracy: 1e-5, "PD ⟨X⟩ λ=\(lambda)")
            XCTAssertEqual(Double(z), 0, accuracy: 1e-5, "PD ⟨Z⟩ λ=\(lambda)")
        }
    }

    // MARK: - Localized Kraus / Pauli on CPU DM

    func testNoiseStress_LocalizedPauliChannel_onCPU_DM() throws {
        // Idle Z on |0⟩ + localized Pauli channel: X/Y flip, Z does not.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.z(0)

        let px: QFloat = 0.12
        let py: QFloat = 0.08
        let pz: QFloat = 0.2
        let channel = try QuantumChannel.makePauliChannel(px: px, py: py, pz: pz)
        let noise = NoiseModel().adding(channel, for: .gate(.z))

        let density = try runCPUDensity(circuit: circuit, noise: noise)
        let probs = density.probabilitiesDouble()
        XCTAssertEqual(probs[0], Double(1 - px - py), accuracy: 1e-5)
        XCTAssertEqual(probs[1], Double(px + py), accuracy: 1e-5)
    }

    func testNoiseStress_LocalizedKrausBitFlip_onCPU_DM() throws {
        // Deterministic X Kraus after Z on |0⟩ → |1⟩.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.z(0)

        let xOp: [ComplexAmplitude] = [
            .init(real: 0, imaginary: 0), .init(real: 1, imaginary: 0),
            .init(real: 1, imaginary: 0), .init(real: 0, imaginary: 0),
        ]
        let channel = try QuantumChannel.fromKraus1Q([xOp])
        let noise = NoiseModel().adding(channel, for: .gate(.z))

        let density = try runCPUDensity(circuit: circuit, noise: noise)
        let probs = density.probabilitiesDouble()
        XCTAssertEqual(probs[0], 0, accuracy: 1e-5)
        XCTAssertEqual(probs[1], 1, accuracy: 1e-5)
    }

    // MARK: - One shared global channel: CPU DM ↔ trajectory ensemble

    func testNoiseStress_TrajectoryEnsemble_vs_CPU_DM_globalAD() throws {
        // Single shared global channel (AD) — Phase11-style parity, CPU only, one case.
        let gamma: QFloat = 0.25
        let noise = NoiseModel(amplitudeDampingProbability: gamma)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let density = try runCPUDensity(circuit: circuit, noise: noise)
        let dmProbs = density.probabilitiesDouble()
        XCTAssertEqual(dmProbs[0], Double(gamma), accuracy: 1e-5)
        XCTAssertEqual(dmProbs[1], Double(1 - gamma), accuracy: 1e-5)

        let traj = TrajectoryBackend(engine: CPUStatevectorEngine())
        let avg = try traj.averageProbabilities(
            circuit: circuit,
            trajectories: 6_000,
            seed: 4_242,
            noise: noise
        )
        // Binomial SE at 6k ≈ √(p(1-p)/n) ≲ 0.006; 0.035 leaves comfortable margin.
        XCTAssertEqual(Double(avg[0]), dmProbs[0], accuracy: 0.035, "traj↔DM P0")
        XCTAssertEqual(Double(avg[1]), dmProbs[1], accuracy: 0.035, "traj↔DM P1")
    }

    // MARK: - Zero-noise NoiseModel ↔ CPU SV

    func testNoiseStress_ZeroRateNoiseModel_CPU_DM_matches_CPU_SV() throws {
        // NoiseModel enabled (channels present) but all rates/probs = 0 → Born ≡ noiseless SV.
        let zeroNoise = NoiseModel(
            depolarizingProbability: 0,
            amplitudeDampingProbability: 0,
            t1: 0,
            t2: 0,
            gateTime: 0,
            phaseDampingProbability: 0,
            readoutFlip0To1: 0,
            readoutFlip1To0: 0
        )
        XCTAssertFalse(zeroNoise.appliesDepolarizing)
        XCTAssertFalse(zeroNoise.appliesAmplitudeDamping)
        XCTAssertFalse(zeroNoise.appliesPhaseDamping)

        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.ry(theta: QFloat(0.37), 2)
        try circuit.cz(1, 2)
        try circuit.rz(theta: QFloat(-0.22), 0)

        let sv = try CPUStateVector(qubitCount: 3)
        _ = try CPUStatevectorEngine().execute(circuit, on: sv)
        let svProbs = sv.probabilitiesDouble()

        let dm = try runCPUDensity(circuit: circuit, noise: zeroNoise)
        let dmProbs = dm.probabilitiesDouble()

        XCTAssertEqual(svProbs.count, dmProbs.count)
        for index in svProbs.indices {
            XCTAssertEqual(
                svProbs[index], dmProbs[index],
                accuracy: 1e-10,
                "zero-noise Born idx=\(index)"
            )
        }

        let svZ = try QuantumMeasurement.expectation(state: sv, paulis: [0: .z, 1: .z])
        let dmZ = try QuantumMeasurement.expectation(density: dm, paulis: [0: .z, 1: .z])
        XCTAssertEqual(Double(svZ), Double(dmZ), accuracy: 1e-10, "zero-noise ⟨ZZ⟩")
    }

    // MARK: - Helpers

    private func runCPUDensity(
        circuit: QuantumCircuit,
        noise: NoiseModel
    ) throws -> CPUDensityMatrix {
        let engine = CPUDensityMatrixEngine()
        let density = try CPUDensityMatrix(qubitCount: circuit.qubitCount)
        _ = try engine.execute(circuit, on: density, noise: noise)
        return density
    }
}
