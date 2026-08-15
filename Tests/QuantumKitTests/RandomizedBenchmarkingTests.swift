import XCTest
@testable import QuantumKit

/// Clifford RB–style exponential survival decay under depolarizing noise.
///
/// **Clifford set (n=1):** 24-element 1Q Clifford group (mod global phase), via `H`/`S`
/// generation, applied as ``Gate/unitary1`` (one depolarizing hit per step).
/// **n=2 (optional):** per-step 1Q Cliffords on both qubits + CX with prob ½; inverse
/// via accumulated `4×4` adjoint (`customUnitary`).
///
/// **Metric:** `P(m) = ⟨0…0|ρ|0…0⟩` from exact ``CPUDensityMatrix``.
/// **Noise:** global ``NoiseModel(depolarizingProbability:)``.
/// **Decay:** n=1 analytic `P(m) ≈ ½ + ½·α^{m+1}`, `α = 1 − 4p/3`.
///
/// No XEB. Leaves Quantum Volume alone.
extension QuantumKitTests {

    // MARK: - Clifford group + inverse correctness

    func testOneQubitCliffordGroupHas24Elements() {
        XCTAssertEqual(RandomizedBenchmarking.oneQubitCliffordGroup.count, 24)
    }

    func testOneQubitCliffordGroupElementsAreDistinctModGlobalPhase() {
        let group = RandomizedBenchmarking.oneQubitCliffordGroup
        XCTAssertEqual(group.count, 24)
        // Entries are already phase-canonicalized by construction; require pairwise separation.
        for i in 0..<group.count {
            for j in (i + 1)..<group.count {
                var maxDiff = 0.0
                for k in 0..<4 {
                    let a = group[i][k]
                    let b = group[j][k]
                    maxDiff = max(
                        maxDiff,
                        abs(Double(a.real - b.real)) + abs(Double(a.imaginary - b.imaginary))
                    )
                }
                XCTAssertGreaterThan(
                    maxDiff,
                    1e-6,
                    "Cliffords \(i) and \(j) look identical mod recorded phase"
                )
            }
        }
    }

    func testRBSequenceNoiselessSurvivalIsOneForVariousLengths() throws {
        let lengths = [0, 1, 2, 5, 10, 20]
        for m in lengths {
            let circuit = try RandomizedBenchmarking.makeSequence(
                options: RandomizedBenchmarkingOptions(qubitCount: 1, sequenceLength: m),
                seed: UInt64(100 + m)
            )
            // m unitary1 Cliffords + 1 inverse
            XCTAssertEqual(circuit.gates.count, m + 1)

            let pDM = try RandomizedBenchmarking.survivalProbability(circuit: circuit, noise: nil)
            XCTAssertEqual(pDM, 1.0, accuracy: 1e-9, "DM survival m=\(m)")

            // Cross-check CPU SV Born
            let engine = CPUStatevectorEngine()
            let state = try CPUStateVector(qubitCount: 1)
            _ = try engine.execute(circuit, on: state)
            XCTAssertEqual(state.probabilitiesDouble()[0], 1.0, accuracy: 1e-9, "SV survival m=\(m)")
        }
    }

    func testRBInverseWrongWouldFailNoiselessCheck() throws {
        // Sanity: a non-inverted random Clifford sequence is NOT identity → survival << 1.
        var rng = QuantumRNG.seeded(7)
        let group = RandomizedBenchmarking.oneQubitCliffordGroup
        var circuit = try QuantumCircuit(qubitCount: 1)
        for _ in 0..<5 {
            let u = group[rng.nextInt(upperBound: group.count)]
            try circuit.unitary1(matrix: u, target: 0)
        }
        // Omit inverse on purpose.
        let p = try RandomizedBenchmarking.survivalProbability(circuit: circuit, noise: nil)
        XCTAssertLessThan(p, 0.99, "unterminated Clifford walk should not stay at |0⟩")
    }

    // MARK: - Noisy decay (n=1 analytic)

    func testDepolarizingSurvivalDecreasesWithSequenceLength() throws {
        let p: QFloat = 0.05
        let noise = NoiseModel(depolarizingProbability: p)
        let seeds: [UInt64] = [1, 2, 3, 4, 5, 6, 7, 8]
        let lengths = [1, 4, 8, 16]

        let curve = try RandomizedBenchmarking.survivalCurve(
            qubitCount: 1,
            lengths: lengths,
            noise: noise,
            seedsPerLength: seeds
        )

        for i in 0..<(curve.survivals.count - 1) {
            XCTAssertGreaterThan(
                curve.survivals[i],
                curve.survivals[i + 1] - 1e-6,
                "P(m) should decrease: m=\(lengths[i]) → \(lengths[i + 1])"
            )
        }
        XCTAssertLessThan(curve.survivals.last!, curve.survivals.first! - 0.02)
    }

    func testIdentityGateTrainMatchesAlphaToThePowerGateCount() throws {
        // Locks noise application count: m+1 identity unitary1s → P = ½ + ½ α^{m+1}
        // exactly (no Clifford sampling noise). Catches α^m vs α^{m+1} off-by-one.
        let p = 0.04
        let alpha = RBDecayFit.depolarizingDecayParameter(p: p)
        let noise = NoiseModel(depolarizingProbability: QFloat(p))
        let identity: [ComplexAmplitude] = [
            ComplexAmplitude(real: 1, imaginary: 0), ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0), ComplexAmplitude(real: 1, imaginary: 0),
        ]
        for m in [0, 1, 2, 5, 10] {
            var circuit = try QuantumCircuit(qubitCount: 1)
            for _ in 0..<(m + 1) {
                try circuit.unitary1(matrix: identity, target: 0)
            }
            let measured = try RandomizedBenchmarking.survivalProbability(
                circuit: circuit,
                noise: noise,
                seed: 0
            )
            let theory = 0.5 + 0.5 * pow(alpha, Double(m + 1))
            XCTAssertEqual(measured, theory, accuracy: 1e-8, "m=\(m) identity×\(m + 1)")
        }
    }

    func testOneQubitDepolarizingDecayMatchesAnalyticAlpha() throws {
        let p = 0.04
        let noise = NoiseModel(depolarizingProbability: QFloat(p))
        let seeds: [UInt64] = Array(1...24).map { UInt64($0) }
        let lengths = [2, 4, 6, 8, 10, 12]

        let curve = try RandomizedBenchmarking.survivalCurve(
            qubitCount: 1,
            lengths: lengths,
            noise: noise,
            seedsPerLength: seeds
        )

        let theory = RBDecayFit.idealOneQubitDepolarizing(lengths: lengths, depolarizingProbability: p)
        for (measured, expected) in zip(curve.survivals, theory.survivals) {
            XCTAssertEqual(
                measured,
                expected,
                accuracy: 0.04,
                "mean P(m) vs analytic (tol allows finite sequence sample)"
            )
        }

        let fit = try RandomizedBenchmarking.fitExponentialDecay(
            lengths: curve.lengths,
            survivals: curve.survivals
        )
        let alphaTheory = RBDecayFit.depolarizingDecayParameter(p: p)
        XCTAssertEqual(fit.decay, alphaTheory, accuracy: 0.05)
        XCTAssertEqual(fit.offset, 0.5, accuracy: 0)
        XCTAssertGreaterThan(fit.decay, 0.5)
        XCTAssertLessThan(fit.decay, 1.0)
    }

    func testNoiselessCurveIsFlatNearOne() throws {
        let seeds: [UInt64] = [10, 11, 12, 13]
        let lengths = [1, 5, 10, 15]
        let curve = try RandomizedBenchmarking.survivalCurve(
            qubitCount: 1,
            lengths: lengths,
            noise: nil,
            seedsPerLength: seeds
        )
        for p in curve.survivals {
            XCTAssertEqual(p, 1.0, accuracy: 1e-9)
        }
    }

    func testNoisyCurveIsNotFlatOrIncreasing() throws {
        let noise = NoiseModel(depolarizingProbability: 0.08)
        let seeds: [UInt64] = Array(20...35).map { UInt64($0) }
        let lengths = [1, 3, 6, 12]
        let curve = try RandomizedBenchmarking.survivalCurve(
            qubitCount: 1,
            lengths: lengths,
            noise: noise,
            seedsPerLength: seeds
        )
        // Fail if flat or increasing without reason.
        let delta = curve.survivals.first! - curve.survivals.last!
        XCTAssertGreaterThan(delta, 0.05, "expected clear decay under depolarizing")
        XCTAssertFalse(
            zip(curve.survivals, curve.survivals.dropFirst()).allSatisfy { $0 <= $1 + 1e-9 },
            "survival must not be monotonically non-decreasing under noise"
        )
    }

    // MARK: - n = 2 (solid optional)

    func testTwoQubitRBNoiselessSurvivalNearOne() throws {
        for m in [0, 1, 2, 4, 8] {
            let circuit = try RandomizedBenchmarking.makeSequence(
                options: RandomizedBenchmarkingOptions(qubitCount: 2, sequenceLength: m),
                seed: UInt64(200 + m)
            )
            let p = try RandomizedBenchmarking.survivalProbability(circuit: circuit, noise: nil)
            XCTAssertEqual(p, 1.0, accuracy: 1e-8, "n=2 noiseless m=\(m)")
        }
    }

    func testTwoQubitRBDepolarizingDecays() throws {
        let noise = NoiseModel(depolarizingProbability: 0.06)
        let seeds: [UInt64] = Array(1...12).map { UInt64($0) }
        let short = try RandomizedBenchmarking.meanSurvival(
            options: RandomizedBenchmarkingOptions(qubitCount: 2, sequenceLength: 1),
            noise: noise,
            seeds: seeds
        )
        let long = try RandomizedBenchmarking.meanSurvival(
            options: RandomizedBenchmarkingOptions(qubitCount: 2, sequenceLength: 8),
            noise: noise,
            seeds: seeds
        )
        XCTAssertGreaterThan(short, long + 0.02)
        XCTAssertLessThan(long, 0.95)
    }
}
