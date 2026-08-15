import XCTest
@testable import QuantumKit

/// Linear cross-entropy benchmarking (XEB): Sampler shots vs ideal CPU SV probs.
///
/// **Formula:** `F_XEB = ⟨2ⁿ p_U(x) − 1⟩_samples`
/// (`D · p_U(x_i) − 1` averaged over shots; histogram form uses the same weights).
/// Population value under ideal sampling: `E[F] = 2ⁿ Σ p_U(x)² − 1`.
///
/// **Circuits:** seeded random `u(θ,φ,λ)` on each qubit + alternating adjacent CX layers
/// (`n ≤ 4`, modest depth). Ideal `p_U` from ``CPUStateVector``.
///
/// **Backends:** primary ``Sampler`` + ``CPUStatevectorBackend``; optional Metal
/// ``StatevectorBackend`` when available. No QV/RB rework; no Sycamore / full-PT claim.
///
/// **Shot counts (documented):** noiseless match uses `S = 8_192` shots. With chaotic
/// circuits `Var(D p − 1)` is O(1), so SE ≈ `1/√S ≈ 0.011`. Threshold `|F̂ − E[F]| ≤ 0.08`
/// is many σ below Hoeffding / CLT flake rates for CI.
extension QuantumKitTests {

    /// Shots for Sampler ↔ ideal expectation match (CPU SV backend).
    private static let xebShotCount = 8_192

    /// Max `|F̂ − E[F]|` for noiseless seeded runs (`S = xebShotCount`).
    private static let xebMatchTolerance = 0.08

    /// Lower bound on population `E[F]` for highly entangling circuits (n≥2).
    /// Porter–Thomas asymptote `(D−1)/(D+1)` is ~0.60 (n=2) … ~0.88 (n=4).
    private static let minExpectedFXEB = 0.45

    /// Negative-control upper bound when scoring with a wrong unitary's probs.
    private static let maxWrongUnitaryFXEB = 0.35

    // MARK: - Formula + exact bias

    func testLinearXEBPopulationEqualsSumSquaresFormula() throws {
        // Hand distribution on n=2 (D=4): p = (0.5, 0.25, 0.25, 0)
        // Σ p² = 0.25 + 0.0625 + 0.0625 = 0.375
        // E[F] = 4 * 0.375 − 1 = 0.5
        let probs = [0.5, 0.25, 0.25, 0.0]
        let expected = try CrossEntropyBenchmark.expectedLinearXEB(idealProbabilities: probs)
        XCTAssertEqual(expected, 0.5, accuracy: 1e-15)

        // Exact enumeration: average of (D p(x) − 1) weighted by p(x).
        var weighted = 0.0
        for (_, p) in probs.enumerated() {
            weighted += p * (4.0 * p - 1.0)
        }
        XCTAssertEqual(weighted, expected, accuracy: 1e-15)
    }

    func testLinearXEBFromHistogramMatchesExpandedOutcomes() throws {
        let probs = [0.4, 0.3, 0.2, 0.1]
        let counts = ShotCounts(shots: 10, counts: [0: 4, 1: 3, 2: 2, 3: 1])
        let fromHist = try CrossEntropyBenchmark.linearXEB(
            idealProbabilities: probs,
            shotCounts: counts
        )
        var outcomes: [Int] = []
        for (x, c) in counts.counts {
            outcomes.append(contentsOf: repeatElement(x, count: c))
        }
        let fromList = try CrossEntropyBenchmark.linearXEB(
            idealProbabilities: probs,
            outcomes: outcomes
        )
        XCTAssertEqual(fromHist, fromList, accuracy: 1e-15)
        // Manual: (1/10)[4*(4*0.4-1) + 3*(4*0.3-1) + 2*(4*0.2-1) + 1*(4*0.1-1)]
        // = (1/10)[4*0.6 + 3*0.2 + 2*(-0.2) + 1*(-0.6)] = (2.4+0.6-0.4-0.6)/10 = 0.2
        XCTAssertEqual(fromHist, 0.2, accuracy: 1e-15)
    }

    // MARK: - Generator

    func testXEBRandomCircuitIsSeededReproducible() throws {
        let options = CrossEntropyBenchmarkOptions(qubitCount: 3, depth: 4)
        let a = try CrossEntropyBenchmark.makeRandomCircuit(options: options, seed: 42)
        let b = try CrossEntropyBenchmark.makeRandomCircuit(options: options, seed: 42)
        let c = try CrossEntropyBenchmark.makeRandomCircuit(options: options, seed: 43)

        XCTAssertEqual(a.gates.count, b.gates.count)
        for (lhs, rhs) in zip(a.gates, b.gates) {
            XCTAssertEqual(xebGateFingerprint(lhs), xebGateFingerprint(rhs))
        }
        XCTAssertNotEqual(
            a.gates.map(xebGateFingerprint),
            c.gates.map(xebGateFingerprint)
        )
    }

    func testXEBCircuitStructureIsRandom1QPlusAdjacentCX() throws {
        let n = 4
        let depth = 3
        let circuit = try CrossEntropyBenchmark.makeRandomCircuit(
            options: CrossEntropyBenchmarkOptions(qubitCount: n, depth: depth),
            seed: 7
        )
        // Per layer: n single-qubit u + floor(n/2) or similar CX count.
        // even: CX on (0,1),(2,3) → 2; odd: (1,2) → 1; even: 2 → total CX = 5
        // 1Q: 4*3 = 12; total gates = 17
        XCTAssertEqual(circuit.gates.count, 12 + 5)

        var uCount = 0
        var cxCount = 0
        for gate in circuit.gates {
            switch gate {
            case .u:
                uCount += 1
            case .cx:
                cxCount += 1
            default:
                XCTFail("unexpected gate \(gate)")
            }
        }
        XCTAssertEqual(uCount, 12)
        XCTAssertEqual(cxCount, 5)

        XCTAssertEqual(
            CrossEntropyBenchmark.adjacentCXPairs(layer: 0, qubitCount: 4).map { [$0.0, $0.1] },
            [[0, 1], [2, 3]]
        )
        XCTAssertEqual(
            CrossEntropyBenchmark.adjacentCXPairs(layer: 1, qubitCount: 4).map { [$0.0, $0.1] },
            [[1, 2]]
        )
    }

    // MARK: - Noiseless Sampler vs ideal (CPU SV)

    func testNoiselessLinearXEBMatchesIdealExpectationCPUSampler() throws {
        // Seeded circuits; F̂ within statistical tol of E[F] = D Σ p² − 1.
        // Also require E[F] high enough that the circuit is chaotically entangling.
        let cases: [(n: Int, depth: Int, circuitSeed: UInt64, sampleSeed: UInt64)] = [
            (2, 4, 101, 201),
            (3, 5, 102, 202),
            (4, 6, 103, 203),
        ]
        let backend = CPUStatevectorBackend()
        for (n, depth, circuitSeed, sampleSeed) in cases {
            let (_, result) = try CrossEntropyBenchmark.evaluateLinearXEB(
                options: CrossEntropyBenchmarkOptions(qubitCount: n, depth: depth),
                circuitSeed: circuitSeed,
                sampleSeed: sampleSeed,
                shots: Self.xebShotCount,
                backend: backend
            )
            XCTAssertEqual(result.hilbertDimension, 1 << n)
            XCTAssertEqual(result.shots, Self.xebShotCount)
            XCTAssertGreaterThan(
                result.expectedUnderIdeal,
                Self.minExpectedFXEB,
                "n=\(n) E[F]=\(result.expectedUnderIdeal) too low for entangling circuit"
            )
            XCTAssertEqual(
                result.fxeb,
                result.expectedUnderIdeal,
                accuracy: Self.xebMatchTolerance,
                "n=\(n) F̂=\(result.fxeb) vs E[F]=\(result.expectedUnderIdeal)"
            )
        }
    }

    func testNoiselessLinearXEBApproachesOneForChaoticCircuits() throws {
        // For n=4, PT asymptote (D−1)/(D+1) ≈ 0.882. Require F̂ within tol of E[F]
        // and E[F] itself close to that band (not a Sycamore claim — just noiseless sanity).
        let options = CrossEntropyBenchmarkOptions(qubitCount: 4, depth: 8)
        let (_, result) = try CrossEntropyBenchmark.evaluateLinearXEB(
            options: options,
            circuitSeed: 911,
            sampleSeed: 912,
            shots: Self.xebShotCount,
            backend: CPUStatevectorBackend()
        )
        let ptAsymptote = (16.0 - 1.0) / (16.0 + 1.0) // 15/17
        XCTAssertEqual(
            result.expectedUnderIdeal,
            ptAsymptote,
            accuracy: 0.25,
            "E[F]=\(result.expectedUnderIdeal) should be near PT asymptote \(ptAsymptote)"
        )
        XCTAssertEqual(
            result.fxeb,
            result.expectedUnderIdeal,
            accuracy: Self.xebMatchTolerance
        )
        // Soft “→ 1” check for n=4: both should sit well above 0.6.
        XCTAssertGreaterThan(result.fxeb, 0.6)
        XCTAssertGreaterThan(result.expectedUnderIdeal, 0.6)
    }

    // MARK: - Negative control (wrong ideal)

    func testWrongIdealProbabilitiesYieldLowLinearXEB() throws {
        let options = CrossEntropyBenchmarkOptions(qubitCount: 3, depth: 5)
        let circuitA = try CrossEntropyBenchmark.makeRandomCircuit(options: options, seed: 50)
        let circuitB = try CrossEntropyBenchmark.makeRandomCircuit(options: options, seed: 99)
        let probsA = try CrossEntropyBenchmark.idealProbabilities(of: circuitA)
        let probsB = try CrossEntropyBenchmark.idealProbabilities(of: circuitB)

        let sampler = Sampler()
        let sampleOpts = QuantumRunOptions(seed: 777, shots: Self.xebShotCount)
        let shotsA = try sampler.run(
            circuit: circuitA,
            backend: CPUStatevectorBackend(),
            options: sampleOpts
        )
        guard let counts = shotsA.shotCounts else {
            return XCTFail("missing shotCounts")
        }

        let matched = try CrossEntropyBenchmark.linearXEB(
            idealProbabilities: probsA,
            shotCounts: counts
        )
        let mismatched = try CrossEntropyBenchmark.linearXEB(
            idealProbabilities: probsB,
            shotCounts: counts
        )
        let expectedA = try CrossEntropyBenchmark.expectedLinearXEB(idealProbabilities: probsA)

        XCTAssertEqual(matched, expectedA, accuracy: Self.xebMatchTolerance)
        XCTAssertLessThan(
            mismatched,
            Self.maxWrongUnitaryFXEB,
            "wrong p_U F_XEB=\(mismatched) should be near 0 / clearly below matched \(matched)"
        )
        XCTAssertLessThan(
            mismatched,
            matched - 0.25,
            "wrong unitary must be discriminative vs matched"
        )
    }

    func testLinearXEBEndiannessBellAndHadamardOracles() throws {
        // Independent of the random-circuit generator: fixed analytic states catch LSB / D bugs.
        var hadamard = try QuantumCircuit(qubitCount: 1)
        try hadamard.h(0)
        let pH = try CrossEntropyBenchmark.idealProbabilities(of: hadamard)
        XCTAssertEqual(pH[0], 0.5, accuracy: 1e-12)
        XCTAssertEqual(pH[1], 0.5, accuracy: 1e-12)
        XCTAssertEqual(
            try CrossEntropyBenchmark.expectedLinearXEB(idealProbabilities: pH),
            0.0,
            accuracy: 1e-12
        )

        // Bell: H(0); CX(0→1) → mass on |00⟩ (index 0) and |11⟩ (index 3) under engine LSB.
        var bell = try QuantumCircuit(qubitCount: 2)
        try bell.h(0)
        try bell.cx(0, 1)
        let pB = try CrossEntropyBenchmark.idealProbabilities(of: bell)
        XCTAssertEqual(pB[0], 0.5, accuracy: 1e-12)
        XCTAssertEqual(pB[1], 0.0, accuracy: 1e-12)
        XCTAssertEqual(pB[2], 0.0, accuracy: 1e-12)
        XCTAssertEqual(pB[3], 0.5, accuracy: 1e-12)
        // E[F] = 4*(0.25+0.25) − 1 = 1
        XCTAssertEqual(
            try CrossEntropyBenchmark.expectedLinearXEB(idealProbabilities: pB),
            1.0,
            accuracy: 1e-12
        )

        // Score Bell shots against Bell ideal (should sit near 1) and against bit-reversed
        // ideal (mass on 0 and 1) which is a classic endianness trap → near 0 / much lower.
        let sampler = Sampler()
        let shots = try sampler.run(
            circuit: bell,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 9, shots: Self.xebShotCount)
        )
        guard let counts = shots.shotCounts else {
            return XCTFail("missing shotCounts")
        }
        let matched = try CrossEntropyBenchmark.linearXEB(
            idealProbabilities: pB,
            shotCounts: counts
        )
        let bitReversedIdeal = [0.5, 0.5, 0.0, 0.0] // wrong endian story for this Bell
        let mismatched = try CrossEntropyBenchmark.linearXEB(
            idealProbabilities: bitReversedIdeal,
            shotCounts: counts
        )
        XCTAssertEqual(matched, 1.0, accuracy: Self.xebMatchTolerance)
        XCTAssertLessThan(mismatched, matched - 0.4)
    }

    // MARK: - Optional Metal Sampler

    func testNoiselessLinearXEBMetalSamplerWhenAvailable() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }
        let options = CrossEntropyBenchmarkOptions(qubitCount: 3, depth: 5)
        let (_, result) = try CrossEntropyBenchmark.evaluateLinearXEB(
            options: options,
            circuitSeed: 55,
            sampleSeed: 56,
            shots: Self.xebShotCount,
            backend: try StatevectorBackend()
        )
        XCTAssertEqual(
            result.fxeb,
            result.expectedUnderIdeal,
            accuracy: Self.xebMatchTolerance,
            "Metal F̂=\(result.fxeb) vs E[F]=\(result.expectedUnderIdeal)"
        )
        XCTAssertGreaterThan(result.expectedUnderIdeal, Self.minExpectedFXEB)
    }

    // MARK: - Porter–Thomas smoke (non-blocking)

    func testPorterThomasSmokeIdealProbsLookChaotic() throws {
        // Smoke only — not a Porter–Thomas proof / Sycamore claim.
        let options = CrossEntropyBenchmarkOptions(qubitCount: 4, depth: 8)
        let circuit = try CrossEntropyBenchmark.makeRandomCircuit(options: options, seed: 314)
        let probs = try CrossEntropyBenchmark.idealProbabilities(of: circuit)
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 1e-10)

        let d = Double(probs.count)
        let meanP = CrossEntropyBenchmark.meanProbability(probs)
        XCTAssertEqual(meanP, 1.0 / d, accuracy: 1e-12)

        let collision = CrossEntropyBenchmark.collisionProbability(probs)
        let ptCollision = 2.0 / (d + 1.0)
        // Smoke only: collision mass near Porter–Thomas, not a heavy PT suite.
        XCTAssertEqual(collision, ptCollision, accuracy: 0.08)
        XCTAssertGreaterThan(collision, 1.0 / d) // not fully uniform
        XCTAssertLessThan(collision, 0.5) // not a basis state
    }

    // MARK: - Helpers

    private func xebGateFingerprint(_ gate: Gate) -> String {
        String(describing: gate)
    }
}
