import XCTest
@testable import QuantumKit

/// Sampling statistical conformance (item 41).
///
/// - Seeded: bit-identical histogram replay (extends DeterminismTests; does not replace it).
/// - Unseeded: large-shot TV / χ² against frozen Born refs — thresholds chosen so flaky CI
///   is unlikely under Hoeffding / χ² tail bounds (documented below).
/// - Bias smoke: all-zero counts on `|+⟩` must fail loudly.
extension QuantumKitTests {

    // MARK: - Thresholds (document intentional CI risk budget)

    /// Unseeded fair-coin / Bell runs use this many shots.
    ///
    /// Hoeffding: `P(|p̂ − p| ≥ ε) ≤ 2 exp(−2 n ε²)`.
    /// With `n = 12_000` and TV / per-bin ε ≈ 0.04, the two-sided miss probability is
    /// ≪ 10⁻⁸ — far below typical CI flake rates. χ² critical values below use α = 10⁻⁴.
    private static let statisticalShotCount = 12_000

    /// Max total-variation distance `½ Σᵢ |p̂ᵢ − pᵢ|` for unseeded runs.
    private static let maxTotalVariation = 0.04

    /// χ² critical value, α = 1e-4, df = 1 (fair coin / Bell support collapse).
    private static let chiSquaredCriticalDF1 = 15.1367

    /// χ² critical value, α = 1e-4, df = 3 (RY Bernoulli treated as 2-bin → use df1;
    /// 4-outcome uniform would use this).
    private static let chiSquaredCriticalDF3 = 21.1075

    // MARK: - Seeded reproducibility (sampling path)

    func testSeededFairCoinShotCountsAreReproducible() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let options = QuantumRunOptions(
            seed: 41_041,
            shots: 2048,
            sampleOptions: SampleCountOptions(batchSize: 16)
        )
        let backend = CPUStatevectorBackend()
        let a = try backend.run(circuit: circuit, options: options)
        let b = try backend.run(circuit: circuit, options: options)
        XCTAssertEqual(a.shotCounts, b.shotCounts)
    }

    // MARK: - Unseeded statistical tests vs frozen Born

    func testUnseededFairCoinMatchesFrozenPlusWithinTVAndChiSquared() throws {
        let entry = try ReferenceOracleCatalog.entry(id: "plus_state")
        let circuit = try ReferenceOracleCatalog.makeCircuit(entry)
        let counts = try sampleUnseededCounts(circuit: circuit, shots: Self.statisticalShotCount)

        let tv = ShotStatistics.totalVariation(
            counts: counts,
            reference: entry.probabilities,
            shots: Self.statisticalShotCount
        )
        XCTAssertLessThanOrEqual(tv, Self.maxTotalVariation, "fair-coin TV=\(tv)")

        let chi2 = ShotStatistics.chiSquared(
            counts: counts,
            reference: entry.probabilities,
            shots: Self.statisticalShotCount
        )
        XCTAssertLessThan(
            chi2, Self.chiSquaredCriticalDF1,
            "fair-coin χ²=\(chi2) ≥ critical \(Self.chiSquaredCriticalDF1) (α=1e-4, df=1)"
        )
    }

    func testUnseededBellMatchesFrozenPhiPlusWithinTVAndChiSquared() throws {
        let entry = try ReferenceOracleCatalog.entry(id: "bell_phi_plus")
        let circuit = try ReferenceOracleCatalog.makeCircuit(entry)
        let counts = try sampleUnseededCounts(circuit: circuit, shots: Self.statisticalShotCount)

        let tv = ShotStatistics.totalVariation(
            counts: counts,
            reference: entry.probabilities,
            shots: Self.statisticalShotCount
        )
        XCTAssertLessThanOrEqual(tv, Self.maxTotalVariation, "Bell TV=\(tv)")

        // Collapse to support {00,11} for χ² df=1 (off-support bins must be empty).
        XCTAssertEqual(counts[1, default: 0], 0, "Bell leaked into |01⟩")
        XCTAssertEqual(counts[2, default: 0], 0, "Bell leaked into |10⟩")
        let supportRef = [0.5, 0.5]
        let supportCounts: [Int: Int] = [
            0: counts[0, default: 0],
            1: counts[3, default: 0]
        ]
        let chi2 = ShotStatistics.chiSquared(
            counts: supportCounts,
            reference: supportRef,
            shots: Self.statisticalShotCount
        )
        XCTAssertLessThan(
            chi2, Self.chiSquaredCriticalDF1,
            "Bell χ²=\(chi2) ≥ critical \(Self.chiSquaredCriticalDF1)"
        )
    }

    func testUnseededRYPiOver3MatchesFrozenBornWithinTV() throws {
        let entry = try ReferenceOracleCatalog.entry(id: "ry_pi_over_3")
        let circuit = try ReferenceOracleCatalog.makeCircuit(entry)
        let counts = try sampleUnseededCounts(circuit: circuit, shots: Self.statisticalShotCount)

        let tv = ShotStatistics.totalVariation(
            counts: counts,
            reference: entry.probabilities,
            shots: Self.statisticalShotCount
        )
        XCTAssertLessThanOrEqual(tv, Self.maxTotalVariation, "RY(π/3) TV=\(tv)")

        let chi2 = ShotStatistics.chiSquared(
            counts: counts,
            reference: entry.probabilities,
            shots: Self.statisticalShotCount
        )
        XCTAssertLessThan(chi2, Self.chiSquaredCriticalDF1, "RY(π/3) χ²=\(chi2)")
        _ = Self.chiSquaredCriticalDF3 // retained for future multi-outcome fixtures
    }

    // MARK: - Bias smoke (must fail on obviously broken sampler)

    func testBiasedAllZeroHistogramFailsFairCoinChecks() {
        // Synthetic broken sampler: always |0⟩ while reference is |+⟩.
        let reference = [0.5, 0.5]
        let shots = 1000
        let biased: [Int: Int] = [0: shots]

        let tv = ShotStatistics.totalVariation(counts: biased, reference: reference, shots: shots)
        XCTAssertGreaterThan(tv, Self.maxTotalVariation)

        let chi2 = ShotStatistics.chiSquared(counts: biased, reference: reference, shots: shots)
        XCTAssertGreaterThan(chi2, Self.chiSquaredCriticalDF1)

        XCTAssertFalse(
            ShotStatistics.hasBothOutcomes(counts: biased, shots: shots),
            "all-zero histogram must not pass the |+⟩ support smoke check"
        )
    }

    func testPlusStateSamplerIsNotAllZero() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let counts = try sampleUnseededCounts(circuit: circuit, shots: 512)
        XCTAssertTrue(
            ShotStatistics.hasBothOutcomes(counts: counts, shots: 512),
            "fair coin / |+⟩ must produce both |0⟩ and |1⟩ counts"
        )
        XCTAssertGreaterThan(counts[1, default: 0], 0)
        XCTAssertGreaterThan(counts[0, default: 0], 0)
    }

    // MARK: - Helpers

    private func sampleUnseededCounts(circuit: QuantumCircuit, shots: Int) throws -> [Int: Int] {
        let result = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: nil,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 32)
            )
        )
        return try XCTUnwrap(result.shotCounts?.counts)
    }
}

/// Lightweight multinomial diagnostics for shot histograms (test-only).
enum ShotStatistics {
    /// `½ Σ |nᵢ/n − pᵢ|` over `reference.indices` (missing counts treated as 0).
    static func totalVariation(counts: [Int: Int], reference: [Double], shots: Int) -> Double {
        precondition(shots > 0)
        var sum = 0.0
        for (index, p) in reference.enumerated() {
            let empirical = Double(counts[index, default: 0]) / Double(shots)
            sum += abs(empirical - p)
        }
        // Any mass outside the reference support also contributes.
        let known = Set(reference.indices)
        for (key, value) in counts where !known.contains(key) && value > 0 {
            sum += Double(value) / Double(shots)
        }
        return 0.5 * sum
    }

    /// Pearson χ² = Σ (O − E)² / E over bins with `E = n p > 0`.
    static func chiSquared(counts: [Int: Int], reference: [Double], shots: Int) -> Double {
        precondition(shots > 0)
        var chi2 = 0.0
        for (index, p) in reference.enumerated() {
            let expected = Double(shots) * p
            guard expected > 0 else {
                precondition(counts[index, default: 0] == 0, "mass in zero-probability bin \(index)")
                continue
            }
            let observed = Double(counts[index, default: 0])
            let delta = observed - expected
            chi2 += (delta * delta) / expected
        }
        return chi2
    }

    static func hasBothOutcomes(counts: [Int: Int], shots: Int) -> Bool {
        let c0 = counts[0, default: 0]
        let c1 = counts[1, default: 0]
        return c0 > 0 && c1 > 0 && (c0 + c1) <= shots
    }
}
