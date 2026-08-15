import XCTest
@testable import QuantumKit

/// Quantum Volume–**style** ideal HOP (noiseless CPU SV) — not an IBM QV certificate.
///
/// **SU(4):** Haar U(4) via Ginibre QR (Mezzadri), projected to SU(4); applied as
/// row-major ``customUnitary`` 4×4. See ``QuantumVolume``.
///
/// **Pairing:** default ``nearestNeighborAlternating`` (routing-friendly). Cross et al.
/// use per-layer random pairing — covered via ``randomPermutation`` + SWAP routing tests.
///
/// **HOP:** ideal Born probs from ``CPUStateVector/probabilitiesDouble()``;
/// heavy = `{x : p(x) > median(p)}`; `HOP = Σ_{heavy} p(x)`. Noiseless check uses
/// this ideal HOP (not shot estimates / CI). Oracle is CPU SV — not ``CircuitUnitary``.
///
/// **n range:** acceptance tests cover square model circuits for `n ∈ {2,3,4}`.
///
/// **Transpile:** ibmEagle basis translate deferred (`customUnitary` unsupported).
/// Covered: linear-coupling ``BasicSwapRoutingPass`` (NN bit-identical; random-perm + SWAPs).
extension QuantumKitTests {

    // MARK: - HOP formula (hand case)

    func testHeavyOutputProbabilityFormulaOnHandDistribution() throws {
        // Sorted: 0.1, 0.2, 0.3, 0.4 → median = (0.2+0.3)/2 = 0.25
        // Heavy: 0.3, 0.4 → HOP = 0.7
        let probs = [0.4, 0.3, 0.2, 0.1]
        let result = try QuantumVolume.heavyOutputProbability(probabilities: probs)
        XCTAssertEqual(result.median, 0.25, accuracy: 1e-15)
        XCTAssertEqual(result.hop, 0.7, accuracy: 1e-15)
        XCTAssertEqual(Set(result.heavyIndices), Set([0, 1]))
    }

    func testHeavyOutputProbabilityUniformIsHalfPlusEpsilonBoundary() throws {
        // Four equal probs: median = 0.25; none strictly greater → HOP = 0.
        let probs = [Double](repeating: 0.25, count: 4)
        let result = try QuantumVolume.heavyOutputProbability(probabilities: probs)
        XCTAssertEqual(result.median, 0.25, accuracy: 1e-15)
        XCTAssertEqual(result.hop, 0.0, accuracy: 1e-15)
        XCTAssertTrue(result.heavyIndices.isEmpty)
    }

    // MARK: - Generator reproducibility + structure

    func testQuantumVolumeGeneratorIsSeededReproducible() throws {
        let options = QuantumVolumeOptions(qubitCount: 3)
        let a = try QuantumVolume.makeModelCircuit(options: options, seed: 99)
        let b = try QuantumVolume.makeModelCircuit(options: options, seed: 99)
        let c = try QuantumVolume.makeModelCircuit(options: options, seed: 100)

        XCTAssertEqual(a.gates.count, b.gates.count)
        for (lhs, rhs) in zip(a.gates, b.gates) {
            XCTAssertEqual(gateFingerprint(lhs), gateFingerprint(rhs))
        }
        XCTAssertNotEqual(
            a.gates.map(gateFingerprint),
            c.gates.map(gateFingerprint)
        )
    }

    func testNearestNeighborPairingAlternatesAsDocumented() {
        XCTAssertEqual(
            QuantumVolume.nearestNeighborPairs(layer: 0, qubitCount: 4).map { [$0.0, $0.1] },
            [[0, 1], [2, 3]]
        )
        XCTAssertEqual(
            QuantumVolume.nearestNeighborPairs(layer: 1, qubitCount: 4).map { [$0.0, $0.1] },
            [[1, 2]]
        )
        XCTAssertEqual(
            QuantumVolume.nearestNeighborPairs(layer: 0, qubitCount: 3).map { [$0.0, $0.1] },
            [[0, 1]]
        )
        XCTAssertEqual(
            QuantumVolume.nearestNeighborPairs(layer: 1, qubitCount: 3).map { [$0.0, $0.1] },
            [[1, 2]]
        )
    }

    func testSquareModelCircuitGateCountMatchesPairing() throws {
        let n = 4
        let circuit = try QuantumVolume.makeModelCircuit(
            options: QuantumVolumeOptions(qubitCount: n),
            seed: 7
        )
        // depth n; even layers: 2 pairs; odd layers: 1 pair → 2+1+2+1 = 6 SU(4)s
        XCTAssertEqual(circuit.gates.count, 6)
        for gate in circuit.gates {
            guard case .customUnitary(let matrix, let qubits) = gate else {
                return XCTFail("expected customUnitary, got \(gate)")
            }
            XCTAssertEqual(matrix.count, 16)
            XCTAssertEqual(qubits.count, 2)
        }

        let n2 = try QuantumVolume.makeModelCircuit(
            options: QuantumVolumeOptions(qubitCount: 2),
            seed: 3
        )
        // n=2: every layer is the single edge (0,1) → depth 2 gates
        XCTAssertEqual(n2.gates.count, 2)
    }

    func testSampledSU4PassesUnitaryValidation() throws {
        var rng = QuantumRNG.seeded(12345)
        for _ in 0..<8 {
            let matrix = QuantumVolume.sampleHaarSU4(rng: &rng)
            XCTAssertNoThrow(try UnitaryValidation.validateUnitary(matrix: matrix, dimension: 4))
            let (detAbs, detArg) = complexDeterminantAbsArg(matrix, dimension: 4)
            XCTAssertEqual(detAbs, 1.0, accuracy: 1e-6, "SU(4) |det|")
            XCTAssertEqual(detArg, 0.0, accuracy: 1e-5, "SU(4) arg(det) after special projection")
        }
    }

    // MARK: - Noiseless HOP > 2/3 (CPU SV)

    func testIdealHOPExceedsTwoThirdsForWidths2Through4() throws {
        // Seeded, non-flaky: fixed seeds per width. Ideal HOP from CPU SV Born probs.
        // Cross requires strict HOP > 2/3 (not ≥).
        let cases: [(n: Int, seed: UInt64)] = [
            (2, 1),
            (2, 2),
            (3, 11),
            (3, 12),
            (4, 21),
            (4, 22),
        ]
        for (n, seed) in cases {
            let (_, hop) = try QuantumVolume.evaluateIdealHOP(
                options: QuantumVolumeOptions(qubitCount: n),
                seed: seed
            )
            XCTAssertEqual(hop.probabilityCount, 1 << n)
            XCTAssertGreaterThan(
                hop.hop,
                HeavyOutputProbability.acceptanceThreshold,
                "n=\(n) seed=\(seed) HOP=\(hop.hop)"
            )
            // Probabilities are a distribution.
            let probs = try QuantumVolume.idealProbabilities(
                of: try QuantumVolume.makeModelCircuit(
                    options: QuantumVolumeOptions(qubitCount: n),
                    seed: seed
                )
            )
            XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 1e-10)
        }
    }

    func testIdealHOPMeanOverManySeedsStaysAboveTwoThirds() throws {
        // Coverage beyond cherry-picked seeds: Haar QV expectation is ~(1+ln2)/2 ≈ 0.85.
        // Require every draw > 2/3 and the mean comfortably in the Haar band.
        let n = 3
        let seeds = (0..<24).map { UInt64($0 + 1) }
        var hops: [Double] = []
        for seed in seeds {
            let hop = try QuantumVolume.evaluateIdealHOP(
                options: QuantumVolumeOptions(qubitCount: n),
                seed: seed
            ).hop.hop
            XCTAssertGreaterThan(
                hop,
                HeavyOutputProbability.acceptanceThreshold,
                "n=\(n) seed=\(seed) HOP=\(hop)"
            )
            hops.append(hop)
        }
        let mean = hops.reduce(0, +) / Double(hops.count)
        XCTAssertGreaterThan(mean, 0.75, "mean HOP=\(mean) should sit near Haar ~0.85")
        XCTAssertLessThan(mean, 0.95, "mean HOP=\(mean) should not look stuck near 1")
    }

    func testIdealHOPIsDeterministicForFixedSeed() throws {
        let options = QuantumVolumeOptions(qubitCount: 3, depth: 3)
        let a = try QuantumVolume.evaluateIdealHOP(options: options, seed: 42).hop
        let b = try QuantumVolume.evaluateIdealHOP(options: options, seed: 42).hop
        XCTAssertEqual(a.hop, b.hop, accuracy: 0)
        XCTAssertEqual(a.median, b.median, accuracy: 0)
        XCTAssertEqual(a.heavyIndices, b.heavyIndices)
    }

    // MARK: - Safe linear routing (basis translate deferred)

    func testLinearCouplingRoutingPreservesIdealHOPForNearestNeighborQV() throws {
        // NN layers already lie on CouplingMap.linear edges → no SWAPs with identity layout.
        // Compare ideal HOP of logical vs routed; also require CPU Born match.
        let options = QuantumVolumeOptions(qubitCount: 3)
        let logical = try QuantumVolume.makeModelCircuit(options: options, seed: 55)
        let routed = try QuantumVolume.routeOntoLinearCoupling(logical, seed: nil)

        XCTAssertFalse(routed.gates.contains { if case .swap = $0 { return true }; return false })

        let hopLogical = try QuantumVolume.idealHeavyOutputProbability(of: logical)
        let hopRouted = try QuantumVolume.idealHeavyOutputProbability(of: routed)
        XCTAssertEqual(hopLogical.hop, hopRouted.hop, accuracy: 1e-12)
        XCTAssertGreaterThan(hopRouted.hop, HeavyOutputProbability.acceptanceThreshold)

        let pL = try QuantumVolume.idealProbabilities(of: logical)
        let pR = try QuantumVolume.idealProbabilities(of: routed)
        XCTAssertEqual(pL.count, pR.count)
        for i in pL.indices {
            XCTAssertEqual(pL[i], pR[i], accuracy: 1e-12)
        }
    }

    func testRoutedHOPStillAboveThresholdVersusLogicalIdealHeavySet() throws {
        // Documented comparison: heavy set from *logical* ideal; HOP of routed circuit
        // summed over that set (compiled-vs-logical). For NN+linear with no SWAPs this
        // equals ideal HOP of either circuit.
        let options = QuantumVolumeOptions(qubitCount: 4)
        let logical = try QuantumVolume.makeModelCircuit(options: options, seed: 77)
        let routed = try QuantumVolume.routeOntoLinearCoupling(logical)

        let logicalHOP = try QuantumVolume.idealHeavyOutputProbability(of: logical)
        let routedProbs = try QuantumVolume.idealProbabilities(of: routed)
        let hopOnLogicalHeavy = logicalHOP.heavyIndices.reduce(0.0) { $0 + routedProbs[$1] }

        XCTAssertEqual(hopOnLogicalHeavy, logicalHOP.hop, accuracy: 1e-12)
        XCTAssertGreaterThan(hopOnLogicalHeavy, HeavyOutputProbability.acceptanceThreshold)
    }

    func testRandomPermutationRoutingInsertsSwapsAndPreservesHOPMultiset() throws {
        // Non-NN layers force BasicSwapRoutingPass to insert SWAPs on a line.
        // Final layout may permute wires → bitstrings need not match index-wise, but
        // the probability *multiset* (hence ideal HOP) is invariant under qubit relabeling.
        let options = QuantumVolumeOptions(
            qubitCount: 4,
            depth: 3,
            pairing: .randomPermutation
        )
        let logical = try QuantumVolume.makeModelCircuit(options: options, seed: 404)
        let routed = try QuantumVolume.routeOntoLinearCoupling(logical, seed: 1)

        let swapCount = routed.gates.reduce(0) { partial, gate in
            if case .swap = gate { return partial + 1 }
            return partial
        }
        XCTAssertGreaterThan(swapCount, 0, "expected SWAPs for random-permutation QV on a line")

        let pL = try QuantumVolume.idealProbabilities(of: logical)
        let pR = try QuantumVolume.idealProbabilities(of: routed)
        let hopL = try QuantumVolume.heavyOutputProbability(probabilities: pL)
        let hopR = try QuantumVolume.heavyOutputProbability(probabilities: pR)

        XCTAssertEqual(hopL.hop, hopR.hop, accuracy: 1e-10)
        XCTAssertGreaterThan(hopL.hop, HeavyOutputProbability.acceptanceThreshold)

        let sortedL = pL.sorted()
        let sortedR = pR.sorted()
        XCTAssertEqual(sortedL.count, sortedR.count)
        for i in sortedL.indices {
            XCTAssertEqual(sortedL[i], sortedR[i], accuracy: 1e-10)
        }
        // Index-wise p[i] may differ after leftover SWAPs; HOP/multiset is the oracle check.
    }

    // MARK: - Helpers

    private func gateFingerprint(_ gate: Gate) -> String {
        switch gate {
        case .customUnitary(let matrix, let qubits):
            let body = matrix.map { String(format: "%.6g%+.6gi", Double($0.real), Double($0.imaginary)) }
                .joined(separator: ",")
            return "U4(\(qubits))[\(body)]"
        default:
            return String(describing: gate)
        }
    }

    /// |det| and arg(det) for a row-major complex matrix (Gaussian elimination).
    private func complexDeterminantAbsArg(
        _ matrix: [ComplexAmplitude],
        dimension: Int
    ) -> (abs: Double, arg: Double) {
        typealias C = (re: Double, im: Double)
        func mul(_ a: C, _ b: C) -> C {
            (a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
        }
        func sub(_ a: C, _ b: C) -> C { (a.re - b.re, a.im - b.im) }

        var a = matrix.map { (re: Double($0.real), im: Double($0.imaginary)) }
        var det: C = (1, 0)
        for k in 0..<dimension {
            var pivot = k
            var best = a[k * dimension + k].re * a[k * dimension + k].re
                + a[k * dimension + k].im * a[k * dimension + k].im
            for r in (k + 1)..<dimension {
                let nrm = a[r * dimension + k].re * a[r * dimension + k].re
                    + a[r * dimension + k].im * a[r * dimension + k].im
                if nrm > best {
                    best = nrm
                    pivot = r
                }
            }
            if pivot != k {
                for c in 0..<dimension {
                    a.swapAt(k * dimension + c, pivot * dimension + c)
                }
                det = (-det.re, -det.im)
            }
            let diag = a[k * dimension + k]
            det = mul(det, diag)
            let invN = diag.re * diag.re + diag.im * diag.im
            guard invN > 1e-30 else { return (0, 0) }
            let inv: C = (diag.re / invN, -diag.im / invN)
            for r in (k + 1)..<dimension {
                let factor = mul(a[r * dimension + k], inv)
                for c in k..<dimension {
                    a[r * dimension + c] = sub(a[r * dimension + c], mul(factor, a[k * dimension + c]))
                }
            }
        }
        let absDet = sqrt(det.re * det.re + det.im * det.im)
        return (absDet, atan2(det.im, det.re))
    }
}
