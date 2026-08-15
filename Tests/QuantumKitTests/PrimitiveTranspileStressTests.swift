import XCTest
@testable import QuantumKit

/// Estimator / Sampler / transpile stress (seeded, CPU-first).
///
/// **QWC:** ``EstimatorOptions/groupCommutingPaulis`` defaults `true` (shared ensemble per
/// qubit-wise commuting group). `false` = legacy per-term ensembles (different seed
/// schedule; both remain unbiased vs exact).
///
/// **Transpile:** ibmEagle (+ optional linear map) on a **Born-safe** gate subset
/// (H/Z/S/SX/RZ/CX). Excludes X/Y/RX (basis expand not Born-faithful). Random circuits
/// use ``optimizationLevel`` 1 — level 2 can fold `SX·SX→X` then hit the bad X expand.
/// Non-adjacent CX on a line asserts routing (CX inflation after SWAP→basis) and Born **multisets**.
extension QuantumKitTests {

    // MARK: - Estimator Pauli sums: exact vs shots (± SE), QWC on/off

    func testPrimitiveStress_Estimator_exactVsShots_withinSE_QWConAndOff() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.ry(theta: QFloat(0.55), 2)
        try circuit.cz(1, 2)

        let hamiltonian = try Hamiltonian(terms: [
            PauliTerm(coefficient: 0.4, label: "Z0"),
            PauliTerm(coefficient: 0.3, label: "Z1"),
            PauliTerm(coefficient: 0.5, label: "Z0 Z1"),
            PauliTerm(coefficient: -0.2, label: "X2"),
            PauliTerm(coefficient: 0.15, label: "Z2"),
        ])
        // Greedy QWC: {Z0,Z1,Z0Z1,X2} + {Z2} (X2 joins Z-axes before Z2 conflicts).
        XCTAssertEqual(PauliCommutingGroups.partition(hamiltonian).groups.count, 2)

        let backend = CPUStatevectorBackend()
        let exact = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend
        )
        XCTAssertNil(exact.shots)
        XCTAssertNil(exact.standardError)

        // QWC on (default): two ensembles for this H.
        PauliShotEstimator.resetSamplingEnsembleCountForTests()
        let qwcOn = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 2_041),
            estimatorOptions: EstimatorOptions(shots: 8_192, groupCommutingPaulis: true)
        )
        XCTAssertEqual(
            PauliShotEstimator.samplingEnsembleCountForTests, 2,
            "QWC should yield exactly 2 ensembles for this Hamiltonian"
        )

        // QWC off: one ensemble per non-identity term.
        PauliShotEstimator.resetSamplingEnsembleCountForTests()
        let qwcOff = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 9_901),
            estimatorOptions: EstimatorOptions(shots: 8_192, groupCommutingPaulis: false)
        )
        XCTAssertEqual(PauliShotEstimator.samplingEnsembleCountForTests, 5)

        assertShotNearExactWithinSE(shot: qwcOn, exact: exact.value, label: "QWC on")
        assertShotNearExactWithinSE(shot: qwcOff, exact: exact.value, label: "QWC off")
    }

    // MARK: - Sampler + readout mitigation round-trip

    func testPrimitiveStress_Sampler_readoutMitigation_syntheticConfusion() throws {
        // ReadoutConfusionMatrix is present — forward via NoiseModel, inverse via resilience.
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.22, p10: 0.18)
        var noise = NoiseModel()
        noise.readoutConfusion = matrix

        let circuit = try QuantumCircuit(qubitCount: 1)
        // Leave |0⟩; confusion alone drives measured flips.
        let backend = CPUStatevectorBackend()
        let seed: UInt64 = 77
        let shots = 6_000

        let raw = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(noise: noise, seed: seed, shots: shots)
        )
        let mitigated = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(
                noise: noise,
                seed: seed,
                shots: shots,
                resilience: ResilienceOptions(readoutMitigation: matrix)
            )
        )

        let rawP0 = Double(raw.quasiProbabilities["0"] ?? 0)
        let mitP0 = Double(mitigated.quasiProbabilities["0"] ?? 0)
        XCTAssertLessThan(rawP0, 0.95, "confusion should flip some |0⟩ shots")
        XCTAssertGreaterThan(mitP0, rawP0)
        XCTAssertGreaterThan(mitP0, 0.90, "mitigation should recover prepared |0⟩")

        // Histogram-only round-trip: synthetic noisy counts → inverse ≈ prepared.
        let synthetic = ShotCounts(shots: 2_000, counts: [0: 1_560, 1: 440])
        let recovered = try ReadoutMitigation.apply(
            to: synthetic,
            matrix: matrix,
            qubitCount: 1
        )
        let recoveredP0 = Double(recovered.counts[0, default: 0]) / 2_000.0
        XCTAssertGreaterThan(recoveredP0, 0.90)
    }

    // MARK: - Transpile ibmEagle / linear map: CPU Born TV

    func testPrimitiveStress_Transpile_ibmEagle_CPUBornTV() throws {
        // Basis translate only (no coupling map) — compares logical Born directly.
        let seeds: [UInt64] = [3, 41, 128]
        let qubitCounts = [2, 3, 4]

        for n in qubitCounts {
            for seed in seeds {
                let original = try makeIbmEagleSafeRandomCircuit(
                    qubitCount: n,
                    depth: n + 3,
                    seed: seed
                )
                let transpiled = try Transpiler.transpile(
                    original,
                    options: TranspileOptions(
                        targetBasis: .ibmEagle,
                        optimizationLevel: 1,
                        seedTranspiler: seed &+ 17
                    )
                )

                let tv = try cpuBornTotalVariation(original, transpiled)
                XCTAssertLessThan(
                    tv, 1e-6,
                    "ibmEagle Born TV n=\(n) seed=\(seed) TV=\(tv)"
                )
            }
        }
    }

    func testPrimitiveStress_Transpile_linearMap_adjacentCX_CPUBornTV() throws {
        // Linear map + pinned identity layout + adjacent-only CX (no wrap) so routing
        // does not leave a logical↔physical permutation that would invalidate raw Born TV.
        let seeds: [UInt64] = [3, 41, 128]
        for n in [2, 3, 4] {
            for seed in seeds {
                let original = try makeIbmEagleSafeRandomCircuit(
                    qubitCount: n,
                    depth: n + 3,
                    seed: seed,
                    adjacentCXOnly: true
                )
                let transpiled = try Transpiler.transpile(
                    original,
                    options: TranspileOptions(
                        targetBasis: .ibmEagle,
                        couplingMap: try CouplingMap.linear(n),
                        initialLayout: try Layout.identity(qubitCount: n),
                        optimizationLevel: 1,
                        seedTranspiler: seed &+ 17
                    )
                )
                let tv = try cpuBornTotalVariation(original, transpiled)
                XCTAssertLessThan(
                    tv, 1e-6,
                    "linear+ibmEagle Born TV n=\(n) seed=\(seed) TV=\(tv)"
                )
            }
        }
    }

    func testPrimitiveStress_Transpile_linearMap_nonAdjacentCX_routing_BornMultiset() throws {
        // Force BasicSwapRoutingPass on a line: CX(0,n-1) is not an edge for n≥3.
        // ibmEagle has no native SWAP — routing SWAPs expand to CX — so assert CX inflation,
        // then Born *multisets* (leftover wire permutation may scramble index-wise probs).
        let seeds: [UInt64] = [3, 41, 128]
        for n in [3, 4] {
            for seed in seeds {
                var original = try QuantumCircuit(qubitCount: n)
                try original.h(0)
                try original.cx(0, n - 1)
                try original.rz(theta: QFloat(0.31 + 0.01 * Double(seed % 7)), 1)
                try original.s(n - 1)

                let transpiled = try Transpiler.transpile(
                    original,
                    options: TranspileOptions(
                        targetBasis: .ibmEagle,
                        couplingMap: try CouplingMap.linear(n),
                        initialLayout: try Layout.identity(qubitCount: n),
                        optimizationLevel: 1,
                        seedTranspiler: seed &+ 17
                    )
                )
                let cxOriginal = original.gates.reduce(0) { partial, gate in
                    if case .cx = gate { return partial + 1 }
                    return partial
                }
                let cxTranspiled = transpiled.gates.reduce(0) { partial, gate in
                    if case .cx = gate { return partial + 1 }
                    return partial
                }
                XCTAssertGreaterThan(
                    cxTranspiled, cxOriginal,
                    "routing should expand CX(0,\(n - 1)) via SWAP→CX on linear n=\(n) seed=\(seed)"
                )

                let pa = try cpuBornProbabilities(original)
                let pb = try cpuBornProbabilities(transpiled)
                let sortedA = pa.sorted()
                let sortedB = pb.sorted()
                XCTAssertEqual(sortedA.count, sortedB.count)
                for index in sortedA.indices {
                    XCTAssertEqual(
                        sortedA[index], sortedB[index],
                        accuracy: 1e-6,
                        "Born multiset n=\(n) seed=\(seed) idx=\(index)"
                    )
                }
            }
        }
    }

    func testPrimitiveStress_Transpile_Bell_ibmEagle_CPUBornTV() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let out = try Transpiler.transpile(
            circuit,
            options: TranspileOptions(
                targetBasis: .ibmEagle,
                couplingMap: try CouplingMap.linear(2),
                initialLayout: try Layout.identity(qubitCount: 2),
                optimizationLevel: 2,
                seedTranspiler: 11
            )
        )
        let tv = try cpuBornTotalVariation(circuit, out)
        XCTAssertLessThan(tv, 1e-6)
    }

    // MARK: - Helpers

    /// |Ê − E| ≤ 5·SE (≈ non-flaky). Tiny floor only guards SE≈0 numerical edges — not a 0.06 always-pass.
    private func assertShotNearExactWithinSE(
        shot: EstimatorResult,
        exact: QFloat,
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let se = shot.standardError
        XCTAssertNotNil(se, "\(label): expected SE", file: file, line: line)
        guard let se else { return }
        let err = abs(Double(shot.value) - Double(exact))
        let bound = max(5.0 * Double(se), 1e-3)
        XCTAssertLessThanOrEqual(
            err, bound,
            "\(label): |shot−exact|=\(err) SE=\(se) bound=\(bound)",
            file: file,
            line: line
        )
    }

    private func cpuBornProbabilities(_ circuit: QuantumCircuit) throws -> [Double] {
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        _ = try engine.execute(circuit, on: state)
        return state.probabilitiesDouble()
    }

    private func cpuBornTotalVariation(
        _ lhs: QuantumCircuit,
        _ rhs: QuantumCircuit
    ) throws -> Double {
        let pa = try cpuBornProbabilities(lhs)
        let pb = try cpuBornProbabilities(rhs)
        XCTAssertEqual(pa.count, pb.count)
        var sum = 0.0
        for index in pa.indices {
            sum += abs(pa[index] - pb[index])
        }
        return 0.5 * sum
    }

    /// Gate subset safe under ibmEagle basis translate (no X / Y / RX).
    private func makeIbmEagleSafeRandomCircuit(
        qubitCount: Int,
        depth: Int,
        seed: UInt64,
        adjacentCXOnly: Bool = false
    ) throws -> QuantumCircuit {
        var rng = QuantumRNG.seeded(seed)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)
        for _ in 0..<depth {
            let q0 = rng.nextInt(upperBound: qubitCount)
            switch rng.nextInt(upperBound: 7) {
            case 0: try circuit.h(q0)
            case 1: try circuit.z(q0)
            case 2: try circuit.s(q0)
            case 3: try circuit.sx(q0)
            case 4: try circuit.rz(theta: QFloat(rng.nextUnitDouble() * Double.pi), q0)
            case 5 where qubitCount >= 2:
                if adjacentCXOnly {
                    let q = rng.nextInt(upperBound: qubitCount - 1)
                    try circuit.cx(q, q + 1)
                } else {
                    let q1 = (q0 + 1) % qubitCount
                    try circuit.cx(q0, q1)
                }
            default: try circuit.h(q0)
            }
        }
        return circuit
    }
}
