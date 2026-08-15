import XCTest
@testable import QuantumKit

/// Cross-backend Born / Pauli / histogram stress on the same seeded circuits (n ≤ 6).
///
/// Compares exact engine outputs where available; Stabilizer uses seeded shot TV vs CPU SV Born.
/// Metal paths XCTSkip when `MetalRuntime.isAvailable` is false.
/// MPS: local path + χ-budget honesty (GHZ stays χ≤2; long-range CX needs χ>1).
extension QuantumKitTests {

    // MARK: - CPU SV ↔ Metal SV

    func testParityStress_CPUSV_vs_MetalSV_seededCircuits() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let metalEngine = try QuantumEngine()
        let cpuEngine = CPUStatevectorEngine()
        let seeds: [UInt64] = [7, 42, 99_001]
        let qubitCounts = [3, 4, 5]

        for n in qubitCounts {
            for seed in seeds {
                let circuit = try makeParityStressUnitaryCircuit(qubitCount: n, depth: n + 4, seed: seed)

                let cpu = try CPUStateVector(qubitCount: n)
                _ = try cpuEngine.execute(circuit, on: cpu)
                let cpuProbs = cpu.probabilitiesDouble()

                let metal = try StateVector(qubitCount: n, device: metalEngine.device)
                _ = try metalEngine.execute(circuit, on: metal)
                let metalProbs = try QuantumMeasurement.probabilities(state: metal, engine: metalEngine)
                let metalAmps = QuantumMeasurement.amplitudes(state: metal)

                XCTAssertEqual(cpuProbs.count, metalProbs.count)
                XCTAssertEqual(metalAmps.count, cpu.stateCount)
                for index in cpuProbs.indices {
                    XCTAssertEqual(
                        cpuProbs[index], Double(metalProbs[index]),
                        accuracy: 1e-4,
                        "CPU↔Metal Born n=\(n) seed=\(seed) idx=\(index)"
                    )
                    let ampProb =
                        Double(metalAmps[index].real) * Double(metalAmps[index].real)
                        + Double(metalAmps[index].imaginary) * Double(metalAmps[index].imaginary)
                    XCTAssertEqual(
                        cpuProbs[index], ampProb,
                        accuracy: 1e-4,
                        "Metal amp Born n=\(n) seed=\(seed) idx=\(index)"
                    )
                }
            }
        }
    }

    // MARK: - CPU SV ↔ CPU DM (pure)

    func testParityStress_CPUSV_vs_CPUDM_probsAndPauliZ() throws {
        let cpuSV = CPUStatevectorEngine()
        let cpuDM = CPUDensityMatrixEngine()
        let seeds: [UInt64] = [11, 2024, 88_877]
        let qubitCounts = [3, 4, 5]

        for n in qubitCounts {
            for seed in seeds {
                let circuit = try makeParityStressUnitaryCircuit(qubitCount: n, depth: n + 5, seed: seed)

                let sv = try CPUStateVector(qubitCount: n)
                _ = try cpuSV.execute(circuit, on: sv)
                let svProbs = sv.probabilitiesDouble()

                let dm = try CPUDensityMatrix(qubitCount: n)
                _ = try cpuDM.execute(circuit, on: dm)
                let dmProbs = dm.probabilitiesDouble()

                XCTAssertEqual(svProbs.count, dmProbs.count)
                for index in svProbs.indices {
                    XCTAssertEqual(
                        svProbs[index], dmProbs[index],
                        accuracy: 1e-10,
                        "CPU SV↔DM Born n=\(n) seed=\(seed) idx=\(index)"
                    )
                }

                // Single-qubit ⟨Z⟩ and all-Z string on the same pure state.
                for q in 0..<n {
                    let svZ = try QuantumMeasurement.expectation(state: sv, paulis: [q: .z])
                    let dmZ = try QuantumMeasurement.expectation(density: dm, paulis: [q: .z])
                    XCTAssertEqual(
                        Double(svZ), Double(dmZ),
                        accuracy: 1e-9,
                        "⟨Z\(q)⟩ SV↔DM n=\(n) seed=\(seed)"
                    )
                }
                var allZ: [Int: Pauli] = [:]
                for q in 0..<n { allZ[q] = .z }
                let svZZ = try QuantumMeasurement.expectation(state: sv, paulis: allZ)
                let dmZZ = try QuantumMeasurement.expectation(density: dm, paulis: allZ)
                XCTAssertEqual(
                    Double(svZZ), Double(dmZZ),
                    accuracy: 1e-9,
                    "⟨Z…Z⟩ SV↔DM n=\(n) seed=\(seed)"
                )
            }
        }
    }

    // MARK: - Stabilizer histogram ↔ CPU SV (Clifford)

    func testParityStress_Stabilizer_vs_CPUSV_cliffordHistogram() throws {
        let stab = StabilizerBackend()
        let cpuSV = CPUStatevectorEngine()
        let shots = 8_000
        let seeds: [UInt64] = [3, 17, 501]
        let qubitCounts = [3, 4, 5]

        for n in qubitCounts {
            for seed in seeds {
                let circuit = try makeParityStressCliffordCircuit(qubitCount: n, depth: n + 6, seed: seed)

                let sv = try CPUStateVector(qubitCount: n)
                _ = try cpuSV.execute(circuit, on: sv)
                let exact = sv.probabilitiesDouble()

                let result = try stab.run(
                    circuit: circuit,
                    options: QuantumRunOptions(seed: seed, shots: shots)
                )
                let counts = try XCTUnwrap(result.shotCounts?.counts)
                let tv = ShotStatistics.totalVariation(counts: counts, reference: exact, shots: shots)
                // 8k shots: TV ≲ 0.04 is typical for these Cliffords; 0.05 leaves margin without
                // being an always-pass floor.
                XCTAssertLessThan(
                    tv, 0.05,
                    "Stabilizer TV vs CPU SV Born n=\(n) seed=\(seed) TV=\(tv)"
                )
                for index in exact.indices where exact[index] < 1e-14 {
                    XCTAssertEqual(
                        counts[index, default: 0], 0,
                        "Stabilizer mass outside SV support n=\(n) seed=\(seed) idx=\(index)"
                    )
                }
            }
        }
    }

    // MARK: - MPS local path ↔ CPU SV (bond growth under large χ budget)

    // MARK: - MPS local path ↔ CPU SV

    /// Adjacent CX chain is GHZ-like (χ≤2). χ_max=128 is only a budget — assert low bond + Born.
    func testParityStress_MPS_adjacentCXChain_vs_CPUSV_lowBond() throws {
        let cpuSV = CPUStatevectorEngine()
        let config = MPSConfiguration(maxBondDimension: 128, svdTruncationThreshold: 1e-14)
        let mpsEngine = MPSEngine(configuration: config)

        for n in [4, 5, 6] {
            let circuit = try makeParityStressAdjacentCXChain(qubitCount: n, seed: UInt64(10 + n))
            let sv = try CPUStateVector(qubitCount: n)
            _ = try cpuSV.execute(circuit, on: sv)
            let svProbs = sv.probabilitiesDouble()

            var mps = try MPSState(qubitCount: n, configuration: config)
            XCTAssertFalse(mps.usesDenseEvolution)
            _ = try mpsEngine.execute(circuit, on: &mps)
            let maxBond = mps.bondDimensions.max() ?? 0
            XCTAssertLessThanOrEqual(maxBond, 2, "GHZ-like chain should stay χ≤2, got \(maxBond)")

            let mpsProbs = try mps.probabilities()
            for index in svProbs.indices {
                XCTAssertEqual(
                    svProbs[index], Double(mpsProbs[index]),
                    accuracy: 1e-5,
                    "MPS GHZ↔CPU SV n=\(n) idx=\(index)"
                )
            }
        }
    }

    /// Non-adjacent CX uses the local SWAP-bubble path; χ=1 must truncate vs CPU SV.
    func testParityStress_MPS_nonAdjacentCX_vs_CPUSV_chiBudgetMatters() throws {
        let cpuSV = CPUStatevectorEngine()
        var circuit = try QuantumCircuit(qubitCount: 5)
        try circuit.h(0)
        try circuit.ry(theta: QFloat(0.41), 2)
        try circuit.cx(0, 4) // forces SWAP bubble on the 1D chain
        try circuit.rz(theta: QFloat(-0.27), 4)

        let sv = try CPUStateVector(qubitCount: 5)
        _ = try cpuSV.execute(circuit, on: sv)
        let svProbs = sv.probabilitiesDouble()

        let loose = MPSConfiguration(maxBondDimension: 32, svdTruncationThreshold: 1e-14)
        var mpsLoose = try MPSState(qubitCount: 5, configuration: loose)
        XCTAssertFalse(mpsLoose.usesDenseEvolution)
        _ = try MPSEngine(configuration: loose).execute(circuit, on: &mpsLoose)
        XCTAssertGreaterThanOrEqual(
            mpsLoose.bondDimensions.max() ?? 0, 2,
            "long-range CX should grow bonds beyond product state"
        )
        let looseProbs = try mpsLoose.probabilities()
        for index in svProbs.indices {
            XCTAssertEqual(
                svProbs[index], Double(looseProbs[index]),
                accuracy: 1e-5,
                "MPS loose χ↔CPU SV idx=\(index)"
            )
        }

        let tight = MPSConfiguration(maxBondDimension: 1, svdTruncationThreshold: 1e-14)
        var mpsTight = try MPSState(qubitCount: 5, configuration: tight)
        _ = try MPSEngine(configuration: tight).execute(circuit, on: &mpsTight)
        let tightProbs = try mpsTight.probabilities()
        var tv = 0.0
        for index in svProbs.indices {
            tv += abs(svProbs[index] - Double(tightProbs[index]))
        }
        tv *= 0.5
        XCTAssertGreaterThan(tv, 1e-3, "χ=1 must truncate vs exact Born TV=\(tv)")
    }

    // MARK: - Fixtures (seeded)

    /// Mixed unitary stress circuit (rotations + CX/CZ) for SV/DM/Metal Born parity.
    private func makeParityStressUnitaryCircuit(
        qubitCount: Int,
        depth: Int,
        seed: UInt64
    ) throws -> QuantumCircuit {
        var rng = QuantumRNG.seeded(seed)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)
        for layer in 0..<depth {
            for q in 0..<qubitCount {
                switch (layer + q + Int(seed % 5)) % 6 {
                case 0: try circuit.h(q)
                case 1: try circuit.rx(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
                case 2: try circuit.ry(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
                case 3: try circuit.rz(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
                case 4: try circuit.s(q)
                default: try circuit.t(q)
                }
            }
            if qubitCount >= 2 {
                for q in 0..<(qubitCount - 1) {
                    if rng.nextInt(upperBound: 2) == 0 {
                        try circuit.cx(q, q + 1)
                    } else {
                        try circuit.cz(q, q + 1)
                    }
                }
            }
        }
        return circuit
    }

    /// Clifford-only circuit for Stabilizer ↔ CPU SV histogram stress.
    private func makeParityStressCliffordCircuit(
        qubitCount: Int,
        depth: Int,
        seed: UInt64
    ) throws -> QuantumCircuit {
        var rng = QuantumRNG.seeded(seed)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)
        for _ in 0..<depth {
            let q0 = rng.nextInt(upperBound: qubitCount)
            let q1 = (q0 + 1) % qubitCount
            switch rng.nextInt(upperBound: 10) {
            case 0: try circuit.h(q0)
            case 1: try circuit.s(q0)
            case 2: try circuit.sdg(q0)
            case 3: try circuit.x(q0)
            case 4: try circuit.y(q0)
            case 5: try circuit.z(q0)
            case 6: try circuit.sx(q0)
            case 7 where qubitCount >= 2: try circuit.cx(q0, q1)
            case 8 where qubitCount >= 2: try circuit.cz(q0, q1)
            case 9 where qubitCount >= 2: try circuit.swap(q0, q1)
            default: try circuit.h(q0)
            }
        }
        return circuit
    }

    /// Seeded product rotations + adjacent CX chain (GHZ-like; χ≤2).
    private func makeParityStressAdjacentCXChain(
        qubitCount: Int,
        seed: UInt64
    ) throws -> QuantumCircuit {
        var rng = QuantumRNG.seeded(seed)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)
        for q in 0..<qubitCount {
            try circuit.ry(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
            try circuit.rz(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
        }
        try circuit.h(0)
        for q in 0..<(qubitCount - 1) {
            try circuit.cx(q, q + 1)
        }
        for q in 0..<qubitCount {
            try circuit.rz(theta: QFloat(rng.nextUnitDouble() * 0.5), q)
        }
        return circuit
    }
}
