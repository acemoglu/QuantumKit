import XCTest
@testable import QuantumKit

/// Golden-vector and randomized rewrite equivalence (item 39 lite).
///
/// **Oracle:** CPU statevector amplitudes / Born probabilities — not
/// ``CircuitUnitary`` alone. Safe gate subset excludes native `.crx` / `.cp`
/// (known ``CircuitUnitary`` mismatch vs engines).
///
/// **Transforms covered:** identity insertion + ``IdleIdentityRemovalPass``,
/// ``QuantumCircuit/expandedForExecution()``, safe rewrite stack
/// (``AlgebraicOptimizationPass`` / ``CliffordSimplificationPass`` /
/// ``LocalUnitarySynthesisPass`` — no ibmEagle basis translate), ``SchedulingPass``.
extension QuantumKitTests {

    // MARK: - Fixed golden amplitude vectors

    func testGoldenBellAmplitudesMatchCPUStatevector() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let invSqrt2 = 1.0 / sqrt(2.0)
        // |Φ+⟩ = (|00⟩ + |11⟩)/√2  (engine LSB: index 0 = |00⟩, 3 = |11⟩)
        let goldenReal = [invSqrt2, 0.0, 0.0, invSqrt2]
        let goldenImag = [0.0, 0.0, 0.0, 0.0]

        let state = try cpuAmplitudes(of: circuit)
        assertAmplitudesEqual(
            real: state.real, imag: state.imag,
            expectedReal: goldenReal, expectedImag: goldenImag
        )
    }

    func testGoldenGHZAmplitudesMatchCPUStatevector() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.cx(0, 2)

        let invSqrt2 = 1.0 / sqrt(2.0)
        // (|000⟩ + |111⟩)/√2
        var goldenReal = [Double](repeating: 0, count: 8)
        goldenReal[0] = invSqrt2
        goldenReal[7] = invSqrt2
        let goldenImag = [Double](repeating: 0, count: 8)

        let state = try cpuAmplitudes(of: circuit)
        assertAmplitudesEqual(
            real: state.real, imag: state.imag,
            expectedReal: goldenReal, expectedImag: goldenImag
        )
    }

    func testGoldenQFTIshUniformSuperpositionAmplitudesMatchCPUStatevector() throws {
        // QFT-adjacent fixture: H⊗n on |0⟩^n (textbook QFT(|0…0⟩) up to global phase).
        // Prefer this over ``applyQFT()`` for amplitude goldens: the CPHASE sandwich in
        // ``applyQFT`` injects relative RZ phases even on |0⟩ controls.
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.h(1)
        try circuit.h(2)

        let amp = 1.0 / sqrt(8.0)
        let goldenReal = [Double](repeating: amp, count: 8)
        let goldenImag = [Double](repeating: 0, count: 8)

        let state = try cpuAmplitudes(of: circuit)
        assertAmplitudesEqual(
            real: state.real, imag: state.imag,
            expectedReal: goldenReal, expectedImag: goldenImag,
            accuracy: 1e-12
        )
    }

    func testApplyQFTOnZeroHasUniformBornProbabilities() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.applyQFT()
        let probs = try cpuAmplitudes(of: circuit).probabilitiesDouble()
        let expected = 1.0 / 8.0
        for p in probs {
            XCTAssertEqual(p, expected, accuracy: 1e-12)
        }
    }

    // MARK: - Rewrite equivalence (CPU SV oracle)

    func testIdentityInsertionThenIdleRemovalPreservesCPUAmplitudes() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try circuit.rz(theta: QFloat(0.4), 0)
        try circuit.cz(0, 1)

        let withIds = try insertIdentities(into: circuit, every: 1, seed: 7)
        XCTAssertGreaterThan(withIds.gates.count, circuit.gates.count)

        let cleaned = try IdleIdentityRemovalPass().run(on: withIds)
        assertCPUProbabilitiesMatch(circuit, cleaned)
        assertCPUProbabilitiesMatch(circuit, withIds) // engines drop .id via expand
    }

    func testExpandedForExecutionPreservesCPUAmplitudesOnCompositeGates() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.apply(.iswap(q1: 0, q2: 1))
        try circuit.apply(.id(target: 0))
        try circuit.apply(.rxx(theta: QFloatExpr(QFloat(0.35)), q1: 0, q2: 1))
        try circuit.rz(theta: QFloat(-0.2), 1)

        let expanded = try circuit.expandedForExecution()
        XCTAssertFalse(expanded.gates.contains { if case .iswap = $0 { return true }; return false })
        XCTAssertFalse(expanded.gates.contains { if case .id = $0 { return true }; return false })
        XCTAssertFalse(expanded.gates.contains { if case .rxx = $0 { return true }; return false })
        assertCPUProbabilitiesMatch(circuit, expanded)
    }

    func testSafeRewritePassesPreserveCPUProbabilitiesOnSafeRandomCircuits() throws {
        // Intentionally skips ibmEagle ``BasisTranslatorPass``: several 1Q expansions
        // (Y / RX / …) are not Born-faithful vs native CPU kernels — out of scope here.
        let level1 = PassManager(passes: [AlgebraicOptimizationPass()])
        let level2 = PassManager(passes: [
            AlgebraicOptimizationPass(),
            CliffordSimplificationPass(),
            LocalUnitarySynthesisPass(),
        ])
        for seed in UInt64(1)...UInt64(8) {
            let circuit = try makeSafeRandomUnitaryCircuit(
                qubitCount: seed % 2 == 0 ? 3 : 2,
                depth: 10,
                seed: 100 &+ seed,
                injectCancellablePairs: true
            )
            let opt1 = try level1.run(on: circuit)
            let opt2 = try level2.run(on: circuit)
            assertCPUProbabilitiesMatch(circuit, opt1, accuracy: 1e-9)
            assertCPUProbabilitiesMatch(circuit, opt2, accuracy: 1e-9)
            // Secondary unitary check only on the safe gate subset (no CRX/CP).
            XCTAssertTrue(try CircuitEquivalenceVerifier.areEquivalent(circuit, opt1, tolerance: 1e-4))
            XCTAssertTrue(try CircuitEquivalenceVerifier.areEquivalent(circuit, opt2, tolerance: 1e-4))
        }
    }

    func testSchedulingDelaysPreserveCPUProbabilities() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.cx(1, 2)
        try circuit.rz(theta: QFloat(0.55), 2)

        let durations = GateDurationTable(defaultDuration: 3)
        let scheduled = try SchedulingPass(durations: durations, method: .asap).run(on: circuit)
        XCTAssertTrue(scheduled.gates.contains { if case .delay = $0 { return true }; return false })
        assertCPUProbabilitiesMatch(circuit, scheduled)
    }

    func testTranspileBellToIbmEaglePreservesCPUProbabilities() throws {
        // Narrow transpile smoke: H/CX → ibmEagle is covered by Dilim/Phase3; pin Born via CPU SV.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let transpiled = try Transpiler.transpile(
            circuit,
            options: TranspileOptions(
                targetBasis: .ibmEagle,
                optimizationLevel: 2,
                seedTranspiler: 11
            )
        )
        assertCPUProbabilitiesMatch(circuit, transpiled, accuracy: 1e-6)
    }

    func testRandomizedRewriteBundlePreservesCPUProbabilities() throws {
        // One seed drives circuit gen + identity insertion; rewrite chain must stay Born-equivalent.
        let seed: UInt64 = 4242
        let original = try makeSafeRandomUnitaryCircuit(
            qubitCount: 3,
            depth: 10,
            seed: seed,
            injectCancellablePairs: true
        )
        let withIds = try insertIdentities(into: original, every: 2, seed: seed &+ 1)
        let afterIdle = try IdleIdentityRemovalPass().run(on: withIds)
        let afterExpand = try withIds.expandedForExecution()
        let afterSafeRewrite = try PassManager(passes: [
            AlgebraicOptimizationPass(),
            CliffordSimplificationPass(),
            LocalUnitarySynthesisPass(),
        ]).run(on: original)
        let scheduled = try SchedulingPass(
            durations: GateDurationTable(defaultDuration: 2),
            method: .alap
        ).run(on: original)

        assertCPUProbabilitiesMatch(original, afterIdle)
        assertCPUProbabilitiesMatch(original, afterExpand)
        assertCPUProbabilitiesMatch(original, afterSafeRewrite)
        assertCPUProbabilitiesMatch(original, scheduled)
    }

    // MARK: - Helpers

    private func cpuAmplitudes(of circuit: QuantumCircuit) throws -> CPUStateVector {
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        _ = try engine.execute(circuit, on: state)
        return state
    }

    private func assertAmplitudesEqual(
        real: [Double],
        imag: [Double],
        expectedReal: [Double],
        expectedImag: [Double],
        accuracy: Double = 1e-12,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(real.count, expectedReal.count, file: file, line: line)
        XCTAssertEqual(imag.count, expectedImag.count, file: file, line: line)
        for index in real.indices {
            XCTAssertEqual(real[index], expectedReal[index], accuracy: accuracy, file: file, line: line)
            XCTAssertEqual(imag[index], expectedImag[index], accuracy: accuracy, file: file, line: line)
        }
    }

    private func assertCPUProbabilitiesMatch(
        _ lhs: QuantumCircuit,
        _ rhs: QuantumCircuit,
        accuracy: Double = 1e-9,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            let engine = CPUStatevectorEngine()
            let a = try CPUStateVector(qubitCount: lhs.qubitCount)
            let b = try CPUStateVector(qubitCount: rhs.qubitCount)
            _ = try engine.execute(lhs, on: a)
            _ = try engine.execute(rhs, on: b)
            let pa = a.probabilitiesDouble()
            let pb = b.probabilitiesDouble()
            XCTAssertEqual(pa.count, pb.count, file: file, line: line)
            for index in pa.indices {
                XCTAssertEqual(pa[index], pb[index], accuracy: accuracy, file: file, line: line)
            }
        } catch {
            XCTFail("CPU SV compare failed: \(error)", file: file, line: line)
        }
    }

    /// Gate subset without native `.crx` / `.cp` (CircuitUnitary pitfall vs engines).
    private func makeSafeRandomUnitaryCircuit(
        qubitCount: Int,
        depth: Int,
        seed: UInt64,
        injectCancellablePairs: Bool = false
    ) throws -> QuantumCircuit {
        var rng = QuantumRNG.seeded(seed)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)
        for _ in 0..<depth {
            if injectCancellablePairs, rng.nextInt(upperBound: 5) == 0 {
                let q = rng.nextInt(upperBound: qubitCount)
                try circuit.h(q)
                try circuit.h(q) // algebraic cancel
                continue
            }
            let q0 = rng.nextInt(upperBound: qubitCount)
            let q1 = (q0 + 1 + rng.nextInt(upperBound: max(qubitCount - 1, 1))) % qubitCount
            switch rng.nextInt(upperBound: 12) {
            case 0:
                try circuit.h(q0)
            case 1:
                try circuit.x(q0)
            case 2:
                try circuit.y(q0)
            case 3:
                try circuit.z(q0)
            case 4:
                try circuit.s(q0)
            case 5:
                try circuit.t(q0)
            case 6:
                try circuit.rx(theta: QFloat(rng.nextUnitDouble() * Double.pi), q0)
            case 7:
                try circuit.ry(theta: QFloat(rng.nextUnitDouble() * Double.pi), q0)
            case 8:
                try circuit.rz(theta: QFloat(rng.nextUnitDouble() * Double.pi), q0)
            case 9 where qubitCount >= 2:
                try circuit.cx(q0, q1)
            case 10 where qubitCount >= 2:
                try circuit.cz(q0, q1)
            case 11 where qubitCount >= 2:
                try circuit.swap(q0, q1)
            default:
                try circuit.h(q0)
            }
        }
        return circuit
    }

    private func insertIdentities(
        into circuit: QuantumCircuit,
        every: Int,
        seed: UInt64
    ) throws -> QuantumCircuit {
        var rng = QuantumRNG.seeded(seed)
        var output = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for (index, gate) in circuit.gates.enumerated() {
            try output.apply(gate)
            if every > 0, (index + 1) % every == 0 {
                let target = rng.nextInt(upperBound: circuit.qubitCount)
                try output.apply(.id(target: target))
            }
        }
        return output
    }
}
