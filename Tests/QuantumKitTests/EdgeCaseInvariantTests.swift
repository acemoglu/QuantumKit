import XCTest
@testable import QuantumKit

/// Edge-case I/O invariants (identity chains, global-phase blindness, empty circuits).
///
/// **Identity count:** `5_000` `.id` ops on `n = 3` (cycling targets). CI-safe; after
/// ``IdleIdentityRemovalPass`` / ``expandedForExecution()`` the gate list is empty.
/// Raw CPU SV still walks the list, but `.id` is a no-op `break` (cheap hang smoke, not a
/// matrix-apply stress). Wall bound is a hang guard only — correctness is Born `|0…0⟩`.
///
/// **Phase construction:** small unitary `U`, then `U'` = `U` followed by
/// ``Gate/unitary1`` with matrix `e^{iφ} I` on qubit 0 (`φ = π/3`). That 1Q op is a true
/// global phase on the full space; Born probs (and same-seed shot histograms) must match.
extension QuantumKitTests {

    /// Documented identity-chain length for this suite (1_000…10_000 band).
    private static let identityChainCount = 5_000

    /// Global phase angle used in `e^{iφ} I` (radians).
    private static let globalPhasePhi = Double.pi / 3.0

    /// Hang smoke only (not a perf SLO). Cleaned path is empty; raw `.id` loop is O(n) no-ops.
    private static let identityChainTimeoutSeconds = 1.0

    // MARK: - Identity chain

    func testEdgeCase_identityChain_idleRemovalThenCPUSV_isNoop() throws {
        let n = 3
        let identityCount = Self.identityChainCount

        var circuit = try QuantumCircuit(qubitCount: n)
        for index in 0..<identityCount {
            try circuit.id(index % n)
        }
        XCTAssertEqual(circuit.gates.count, identityCount)

        let cleaned = try IdleIdentityRemovalPass().run(on: circuit)
        XCTAssertTrue(cleaned.gates.isEmpty, "IdleIdentityRemovalPass must drop all .id")

        let expanded = try circuit.expandedForExecution()
        XCTAssertTrue(expanded.gates.isEmpty, "expandedForExecution must drop all .id")

        let engine = CPUStatevectorEngine()

        let t0 = DispatchTime.now()
        let stateClean = try CPUStateVector(qubitCount: n)
        _ = try engine.execute(cleaned, on: stateClean)
        let cleanedSeconds = secondsSince(t0)

        let t1 = DispatchTime.now()
        let stateRaw = try CPUStateVector(qubitCount: n)
        _ = try engine.execute(circuit, on: stateRaw)
        let rawSeconds = secondsSince(t1)

        XCTAssertLessThan(
            cleanedSeconds,
            Self.identityChainTimeoutSeconds,
            "cleaned identity chain hang smoke exceeded \(Self.identityChainTimeoutSeconds)s"
        )
        XCTAssertLessThan(
            rawSeconds,
            Self.identityChainTimeoutSeconds,
            "raw \(identityCount) .id hang smoke exceeded \(Self.identityChainTimeoutSeconds)s"
        )

        assertComputationalZeroProbabilities(stateClean.probabilitiesDouble(), qubitCount: n)
        assertComputationalZeroProbabilities(stateRaw.probabilitiesDouble(), qubitCount: n)
    }

    func testEdgeCase_identityChainWithBarriers_idleAndExpanded_staysFiniteNoop() throws {
        // Mix of removable `.id` and kept barriers; idle + expand paths stay Born-noop on |0⟩.
        let n = 2
        let identityCount = Self.identityChainCount
        var circuit = try QuantumCircuit(qubitCount: n)
        for index in 0..<identityCount {
            try circuit.id(index % n)
            if index % 500 == 499 {
                try circuit.barrier([0, 1])
            }
        }

        let cleaned = try IdleIdentityRemovalPass().run(on: circuit)
        XCTAssertFalse(cleaned.gates.contains { if case .id = $0 { return true }; return false })
        XCTAssertTrue(cleaned.gates.allSatisfy {
            if case .barrier = $0 { return true }
            return false
        })

        let expanded = try circuit.expandedForExecution()
        XCTAssertFalse(expanded.gates.contains { if case .id = $0 { return true }; return false })
        XCTAssertTrue(expanded.gates.allSatisfy {
            if case .barrier = $0 { return true }
            return false
        })
        XCTAssertEqual(expanded.gates.count, cleaned.gates.count)

        let engine = CPUStatevectorEngine()
        let t0 = DispatchTime.now()
        let stateIdle = try CPUStateVector(qubitCount: n)
        _ = try engine.execute(cleaned, on: stateIdle)
        let stateExpanded = try CPUStateVector(qubitCount: n)
        _ = try engine.execute(expanded, on: stateExpanded)
        XCTAssertLessThan(secondsSince(t0), Self.identityChainTimeoutSeconds)

        assertComputationalZeroProbabilities(stateIdle.probabilitiesDouble(), qubitCount: n)
        assertComputationalZeroProbabilities(stateExpanded.probabilitiesDouble(), qubitCount: n)
    }

    // MARK: - Global phase blindness

    func testEdgeCase_globalPhaseUnitary1_bornAndSeededShotsMatch() throws {
        var base = try QuantumCircuit(qubitCount: 2)
        try base.h(0)
        try base.cx(0, 1)
        try base.ry(theta: QFloat(0.4), 1)
        try base.rz(theta: QFloat(-0.25), 0)

        var phased = base
        try phased.unitary1(matrix: Self.globalPhaseIdentityMatrix(), target: 0)

        let engine = CPUStatevectorEngine()
        let stateU = try CPUStateVector(qubitCount: 2)
        let stateUP = try CPUStateVector(qubitCount: 2)
        _ = try engine.execute(base, on: stateU)
        _ = try engine.execute(phased, on: stateUP)

        let probsU = stateU.probabilitiesDouble()
        let probsUP = stateUP.probabilitiesDouble()
        XCTAssertEqual(probsU.count, probsUP.count)
        for index in probsU.indices {
            XCTAssertFalse(probsU[index].isNaN)
            XCTAssertFalse(probsUP[index].isNaN)
            XCTAssertEqual(
                probsU[index],
                probsUP[index],
                accuracy: 1e-12,
                "Born mismatch at |\(index)⟩ under global phase e^{iπ/3} I"
            )
        }

        // Amplitudes may differ by a common phase; check |⟨ψ|φ⟩| ≈ 1 when norms match.
        let overlap = complexOverlap(
            realA: stateU.real, imagA: stateU.imag,
            realB: stateUP.real, imagB: stateUP.imag
        )
        let overlapMag = sqrt(overlap.re * overlap.re + overlap.im * overlap.im)
        XCTAssertEqual(overlapMag, 1.0, accuracy: 1e-10)

        let seed: UInt64 = 42_424
        let shots = 512
        let options = QuantumRunOptions(
            seed: seed,
            shots: shots,
            sampleOptions: SampleCountOptions(batchSize: 8)
        )
        let backend = CPUStatevectorBackend()
        let countsU = try backend.run(circuit: base, options: options).shotCounts
        let countsUP = try backend.run(circuit: phased, options: options).shotCounts
        XCTAssertEqual(countsU, countsUP, "seeded histograms must match for true global phase")
        XCTAssertEqual(countsU?.shots, shots)
    }

    func testEdgeCase_globalPhaseInitialAmplitudes_bornMatch() throws {
        // Alternate construction: multiply |0…0⟩ by e^{iφ} via initialize, then same U.
        let phi = Self.globalPhasePhi
        let c = QFloat(cos(phi))
        let s = QFloat(sin(phi))

        var plain = try QuantumCircuit(qubitCount: 2)
        try plain.h(0)
        try plain.cx(0, 1)

        var phasedInit = try QuantumCircuit(qubitCount: 2)
        try phasedInit.initialize(
            qubits: [0, 1],
            amplitudes: [
                ComplexAmplitude(real: c, imaginary: s),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
            ]
        )
        try phasedInit.h(0)
        try phasedInit.cx(0, 1)

        let engine = CPUStatevectorEngine()
        let a = try CPUStateVector(qubitCount: 2)
        let b = try CPUStateVector(qubitCount: 2)
        _ = try engine.execute(plain, on: a)
        _ = try engine.execute(phasedInit, on: b)

        let pa = a.probabilitiesDouble()
        let pb = b.probabilitiesDouble()
        for index in pa.indices {
            XCTAssertEqual(pa[index], pb[index], accuracy: 1e-12)
        }

        let overlap = complexOverlap(
            realA: a.real, imagA: a.imag,
            realB: b.real, imagB: b.imag
        )
        let overlapMag = sqrt(overlap.re * overlap.re + overlap.im * overlap.im)
        XCTAssertEqual(overlapMag, 1.0, accuracy: 1e-10)
    }

    // MARK: - Empty / near-empty

    func testEdgeCase_emptyCircuit_probsSumToOne() throws {
        for n in 1...4 {
            let circuit = try QuantumCircuit(qubitCount: n)
            XCTAssertTrue(circuit.gates.isEmpty)

            let engine = CPUStatevectorEngine()
            let state = try CPUStateVector(qubitCount: n)
            _ = try engine.execute(circuit, on: state)
            let probs = state.probabilitiesDouble()

            XCTAssertEqual(probs.count, 1 << n)
            var sum = 0.0
            for (index, p) in probs.enumerated() {
                XCTAssertFalse(p.isNaN, "NaN at n=\(n) idx=\(index)")
                XCTAssertTrue(p.isFinite)
                sum += p
            }
            XCTAssertEqual(sum, 1.0, accuracy: 1e-12, "empty n=\(n) ‖ψ‖²")
            XCTAssertEqual(probs[0], 1.0, accuracy: 1e-12)
            for index in 1..<probs.count {
                XCTAssertEqual(probs[index], 0.0, accuracy: 1e-12)
            }
        }
    }

    // MARK: - Helpers

    private static func globalPhaseIdentityMatrix() -> [ComplexAmplitude] {
        let c = QFloat(cos(globalPhasePhi))
        let s = QFloat(sin(globalPhasePhi))
        return [
            ComplexAmplitude(real: c, imaginary: s),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: c, imaginary: s),
        ]
    }

    private func secondsSince(_ start: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
    }

    private func assertComputationalZeroProbabilities(
        _ probs: [Double],
        qubitCount: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(probs.count, 1 << qubitCount, file: file, line: line)
        var sum = 0.0
        for (index, p) in probs.enumerated() {
            XCTAssertFalse(p.isNaN, "NaN Born idx=\(index)", file: file, line: line)
            XCTAssertTrue(p.isFinite, file: file, line: line)
            sum += p
            if index == 0 {
                XCTAssertEqual(p, 1.0, accuracy: 1e-12, file: file, line: line)
            } else {
                XCTAssertEqual(p, 0.0, accuracy: 1e-12, file: file, line: line)
            }
        }
        XCTAssertEqual(sum, 1.0, accuracy: 1e-12, file: file, line: line)
    }

    private func complexOverlap(
        realA: [Double],
        imagA: [Double],
        realB: [Double],
        imagB: [Double]
    ) -> (re: Double, im: Double) {
        var re = 0.0
        var im = 0.0
        for index in realA.indices {
            // conj(a)·b
            re += realA[index] * realB[index] + imagA[index] * imagB[index]
            im += realA[index] * imagB[index] - imagA[index] * realB[index]
        }
        return (re, im)
    }
}
