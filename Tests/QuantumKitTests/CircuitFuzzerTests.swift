import XCTest
@testable import QuantumKit

/// I7 lite: seeded random-circuit fuzzer on CPU SV (no Aer).
///
/// Gate set + bounds: ``CircuitFuzzerHarness``. Mid-measure trials are rejected/skipped
/// for unitary Born asserts; unitary circuits must not throw / NaN and ‖ψ‖² ≈ 1.
extension QuantumKitTests {

    func testCircuitFuzzer_seededUnitaryTrials_probsSumAndFinite() throws {
        let trials = CircuitFuzzerHarness.defaultTrialCount
        let rootSeed: UInt64 = 77_001
        let engine = CPUStatevectorEngine()
        var executed = 0
        var skippedMeasure = 0

        for trial in 0..<trials {
            let seed = rootSeed &+ UInt64(trial) &* 0xD1B5_4A32_D192_ED03
            let generated = try CircuitFuzzerHarness.makeCircuit(seed: seed, allowMidMeasure: true)

            if generated.containsMidMeasure || !generated.circuit.isUnitaryOnly {
                // Mid-measure (or reset/c_if if ever added): skip unitary Born path.
                skippedMeasure += 1
                continue
            }

            XCTAssertLessThanOrEqual(generated.qubitCount, CircuitFuzzerHarness.maxQubitCount)
            XCTAssertGreaterThanOrEqual(generated.qubitCount, CircuitFuzzerHarness.minQubitCount)
            XCTAssertLessThanOrEqual(generated.depth, CircuitFuzzerHarness.maxDepth)

            let state = try CPUStateVector(qubitCount: generated.qubitCount)
            XCTAssertNoThrow(try engine.execute(generated.circuit, on: state))
            let probs = state.probabilitiesDouble()
            XCTAssertEqual(probs.count, 1 << generated.qubitCount)

            var sum = 0.0
            for (index, p) in probs.enumerated() {
                XCTAssertFalse(p.isNaN, "trial \(trial) seed=\(seed) NaN idx=\(index)")
                XCTAssertTrue(p.isFinite, "trial \(trial) seed=\(seed) non-finite idx=\(index)")
                XCTAssertGreaterThanOrEqual(p, -1e-15)
                sum += p
            }
            XCTAssertEqual(sum, 1.0, accuracy: 1e-9, "trial \(trial) seed=\(seed) ‖ψ‖²")
            executed += 1
        }

        XCTAssertGreaterThan(executed, trials / 2, "too many mid-measure skips; check fuzzer rates")
        // Document that some measure draws occurred (or zero is fine if unlucky).
        _ = skippedMeasure
    }

    func testCircuitFuzzer_makeUnitaryCircuit_neverEmitsMeasure() throws {
        let root: UInt64 = 90_011
        for trial in 0..<24 {
            let generated = try CircuitFuzzerHarness.makeUnitaryCircuit(
                seed: root &+ UInt64(trial) &* 17
            )
            XCTAssertFalse(generated.containsMidMeasure)
            XCTAssertTrue(generated.circuit.isUnitaryOnly)
            XCTAssertFalse(generated.circuit.containsMeasure)
        }
    }

    func testCircuitFuzzer_midMeasureDraw_rejectedFromUnitaryPath() throws {
        // Find a seeded draw that inserted mid-measure; unitary Born path rejects it.
        var found: CircuitFuzzerHarness.GeneratedCircuit?
        for offset in 0..<400 {
            let generated = try CircuitFuzzerHarness.makeCircuit(
                seed: 5_500 &+ UInt64(offset),
                allowMidMeasure: true
            )
            if generated.containsMidMeasure {
                found = generated
                break
            }
        }
        guard let generated = found else {
            throw XCTSkip("no mid-measure draw in 400 seeds (~5%/layer); rate OK to skip")
        }
        XCTAssertTrue(generated.circuit.containsMeasure)
        XCTAssertFalse(generated.circuit.isUnitaryOnly)
        // Do not fail the suite: unitary fuzz asserts are intentionally not applied.
    }

    func testCircuitFuzzer_reproducibleUnderSameSeed() throws {
        let seed: UInt64 = 42_424
        let a = try CircuitFuzzerHarness.makeCircuit(seed: seed, allowMidMeasure: false)
        let b = try CircuitFuzzerHarness.makeCircuit(seed: seed, allowMidMeasure: false)
        XCTAssertEqual(a.qubitCount, b.qubitCount)
        XCTAssertEqual(a.depth, b.depth)
        XCTAssertEqual(a.circuit.gates.count, b.circuit.gates.count)
        XCTAssertEqual(a.circuit.gates, b.circuit.gates)

        let engine = CPUStatevectorEngine()
        let sa = try CPUStateVector(qubitCount: a.qubitCount)
        let sb = try CPUStateVector(qubitCount: b.qubitCount)
        _ = try engine.execute(a.circuit, on: sa)
        _ = try engine.execute(b.circuit, on: sb)
        XCTAssertEqual(sa.probabilitiesDouble(), sb.probabilitiesDouble())
    }
}
