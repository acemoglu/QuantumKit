import XCTest
@testable import QuantumKit

/// I5 lite: wall-time / memory **canaries** (not SLAs). Soft trips → ``XCTSkip``;
/// only hang / absurd bounds fail. See ``PerfCanaryHarness``.
extension QuantumKitTests {

    func testPerfCanary_CPUSV_n8_shallow_wallTime() throws {
        let n = 8
        let circuit = try PerfCanaryHarness.makeShallowLayeredCircuit(
            qubitCount: n,
            depth: 4,
            seed: 8_001
        )
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: n)

        let t0 = DispatchTime.now()
        _ = try engine.execute(circuit, on: state)
        let elapsed = PerfCanaryHarness.secondsSince(t0)

        let probs = state.probabilitiesDouble()
        XCTAssertEqual(probs.count, 1 << n)
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 1e-9)

        try PerfCanaryHarness.assertWallCanary(
            elapsedSeconds: elapsed,
            budget: .cpuSV_n8_shallow,
            label: "CPU SV n=8 shallow"
        )
    }

    func testPerfCanary_CPUSV_n10_depth8_wallTime() throws {
        let n = 10
        let circuit = try PerfCanaryHarness.makeShallowLayeredCircuit(
            qubitCount: n,
            depth: 8,
            seed: 10_042
        )
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: n)

        let t0 = DispatchTime.now()
        _ = try engine.execute(circuit, on: state)
        let elapsed = PerfCanaryHarness.secondsSince(t0)

        let probs = state.probabilitiesDouble()
        XCTAssertFalse(probs.contains { $0.isNaN })
        XCTAssertEqual(probs.reduce(0, +), 1.0, accuracy: 1e-9)

        try PerfCanaryHarness.assertWallCanary(
            elapsedSeconds: elapsed,
            budget: .cpuSV_n10_depth8,
            label: "CPU SV n=10 depth=8"
        )
    }

    func testPerfCanary_CPUSV_n12_shallow_wallTime() throws {
        let n = 12
        let peak = PerfCanaryHarness.theoreticalStateBytes(qubitCount: n) * 2
        try PerfCanaryHarness.requireHostMemory(peakBytes: peak, label: "CPU SV n=12 canary")

        let circuit = try PerfCanaryHarness.makeShallowLayeredCircuit(
            qubitCount: n,
            depth: max(2, n / 4),
            seed: 12_100
        )
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: n)

        let t0 = DispatchTime.now()
        _ = try engine.execute(circuit, on: state)
        let elapsed = PerfCanaryHarness.secondsSince(t0)

        XCTAssertEqual(state.probabilitiesDouble().reduce(0, +), 1.0, accuracy: 1e-9)

        try PerfCanaryHarness.assertWallCanary(
            elapsedSeconds: elapsed,
            budget: .cpuSV_n12_shallow,
            label: "CPU SV n=12 shallow"
        )
    }

    func testPerfCanary_CPUSV_n14_estimatedPeakMemory() throws {
        let budget = PerfCanaryHarness.MemoryBudget.cpuSV_n14
        let n = budget.qubitCount

        let estimate = try QuantumBackendFactory.estimateResources(
            qubitCount: n,
            policy: SimulationPolicy(devicePreference: .cpu)
        )
        try PerfCanaryHarness.requireHostMemory(
            peakBytes: estimate.estimatedPeakMemoryBytes,
            label: "CPU SV n=14 memory canary"
        )

        // Theoretical lower bound: estimate must at least cover the state buffer.
        let stateBytes = PerfCanaryHarness.theoreticalStateBytes(qubitCount: n)
        XCTAssertGreaterThanOrEqual(estimate.estimatedStateBytes, stateBytes / 2)
        XCTAssertGreaterThanOrEqual(estimate.estimatedPeakMemoryBytes, estimate.estimatedStateBytes)

        try PerfCanaryHarness.assertMemoryEstimateCanary(
            peakBytes: estimate.estimatedPeakMemoryBytes,
            budget: budget,
            label: "CPU SV n=14 estimate"
        )

        // Profiled run peak (estimated source) also stays in the soft band.
        var circuit = try QuantumCircuit(qubitCount: min(n, 8))
        try circuit.h(0)
        for q in 0..<(circuit.qubitCount - 1) {
            try circuit.cx(q, q + 1)
        }
        let result = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 14_001, profiling: .enabled)
        )
        let profile = try XCTUnwrap(result.profile)
        XCTAssertGreaterThan(profile.peakMemoryBytes, 0)
        XCTAssertEqual(profile.memorySource, .estimated)
        // Small-n profiled peak must remain finite and below hang multiple of its own state.
        let hang = PerfCanaryHarness.theoreticalStateBytes(qubitCount: circuit.qubitCount) * 64
        XCTAssertLessThanOrEqual(profile.peakMemoryBytes, hang)
    }
}
