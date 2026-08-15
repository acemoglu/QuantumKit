import XCTest
@testable import QuantumKit

/// Host-scale smoke: large-ish CPU SV runs stay finite, memory budgets fail cleanly,
/// async cancel maps to ``CircuitExecutionCancellationError``, and CPU canBatch
/// `batchSize` does not change seeded histograms.
///
/// Soft-skips only when ``ProcessInfo/physicalMemory`` is too small for the peak estimate.
/// Does **not** change default qubit/memory limits, multi-GPU, or out-of-core paths.
extension QuantumKitTests {

    // MARK: - CPU SV n=12…16 shallow random

    func testScaleSmoke_CPUSV_n12to16_shallowRandom_finiteUnitProbs() throws {
        let engine = CPUStatevectorEngine()
        // Depths tried: shallow (n/4 + 2 layers of seeded 1Q + adjacent CX).
        let widths = Array(12...min(16, CPUStateVector.maxQubitCount))

        for n in widths {
            let peak = cpuSVPeakBytes(qubitCount: n)
            try requireHostMemory(bytes: peak, label: "CPU SV n=\(n)")

            let circuit = try makeScaleSmokeShallowRandom(qubitCount: n, seed: UInt64(1_000 + n))
            let state = try CPUStateVector(qubitCount: n)
            _ = try engine.execute(circuit, on: state)
            let probs = state.probabilitiesDouble()

            XCTAssertEqual(probs.count, 1 << n)
            var sum = 0.0
            for (index, p) in probs.enumerated() {
                XCTAssertFalse(p.isNaN, "NaN Born n=\(n) idx=\(index)")
                XCTAssertTrue(p.isFinite, "non-finite Born n=\(n) idx=\(index)")
                XCTAssertGreaterThanOrEqual(p, -1e-15, "negative Born n=\(n) idx=\(index)")
                sum += p
            }
            XCTAssertFalse(sum.isNaN)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-9, "‖ψ‖² n=\(n)")
        }
    }

    // MARK: - Memory budget / recommendMethod

    func testScaleSmoke_overBudget_estimateAndRecommend_failCleanly() throws {
        let policy = SimulationPolicy(
            devicePreference: .cpu,
            maxPeakMemoryBytes: 64
        )

        XCTAssertThrowsError(
            try QuantumBackendFactory.estimateResources(qubitCount: 14, policy: policy)
        ) { error in
            guard case SimulationPolicyError.estimatedMemoryExceedsBudget(let estimated, let budget) = error else {
                return XCTFail("expected estimatedMemoryExceedsBudget, got \(error)")
            }
            XCTAssertEqual(budget, 64)
            XCTAssertGreaterThan(estimated, budget)
        }

        XCTAssertThrowsError(
            try QuantumBackendFactory.makeRecommended(qubitCount: 14, policy: policy)
        ) { error in
            guard case SimulationPolicyError.estimatedMemoryExceedsBudget = error else {
                return XCTFail("expected estimatedMemoryExceedsBudget, got \(error)")
            }
        }

        XCTAssertThrowsError(
            try QuantumBackendFactory.makeStatevector(
                devicePreference: .cpu,
                qubitCount: 14,
                policy: policy
            )
        ) { error in
            guard case SimulationPolicyError.estimatedMemoryExceedsBudget = error else {
                return XCTFail("expected estimatedMemoryExceedsBudget, got \(error)")
            }
        }
    }

    func testScaleSmoke_recommendMethod_underGenerousBudget() throws {
        let n = 14
        try requireHostMemory(bytes: cpuSVPeakBytes(qubitCount: n), label: "recommend n=\(n)")

        let policy = SimulationPolicy(
            devicePreference: .cpu,
            maxPeakMemoryBytes: 64 * 1024 * 1024
        )
        let method = try QuantumBackendFactory.recommendMethod(qubitCount: n, policy: policy)
        XCTAssertEqual(method, .statevector)

        let backend = try QuantumBackendFactory.makeRecommended(qubitCount: n, policy: policy)
        XCTAssertTrue(backend is CPUStatevectorBackend)
        XCTAssertEqual(backend.method, .statevector)
    }

    // MARK: - Async cancel

    func testScaleSmoke_asyncCancelMidShotBatch_returnsCancelled() async throws {
        var circuit = try QuantumCircuit(qubitCount: 8)
        try circuit.h(0)
        for q in 0..<(circuit.qubitCount - 1) {
            try circuit.cx(q, q + 1)
        }

        let backend = CPUStatevectorBackend()
        let task = Task {
            try await backend.runAsync(
                circuit: circuit,
                options: QuantumRunOptions(
                    seed: 19,
                    shots: 80_000,
                    sampleOptions: SampleCountOptions(batchSize: 32)
                )
            )
        }
        // Short delay so sampling starts; cancel must map to kit error (not raw CancellationError).
        try await Task.sleep(nanoseconds: 2_000_000)
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("expected CircuitExecutionCancellationError.cancelled")
        } catch is CircuitExecutionCancellationError {
            // ok
        } catch is CancellationError {
            XCTFail("CancellationError must be mapped to CircuitExecutionCancellationError")
        }
    }

    // MARK: - Shot parallel: seeded batch ≡ serial (CPU canBatch)

    func testScaleSmoke_CPU_seededBatchMatchesSerial_canBatch() throws {
        // Contract: canBatch + independentShotStream → batchSize does not change histogram.
        // Intentional diverge (not asserted here): CPU canBatch ≠ Metal sequential under same seed;
        // mustSerial mid-circuit measure uses a different schedule than independent streams.
        var circuit = try QuantumCircuit(qubitCount: 5)
        try circuit.h(0)
        for q in 0..<4 {
            try circuit.cx(q, q + 1)
        }
        try circuit.ry(theta: QFloat(0.41), 2)
        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: circuit, noise: nil))

        let seed: UInt64 = 55_021
        let shots = 512
        let backend = CPUStatevectorBackend()

        let serial = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 1)
            )
        )
        let batched = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 32)
            )
        )
        XCTAssertEqual(serial.shotCounts, batched.shotCounts)
    }

    // MARK: - Helpers

    /// Estimated CPU SV peak (matches policy heuristic: 2× state for real+imag Double buffers).
    private func cpuSVPeakBytes(qubitCount: Int) -> Int {
        let stateBytes = (1 << qubitCount) * 2 * MemoryLayout<Double>.stride
        return stateBytes * 2
    }

    private func requireHostMemory(bytes needed: Int, label: String) throws {
        let available = ProcessInfo.processInfo.physicalMemory
        // 4× headroom for OS + workers; soft-skip rather than OOM on tiny hosts.
        let required = UInt64(max(needed, 1)) * 4
        if available < required {
            throw XCTSkip(
                "\(label): need ~\(required) B headroom, physicalMemory=\(available)"
            )
        }
    }

    /// Shallow seeded random: few 1Q layers + adjacent CX (cheap at n≤16).
    private func makeScaleSmokeShallowRandom(qubitCount n: Int, seed: UInt64) throws -> QuantumCircuit {
        var rng = QuantumRNG.seeded(seed)
        var circuit = try QuantumCircuit(qubitCount: n)
        let depth = max(2, n / 4) // depths tried: 3…4 for n=12…16
        for layer in 0..<depth {
            for q in 0..<n {
                switch (Int(seed) &+ layer &+ q) % 4 {
                case 0: try circuit.h(q)
                case 1: try circuit.rx(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
                case 2: try circuit.ry(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
                default: try circuit.rz(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
                }
            }
            for q in 0..<(n - 1) {
                try circuit.cx(q, q + 1)
            }
        }
        return circuit
    }
}
