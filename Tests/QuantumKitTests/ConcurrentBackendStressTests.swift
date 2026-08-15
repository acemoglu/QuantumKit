import XCTest
@testable import QuantumKit

/// Concurrent submission smoke (races / crashes), not a 10k Metal bomb or deep scale stress.
///
/// **N (CI default):** `256` independent shallow jobs (band 100–500). Override with
/// `QUANTUMKIT_CONCURRENT_STRESS_N` (clamped to `1…2_000`). Set
/// `QUANTUMKIT_LONG_TESTS=1` to raise the default to `1_000` when the env N is unset.
///
/// **Share vs per-task policy (matches source docs):**
/// - ``CPUStatevectorBackend`` / ``CPUStatevectorEngine``: **shareable**; each ``run`` /
///   ``execute`` uses a distinct ``CPUStateVector`` (do not mutate one state concurrently).
/// - Mass parallelism here prefers **per-task backends** for isolation; a separate test
///   exercises **one shared** CPU backend under ``TaskGroup`` vs serial baselines.
/// - Metal is **not** mass-submitted (avoid unbounded GPU contexts / OOM); CPU only.
extension QuantumKitTests {

    private static var concurrentJobCount: Int {
        if let raw = ProcessInfo.processInfo.environment["QUANTUMKIT_CONCURRENT_STRESS_N"],
           let parsed = Int(raw), parsed > 0 {
            return min(parsed, 2_000)
        }
        let long = ProcessInfo.processInfo.environment["QUANTUMKIT_LONG_TESTS"]
            .map { $0 == "1" || $0.lowercased() == "true" } ?? false
        return long ? 1_000 : 256
    }

    // MARK: - Per-task backends (preferred isolation)

    func testConcurrent_perTaskCPUEngines_exactSV_matchSerial() async throws {
        let nJobs = Self.concurrentJobCount
        let circuits = try (0..<nJobs).map { job in
            try makeConcurrentStressCircuit(qubitCount: 2 + (job % 2), seed: UInt64(10_000 &+ job))
        }

        let serialProbs: [[Double]] = try circuits.map { circuit in
            let engine = CPUStatevectorEngine()
            let state = try CPUStateVector(qubitCount: circuit.qubitCount)
            _ = try engine.execute(circuit, on: state)
            return state.probabilitiesDouble()
        }

        struct JobProbs: Sendable {
            let job: Int
            let probs: [Double]
        }

        let parallel: [JobProbs] = try await withThrowingTaskGroup(of: JobProbs.self) { group in
            for (job, circuit) in circuits.enumerated() {
                group.addTask {
                    let engine = CPUStatevectorEngine()
                    let state = try CPUStateVector(qubitCount: circuit.qubitCount)
                    _ = try engine.execute(circuit, on: state)
                    return JobProbs(job: job, probs: state.probabilitiesDouble())
                }
            }
            var collected: [JobProbs] = []
            collected.reserveCapacity(nJobs)
            for try await item in group {
                collected.append(item)
            }
            return collected
        }

        XCTAssertEqual(parallel.count, nJobs)
        for item in parallel {
            assertUnitBorn(item.probs, label: "per-task job \(item.job)")
            let expected = serialProbs[item.job]
            XCTAssertEqual(item.probs.count, expected.count, "job \(item.job) length")
            for index in expected.indices {
                XCTAssertEqual(
                    item.probs[index],
                    expected[index],
                    accuracy: 1e-12,
                    "per-task job \(item.job) Born[\(index)] vs serial"
                )
            }
        }
    }

    func testConcurrent_perTaskCPUBackends_unseededShotsComplete() async throws {
        let nJobs = Self.concurrentJobCount
        let circuits = try (0..<nJobs).map { job in
            try makeConcurrentStressCircuit(qubitCount: 2 + (job % 2), seed: UInt64(20_000 &+ job))
        }

        struct JobShotSummary: Sendable {
            let shots: Int
            let histogramSum: Int
        }

        let summaries: [JobShotSummary] = try await withThrowingTaskGroup(of: JobShotSummary.self) { group in
            for circuit in circuits {
                group.addTask {
                    let backend = CPUStatevectorBackend()
                    let result = try backend.run(
                        circuit: circuit,
                        options: QuantumRunOptions(
                            shots: 32,
                            sampleOptions: SampleCountOptions(batchSize: 8)
                        )
                    )
                    guard let counts = result.shotCounts else {
                        throw ConcurrentStressError.missingShotCounts
                    }
                    return JobShotSummary(
                        shots: counts.shots,
                        histogramSum: counts.counts.values.reduce(0, +)
                    )
                }
            }
            var collected: [JobShotSummary] = []
            collected.reserveCapacity(nJobs)
            for try await summary in group {
                collected.append(summary)
            }
            return collected
        }

        XCTAssertEqual(summaries.count, nJobs)
        for (job, summary) in summaries.enumerated() {
            XCTAssertEqual(summary.shots, 32, "job \(job)")
            XCTAssertEqual(summary.histogramSum, summary.shots, "job \(job) histogram sum")
        }
    }

    // MARK: - Shared backend (documented OK for CPU)

    func testConcurrent_sharedCPUBackend_seededShotsMatchSerial() async throws {
        // Docs: CPUStatevectorBackend may be shared; each run allocates its own state(s).
        let nJobs = Self.concurrentJobCount
        let backend = CPUStatevectorBackend()
        let circuits = try (0..<nJobs).map { job in
            try makeConcurrentStressCircuit(qubitCount: 2 + (job % 2), seed: UInt64(30_000 &+ job))
        }

        func options(for job: Int) -> QuantumRunOptions {
            QuantumRunOptions(
                seed: UInt64(90_000 &+ job),
                shots: 48,
                sampleOptions: SampleCountOptions(batchSize: 8)
            )
        }

        var serialBaselines: [ShotCounts] = []
        serialBaselines.reserveCapacity(nJobs)
        for (job, circuit) in circuits.enumerated() {
            let result = try backend.run(circuit: circuit, options: options(for: job))
            serialBaselines.append(try XCTUnwrap(result.shotCounts))
        }

        struct JobCounts: Sendable {
            let job: Int
            let counts: ShotCounts
        }

        let parallel: [JobCounts] = try await withThrowingTaskGroup(of: JobCounts.self) { group in
            for (job, circuit) in circuits.enumerated() {
                group.addTask {
                    let result = try backend.run(circuit: circuit, options: options(for: job))
                    guard let counts = result.shotCounts else {
                        throw ConcurrentStressError.missingShotCounts
                    }
                    return JobCounts(job: job, counts: counts)
                }
            }
            var collected: [JobCounts] = []
            collected.reserveCapacity(nJobs)
            for try await item in group {
                collected.append(item)
            }
            return collected
        }

        XCTAssertEqual(parallel.count, nJobs)
        for item in parallel {
            XCTAssertEqual(item.counts.shots, 48, "shared job \(item.job)")
            XCTAssertEqual(
                item.counts.counts.values.reduce(0, +),
                item.counts.shots,
                "shared job \(item.job) histogram sum"
            )
            XCTAssertEqual(
                item.counts,
                serialBaselines[item.job],
                "shared parallel job \(item.job) must match serial baseline on same backend"
            )
        }
    }

    // MARK: - Serial vs parallel (independent backends)

    func testConcurrent_perTaskBackends_seededShotsMatchSerialLaunch() async throws {
        var built = try QuantumCircuit(qubitCount: 2)
        try built.h(0)
        try built.cx(0, 1)
        try built.ry(theta: QFloat(0.37), 1)
        let circuit = built

        let seed: UInt64 = 55_901
        let shots = 128
        let options = QuantumRunOptions(
            seed: seed,
            shots: shots,
            sampleOptions: SampleCountOptions(batchSize: 8)
        )
        // Modest fan-out for bit-identical replay (not the full CI N).
        let replicas = min(64, Self.concurrentJobCount)

        var serialCounts: [ShotCounts] = []
        serialCounts.reserveCapacity(replicas)
        for _ in 0..<replicas {
            let backend = CPUStatevectorBackend()
            let result = try backend.run(circuit: circuit, options: options)
            serialCounts.append(try XCTUnwrap(result.shotCounts))
        }
        let baseline = serialCounts[0]
        for counts in serialCounts {
            XCTAssertEqual(counts, baseline, "serial independent backends must be bit-identical")
        }

        let parallelCounts: [ShotCounts] = try await withThrowingTaskGroup(of: ShotCounts.self) { group in
            for _ in 0..<replicas {
                group.addTask {
                    let backend = CPUStatevectorBackend()
                    let result = try backend.run(circuit: circuit, options: options)
                    guard let counts = result.shotCounts else {
                        throw ConcurrentStressError.missingShotCounts
                    }
                    return counts
                }
            }
            var collected: [ShotCounts] = []
            collected.reserveCapacity(replicas)
            for try await counts in group {
                collected.append(counts)
            }
            return collected
        }

        XCTAssertEqual(parallelCounts.count, replicas)
        for counts in parallelCounts {
            XCTAssertEqual(counts, baseline, "parallel per-task backends must match serial")
        }
    }

    // MARK: - Helpers

    private enum ConcurrentStressError: Error {
        case missingShotCounts
    }

    /// Shallow n=2…3 circuit; seed only drives structure (deterministic, no RNG needed).
    private func makeConcurrentStressCircuit(qubitCount n: Int, seed: UInt64) throws -> QuantumCircuit {
        var circuit = try QuantumCircuit(qubitCount: n)
        try circuit.h(0)
        if n > 1 {
            try circuit.cx(0, 1)
        }
        if n > 2 {
            try circuit.cx(1, 2)
        }
        let angle = QFloat(Double((seed % 17) &+ 1) * 0.11)
        try circuit.rz(theta: angle, 0)
        if n > 1 {
            try circuit.ry(theta: angle, n - 1)
        }
        return circuit
    }

    private func assertUnitBorn(
        _ probs: [Double],
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(probs.isEmpty, label, file: file, line: line)
        var sum = 0.0
        for (index, p) in probs.enumerated() {
            XCTAssertFalse(p.isNaN, "\(label) NaN idx=\(index)", file: file, line: line)
            XCTAssertTrue(p.isFinite, "\(label) non-finite idx=\(index)", file: file, line: line)
            XCTAssertGreaterThanOrEqual(p, -1e-15, file: file, line: line)
            sum += p
        }
        XCTAssertEqual(sum, 1.0, accuracy: 1e-9, "\(label) ‖ψ‖²", file: file, line: line)
    }
}
