import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Shot independence predicate

    func testShotPolicyUnitaryCanBatchAndMetalUnitaryBatch() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: circuit, noise: nil))
        XCTAssertFalse(ShotExecutionPolicy.mustSerial(circuit: circuit, noise: nil))
        XCTAssertTrue(ShotExecutionPolicy.canUseMetalUnitaryBatch(circuit: circuit, noise: nil))
        XCTAssertEqual(
            ShotExecutionPolicy.metalUnitaryBatchSize(
                circuit: circuit, noise: nil, requested: 32, shots: 100
            ),
            32
        )
        XCTAssertEqual(
            ShotExecutionPolicy.cpuWorkerPoolSize(
                circuit: circuit, noise: nil, requested: 32, shots: 100
            ),
            32
        )
        let live = ShotExecutionPolicy.cpuLiveStateCopies(
            circuit: circuit, noise: nil, requested: 32, shots: 100
        )
        XCTAssertLessThanOrEqual(live, 32)
        XCTAssertEqual(
            live,
            min(32, max(ProcessInfo.processInfo.activeProcessorCount, 1), 100)
        )
    }

    func testShotPolicyMidCircuitMeasureAndCIFMustSerial() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        try circuit.c_if(classicalRegister: 0, equals: 1, x: 1)

        XCTAssertTrue(ShotExecutionPolicy.mustSerial(circuit: circuit, noise: nil))
        XCTAssertFalse(ShotExecutionPolicy.canBatch(circuit: circuit, noise: nil))
        XCTAssertFalse(ShotExecutionPolicy.canUseMetalUnitaryBatch(circuit: circuit, noise: nil))
        XCTAssertEqual(
            ShotExecutionPolicy.metalUnitaryBatchSize(
                circuit: circuit, noise: nil, requested: 32, shots: 64
            ),
            1
        )
        XCTAssertEqual(
            ShotExecutionPolicy.cpuWorkerPoolSize(
                circuit: circuit, noise: nil, requested: 32, shots: 64
            ),
            1
        )
        XCTAssertFalse(circuit.allowsPreparedDensityShotBatching(noise: nil))
    }

    func testShotPolicyNestedCIFMustSerial() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        let inner = Gate.c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 1))
        try circuit.apply(.c_if(classicalRegister: 0, expectedValue: 1, gate: inner))

        XCTAssertTrue(ShotExecutionPolicy.mustSerial(circuit: circuit, noise: nil))
        XCTAssertFalse(ShotExecutionPolicy.canBatch(circuit: circuit, noise: nil))
        XCTAssertFalse(circuit.allowsPreparedDensityShotBatching(noise: nil))
    }

    func testNestedCIFForcedSerialMatchesSingleThreadHistogram() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        let inner = Gate.c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 1))
        try circuit.apply(.c_if(classicalRegister: 0, expectedValue: 1, gate: inner))

        let seed: UInt64 = 27
        let shots = 48
        let serial = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 1)
            )
        )
        let requestedParallel = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 32)
            )
        )
        XCTAssertEqual(serial.shotCounts, requestedParallel.shotCounts)

        let legacy = try cpuLegacySequentialShotHistogram(
            circuit: circuit,
            shots: shots,
            seed: seed,
            noise: nil
        )
        // mustSerial keeps the shared sequential stream — matches legacy schedule.
        XCTAssertEqual(serial.shotCounts, legacy)
    }

    func testShotPolicyGlobalDepolarizingCPUParallelMetalSerial() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let noise = NoiseModel(depolarizingProbability: 0.2)

        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: circuit, noise: noise))
        XCTAssertFalse(ShotExecutionPolicy.mustSerial(circuit: circuit, noise: noise))
        // executeUnitaryBatch cannot apply per-shot unraveling RNG.
        XCTAssertFalse(ShotExecutionPolicy.canUseMetalUnitaryBatch(circuit: circuit, noise: noise))
        XCTAssertEqual(
            ShotExecutionPolicy.metalUnitaryBatchSize(
                circuit: circuit, noise: noise, requested: 32, shots: 64
            ),
            1
        )
        XCTAssertEqual(
            ShotExecutionPolicy.cpuWorkerPoolSize(
                circuit: circuit, noise: noise, requested: 32, shots: 64
            ),
            32
        )
    }

    func testShotPolicyCIFAloneMustSerial() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1, classicalRegisters: [creg])
        try circuit.c_if(classicalRegister: 0, equals: 0, x: 0)
        XCTAssertTrue(ShotExecutionPolicy.mustSerial(circuit: circuit, noise: nil))
    }

    func testContainsDelayRecursesIntoCIF() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1, classicalRegisters: [creg])
        try circuit.apply(
            .c_if(
                classicalRegister: 0,
                expectedValue: 0,
                gate: .delay(duration: 1e-6, qubit: 0)
            )
        )
        XCTAssertTrue(circuit.containsDelay)
        let idle = NoiseModel(t1: 50e-6, t2: 70e-6, thermalRelaxationOnDelay: true)
        XCTAssertTrue(ShotExecutionPolicy.requiresEvolutionNoise(idle, circuit: circuit))
        // Still serial because of c_if — no parallel + nested delay hazard.
        XCTAssertTrue(ShotExecutionPolicy.mustSerial(circuit: circuit, noise: idle))
    }

    func testDensityPreparedBatchingAlignedWithCanBatch() throws {
        var unitary = try QuantumCircuit(qubitCount: 1)
        try unitary.h(0)
        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: unitary, noise: nil))
        XCTAssertTrue(unitary.allowsPreparedDensityShotBatching(noise: nil))
        XCTAssertTrue(
            unitary.allowsPreparedDensityShotBatching(
                noise: NoiseModel(depolarizingProbability: 0.1)
            )
        )

        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var measured = try QuantumCircuit(qubitCount: 1, classicalRegisters: [creg])
        try measured.h(0)
        try measured.measure(qubits: [0], classicalRegister: 0)
        // Projective measure still forces trajectory *parallelism* off…
        XCTAssertFalse(ShotExecutionPolicy.canBatch(circuit: measured, noise: nil))
        // …but trailing terminal measures allow evolve-once sampling.
        XCTAssertTrue(measured.allowsPreparedDensityShotBatching(noise: nil))
        XCTAssertTrue(measured.allowsPreparedStatevectorShotSampling(noise: nil))
        XCTAssertNotNil(measured.preparedShotUnitaryPrefix())
    }

    func testPreparedSamplingHostCDFThreshold() {
        XCTAssertFalse(ShotExecutionPolicy.usesDevicePreparedSampling(qubitCount: 16))
        XCTAssertFalse(ShotExecutionPolicy.usesDevicePreparedSampling(qubitCount: 20))
        XCTAssertTrue(ShotExecutionPolicy.usesDevicePreparedSampling(qubitCount: 21))
        XCTAssertTrue(ShotExecutionPolicy.usesDevicePreparedSampling(qubitCount: 30))
        XCTAssertEqual(ShotExecutionPolicy.hostPreparedSamplingMaxQubitCount, 20)
    }

    func testPreparedSamplingMidCircuitMeasureStillRejected() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        try circuit.x(1)
        XCTAssertNil(circuit.preparedShotUnitaryPrefix())
        XCTAssertFalse(circuit.allowsPreparedStatevectorShotSampling(noise: nil))
    }

    func testPreparedSamplingMatchesBornDistribution() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 2)
        var withMeasure = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try withMeasure.h(0)
        try withMeasure.cx(0, 1)
        try withMeasure.measure(qubits: [0, 1], classicalRegister: 0)

        var unitary = try QuantumCircuit(qubitCount: 2)
        try unitary.h(0)
        try unitary.cx(0, 1)

        let seed: UInt64 = 123
        let shots = 512
        let measured = try CPUStatevectorBackend().run(
            circuit: withMeasure,
            options: QuantumRunOptions(seed: seed, shots: shots)
        )
        let plain = try CPUStatevectorBackend().run(
            circuit: unitary,
            options: QuantumRunOptions(seed: seed, shots: shots)
        )
        XCTAssertEqual(measured.shotCounts, plain.shotCounts)

        // Same seed + prepared path must not depend on batchSize.
        let batch1 = try CPUStatevectorBackend().run(
            circuit: withMeasure,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 1)
            )
        )
        let batch32 = try CPUStatevectorBackend().run(
            circuit: withMeasure,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 32)
            )
        )
        XCTAssertEqual(batch1.shotCounts, batch32.shotCounts)
    }

    func testPreparedSamplingOptOutKeepsTrajectory() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let seed: UInt64 = 7
        let shots = 64
        let prepared = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(preferPreparedSampling: true)
            )
        )
        let trajectory = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(preferPreparedSampling: false)
            )
        )
        // Different RNG schedules — not required to match; both must be valid histograms.
        XCTAssertEqual(prepared.shotCounts?.shots, shots)
        XCTAssertEqual(trajectory.shotCounts?.shots, shots)
        XCTAssertEqual(
            prepared.shotCounts?.counts.values.reduce(0, +),
            shots
        )
        XCTAssertEqual(
            trajectory.shotCounts?.counts.values.reduce(0, +),
            shots
        )
    }

    // MARK: - Independent unitary parity

    func testCPUIndependentUnitaryParallelMatchesPerShotSeeds() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let seed: UInt64 = 42
        let shots = 64

        let parallel = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 32, preferPreparedSampling: false)
            )
        )
        let reference = try cpuIndependentShotHistogram(
            circuit: circuit,
            shots: shots,
            seed: seed,
            noise: nil
        )
        XCTAssertEqual(parallel.shotCounts, reference)
    }

    func testCPUBatchSizeDoesNotChangeIndependentHistogram() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let seed: UInt64 = 99
        let shots = 48

        let serialPool = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 1, preferPreparedSampling: false)
            )
        )
        let parallelPool = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 32, preferPreparedSampling: false)
            )
        )
        XCTAssertEqual(serialPool.shotCounts, parallelPool.shotCounts)
        XCTAssertEqual(
            serialPool.shotCounts,
            try cpuIndependentShotHistogram(circuit: circuit, shots: shots, seed: seed, noise: nil)
        )
    }

    func testMetalUnitaryBatchMatchesSerialSequentialRNG() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        let engine = try QuantumEngine()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        var sequentialRNG: QuantumRNG = .seeded(42)
        let sequential = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 128,
            rng: &sequentialRNG,
            options: SampleCountOptions(batchSize: 1)
        )
        var batchedRNG: QuantumRNG = .seeded(42)
        let batched = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 128,
            rng: &batchedRNG,
            options: SampleCountOptions(batchSize: 32)
        )
        XCTAssertEqual(sequential, batched)
    }

    func testCPUIndependentSeedDivergesFromLegacySequentialStream() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let seed: UInt64 = 42
        let shots = 64

        let modern = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 1, preferPreparedSampling: false)
            )
        )
        let legacy = try cpuLegacySequentialShotHistogram(
            circuit: circuit,
            shots: shots,
            seed: seed,
            noise: nil
        )
        // Documented breaking schedule: per-shot streams ≠ one shared seeded stream.
        XCTAssertNotEqual(modern.shotCounts, legacy)
        XCTAssertEqual(
            modern.shotCounts,
            try cpuIndependentShotHistogram(circuit: circuit, shots: shots, seed: seed, noise: nil)
        )
    }

    func testCPUAndMetalPreparedSamplingShareSeededHistogram() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let seed: UInt64 = 42
        let shots = 128

        let cpu = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 8)
            )
        )
        let engine = try QuantumEngine()
        var metalRNG: QuantumRNG = .seeded(seed)
        let metal = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: shots,
            rng: &metalRNG,
            options: SampleCountOptions(batchSize: 1)
        )
        // Evolve-once + multinomial uses the same sequential measurement stream on both.
        XCTAssertEqual(cpu.shotCounts, metal)
        var metalBatchedRNG: QuantumRNG = .seeded(seed)
        let metalBatched = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: shots,
            rng: &metalBatchedRNG,
            options: SampleCountOptions(batchSize: 32)
        )
        XCTAssertEqual(metal, metalBatched)
    }

    func testCPUIndependentSeedDivergesFromMetalSequentialScheduleWhenTrajectoryForced() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let seed: UInt64 = 42
        let shots = 128

        let cpu = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 8, preferPreparedSampling: false)
            )
        )
        let engine = try QuantumEngine()
        var metalRNG: QuantumRNG = .seeded(seed)
        let metal = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: shots,
            rng: &metalRNG,
            options: SampleCountOptions(batchSize: 1, preferPreparedSampling: false)
        )
        XCTAssertNotEqual(cpu.shotCounts, metal)
    }

    // MARK: - Mid-circuit forced serial

    func testMidCircuitMeasureCIFForcedSerialMatchesSingleThread() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        try circuit.c_if(classicalRegister: 0, equals: 1, x: 1)

        XCTAssertTrue(ShotExecutionPolicy.mustSerial(circuit: circuit, noise: nil))

        let seed: UInt64 = 19
        let shots = 48
        let serial = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 1)
            )
        )
        let requestedParallel = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 32)
            )
        )
        XCTAssertEqual(serial.shotCounts, requestedParallel.shotCounts)

        if makeDevice() != nil {
            let engine = try QuantumEngine()
            var rngA: QuantumRNG = .seeded(seed)
            let metalSerial = try QuantumMeasurement.runSampleCountsRNG(
                circuit: circuit,
                engine: engine,
                shots: shots,
                rng: &rngA,
                options: SampleCountOptions(batchSize: 1)
            )
            var rngB: QuantumRNG = .seeded(seed)
            let metalRequested = try QuantumMeasurement.runSampleCountsRNG(
                circuit: circuit,
                engine: engine,
                shots: shots,
                rng: &rngB,
                options: SampleCountOptions(batchSize: 32)
            )
            XCTAssertEqual(metalSerial, metalRequested)
        }
    }

    // MARK: - Global depolarizing / reset+noise

    func testCPUDepolarizingParallelMatchesIndependentStreams() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let noise = NoiseModel(depolarizingProbability: 0.25)
        let seed: UInt64 = 7
        let shots = 80

        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: circuit, noise: noise))
        XCTAssertFalse(ShotExecutionPolicy.canUseMetalUnitaryBatch(circuit: circuit, noise: noise))

        let parallel = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                noise: noise,
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 16)
            )
        )
        let reference = try cpuIndependentShotHistogram(
            circuit: circuit,
            shots: shots,
            seed: seed,
            noise: noise
        )
        XCTAssertEqual(parallel.shotCounts, reference)
    }

    func testCPUResetWithNoiseParallelMatchesIndependentStreams() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try circuit.reset(0)
        let noise = NoiseModel(depolarizingProbability: 0.15, resetErrorProbability: 0.2)
        let seed: UInt64 = 21
        let shots = 64

        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: circuit, noise: noise))
        XCTAssertFalse(ShotExecutionPolicy.canUseMetalUnitaryBatch(circuit: circuit, noise: noise))

        let parallel = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                noise: noise,
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 16)
            )
        )
        let serialPool = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                noise: noise,
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 1)
            )
        )
        let reference = try cpuIndependentShotHistogram(
            circuit: circuit,
            shots: shots,
            seed: seed,
            noise: noise
        )
        XCTAssertEqual(parallel.shotCounts, reference)
        XCTAssertEqual(serialPool.shotCounts, reference)
    }

    func testMetalDepolarizingStaysSerialAndAppliesNoise() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let noise = NoiseModel(depolarizingProbability: 0.35)
        XCTAssertEqual(
            ShotExecutionPolicy.metalUnitaryBatchSize(
                circuit: circuit, noise: noise, requested: 32, shots: 40
            ),
            1
        )

        let engine = try QuantumEngine()
        var rngA: QuantumRNG = .seeded(11)
        let a = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 40,
            rng: &rngA,
            noise: noise,
            options: SampleCountOptions(batchSize: 1)
        )
        var rngB: QuantumRNG = .seeded(11)
        let b = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 40,
            rng: &rngB,
            noise: noise,
            options: SampleCountOptions(batchSize: 32)
        )
        XCTAssertEqual(a, b)

        var rngIdeal: QuantumRNG = .seeded(11)
        let ideal = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 40,
            rng: &rngIdeal,
            noise: nil,
            options: SampleCountOptions(batchSize: 1)
        )
        // Ideal X|0⟩ is always |1⟩; depolarizing must move probability mass.
        XCTAssertEqual(ideal.counts[1], 40)
        XCTAssertNotEqual(a.counts, ideal.counts)
    }

    func testTrajectoryCPUIndependentShotsMatchPerShotSeeds() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let noise = NoiseModel(depolarizingProbability: 0.2)
        let seed: UInt64 = 33
        let shots = 40
        let backend = try TrajectoryBackend(wrapping: CPUStatevectorBackend())

        let result = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(
                noise: noise,
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 16)
            )
        )
        let reference = try cpuIndependentShotHistogram(
            circuit: circuit,
            shots: shots,
            seed: seed,
            noise: noise
        )
        XCTAssertEqual(result.shotCounts, reference)
        XCTAssertEqual(result.metadata.method, .trajectory)
    }

    // MARK: - Cancellation / profiling

    func testCPUParallelShotCancellation() async throws {
        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.cx(1, 2)
        try circuit.cx(2, 3)
        let backend = CPUStatevectorBackend()
        let entered = expectation(description: "independent sampling entered")
        let gate = NSLock()
        var didFulfill = false
        let fulfillOnce: () -> Void = {
            gate.lock()
            defer { gate.unlock() }
            guard !didFulfill else { return }
            didFulfill = true
            entered.fulfill()
        }
        // Prefer the GCD-worker hook when the pool is concurrent; fall back to owner-start
        // on single-core hosts where independent shots stay serial.
        CPUShotSampler.onIndependentSamplingStarted = fulfillOnce
        CPUShotSampler.onIndependentWorkerShotStarted = fulfillOnce
        defer {
            CPUShotSampler.onIndependentWorkerShotStarted = nil
            CPUShotSampler.onIndependentSamplingStarted = nil
        }

        let task = Task {
            try await backend.runAsync(
                circuit: circuit,
                options: QuantumRunOptions(
                    seed: 1,
                    shots: 200_000,
                    sampleOptions: SampleCountOptions(batchSize: 32, preferPreparedSampling: false)
                )
            )
        }
        await fulfillment(of: [entered], timeout: 5.0)
        // Give a concurrent worker a moment to be inside executeRNG before cancel.
        try await Task.sleep(nanoseconds: 2_000_000)
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected cancellation")
        } catch is CircuitExecutionCancellationError {
            // expected
        } catch is CancellationError {
            XCTFail("CancellationError should be mapped to CircuitExecutionCancellationError")
        }
    }

    func testProfilingDetailedOnParallelCPUShots() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let options = QuantumRunOptions(
            seed: 5,
            shots: 24,
            sampleOptions: SampleCountOptions(batchSize: 8, preferPreparedSampling: false),
            profiling: .detailed
        )
        let result = try CPUStatevectorBackend().run(circuit: circuit, options: options)
        let profile = try XCTUnwrap(result.profile)
        XCTAssertEqual(profile.phaseTimings?.map(\.name), ["sample"])
        XCTAssertFalse(profile.phaseTimings?.map(\.name).contains("worker") ?? false)
        let gates = try XCTUnwrap(profile.gateTimings)
        XCTAssertEqual(gates.map(\.index), Array(0..<circuit.gates.count))
        XCTAssertEqual(Set(gates.map(\.index)).count, circuit.gates.count)
        XCTAssertGreaterThan(gates.map(\.wallClockNanoseconds).reduce(0, +), 0)
        let live = ShotExecutionPolicy.cpuLiveStateCopies(
            circuit: circuit,
            noise: nil,
            requested: 8,
            shots: 24
        )
        XCTAssertEqual(profile.peakMemoryBytes, profile.stateBytes * (live + 1))
    }

    func testCPUParallelProfilingDoesNotChangeHistogram() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let off = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: 13,
                shots: 40,
                sampleOptions: SampleCountOptions(batchSize: 8, preferPreparedSampling: false)
            )
        )
        let on = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: 13,
                shots: 40,
                sampleOptions: SampleCountOptions(batchSize: 8, preferPreparedSampling: false),
                profiling: .detailed
            )
        )
        XCTAssertEqual(off.shotCounts, on.shotCounts)
    }

    // MARK: - Audit follow-ups (noise / Metal / RNG footgun)

    func testMeasurementDephasingAloneDoesNotBlockMetalUnitaryBatch() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        // Terminal-only dephasing flag: no mid-circuit measure → not evolution noise.
        let noise = NoiseModel(measurementDephasingProbability: 0.3)
        XCTAssertFalse(circuit.containsMeasure)
        XCTAssertFalse(ShotExecutionPolicy.requiresEvolutionNoise(noise, circuit: circuit))
        XCTAssertTrue(ShotExecutionPolicy.canUseMetalUnitaryBatch(circuit: circuit, noise: noise))
        XCTAssertEqual(
            ShotExecutionPolicy.metalUnitaryBatchSize(
                circuit: circuit, noise: noise, requested: 32, shots: 64
            ),
            32
        )
    }

    func testMeasurementDephasingWithMidCircuitMeasureBlocksMetalBatch() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        let noise = NoiseModel(measurementDephasingProbability: 0.3)
        XCTAssertTrue(circuit.containsMeasure)
        XCTAssertTrue(ShotExecutionPolicy.requiresEvolutionNoise(noise, circuit: circuit))
        // Projective mid-circuit already forces mustSerial / no Metal unitary batch.
        XCTAssertTrue(ShotExecutionPolicy.mustSerial(circuit: circuit, noise: noise))
        XCTAssertFalse(ShotExecutionPolicy.canUseMetalUnitaryBatch(circuit: circuit, noise: noise))
    }

    func testContainsMeasureRecursesIntoCIF() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1, classicalRegisters: [creg])
        try circuit.apply(
            .c_if(
                classicalRegister: 0,
                expectedValue: 0,
                gate: .measure(MeasureSpec(qubits: [0], classicalRegister: 0))
            )
        )
        XCTAssertTrue(circuit.containsMeasure)
        let noise = NoiseModel(measurementDephasingProbability: 0.1)
        XCTAssertTrue(ShotExecutionPolicy.requiresEvolutionNoise(noise, circuit: circuit))
    }

    func testCPUAmplitudeDampingParallelMatchesIndependentStreams() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let noise = NoiseModel(amplitudeDampingProbability: 0.2)
        let seed: UInt64 = 17
        let shots = 64

        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: circuit, noise: noise))
        XCTAssertFalse(ShotExecutionPolicy.canUseMetalUnitaryBatch(circuit: circuit, noise: noise))

        let parallel = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                noise: noise,
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 16)
            )
        )
        let reference = try cpuIndependentShotHistogram(
            circuit: circuit,
            shots: shots,
            seed: seed,
            noise: noise
        )
        XCTAssertEqual(parallel.shotCounts, reference)
    }

    func testHostUnitaryForcesMetalSerialMatchingBatchSizes() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        var circuit = try QuantumCircuit(qubitCount: 1)
        let invSqrt2 = QFloat(1 / sqrt(2.0))
        let hadamard: [ComplexAmplitude] = [
            ComplexAmplitude(real: invSqrt2, imaginary: 0),
            ComplexAmplitude(real: invSqrt2, imaginary: 0),
            ComplexAmplitude(real: invSqrt2, imaginary: 0),
            ComplexAmplitude(real: -invSqrt2, imaginary: 0),
        ]
        try circuit.unitary1(matrix: hadamard, target: 0)
        XCTAssertTrue(circuit.containsHostAppliedUnitaryGates)
        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: circuit, noise: nil))
        XCTAssertFalse(ShotExecutionPolicy.canUseMetalUnitaryBatch(circuit: circuit, noise: nil))
        XCTAssertEqual(
            ShotExecutionPolicy.metalUnitaryBatchSize(
                circuit: circuit, noise: nil, requested: 32, shots: 40
            ),
            1
        )

        let engine = try QuantumEngine()
        var rngA: QuantumRNG = .seeded(3)
        let a = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 40,
            rng: &rngA,
            options: SampleCountOptions(batchSize: 1)
        )
        var rngB: QuantumRNG = .seeded(3)
        let b = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 40,
            rng: &rngB,
            options: SampleCountOptions(batchSize: 32)
        )
        XCTAssertEqual(a, b)
    }

    func testIndependentPathUsesSeededRNGWhenOptionsSeedNil() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let engine = CPUStatevectorEngine()
        var rng: QuantumRNG = .seeded(55)
        let counts = try CPUShotSampler.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 32,
            rng: &rng,
            noise: nil,
            options: SampleCountOptions(batchSize: 8),
            seed: nil
        )
        // `rng` must not be advanced on the independent path.
        if case .seeded(let state) = rng {
            XCTAssertEqual(state, 55)
        } else {
            XCTFail("expected seeded RNG to remain untouched")
        }
        let reference = try cpuIndependentShotHistogram(
            circuit: circuit,
            shots: 32,
            seed: 55,
            noise: nil
        )
        XCTAssertEqual(counts, reference)
    }
}

extension QuantumKitTests {
    fileprivate func cpuIndependentShotHistogram(
        circuit: QuantumCircuit,
        shots: Int,
        seed: UInt64,
        noise: NoiseModel?
    ) throws -> ShotCounts {
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        var histogram: [Int: Int] = [:]
        for shotIndex in 0..<shots {
            var rng = QuantumRNG.independentShotStream(seed: seed, shotIndex: shotIndex)
            let outcome = try CPUShotSampler.runOneShot(
                circuit: circuit,
                engine: engine,
                state: state,
                rng: &rng,
                noise: noise,
                cancellationCheck: nil
            )
            histogram[outcome, default: 0] += 1
        }
        return ShotCounts(shots: shots, counts: histogram)
    }

    /// Pre-shot-parallel CPU schedule: one ``QuantumRNG/seeded`` advanced across all shots.
    fileprivate func cpuLegacySequentialShotHistogram(
        circuit: QuantumCircuit,
        shots: Int,
        seed: UInt64,
        noise: NoiseModel?
    ) throws -> ShotCounts {
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        var rng: QuantumRNG = .seeded(seed)
        var histogram: [Int: Int] = [:]
        for _ in 0..<shots {
            let outcome = try CPUShotSampler.runOneShot(
                circuit: circuit,
                engine: engine,
                state: state,
                rng: &rng,
                noise: noise,
                cancellationCheck: nil
            )
            histogram[outcome, default: 0] += 1
        }
        return ShotCounts(shots: shots, counts: histogram)
    }
}
