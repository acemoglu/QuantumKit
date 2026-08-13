import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Dilim 2: checkpoints / resume / async / concurrency / precision

    func testCPUStatevectorSnapshotRestoreRoundTrip() throws {
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        var rng: QuantumRNG = .seeded(42)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng)
        let snap = state.snapshot()
        let probsBefore = state.probabilitiesDouble()

        state.resetToZero()
        XCTAssertEqual(state.probabilitiesDouble()[0], 1.0, accuracy: 1e-12)

        try state.restore(from: snap)
        let probsAfter = state.probabilitiesDouble()
        for index in 0..<4 {
            XCTAssertEqual(probsBefore[index], probsAfter[index], accuracy: 1e-15)
        }

        var rngContinue: QuantumRNG = .seeded(99)
        let outcomeA = try engine.measureCollapse(on: state, qubits: [0, 1], rng: &rngContinue, noise: nil)

        let stateB = try CPUStateVector(qubitCount: 2)
        try stateB.restore(from: snap)
        var rngB: QuantumRNG = .seeded(99)
        let outcomeB = try engine.measureCollapse(on: stateB, qubits: [0, 1], rng: &rngB, noise: nil)
        XCTAssertEqual(outcomeA, outcomeB)
    }

    func testCPUDensityMatrixSnapshotRestoreRoundTrip() throws {
        let engine = CPUDensityMatrixEngine()
        let density = try CPUDensityMatrix(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let noise = NoiseModel(depolarizingProbability: 0.1)
        _ = try engine.execute(circuit, on: density, noise: noise)
        let snap = density.snapshot()
        let before = density.probabilitiesDouble()

        density.resetToZero()
        try density.restore(from: snap)
        let after = density.probabilitiesDouble()
        XCTAssertEqual(before[0], after[0], accuracy: 1e-15)
        XCTAssertEqual(before[1], after[1], accuracy: 1e-15)
    }

    func testClassicalMemorySnapshotRestore() throws {
        var memory = ClassicalMemory(registerWidths: [2, 4])
        try memory.writeMeasuredBits([1, 0], register: 0, bitOffset: 0)
        try memory.writeOutcome(3, measuredQubitCount: 2, register: 1, bitOffset: 0)
        let snap = memory.snapshot()

        var mutated = ClassicalMemory(registerWidths: [2, 4])
        try mutated.restore(from: snap)
        XCTAssertEqual(mutated.memorySlots, snap.memorySlots)
        XCTAssertEqual(mutated.value(ofRegister: 0), 1)
        XCTAssertEqual(mutated.value(ofRegister: 1), 3)
    }

    func testCPUResumeMatchesFullRunWithCIFAndMeasure() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        try circuit.c_if(classicalRegister: 0, equals: 1, x: 1)
        try circuit.h(1)

        let seed: UInt64 = 12345
        let engine = CPUStatevectorEngine()

        let fullState = try CPUStateVector(qubitCount: 2)
        var fullRNG: QuantumRNG = .seeded(seed)
        let fullResult = try engine.executeRNG(circuit, on: fullState, rng: &fullRNG)
        let fullProbs = fullState.probabilitiesDouble()

        let splitAt = 2 // after H + measure
        let partialState = try CPUStateVector(qubitCount: 2)
        var partialRNG: QuantumRNG = .seeded(seed)
        let first = try engine.executeRNG(
            circuit,
            on: partialState,
            rng: &partialRNG,
            runState: CircuitRunState(toInstruction: splitAt)
        )
        let checkpoint = CircuitCheckpoint.make(nextInstructionIndex: splitAt, from: first)
        let quantumSnap = partialState.snapshot()

        // Mutate, then restore
        partialState.resetToZero()
        try partialState.restore(from: quantumSnap)
        let second = try engine.executeRNG(
            circuit,
            on: partialState,
            rng: &partialRNG,
            runState: .resume(from: checkpoint)
        )

        XCTAssertEqual(fullResult.measurementOutcomes, second.measurementOutcomes)
        XCTAssertEqual(fullResult.classicalMemory.memorySlots, second.classicalMemory.memorySlots)
        let splitProbs = partialState.probabilitiesDouble()
        for index in 0..<4 {
            XCTAssertEqual(fullProbs[index], splitProbs[index], accuracy: 1e-12)
        }
    }

    func testMetalResumeMatchesFullRunWhenAvailable() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        try circuit.c_if(classicalRegister: 0, equals: 1, x: 1)

        let seed: UInt64 = 777
        let engine = try QuantumEngine()

        let fullState = try StateVector(qubitCount: 2, device: engine.device)
        var fullRNG: QuantumRNG = .seeded(seed)
        let fullResult = try engine.executeRNG(circuit, on: fullState, rng: &fullRNG)
        let fullProbs = try QuantumMeasurement.probabilities(state: fullState, engine: engine)

        let splitAt = 2
        let partialState = try StateVector(qubitCount: 2, device: engine.device)
        var partialRNG: QuantumRNG = .seeded(seed)
        let first = try engine.executeRNG(
            circuit,
            on: partialState,
            rng: &partialRNG,
            runState: CircuitRunState(toInstruction: splitAt)
        )
        let checkpoint = CircuitCheckpoint.make(nextInstructionIndex: splitAt, from: first)
        let snap = try engine.snapshot(partialState)

        partialState.resetToZero()
        try engine.restore(partialState, from: snap)
        let second = try engine.executeRNG(
            circuit,
            on: partialState,
            rng: &partialRNG,
            runState: .resume(from: checkpoint)
        )

        XCTAssertEqual(fullResult.measurementOutcomes, second.measurementOutcomes)
        XCTAssertEqual(fullResult.classicalMemory.memorySlots, second.classicalMemory.memorySlots)
        let splitProbs = try QuantumMeasurement.probabilities(state: partialState, engine: engine)
        for index in 0..<4 {
            XCTAssertEqual(fullProbs[index], splitProbs[index], accuracy: 1e-5)
        }
    }

    func testAsyncCPURunMatchesSync() async throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let backend = CPUStatevectorBackend()
        let sync = try backend.run(circuit: circuit, options: QuantumRunOptions(seed: 11, shots: 64))
        let asyncResult = try await backend.runAsync(
            circuit: circuit,
            options: QuantumRunOptions(seed: 11, shots: 64)
        )
        XCTAssertEqual(sync.shotCounts?.counts, asyncResult.shotCounts?.counts)
    }

    func testAsyncCancellationMidShotBatch() async throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.cx(1, 2)

        let backend = CPUStatevectorBackend()
        let task = Task {
            try await backend.runAsync(
                circuit: circuit,
                options: QuantumRunOptions(seed: 1, shots: 10_000)
            )
        }
        // Allow a few shots to start, then cancel.
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

    func testConcurrentCPUEnginesOnDistinctStates() throws {
        final class FailureBox: @unchecked Sendable {
            private let lock = NSLock()
            private var values: [String] = []
            func append(_ value: String) {
                lock.lock(); values.append(value); lock.unlock()
            }
            var snapshot: [String] {
                lock.lock(); defer { lock.unlock() }
                return values
            }
        }
        let engine = CPUStatevectorEngine()
        let group = DispatchGroup()
        let failures = FailureBox()

        for seed in 0..<8 {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }
                do {
                    var circuit = try QuantumCircuit(qubitCount: 2)
                    try circuit.h(0)
                    try circuit.cx(0, 1)
                    let state = try CPUStateVector(qubitCount: 2)
                    var rng: QuantumRNG = .seeded(UInt64(seed))
                    _ = try engine.executeRNG(circuit, on: state, rng: &rng)
                    let probs = state.probabilitiesDouble()
                    XCTAssertEqual(probs[0], 0.5, accuracy: 1e-12)
                    XCTAssertEqual(probs[3], 0.5, accuracy: 1e-12)
                } catch {
                    failures.append(String(describing: error))
                }
            }
        }
        group.wait()
        XCTAssertTrue(failures.snapshot.isEmpty, "concurrent failures: \(failures.snapshot)")
    }

    func testFloat64CPUPrecisionMatchesAnalyticTight() throws {
        let policy = SimulationPolicy(devicePreference: .cpu, precision: .float64)
        let backend = try QuantumBackendFactory.makeStatevector(
            devicePreference: .cpu,
            qubitCount: 1,
            policy: policy
        )
        XCTAssertTrue(backend is CPUStatevectorBackend)

        // RX(θ)|0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        let theta: QFloat = QFloat.pi / 5
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: theta, 0)

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 1)
        _ = try engine.execute(circuit, on: state)
        let probs = state.probabilitiesDouble()
        let expected0 = Double(cos(Double(theta) / 2) * cos(Double(theta) / 2))
        let expected1 = Double(sin(Double(theta) / 2) * sin(Double(theta) / 2))
        XCTAssertEqual(probs[0], expected0, accuracy: 1e-14)
        XCTAssertEqual(probs[1], expected1, accuracy: 1e-14)
    }

    func testMetalFloat64UnsupportedWhenForced() throws {
        let policy = SimulationPolicy(devicePreference: .metal, precision: .float64)
        XCTAssertThrowsError(
            try QuantumBackendFactory.makeStatevector(
                devicePreference: .metal,
                qubitCount: 1,
                policy: policy
            )
        ) { error in
            guard case SimulationPrecisionError.metalFloat64Unsupported = error else {
                return XCTFail("expected metalFloat64Unsupported, got \(error)")
            }
        }
    }

    func testFloat64AutomaticFallsBackToCPU() throws {
        let policy = SimulationPolicy(devicePreference: .automatic, precision: .float64)
        let backend = try QuantumBackendFactory.makeStatevector(
            devicePreference: .automatic,
            qubitCount: 2,
            policy: policy
        )
        XCTAssertTrue(backend is CPUStatevectorBackend)
    }

    func testMetalStatevectorSnapshotWhenAvailable() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }
        let engine = try QuantumEngine()
        let state = try StateVector(qubitCount: 2, device: engine.device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        _ = try engine.execute(circuit, on: state)
        let snap = try engine.snapshot(state)
        let before = try QuantumMeasurement.probabilities(state: state, engine: engine)

        state.resetToZero()
        try engine.restore(state, from: snap)
        let after = try QuantumMeasurement.probabilities(state: state, engine: engine)
        for index in 0..<4 {
            XCTAssertEqual(before[index], after[index], accuracy: 1e-6)
        }
    }
}
