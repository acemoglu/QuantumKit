import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Dilim 3: trajectory method / DM shot batching / resource estimation

    // MARK: A — Trajectory as first-class method

    func testRecommendTrajectoryWhenNoisyExceedsDensityMatrixLimit() throws {
        let noise = NoiseModel(depolarizingProbability: 0.05)
        let policy = SimulationPolicy(
            statevectorQubitLimit: 20,
            densityMatrixQubitLimit: 4,
            preferDensityMatrixWhenNoisy: true,
            preferTrajectoryWhenDensityMatrixTooWide: true
        )
        let method = try QuantumBackendFactory.recommendMethod(
            qubitCount: 6,
            noise: noise,
            policy: policy
        )
        XCTAssertEqual(method, .trajectory)
    }

    func testRecommendDensityMatrixForNarrowNoisyCircuit() throws {
        let noise = NoiseModel(amplitudeDampingProbability: 0.1)
        let method = try QuantumBackendFactory.recommendMethod(qubitCount: 3, noise: noise)
        XCTAssertEqual(method, .densityMatrix)
    }

    func testRecommendRejectsLocalizedNoiseWhenDensityMatrixTooWide() throws {
        let noise = NoiseModel().adding(.pauliXFlip(probability: 0.1), for: .gate(.x))
        let policy = SimulationPolicy(densityMatrixQubitLimit: 2)
        XCTAssertThrowsError(
            try QuantumBackendFactory.recommendMethod(qubitCount: 4, noise: noise, policy: policy)
        ) { error in
            guard case SimulationPolicyError.densityMatrixRequiredButTooWide = error else {
                return XCTFail("Expected densityMatrixRequiredButTooWide, got \(error)")
            }
        }
    }

    func testRecommendRejectsDephasingOnlyWhenDensityMatrixTooWide() throws {
        var noise = NoiseModel()
        noise.measurementMode = .dephasingOnly
        let policy = SimulationPolicy(densityMatrixQubitLimit: 2)
        XCTAssertFalse(noise.supportsTrajectorySimulation)
        XCTAssertThrowsError(
            try QuantumBackendFactory.recommendMethod(qubitCount: 4, noise: noise, policy: policy)
        ) { error in
            guard case SimulationPolicyError.densityMatrixRequiredButTooWide = error else {
                return XCTFail("Expected densityMatrixRequiredButTooWide, got \(error)")
            }
        }
    }

    func testTrajectoryBackendRejectsLocalizedAndDephasingOnly() throws {
        let backend = try QuantumBackendFactory.makeTrajectory(
            devicePreference: .cpu,
            qubitCount: 1
        )
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let localized = NoiseModel().adding(.pauliXFlip(probability: 0.1), for: .gate(.x))
        XCTAssertThrowsError(
            try backend.run(
                circuit: circuit,
                options: QuantumRunOptions(noise: localized, seed: 1, shots: 8)
            )
        )

        var dephasingOnly = NoiseModel()
        dephasingOnly.measurementMode = .dephasingOnly
        XCTAssertThrowsError(
            try backend.run(
                circuit: circuit,
                options: QuantumRunOptions(noise: dephasingOnly, seed: 1, shots: 8)
            )
        )
    }

    func testTrajectoryParityVersusCPUDensityMatrixDepolarizing() throws {
        let p: QFloat = 0.15
        let noise = NoiseModel(depolarizingProbability: p)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let dmBackend = CPUDensityMatrixBackend()
        let density = try CPUDensityMatrix(qubitCount: 1)
        var dmRNG: QuantumRNG = .seeded(1)
        _ = try dmBackend.engine.executeRNG(circuit, on: density, rng: &dmRNG, noise: noise)
        let dmProbs = density.probabilities()
        XCTAssertEqual(dmProbs[0], 2 * p / 3, accuracy: 1e-5)
        XCTAssertEqual(dmProbs[1], 1 - 2 * p / 3, accuracy: 1e-5)

        let traj = TrajectoryBackend(engine: CPUStatevectorEngine())
        let trajProbs = try traj.averageProbabilities(
            circuit: circuit,
            trajectories: 4000,
            seed: 42,
            noise: noise
        )
        XCTAssertEqual(trajProbs[0], dmProbs[0], accuracy: 0.04)
        XCTAssertEqual(trajProbs[1], dmProbs[1], accuracy: 0.04)
    }

    func testTrajectoryBackendReportsShotCounts() throws {
        let backend = try QuantumBackendFactory.makeTrajectory(
            devicePreference: .cpu,
            qubitCount: 1
        )
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let noise = NoiseModel(depolarizingProbability: 0.2)
        let result = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(noise: noise, seed: 9, shots: 512)
        )
        XCTAssertEqual(result.metadata.method, .trajectory)
        XCTAssertEqual(result.shotCounts?.shots, 512)
        XCTAssertEqual(result.metadata.deviceName, "CPU")
    }

    func testTrajectoryBackendRequiresShots() throws {
        let backend = try QuantumBackendFactory.makeTrajectory(
            devicePreference: .cpu,
            qubitCount: 1
        )
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        XCTAssertThrowsError(
            try backend.run(circuit: circuit, options: QuantumRunOptions(seed: 1))
        ) { error in
            guard case TrajectoryBackendError.shotsRequired = error else {
                return XCTFail("Expected shotsRequired, got \(error)")
            }
        }
    }

    func testPreferDensityMatrixWhenNoisyFalseSelectsStatevector() throws {
        let noise = NoiseModel(depolarizingProbability: 0.05)
        let policy = SimulationPolicy(
            densityMatrixQubitLimit: 8,
            preferDensityMatrixWhenNoisy: false,
            preferTrajectoryWhenDensityMatrixTooWide: true
        )
        let method = try QuantumBackendFactory.recommendMethod(
            qubitCount: 3,
            noise: noise,
            policy: policy
        )
        XCTAssertEqual(method, .statevector)
    }

    func testPreferDensityMatrixWhenNoisyTrueSelectsDensityMatrix() throws {
        let noise = NoiseModel(depolarizingProbability: 0.05)
        let policy = SimulationPolicy(
            densityMatrixQubitLimit: 8,
            preferDensityMatrixWhenNoisy: true,
            preferTrajectoryWhenDensityMatrixTooWide: true
        )
        let method = try QuantumBackendFactory.recommendMethod(
            qubitCount: 3,
            noise: noise,
            policy: policy
        )
        XCTAssertEqual(method, .densityMatrix)
    }

    func testMakeRecommendedReturnsTrajectoryWhenPolicyRequires() throws {
        let noise = NoiseModel(phaseDampingProbability: 0.2)
        let policy = SimulationPolicy(
            statevectorQubitLimit: 16,
            densityMatrixQubitLimit: 2,
            devicePreference: .cpu
        )
        let backend = try QuantumBackendFactory.makeRecommended(
            qubitCount: 4,
            noise: noise,
            policy: policy
        )
        XCTAssertEqual(backend.method, .trajectory)
    }

    // MARK: B — DM shot batching + expectation parity

    func testCPUDensityMatrixShotBatchMatchesExactExpectation() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let backend = CPUDensityMatrixBackend()

        let density = try CPUDensityMatrix(qubitCount: 1)
        var rng: QuantumRNG = .seeded(1)
        _ = try backend.engine.executeRNG(circuit, on: density, rng: &rng, noise: nil)
        // Exact ⟨Z⟩ after H is 0; P(|0⟩)=P(|1⟩)=0.5
        let probs = density.probabilities()
        XCTAssertEqual(probs[0], 0.5, accuracy: 1e-6)
        XCTAssertEqual(probs[1], 0.5, accuracy: 1e-6)

        let shotResult = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 11, shots: 4096)
        )
        let counts = try XCTUnwrap(shotResult.shotCounts)
        let zEst = DensityMatrixShotSampler.zExpectation(from: counts, qubits: [0])
        XCTAssertEqual(zEst, 0, accuracy: 0.06)

        let p0 = QFloat(counts.counts[0, default: 0]) / QFloat(counts.shots)
        XCTAssertEqual(p0, 0.5, accuracy: 0.05)
    }

    func testCPUDensityMatrixShotParityVersusTrajectory() throws {
        let noise = NoiseModel(amplitudeDampingProbability: 0.3)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let dm = try CPUDensityMatrixBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(noise: noise, seed: 3, shots: 4000)
        )
        let traj = try TrajectoryBackend(engine: CPUStatevectorEngine()).run(
            circuit: circuit,
            options: QuantumRunOptions(noise: noise, seed: 3, shots: 4000)
        )

        let dmP0 = QFloat(dm.shotCounts?.counts[0, default: 0] ?? 0) / 4000
        let trajP0 = QFloat(traj.shotCounts?.counts[0, default: 0] ?? 0) / 4000
        // Analytic AD on |1⟩: P(|0⟩)=γ=0.3
        XCTAssertEqual(dmP0, 0.3, accuracy: 0.04)
        XCTAssertEqual(trajP0, dmP0, accuracy: 0.05)
    }

    func testMetalDensityMatrixShotBatchWhenAvailable() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let noise = NoiseModel(depolarizingProbability: 0.15)
        let backend = try DensityMatrixBackend()

        let exact = try backend.run(circuit: circuit, options: QuantumRunOptions(noise: noise, seed: 1))
        _ = exact
        let density = try DensityMatrix(qubitCount: 1)
        var rng: QuantumRNG = .seeded(1)
        _ = try backend.engine.executeRNG(circuit, on: density, rng: &rng, noise: noise)
        let exactP0 = backend.engine.probabilities(of: density)[0]
        XCTAssertEqual(exactP0, 2 * QFloat(0.15) / 3, accuracy: 1e-5)

        let shots = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(noise: noise, seed: 5, shots: 4000)
        )
        let estP0 = QFloat(shots.shotCounts?.counts[0, default: 0] ?? 0) / 4000
        XCTAssertEqual(estP0, exactP0, accuracy: 0.04)
    }

    func testPreparedDensityShotBatchingFlag() throws {
        var unitary = try QuantumCircuit(qubitCount: 2)
        try unitary.h(0)
        try unitary.cx(0, 1)
        XCTAssertTrue(unitary.allowsPreparedDensityShotBatching(noise: nil))

        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var measured = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try measured.h(0)
        try measured.measure(qubits: [0], classicalRegister: 0)
        XCTAssertFalse(measured.allowsPreparedDensityShotBatching(noise: nil))

        var dephasingOnly = NoiseModel()
        dephasingOnly.measurementMode = .dephasingOnly
        XCTAssertTrue(measured.allowsPreparedDensityShotBatching(noise: dephasingOnly))
    }

    // MARK: C — Expanded resource estimation

    func testResourceEstimateMemoryGrowsWithQubitCount() throws {
        let n3 = try QuantumBackendFactory.estimateResources(qubitCount: 3)
        let n5 = try QuantumBackendFactory.estimateResources(qubitCount: 5)
        XCTAssertEqual(n3.recommendedMethod, .statevector)
        XCTAssertEqual(n5.recommendedMethod, .statevector)
        XCTAssertGreaterThan(n5.estimatedStateBytes, n3.estimatedStateBytes)
        XCTAssertGreaterThan(n5.estimatedPeakMemoryBytes, n3.estimatedPeakMemoryBytes)
        XCTAssertGreaterThan(n5.estimatedRuntimeHintNanoseconds, 0)
    }

    func testResourceEstimateNoisyRecommendsDensityMatrixOrTrajectory() throws {
        let noise = NoiseModel(depolarizingProbability: 0.01)
        let narrow = try QuantumBackendFactory.estimateResources(qubitCount: 3, noise: noise)
        XCTAssertEqual(narrow.recommendedMethod, .densityMatrix)
        XCTAssertGreaterThan(narrow.estimatedPeakMemoryBytes, narrow.estimatedStateBytes / 2)

        let policy = SimulationPolicy(densityMatrixQubitLimit: 2, devicePreference: .cpu)
        let wide = try QuantumBackendFactory.estimateResources(
            qubitCount: 5,
            noise: noise,
            policy: policy
        )
        XCTAssertEqual(wide.recommendedMethod, .trajectory)
        XCTAssertEqual(wide.recommendedDevice, .cpu)
        XCTAssertEqual(wide.assumedTrajectoryShots, SimulationPolicy.default.defaultTrajectoryShots)
        // Noisy trajectories are serial on Metal too; CPU peak ≈ 2× SV.
        XCTAssertEqual(wide.estimatedPeakMemoryBytes, wide.estimatedStateBytes * 2)
    }

    func testResourceEstimateFloat64ForcesCPUDeviceHint() throws {
        let policy = SimulationPolicy(
            devicePreference: .automatic,
            precision: .float64
        )
        let estimate = try QuantumBackendFactory.estimateResources(
            qubitCount: 4,
            policy: policy
        )
        XCTAssertEqual(estimate.recommendedDevice, .cpu)
        XCTAssertEqual(
            estimate.estimatedStateBytes,
            (1 << 4) * 2 * MemoryLayout<Double>.stride
        )
    }

    func testResourceEstimateFailsEarlyWhenOverMemoryBudget() throws {
        let policy = SimulationPolicy(maxPeakMemoryBytes: 64)
        XCTAssertThrowsError(
            try QuantumBackendFactory.estimateResources(qubitCount: 8, policy: policy)
        ) { error in
            guard case SimulationPolicyError.estimatedMemoryExceedsBudget = error else {
                return XCTFail("Expected estimatedMemoryExceedsBudget, got \(error)")
            }
        }
    }

    func testMakeRecommendedFailsEarlyWhenOverMemoryBudget() throws {
        let policy = SimulationPolicy(
            devicePreference: .cpu,
            maxPeakMemoryBytes: 128
        )
        XCTAssertThrowsError(
            try QuantumBackendFactory.makeRecommended(qubitCount: 10, policy: policy)
        ) { error in
            guard case SimulationPolicyError.estimatedMemoryExceedsBudget = error else {
                return XCTFail("Expected estimatedMemoryExceedsBudget, got \(error)")
            }
        }
    }

    func testMakeStatevectorHonorsMemoryBudget() throws {
        let policy = SimulationPolicy(
            devicePreference: .cpu,
            maxPeakMemoryBytes: 64
        )
        XCTAssertThrowsError(
            try QuantumBackendFactory.makeStatevector(
                devicePreference: .cpu,
                qubitCount: 10,
                policy: policy
            )
        ) { error in
            guard case SimulationPolicyError.estimatedMemoryExceedsBudget = error else {
                return XCTFail("Expected estimatedMemoryExceedsBudget, got \(error)")
            }
        }
    }

    func testBudgetedFactoryWithoutQubitCountThrows() throws {
        let policy = SimulationPolicy(
            devicePreference: .cpu,
            maxPeakMemoryBytes: 1_000_000
        )
        XCTAssertThrowsError(
            try QuantumBackendFactory.makeStatevector(
                devicePreference: .cpu,
                policy: policy
            )
        ) { error in
            guard case SimulationPolicyError.memoryBudgetRequiresQubitCount = error else {
                return XCTFail("Expected memoryBudgetRequiresQubitCount, got \(error)")
            }
        }
        XCTAssertThrowsError(
            try QuantumBackendFactory.makeDensityMatrix(
                devicePreference: .cpu,
                policy: policy
            )
        ) { error in
            guard case SimulationPolicyError.memoryBudgetRequiresQubitCount = error else {
                return XCTFail("Expected memoryBudgetRequiresQubitCount, got \(error)")
            }
        }
        XCTAssertThrowsError(
            try QuantumBackendFactory.makeTrajectory(
                devicePreference: .cpu,
                policy: policy
            )
        ) { error in
            guard case SimulationPolicyError.memoryBudgetRequiresQubitCount = error else {
                return XCTFail("Expected memoryBudgetRequiresQubitCount, got \(error)")
            }
        }
    }

    func testCPUStatevectorResourceEstimateIsNotFullMatrix() throws {
        let policy = SimulationPolicy(devicePreference: .cpu)
        let estimate = try QuantumBackendFactory.estimateResources(
            qubitCount: 8,
            policy: policy
        )
        XCTAssertEqual(estimate.recommendedMethod, .statevector)
        let complexBytes = 2 * MemoryLayout<Double>.stride
        let stateBytes = (1 << 8) * complexBytes
        XCTAssertEqual(estimate.estimatedStateBytes, stateBytes)
        XCTAssertEqual(estimate.estimatedPeakMemoryBytes, stateBytes * 2)
        // Full n-qubit unitary would be (2^n)^2 complexes ≫ 2× statevector.
        XCTAssertLessThan(estimate.estimatedPeakMemoryBytes, (1 << 16) * complexBytes)
    }

    func testMakeTrajectoryHonorsPolicyDevicePreference() throws {
        let policy = SimulationPolicy(devicePreference: .cpu)
        let backend = try QuantumBackendFactory.makeTrajectory(qubitCount: 1, policy: policy)
        let trajectory = try XCTUnwrap(backend as? TrajectoryBackend)
        XCTAssertEqual(trajectory.deviceName, "CPU")
    }

    func testMakeTrajectoryNoisyBudgetDoesNotUseNoiselessBatchCharge() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal batching vs serial peak split is GPU-only")
        }
        let noise = NoiseModel(depolarizingProbability: 0.1)
        let n = 8
        let policy = SimulationPolicy(devicePreference: .automatic)
        let device = try QuantumBackendFactory.resolveDeviceHint(policy: policy)
        guard device == .metal else {
            throw XCTSkip("automatic device hint is not Metal")
        }
        let (_, noisyPeak, _) = QuantumBackendFactory.peakMemoryBreakdown(
            qubitCount: n,
            method: .trajectory,
            device: .metal,
            noise: noise,
            policy: policy,
            shots: nil
        )
        let (_, quietPeak, _) = QuantumBackendFactory.peakMemoryBreakdown(
            qubitCount: n,
            method: .trajectory,
            device: .metal,
            noise: nil,
            policy: policy,
            shots: nil
        )
        XCTAssertGreaterThan(quietPeak, noisyPeak)
        let budget = (noisyPeak + quietPeak) / 2
        let tight = SimulationPolicy(
            devicePreference: .automatic,
            maxPeakMemoryBytes: budget
        )
        // Previously makeTrajectory always budgeted noise:nil and would throw here.
        let backend = try QuantumBackendFactory.makeTrajectory(
            qubitCount: n,
            policy: tight,
            noise: noise
        )
        XCTAssertEqual(backend.method, .trajectory)
    }

    func testRecommendMethodUsesCPUWidthCaps() throws {
        let noise = NoiseModel(depolarizingProbability: 0.05)
        let policy = SimulationPolicy(
            densityMatrixQubitLimit: 14,
            devicePreference: .cpu,
            cpuDensityMatrixQubitLimit: 8
        )
        // 10 qubits: Metal DM would fit (14), CPU DM does not (8) → trajectory.
        let method = try QuantumBackendFactory.recommendMethod(
            qubitCount: 10,
            noise: noise,
            policy: policy
        )
        XCTAssertEqual(method, .trajectory)
    }

    func testPreferTrajectoryFalseThrowsWhenDensityMatrixTooWide() throws {
        let noise = NoiseModel(depolarizingProbability: 0.05)
        let policy = SimulationPolicy(
            densityMatrixQubitLimit: 2,
            preferDensityMatrixWhenNoisy: true,
            preferTrajectoryWhenDensityMatrixTooWide: false,
            devicePreference: .cpu
        )
        XCTAssertThrowsError(
            try QuantumBackendFactory.recommendMethod(qubitCount: 5, noise: noise, policy: policy)
        ) { error in
            guard case SimulationPolicyError.densityMatrixRequiredButTooWide = error else {
                return XCTFail("Expected densityMatrixRequiredButTooWide, got \(error)")
            }
        }
    }
}
