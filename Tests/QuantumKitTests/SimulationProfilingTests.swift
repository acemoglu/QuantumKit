import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    func testProfilingOffLeavesProfileNil() throws {
        let backend = CPUStatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let result = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 42)
        )
        XCTAssertNil(result.profile)
        XCTAssertNil(result.metadata.profile)
    }

    func testProfilingOnOffSameShotHistogramCPU() throws {
        let backend = CPUStatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let off = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 99, shots: 256)
        )
        let on = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: 99,
                shots: 256,
                profiling: .detailed
            )
        )

        XCTAssertEqual(off.shotCounts, on.shotCounts)
        XCTAssertEqual(off.metadata.pipelineHash, on.metadata.pipelineHash)
        XCTAssertNil(off.profile)
        try assertProfilePopulated(on.profile, qubitCount: 2, method: .statevector, isCPU: true)
        XCTAssertEqual(on.profile?.wallClockNanoseconds, on.metadata.wallClockNanoseconds)
        try assertAggregatedGateTimings(on.profile?.gateTimings, gateCount: circuit.gates.count)
        XCTAssertEqual(on.profile?.phaseTimings?.map(\.name), ["sample"])
    }

    func testProfilingOnOffSameStatevectorExecutionCPU() throws {
        let backend = CPUStatevectorBackend()
        var circuit = try QuantumCircuit(
            qubitCount: 2,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        try circuit.c_if(classicalRegister: 0, equals: 1, x: 1)

        let off = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 7)
        )
        let on = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 7, profiling: .detailed)
        )

        XCTAssertEqual(off.execution, on.execution)
        XCTAssertEqual(off.metadata.pipelineHash, on.metadata.pipelineHash)
        XCTAssertNil(off.profile)
        try assertProfilePopulated(on.profile, qubitCount: 2, method: .statevector, isCPU: true)
        try assertAggregatedGateTimings(on.profile?.gateTimings, gateCount: circuit.gates.count)
        XCTAssertEqual(on.profile?.phaseTimings?.map(\.name), ["evolve"])
    }

    func testProfilingFieldsAndGateTimingsCPU() throws {
        let backend = CPUStatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 4)
        for qubit in 0..<4 {
            try circuit.h(qubit)
        }
        try circuit.cx(0, 1)
        try circuit.cx(1, 2)
        try circuit.cx(2, 3)
        try circuit.rz(theta: QFloat(0.37), 0)

        let result = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 3, profiling: .detailed)
        )

        let profile = try XCTUnwrap(result.profile)
        try assertProfilePopulated(profile, qubitCount: 4, method: .statevector, isCPU: true)
        XCTAssertGreaterThan(profile.wallClockNanoseconds, 0)

        try assertAggregatedGateTimings(profile.gateTimings, gateCount: circuit.gates.count)

        let phases = try XCTUnwrap(profile.phaseTimings)
        XCTAssertEqual(phases.map(\.name), ["evolve"])
        XCTAssertGreaterThan(phases[0].wallClockNanoseconds, 0)
    }

    func testProfilingDensityMatrixCPU() throws {
        let backend = CPUDensityMatrixBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let off = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 11, shots: 64)
        )
        let on = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 11, shots: 64, profiling: .detailed)
        )

        XCTAssertEqual(off.shotCounts, on.shotCounts)
        let profile = try XCTUnwrap(on.profile)
        try assertProfilePopulated(profile, qubitCount: 2, method: .densityMatrix, isCPU: true)
        XCTAssertEqual(profile.memorySource, .estimated)
        try assertAggregatedGateTimings(profile.gateTimings, gateCount: circuit.gates.count)
        XCTAssertEqual(profile.phaseTimings?.map(\.name), ["sample"])
    }

    func testProfilingOnOffSameShotHistogramMetal() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let backend = try StatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let off = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 99, shots: 256)
        )
        let on = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 99, shots: 256, profiling: .enabled)
        )

        XCTAssertEqual(off.shotCounts, on.shotCounts)
        XCTAssertEqual(off.metadata.pipelineHash, on.metadata.pipelineHash)
        try assertProfilePopulated(on.profile, qubitCount: 2, method: .statevector, isCPU: false)
        XCTAssertGreaterThan(on.profile?.peakMemoryBytes ?? 0, on.profile?.stateBytes ?? 0)
        XCTAssertNil(on.profile?.gateTimings)
        XCTAssertNil(on.profile?.phaseTimings)
    }

    /// Batchable Metal Bell under `.detailed` (recorder + `sample` phase), not merely `.enabled`.
    func testProfilingOnOffSameShotHistogramMetalDetailed() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let backend = try StatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let off = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 99, shots: 256)
        )
        let on = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 99, shots: 256, profiling: .detailed)
        )

        XCTAssertEqual(off.shotCounts, on.shotCounts)
        XCTAssertEqual(off.metadata.pipelineHash, on.metadata.pipelineHash)
        try assertProfilePopulated(on.profile, qubitCount: 2, method: .statevector, isCPU: false)
        // Option B: Metal SV does not emit per-gate host samples.
        XCTAssertNil(on.profile?.gateTimings)
        XCTAssertEqual(on.profile?.phaseTimings?.map(\.name), ["sample"])
        XCTAssertGreaterThan(on.profile?.phaseTimings?.first?.wallClockNanoseconds ?? 0, 0)
    }

    func testProfilingMetalNoiselessOmitsGateTimings() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let backend = try StatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let evolve = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 3, profiling: .detailed)
        )
        XCTAssertNil(evolve.profile?.gateTimings)
        XCTAssertEqual(evolve.profile?.phaseTimings?.map(\.name), ["evolve"])

        let sample = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 3, shots: 64, profiling: .detailed)
        )
        XCTAssertNil(sample.profile?.gateTimings)
        XCTAssertEqual(sample.profile?.phaseTimings?.map(\.name), ["sample"])
        XCTAssertGreaterThan(sample.profile?.peakMemoryBytes ?? 0, sample.profile?.stateBytes ?? 0)
    }

    func testProfilingOnOffSameShotHistogramMetalNonBatchable() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let backend = try StatevectorBackend()
        var circuit = try QuantumCircuit(
            qubitCount: 2,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        try circuit.c_if(classicalRegister: 0, equals: 1, x: 1)

        let off = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 41, shots: 64)
        )
        let on = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 41, shots: 64, profiling: .detailed)
        )

        XCTAssertEqual(off.shotCounts, on.shotCounts)
        XCTAssertEqual(off.execution, on.execution)
        XCTAssertEqual(off.metadata.pipelineHash, on.metadata.pipelineHash)
        XCTAssertNil(on.profile?.gateTimings)
        XCTAssertEqual(on.profile?.phaseTimings?.map(\.name), ["sample"])
    }

    func testProfilingPipelineFingerprintIgnoresProfiling() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let off = QuantumRunOptions(seed: 1, shots: 8)
        let on = QuantumRunOptions(seed: 1, shots: 8, profiling: .detailed)
        XCTAssertEqual(
            PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: off),
            PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: on)
        )
    }

    func testProfilingEstimatorDetailedRecordsEstimatePhase() throws {
        let backend = CPUStatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))

        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 1, profiling: .detailed)
        )

        let profile = try XCTUnwrap(result.metadata.profile)
        XCTAssertEqual(profile.phaseTimings?.map(\.name), ["estimate"])
        try assertAggregatedGateTimings(profile.gateTimings, gateCount: circuit.gates.count)
    }

    func testProfilingShotEstimatorOmitsGateTimings() throws {
        let backend = CPUStatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "X0"))

        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 4, profiling: .detailed),
            estimatorOptions: EstimatorOptions(shots: 32)
        )

        let profile = try XCTUnwrap(result.metadata.profile)
        XCTAssertEqual(profile.phaseTimings?.map(\.name), ["estimate"])
        XCTAssertNil(profile.gateTimings)
    }

    func testProfilingEstimatorTrajectoryPhasesAreEstimateOnly() throws {
        let backend = TrajectoryBackend(engine: CPUStatevectorEngine())
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))

        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 8, shots: 32, profiling: .detailed)
        )

        let profile = try XCTUnwrap(result.metadata.profile)
        XCTAssertEqual(profile.phaseTimings?.map(\.name), ["estimate"])
        XCTAssertFalse(profile.phaseTimings?.map(\.name).contains("sample") ?? true)
        XCTAssertNil(profile.gateTimings)
    }

    func testProfilingSamplerDetailedRecordsPhases() throws {
        let backend = CPUStatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)

        let evolve = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(seed: 2, profiling: .detailed)
        )
        XCTAssertEqual(evolve.metadata.profile?.phaseTimings?.map(\.name), ["evolve"])
        try assertAggregatedGateTimings(evolve.metadata.profile?.gateTimings, gateCount: circuit.gates.count)

        let sample = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(seed: 2, shots: 32, profiling: .detailed)
        )
        XCTAssertEqual(sample.metadata.profile?.phaseTimings?.map(\.name), ["sample"])
        try assertAggregatedGateTimings(sample.metadata.profile?.gateTimings, gateCount: circuit.gates.count)
    }

    func testProfilingSamplerTrajectoryDetailedRecordsSamplePhase() throws {
        let backend = TrajectoryBackend(engine: CPUStatevectorEngine())
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let result = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(seed: 6, shots: 32, profiling: .detailed)
        )
        XCTAssertEqual(result.metadata.profile?.phaseTimings?.map(\.name), ["sample"])
        XCTAssertEqual(result.metadata.method, .trajectory)
        try assertAggregatedGateTimings(result.metadata.profile?.gateTimings, gateCount: circuit.gates.count)
    }

    func testProfilingSamplerMetalBatchedOmitsGateTimings() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let backend = try StatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        let result = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(seed: 5, shots: 32, profiling: .detailed)
        )
        XCTAssertNil(result.metadata.profile?.gateTimings)
        XCTAssertEqual(result.metadata.profile?.phaseTimings?.map(\.name), ["sample"])
    }

    func testProfilingMemoryEstimateUsesEffectiveShots() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        let none = SimulationMemoryFootprint.estimate(
            qubitCount: 3,
            method: .statevector,
            isCPU: false,
            shots: nil,
            batchSize: 32,
            circuit: circuit,
            noise: nil
        )
        let batched = SimulationMemoryFootprint.estimate(
            qubitCount: 3,
            method: .statevector,
            isCPU: false,
            shots: 64,
            batchSize: 32,
            circuit: circuit,
            noise: nil
        )
        XCTAssertEqual(none.peakBytes, none.stateBytes * 2)
        XCTAssertEqual(batched.peakBytes, batched.stateBytes * (32 + 1))
        XCTAssertGreaterThan(batched.peakBytes, none.peakBytes)
    }

    func testProfilingRecorderConcurrentTimeGateIsThreadSafe() throws {
        let recorder = SimulationProfileRecorder(options: .detailed)
        recorder.markGateInstrumentationStarted()
        DispatchQueue.concurrentPerform(iterations: 128) { iteration in
            try! recorder.timeGate(index: iteration % 4) {
                _ = (0..<32).reduce(0, +)
            }
        }
        let timings = try XCTUnwrap(recorder.snapshotGateTimings())
        XCTAssertEqual(timings.map(\.index), [0, 1, 2, 3])
        XCTAssertEqual(Set(timings.map(\.index)).count, 4)
        XCTAssertEqual(timings.count, 4)
    }

    /// Multi-shot gate ns must be summed by index (`&+=`), not last-write-wins.
    func testProfilingMultiShotGateTimingsAreSummed() throws {
        let backend = CPUStatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 4)
        for qubit in 0..<4 {
            try circuit.h(qubit)
        }
        try circuit.cx(0, 1)
        try circuit.cx(1, 2)
        try circuit.cx(2, 3)
        try circuit.rz(theta: QFloat(0.37), 0)

        let shotCount = 8
        let one = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 17, shots: 1, profiling: .detailed)
        )
        let many = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 17, shots: shotCount, profiling: .detailed)
        )

        let oneGates = try XCTUnwrap(one.profile?.gateTimings)
        let manyGates = try XCTUnwrap(many.profile?.gateTimings)
        try assertAggregatedGateTimings(oneGates, gateCount: circuit.gates.count)
        try assertAggregatedGateTimings(manyGates, gateCount: circuit.gates.count)

        var oneTotal: UInt64 = 0
        var manyTotal: UInt64 = 0
        for index in 0..<circuit.gates.count {
            let oneNs = oneGates[index].wallClockNanoseconds
            let manyNs = manyGates[index].wallClockNanoseconds
            XCTAssertEqual(oneGates[index].index, index)
            XCTAssertEqual(manyGates[index].index, index)
            XCTAssertGreaterThanOrEqual(
                manyNs,
                oneNs,
                "gate \(index): \(shotCount)-shot ns must be >= 1-shot (sum, not overwrite)"
            )
            oneTotal &+= oneNs
            manyTotal &+= manyNs
        }
        XCTAssertGreaterThan(oneTotal, 0, "1-shot gate timings must be measurable")
        // Last-write-wins would keep manyTotal ≈ oneTotal; summing scales roughly with shot count.
        XCTAssertGreaterThanOrEqual(
            manyTotal,
            oneTotal &* UInt64(shotCount / 2),
            "\(shotCount)-shot total gate ns must scale with shots (summation)"
        )
    }

    func testProfilingParameterShiftPhasesAreGradientOnly() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let backend = try StatevectorBackend()
        let theta = QFloat(Double.pi / 4)

        let detailed = try GradientCalculator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: backend,
            options: QuantumRunOptions(seed: 11, profiling: .detailed)
        )
        let off = try GradientCalculator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: backend,
            options: QuantumRunOptions(seed: 11)
        )

        XCTAssertEqual(detailed.expectationValue, off.expectationValue, accuracy: 1e-5)
        XCTAssertEqual(detailed.parameterGradients.count, off.parameterGradients.count)
        XCTAssertEqual(
            detailed.parameterGradients[0].gradient,
            off.parameterGradients[0].gradient,
            accuracy: 1e-4
        )
        let profile = try XCTUnwrap(detailed.metadata.profile)
        XCTAssertEqual(profile.phaseTimings?.map(\.name), ["gradient"])
        XCTAssertFalse(profile.phaseTimings?.map(\.name).contains("estimate") ?? true)
    }

    func testProfilingTaskLocalChildTaskRecordsGateNotPhase() throws {
        let options = QuantumRunOptions(seed: 1, profiling: .detailed)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let profileCircuit = circuit

        let profile = try SimulationProfiling.usingRecorder(for: options) {
            let box = ChildTaskErrorBox()
            let group = DispatchGroup()
            group.enter()
            Task {
                defer { group.leave() }
                do {
                    guard let recorder = SimulationProfiling.recorder else {
                        throw ChildTaskProfilingError.missingInheritedRecorder
                    }
                    if SimulationProfiling.isPhaseOwner {
                        throw ChildTaskProfilingError.childTreatedAsPhaseOwner
                    }
                    try recorder.timeGate(index: 0) {
                        usleep(1_000)
                    }
                    _ = try SimulationProfiling.timePhase("child") { () -> Int in
                        usleep(500)
                        return 0
                    }
                    let childSnapshot = SimulationProfiling.finishProfile(
                        options: options,
                        circuit: profileCircuit,
                        method: .statevector,
                        isCPU: true,
                        elapsed: 1
                    )
                    if childSnapshot != nil {
                        throw ChildTaskProfilingError.workerPublishedProfile
                    }
                } catch {
                    box.error = error
                }
            }
            group.wait()
            if let childError = box.error { throw childError }
            _ = try SimulationProfiling.timePhase("sample") { () -> Int in 0 }
            return SimulationProfiling.finishProfile(
                options: options,
                circuit: circuit,
                method: .statevector,
                isCPU: true,
                elapsed: 1
            )
        }

        let finished = try XCTUnwrap(profile)
        let gates = try XCTUnwrap(finished.gateTimings)
        XCTAssertEqual(gates.map(\.index), [0])
        XCTAssertGreaterThan(gates[0].wallClockNanoseconds, 0)
        XCTAssertEqual(finished.phaseTimings?.map(\.name), ["sample"])
        XCTAssertFalse(finished.phaseTimings?.map(\.name).contains("child") ?? false)
    }

    func testProfilingWithWorkerRecorderDisablesPhasePublishing() throws {
        let options = QuantumRunOptions(seed: 1, profiling: .detailed)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)

        let profile = try SimulationProfiling.usingRecorder(for: options) {
            try SimulationProfiling.withWorkerRecorder {
                XCTAssertFalse(SimulationProfiling.isPhaseOwner)
                try SimulationProfiling.recorder?.timeGate(index: 0) {}
                _ = try SimulationProfiling.timePhase("worker") { () -> Int in 0 }
            }
            _ = try SimulationProfiling.timePhase("sample") { () -> Int in 0 }
            return SimulationProfiling.finishProfile(
                options: options,
                circuit: circuit,
                method: .statevector,
                isCPU: true,
                elapsed: 1
            )
        }

        let finished = try XCTUnwrap(profile)
        XCTAssertEqual(finished.gateTimings?.map(\.index), [0])
        XCTAssertEqual(finished.phaseTimings?.map(\.name), ["sample"])
    }

    /// Nested shot Estimator must not `timeGate` basis-changed circuits into the owner recorder.
    func testProfilingNestedShotEstimatorDoesNotRecordMeasureCircuitGates() throws {
        let backend = CPUStatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "X0"))
        let options = QuantumRunOptions(seed: 4, profiling: .detailed)

        let (indices, profile) = try SimulationProfiling.usingRecorder(for: options) {
            _ = try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: backend,
                options: options,
                estimatorOptions: EstimatorOptions(shots: 32)
            )
            let rec = try XCTUnwrap(SimulationProfiling.recorder)
            return (
                rec.recordedGateIndices(),
                SimulationProfiling.finishProfile(
                    options: options,
                    circuit: circuit,
                    method: .statevector,
                    isCPU: true,
                    elapsed: 1
                )
            )
        }

        XCTAssertNil(indices, "shot Estimator must not mark/timeGate measure-circuit indices")
        XCTAssertNil(profile?.gateTimings)
        XCTAssertNil(profile?.phaseTimings)
    }

    func testProfilingNestedExactEstimatorRecordsUserCircuitGates() throws {
        let backend = CPUStatevectorBackend()
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let options = QuantumRunOptions(seed: 1, profiling: .detailed)

        let indices = try SimulationProfiling.usingRecorder(for: options) {
            _ = try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: backend,
                options: options
            )
            return SimulationProfiling.recorder?.recordedGateIndices()
        }

        XCTAssertEqual(indices, [0])
    }

    /// Gradient + Metal DM + `options.shots`: nested shot Estimator must not leak extra
    /// measure-circuit indices; shift evaluations may still time the user-circuit gates.
    func testProfilingGradientShotEstimatorOmitsMeasureCircuitGateIndices() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "X0"))
        let backend = try DensityMatrixBackend()
        let theta = QFloat(Double.pi / 4)

        let result = try GradientCalculator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: backend,
            options: QuantumRunOptions(seed: 11, shots: 16, profiling: .detailed)
        )

        let profile = try XCTUnwrap(result.metadata.profile)
        XCTAssertEqual(profile.phaseTimings?.map(\.name), ["gradient"])
        let gateIndices = profile.gateTimings?.map(\.index) ?? []
        XCTAssertFalse(
            gateIndices.contains { $0 >= circuit.gates.count },
            "measure-circuit indices must not appear on the gradient profile: \(gateIndices)"
        )
    }

    func testSimulationProfileJSONRoundTrip() throws {
        let profile = SimulationProfile(
            wallClockNanoseconds: 1_000,
            stateBytes: 128,
            peakMemoryBytes: 256,
            memorySource: .estimated,
            gateTimings: [SimulationGateTiming(index: 0, wallClockNanoseconds: 10)],
            phaseTimings: [SimulationPhaseTiming(name: "evolve", wallClockNanoseconds: 900)]
        )
        let metadata = QuantumResultMetadata(
            method: .statevector,
            seed: 1,
            deviceName: "CPU",
            wallClockNanoseconds: 1_000,
            qubitCount: 3,
            gateCount: 1,
            noiseSnapshot: nil,
            profile: profile
        )
        let data = try JSONEncoder().encode(metadata)
        let decoded = try JSONDecoder().decode(QuantumResultMetadata.self, from: data)
        XCTAssertEqual(decoded, metadata)
        XCTAssertEqual(decoded.profile?.memorySource, .estimated)
    }

    func testProfilingEmptyCPUInstructionRangeYieldsEmptyGateTimings() throws {
        let backend = CPUStatevectorBackend()
        let circuit = try QuantumCircuit(qubitCount: 1)
        XCTAssertTrue(circuit.gates.isEmpty)

        let result = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 1, profiling: .detailed)
        )
        let timings = try XCTUnwrap(result.profile?.gateTimings)
        XCTAssertEqual(timings, [])
        XCTAssertEqual(result.profile?.phaseTimings?.map(\.name), ["evolve"])
    }

    func testProfilingEmptyMetalInstructionRangeOmitsGateTimings() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        let backend = try StatevectorBackend()
        let circuit = try QuantumCircuit(qubitCount: 1)
        let result = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 1, profiling: .detailed)
        )
        XCTAssertNil(result.profile?.gateTimings)
        XCTAssertEqual(result.profile?.phaseTimings?.map(\.name), ["evolve"])
    }

    func testProfilingOnOffSameShotHistogramMetalDensityMatrixDetailed() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        let backend = try DensityMatrixBackend()
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let off = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 21, shots: 64)
        )
        let on = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 21, shots: 64, profiling: .detailed)
        )
        XCTAssertEqual(off.shotCounts, on.shotCounts)
        XCTAssertEqual(off.metadata.pipelineHash, on.metadata.pipelineHash)
        XCTAssertEqual(on.profile?.phaseTimings?.map(\.name), ["sample"])
        try assertAggregatedGateTimings(on.profile?.gateTimings, gateCount: circuit.gates.count)
    }

    func testProfilingMultiShotGateTimingsAreSummedMetalDensityMatrix() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        let backend = try DensityMatrixBackend()
        // Prepared-ρ Bell shots evolve once; mid-circuit + c_if re-executes per shot.
        var circuit = try QuantumCircuit(
            qubitCount: 2,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.h(0)
        try circuit.measure(qubits: [0], classicalRegister: 0)
        try circuit.c_if(classicalRegister: 0, equals: 1, x: 1)

        _ = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 21, shots: 1, profiling: .detailed)
        )

        let shotCount = 8
        let one = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 21, shots: 1, profiling: .detailed)
        )
        let many = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 21, shots: shotCount, profiling: .detailed)
        )

        let oneGates = try XCTUnwrap(one.profile?.gateTimings)
        let manyGates = try XCTUnwrap(many.profile?.gateTimings)
        try assertAggregatedGateTimings(oneGates, gateCount: circuit.gates.count)
        try assertAggregatedGateTimings(manyGates, gateCount: circuit.gates.count)

        var oneTotal: UInt64 = 0
        var manyTotal: UInt64 = 0
        for index in 0..<circuit.gates.count {
            let oneNs = oneGates[index].wallClockNanoseconds
            let manyNs = manyGates[index].wallClockNanoseconds
            XCTAssertGreaterThanOrEqual(
                manyNs,
                oneNs,
                "gate \(index): \(shotCount)-shot ns must be >= 1-shot (sum, not overwrite)"
            )
            oneTotal &+= oneNs
            manyTotal &+= manyNs
        }
        XCTAssertGreaterThan(oneTotal, 0, "1-shot gate timings must be measurable")
        XCTAssertGreaterThanOrEqual(
            manyTotal,
            oneTotal &* UInt64(shotCount / 2),
            "\(shotCount)-shot total gate ns must scale with shots (summation)"
        )
    }

    func testProfilingOnOffSameShotHistogramTrajectoryDetailed() throws {
        let backend = TrajectoryBackend(engine: CPUStatevectorEngine())
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let off = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 33, shots: 64)
        )
        let on = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 33, shots: 64, profiling: .detailed)
        )
        XCTAssertEqual(off.shotCounts, on.shotCounts)
        XCTAssertEqual(off.metadata.pipelineHash, on.metadata.pipelineHash)
        XCTAssertEqual(on.profile?.phaseTimings?.map(\.name), ["sample"])
        try assertAggregatedGateTimings(on.profile?.gateTimings, gateCount: circuit.gates.count)
    }

    private func assertAggregatedGateTimings(
        _ timings: [SimulationGateTiming]?,
        gateCount: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let timings = try XCTUnwrap(timings, file: file, line: line)
        XCTAssertEqual(timings.count, gateCount, file: file, line: line)
        XCTAssertEqual(timings.map(\.index), Array(0..<gateCount), file: file, line: line)
        XCTAssertEqual(Set(timings.map(\.index)).count, timings.count, file: file, line: line)
    }

    private func assertProfilePopulated(
        _ profile: SimulationProfile?,
        qubitCount: Int,
        method: QuantumSimulationMethod,
        isCPU: Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let profile = try XCTUnwrap(profile, file: file, line: line)
        XCTAssertGreaterThan(profile.wallClockNanoseconds, 0, file: file, line: line)
        XCTAssertEqual(profile.memorySource, .estimated, file: file, line: line)

        let complexBytes = isCPU
            ? 2 * MemoryLayout<Double>.stride
            : 2 * MemoryLayout<Float32>.stride
        let dim = 1 << qubitCount
        let expectedState: Int
        switch method {
        case .statevector, .trajectory:
            expectedState = dim * complexBytes
        case .densityMatrix:
            expectedState = dim * dim * complexBytes
        }
        XCTAssertEqual(profile.stateBytes, expectedState, file: file, line: line)
        XCTAssertGreaterThanOrEqual(profile.peakMemoryBytes, profile.stateBytes, file: file, line: line)
    }
}

private final class ChildTaskErrorBox: @unchecked Sendable {
    var error: Error?
}

private enum ChildTaskProfilingError: Error {
    case missingInheritedRecorder
    case childTreatedAsPhaseOwner
    case workerPublishedProfile
}
