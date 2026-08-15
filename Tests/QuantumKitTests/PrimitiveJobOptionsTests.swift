import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Primitive job options / readout mitigation

    func testShotBudgetMatchesEstimatorOptionsResolution() throws {
        XCTAssertNil(try ShotBudget.exact.resolvedShots())
        XCTAssertEqual(try ShotBudget(shots: 128).resolvedShots(), 128)
        XCTAssertEqual(try ShotBudget(precision: 0.1).resolvedShots(), 100)
        XCTAssertEqual(try EstimatorOptions(shots: 50, precision: 0.01).resolvedShots(), 50)
        XCTAssertEqual(
            try EstimatorOptions(precision: 0.1).shotBudget.resolvedShots(),
            try ShotBudget(precision: 0.1).resolvedShots()
        )
    }

    func testResilienceDisabledLeavesSamplerHistogramUnchanged() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let backend = CPUStatevectorBackend()

        let baseline = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(seed: 42, shots: 512)
        )
        let explicitOff = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(seed: 42, shots: 512, resilience: .disabled)
        )

        XCTAssertEqual(baseline.shotCounts, explicitOff.shotCounts)
        XCTAssertEqual(baseline.quasiProbabilities, explicitOff.quasiProbabilities)
        XCTAssertFalse(ResilienceOptions.disabled.isEnabled)
    }

    func testReadoutMitigationRecoversPreparedGroundState() throws {
        // P(m|p): mild asymmetric flip. True prepared = |0⟩ × 1000 shots.
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.2, p10: 0.1)
        let noisy = ShotCounts(shots: 1000, counts: [0: 800, 1: 200])

        let mitigated = try ReadoutMitigation.apply(
            to: noisy,
            matrix: matrix,
            qubitCount: 1
        )

        XCTAssertEqual(mitigated.shots, 1000)
        XCTAssertEqual(mitigated.counts.values.reduce(0, +), 1000)
        let p0 = QFloat(mitigated.counts[0] ?? 0) / 1000
        XCTAssertGreaterThan(p0, 0.9)
    }

    func testSamplerReadoutMitigationChangesHistogramWhenEnabled() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        let backend = CPUStatevectorBackend()
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.25, p10: 0.25)

        var noise = NoiseModel()
        noise.readoutConfusion = matrix

        let raw = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(noise: noise, seed: 7, shots: 2048)
        )
        let mitigated = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(
                noise: noise,
                seed: 7,
                shots: 2048,
                resilience: ResilienceOptions(readoutMitigation: matrix)
            )
        )

        XCTAssertNotEqual(raw.shotCounts, mitigated.shotCounts)
        XCTAssertEqual(mitigated.shotCounts?.counts.values.reduce(0, +), 2048)
        let rawP0 = raw.quasiProbabilities["0"] ?? 0
        let mitP0 = mitigated.quasiProbabilities["0"] ?? 0
        XCTAssertGreaterThan(mitP0, rawP0)
        XCTAssertGreaterThan(mitP0, 0.9)
    }

    func testEstimatorOptionsResilienceAppliesReadoutMitigation() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.3, p10: 0.3)
        var noise = NoiseModel()
        noise.readoutConfusion = matrix
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))

        let raw = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(noise: noise, seed: 5),
            estimatorOptions: EstimatorOptions(shots: 2048)
        )
        let mitigated = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(noise: noise, seed: 5),
            estimatorOptions: EstimatorOptions(
                shots: 2048,
                resilience: ResilienceOptions(readoutMitigation: matrix)
            )
        )

        // Ideal ⟨Z⟩ on |0⟩ is 1; mitigation should move the noisy estimate upward.
        XCTAssertGreaterThan(mitigated.value, raw.value)
        XCTAssertGreaterThan(mitigated.value, 0.85)
        XCTAssertNotNil(mitigated.metadata.pipelineHash)
        XCTAssertNotEqual(raw.metadata.pipelineHash, mitigated.metadata.pipelineHash)
    }

    func testPipelineFingerprintIncludesReadoutMitigation() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let base = QuantumRunOptions(seed: 1, shots: 64)
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.1, p10: 0.05)
        let withMitigation = QuantumRunOptions(
            seed: 1,
            shots: 64,
            resilience: ResilienceOptions(readoutMitigation: matrix)
        )

        let h0 = PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: base)
        let h1 = PipelineFingerprint.hash(
            circuit: circuit,
            method: .statevector,
            options: withMitigation
        )
        XCTAssertNotEqual(h0, h1)

        // Estimator-only resilience (via resolved options) must also change the hash.
        let estimatorResolved = PipelineFingerprint.optionsForEstimatorFingerprint(
            runOptions: base,
            resolvedShots: 2048,
            resolvedResilience: ResilienceOptions(readoutMitigation: matrix)
        )
        let hEst = PipelineFingerprint.hash(
            circuit: circuit,
            method: .statevector,
            options: estimatorResolved,
            extra: ["qwc:1"]
        )
        let hEstOff = PipelineFingerprint.hash(
            circuit: circuit,
            method: .statevector,
            options: PipelineFingerprint.optionsForEstimatorFingerprint(
                runOptions: base,
                resolvedShots: 2048,
                resolvedResilience: .disabled
            ),
            extra: ["qwc:1"]
        )
        XCTAssertNotEqual(hEst, hEstOff)

        let hQwcOff = PipelineFingerprint.hash(
            circuit: circuit,
            method: .statevector,
            options: estimatorResolved,
            extra: ["qwc:0"]
        )
        XCTAssertNotEqual(hEst, hQwcOff)
    }

    func testMismatchedReadoutMitigationDoesNotRecoverPreparedState() throws {
        // Noise uses strong 0→1 flips; mitigation matrix assumes the opposite asymmetry.
        let noiseMatrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.4, p10: 0.05)
        let wrongMitigation = try ReadoutConfusionMatrix.singleQubit(p01: 0.05, p10: 0.4)
        var noise = NoiseModel()
        noise.readoutConfusion = noiseMatrix

        var circuit = try QuantumCircuit(qubitCount: 1)
        let backend = CPUStatevectorBackend()
        let raw = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(noise: noise, seed: 9, shots: 2048)
        )
        let wrong = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(
                noise: noise,
                seed: 9,
                shots: 2048,
                resilience: ResilienceOptions(readoutMitigation: wrongMitigation)
            )
        )
        let matched = try Sampler().run(
            circuit: circuit,
            backend: backend,
            options: QuantumRunOptions(
                noise: noise,
                seed: 9,
                shots: 2048,
                resilience: ResilienceOptions(readoutMitigation: noiseMatrix)
            )
        )

        let rawP0 = raw.quasiProbabilities["0"] ?? 0
        let wrongP0 = wrong.quasiProbabilities["0"] ?? 0
        let matchedP0 = matched.quasiProbabilities["0"] ?? 0
        XCTAssertGreaterThan(matchedP0, rawP0)
        XCTAssertGreaterThan(matchedP0, 0.85)
        // Wrong inverse must not match the well-mitigated recovery.
        XCTAssertLessThan(wrongP0, matchedP0 - 0.1)
    }

    func testReadoutMitigationThrowsOnNonPositivePreparedMass() throws {
        // Invertible confusion, but empty histogram mass ⇒ prepared solution is 0 after clamp.
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.1, p10: 0.1)
        let counts = ShotCounts(shots: 10, counts: [:])
        XCTAssertThrowsError(
            try ReadoutMitigation.apply(to: counts, matrix: matrix, qubitCount: 1)
        ) { error in
            XCTAssertEqual(error as? ReadoutMitigationError, .nonPositivePreparedMass)
        }
    }
}
