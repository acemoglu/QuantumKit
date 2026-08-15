import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - C13 zero-noise extrapolation

    func testZNELinearExtrapolatorRecoversAffineIntercept() throws {
        // Synthetic E(λ) = 0.4 + 0.1 λ → extrapolate to a = 0.4.
        let scales: [QFloat] = [1, 3, 5]
        let values = scales.map { 0.4 + 0.1 * $0 }
        let a = try ZeroNoiseExtrapolation.extrapolateLinear(
            scaleFactors: scales,
            values: values
        )
        XCTAssertEqual(a, 0.4, accuracy: 1e-5)
    }

    func testZNELinearExtrapolatorTwoPointRichardsonEquivalent() throws {
        // First-order Richardson / linear: a = (λ2 E1 - λ1 E2) / (λ2 - λ1) with λ1=1, λ2=3
        // E = 0.25 + 0.05 λ → a = 0.25.
        let a = try ZeroNoiseExtrapolation.extrapolateLinear(
            scaleFactors: [1, 3],
            values: [0.3, 0.4]
        )
        XCTAssertEqual(a, 0.25, accuracy: 1e-5)
    }

    func testZNEValidateRejectsInsufficientOrDuplicateScales() {
        XCTAssertThrowsError(
            try ZeroNoiseExtrapolation.validate(ZNEOptions(scaleFactors: [1]))
        ) { error in
            guard case ZNEError.insufficientScaleFactors = error else {
                return XCTFail("expected insufficientScaleFactors, got \(error)")
            }
        }
        XCTAssertThrowsError(
            try ZeroNoiseExtrapolation.validate(ZNEOptions(scaleFactors: [1, 1, 3]))
        ) { error in
            guard case ZNEError.nonDistinctScaleFactors = error else {
                return XCTFail("expected nonDistinctScaleFactors, got \(error)")
            }
        }
    }

    func testNoiseModelScalingGlobalDepolarizingOnly() {
        var model = NoiseModel(
            depolarizingProbability: 0.1,
            amplitudeDampingProbability: 0.2
        )
        model = model.adding(.pauliXFlip(probability: 0.05), for: .gate(.x))
        let scaled = model.scalingGlobalDepolarizing(by: 3)
        XCTAssertEqual(scaled.depolarizingProbability, 0.3, accuracy: 1e-6)
        XCTAssertEqual(scaled.amplitudeDampingProbability, 0.2, accuracy: 1e-6)
        XCTAssertEqual(scaled.localizedRules, model.localizedRules)
        XCTAssertEqual(model.scalingGlobalDepolarizing(by: 20).depolarizingProbability, 1, accuracy: 1e-6)
    }

    func testResilienceDisabledEstimatorMatchesExplicitOff() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let noise = NoiseModel(depolarizingProbability: 0.05)
        let backend = CPUStatevectorBackend()

        let baseline = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(noise: noise, seed: 11),
            estimatorOptions: EstimatorOptions(shots: 512)
        )
        let explicitOff = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(noise: noise, seed: 11, resilience: .disabled),
            estimatorOptions: EstimatorOptions(shots: 512, resilience: .disabled)
        )

        XCTAssertEqual(baseline.value, explicitOff.value)
        XCTAssertEqual(baseline.standardError, explicitOff.standardError)
        XCTAssertEqual(baseline.metadata.pipelineHash, explicitOff.metadata.pipelineHash)
        XCTAssertNil(baseline.zne)
        XCTAssertNil(explicitOff.zne)
        XCTAssertFalse(ResilienceOptions.disabled.isEnabled)
    }

    func testPipelineFingerprintIncludesZNE() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let base = QuantumRunOptions(seed: 2, shots: 64)
        let withZNE = QuantumRunOptions(
            seed: 2,
            shots: 64,
            resilience: ResilienceOptions(zne: .default)
        )

        let h0 = PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: base)
        let h1 = PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: withZNE)
        XCTAssertNotEqual(h0, h1)

        let hEst = PipelineFingerprint.hash(
            circuit: circuit,
            method: .statevector,
            options: PipelineFingerprint.optionsForEstimatorFingerprint(
                runOptions: base,
                resolvedShots: 256,
                resolvedResilience: ResilienceOptions(zne: ZNEOptions(scaleFactors: [1, 3]))
            ),
            extra: ["qwc:1"]
        )
        let hEstOff = PipelineFingerprint.hash(
            circuit: circuit,
            method: .statevector,
            options: PipelineFingerprint.optionsForEstimatorFingerprint(
                runOptions: base,
                resolvedShots: 256,
                resolvedResilience: .disabled
            ),
            extra: ["qwc:1"]
        )
        XCTAssertNotEqual(hEst, hEstOff)
    }

    func testEstimatorZNEReturnsExtrapolatedValueAndScalePoints() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.z(0) // idle unitary so global depolarizing has a gate to stretch
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let noise = NoiseModel(depolarizingProbability: 0.08)
        let zne = ZNEOptions(scaleFactors: [1, 3, 5], extrapolator: .linear)

        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(noise: noise, seed: 19),
            estimatorOptions: EstimatorOptions(
                shots: 2048,
                resilience: ResilienceOptions(zne: zne)
            )
        )

        let meta = try XCTUnwrap(result.zne)
        XCTAssertEqual(meta.scaleFactors, [1, 3, 5])
        XCTAssertEqual(meta.valuesAtScale.count, 3)
        XCTAssertEqual(meta.extrapolator, .linear)
        XCTAssertEqual(meta.scalingMethod, .globalDepolarizing)
        XCTAssertEqual(result.value, meta.extrapolatedValue, accuracy: 1e-6)

        // Ideal ⟨Z⟩ on |0⟩ is 1; stretched dep should reduce E(λ) as λ grows.
        XCTAssertGreaterThan(meta.valuesAtScale[0], meta.valuesAtScale[2])
        XCTAssertGreaterThan(result.value, meta.valuesAtScale[2] - 0.05)
        XCTAssertNotNil(result.metadata.pipelineHash)
    }

    /// Independent oracle: |0⟩--Z--⟨Z⟩ under global dep has E(λ)=1-(4/3)pλ (unsaturated).
    /// Exact DM scale points + linear ZNE must recover E(0)=1.
    func testZNEExactDepolarizingScalesExtrapolateToIdeal() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.z(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let p: QFloat = 0.05
        let noise = NoiseModel(depolarizingProbability: p)
        let scales: [QFloat] = [1, 2, 3]
        let engine = CPUDensityMatrixEngine()

        var values: [QFloat] = []
        values.reserveCapacity(scales.count)
        for λ in scales {
            let scaled = noise.scalingGlobalDepolarizing(by: λ)
            XCTAssertLessThanOrEqual(scaled.depolarizingProbability, 1, "unsaturated scales required")
            let density = try CPUDensityMatrix(qubitCount: 1)
            _ = try engine.execute(circuit, on: density, noise: scaled)
            let value = try hamiltonian.expectation(density: density)
            let predicted = 1 - (4 * p * λ) / 3
            XCTAssertEqual(value, predicted, accuracy: 1e-5)
            values.append(value)
        }

        let extrapolated = try ZeroNoiseExtrapolation.extrapolateLinear(
            scaleFactors: scales,
            values: values
        )
        XCTAssertEqual(extrapolated, 1, accuracy: 1e-5)
    }

    /// Shot Estimator ZNE on the same affine model should land near the ideal intercept.
    func testEstimatorZNERecoversNearIdealOnSingleQubitDepolarizing() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.z(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let p: QFloat = 0.05
        let noise = NoiseModel(depolarizingProbability: p)
        let zne = ZNEOptions(scaleFactors: [1, 2, 3], extrapolator: .linear)

        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUDensityMatrixBackend(),
            options: QuantumRunOptions(noise: noise, seed: 31),
            estimatorOptions: EstimatorOptions(
                shots: 8192,
                resilience: ResilienceOptions(zne: zne)
            )
        )

        let meta = try XCTUnwrap(result.zne)
        for (λ, value) in zip(meta.scaleFactors, meta.valuesAtScale) {
            let predicted = 1 - (4 * p * λ) / 3
            XCTAssertEqual(value, predicted, accuracy: 0.06)
        }
        XCTAssertEqual(result.value, 1, accuracy: 0.08)
    }

    func testInactiveZNEIgnoredWithReadoutMitigation() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.01, p10: 0.02)
        let inactive = ZNEOptions(scaleFactors: [1])
        XCTAssertFalse(inactive.isActive)

        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 5),
            estimatorOptions: EstimatorOptions(
                shots: 128,
                resilience: ResilienceOptions(readoutMitigation: matrix, zne: inactive)
            )
        )
        XCTAssertNil(result.zne)
        XCTAssertNotNil(result.metadata.pipelineHash)
    }

    func testZNERequiresGlobalDepolarizing() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))

        XCTAssertThrowsError(
            try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: CPUStatevectorBackend(),
                options: QuantumRunOptions(
                    noise: NoiseModel(depolarizingProbability: 0),
                    seed: 2
                ),
                estimatorOptions: EstimatorOptions(
                    shots: 64,
                    resilience: ResilienceOptions(zne: .default)
                )
            )
        ) { error in
            XCTAssertEqual(error as? ZNEError, .missingGlobalDepolarizing)
        }

        XCTAssertThrowsError(
            try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: CPUStatevectorBackend(),
                options: QuantumRunOptions(seed: 2),
                estimatorOptions: EstimatorOptions(
                    shots: 64,
                    resilience: ResilienceOptions(zne: .default)
                )
            )
        ) { error in
            XCTAssertEqual(error as? ZNEError, .missingGlobalDepolarizing)
        }
    }

    func testZNEValidateRejectsNonFiniteScales() {
        XCTAssertThrowsError(
            try ZeroNoiseExtrapolation.validate(
                ZNEOptions(scaleFactors: [1, QFloat.nan, 3])
            )
        ) { error in
            XCTAssertEqual(error as? ZNEError, .nonFiniteScaleFactor)
        }
    }

    func testZNEValidateRejectsNegativeScales() {
        XCTAssertThrowsError(
            try ZeroNoiseExtrapolation.validate(
                ZNEOptions(scaleFactors: [1, -1, 3])
            )
        ) { error in
            XCTAssertEqual(error as? ZNEError, .negativeScaleFactor)
        }
    }

    func testResilienceWithoutZNEKeepsPEC() throws {
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.01, p10: 0.02)
        let options = ResilienceOptions(
            readoutMitigation: matrix,
            zne: .default,
            pec: .default
        )
        let stripped = options.withoutZNE()
        XCTAssertNil(stripped.zne)
        XCTAssertEqual(stripped.pec, .default)
        XCTAssertEqual(stripped.readoutMitigation, matrix)

        let strippedPEC = options.withoutPEC()
        XCTAssertEqual(strippedPEC.zne, .default)
        XCTAssertNil(strippedPEC.pec)
        XCTAssertEqual(strippedPEC.readoutMitigation, matrix)
    }

    func testSamplerFingerprintIgnoresZNEAndPEC() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let base = QuantumRunOptions(seed: 9, shots: 64)
        let withZNE = QuantumRunOptions(
            seed: 9,
            shots: 64,
            resilience: ResilienceOptions(zne: .default)
        )
        let withPEC = QuantumRunOptions(
            seed: 9,
            shots: 64,
            resilience: ResilienceOptions(pec: .default)
        )

        let backend = CPUStatevectorBackend()
        let r0 = try Sampler().run(circuit: circuit, backend: backend, options: base)
        let r1 = try Sampler().run(circuit: circuit, backend: backend, options: withZNE)
        let r2 = try Sampler().run(circuit: circuit, backend: backend, options: withPEC)

        XCTAssertEqual(r0.metadata.pipelineHash, r1.metadata.pipelineHash)
        XCTAssertEqual(r0.metadata.pipelineHash, r2.metadata.pipelineHash)
        XCTAssertEqual(r0.shotCounts, r1.shotCounts)
        XCTAssertEqual(r0.shotCounts, r2.shotCounts)
    }
}
