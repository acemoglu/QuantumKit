import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - C13 PEC lite

    func testPECDepolarizingQPRGammaAndInverseEigenvalues() throws {
        let p: QFloat = 0.12
        let qpr = try ProbabilisticErrorCancellation.depolarizingQuasiprobability(probability: p)
        let lambda = 1 - (4 * p) / 3
        let expectedGamma = (1 + (2 * p) / 3) / (1 - (4 * p) / 3)
        XCTAssertEqual(qpr.gamma, expectedGamma, accuracy: 1e-5)

        // η recovers 1/λ on the Pauli spectrum: Σ_σ η_σ χ(σ,P) = 1/λ_P.
        let invI = qpr.etaI + qpr.etaX + qpr.etaY + qpr.etaZ
        let invX = qpr.etaI + qpr.etaX - qpr.etaY - qpr.etaZ
        let invY = qpr.etaI - qpr.etaX + qpr.etaY - qpr.etaZ
        let invZ = qpr.etaI - qpr.etaX - qpr.etaY + qpr.etaZ
        XCTAssertEqual(invI, 1, accuracy: 1e-5)
        XCTAssertEqual(invX, 1 / lambda, accuracy: 1e-5)
        XCTAssertEqual(invY, 1 / lambda, accuracy: 1e-5)
        XCTAssertEqual(invZ, 1 / lambda, accuracy: 1e-5)
    }

    func testPECPauliChannelQPRMatchesDepolarizingWhenEqualRates() throws {
        let p: QFloat = 0.09
        let dep = try ProbabilisticErrorCancellation.depolarizingQuasiprobability(probability: p)
        let pauli = try ProbabilisticErrorCancellation.pauliChannelQuasiprobability(
            px: p / 3,
            py: p / 3,
            pz: p / 3
        )
        XCTAssertEqual(dep.etaI, pauli.etaI, accuracy: 1e-5)
        XCTAssertEqual(dep.etaX, pauli.etaX, accuracy: 1e-5)
        XCTAssertEqual(dep.gamma, pauli.gamma, accuracy: 1e-5)
    }

    func testPECRejectsNonInvertibleAndUnsupportedNoise() {
        XCTAssertThrowsError(
            try ProbabilisticErrorCancellation.depolarizingQuasiprobability(probability: 0.8)
        ) { error in
            guard case PECError.nonInvertibleChannel = error else {
                return XCTFail("expected nonInvertibleChannel, got \(error)")
            }
        }

        let amp = NoiseModel(depolarizingProbability: 0.05, amplitudeDampingProbability: 0.1)
        XCTAssertThrowsError(
            try ProbabilisticErrorCancellation.validateNoiseModelForPEC(amp)
        ) { error in
            guard case PECError.unsupportedNoiseModel = error else {
                return XCTFail("expected unsupportedNoiseModel, got \(error)")
            }
        }

        let withReadout = NoiseModel(
            depolarizingProbability: 0.05,
            readoutErrorProbability: 0.02
        )
        XCTAssertThrowsError(
            try ProbabilisticErrorCancellation.validateNoiseModelForPEC(withReadout)
        ) { error in
            guard case PECError.unsupportedNoiseModel(let message) = error else {
                return XCTFail("expected unsupportedNoiseModel, got \(error)")
            }
            XCTAssertTrue(message.contains("readout"))
        }
    }

    func testPECRejectsMultiQubitGatesAndZNEStack() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        let noise = NoiseModel(depolarizingProbability: 0.05)
        XCTAssertThrowsError(
            try ProbabilisticErrorCancellation.validatedPECSites(circuit: circuit, noise: noise)
        ) { error in
            guard case PECError.unsupportedMultiQubitGate = error else {
                return XCTFail("expected unsupportedMultiQubitGate, got \(error)")
            }
        }

        var single = try QuantumCircuit(qubitCount: 1)
        try single.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        XCTAssertThrowsError(
            try Estimator().run(
                circuit: single,
                hamiltonian: hamiltonian,
                backend: CPUStatevectorBackend(),
                options: QuantumRunOptions(noise: noise, seed: 1),
                estimatorOptions: EstimatorOptions(
                    shots: 32,
                    resilience: ResilienceOptions(
                        zne: .default,
                        pec: .default
                    )
                )
            )
        ) { error in
            XCTAssertEqual(error as? PECError, .incompatibleWithZNE)
        }
    }

    func testPECDisabledMatchesBaselineEstimator() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let noise = NoiseModel(depolarizingProbability: 0.05)
        let backend = CPUStatevectorBackend()

        let baseline = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(noise: noise, seed: 21),
            estimatorOptions: EstimatorOptions(shots: 256)
        )
        let explicitOff = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(noise: noise, seed: 21, resilience: .disabled),
            estimatorOptions: EstimatorOptions(shots: 256, resilience: .disabled)
        )
        XCTAssertEqual(baseline.value, explicitOff.value)
        XCTAssertNil(baseline.pec)
        XCTAssertEqual(baseline.metadata.pipelineHash, explicitOff.metadata.pipelineHash)
    }

    func testPipelineFingerprintIncludesPEC() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let base = QuantumRunOptions(seed: 3, shots: 64)
        let withPEC = QuantumRunOptions(
            seed: 3,
            shots: 64,
            resilience: ResilienceOptions(pec: .default)
        )
        let h0 = PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: base)
        let h1 = PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: withPEC)
        XCTAssertNotEqual(h0, h1)
    }

    func testEstimatorPECCloserToIdealThanRaw() throws {
        // |0⟩ --X-- ⟨Z⟩; ideal = -1. Global dep on the X site biases toward 0.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let noise = NoiseModel(depolarizingProbability: 0.1)
        let backend = CPUStatevectorBackend()
        let ideal: QFloat = -1

        let raw = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(noise: noise, seed: 42),
            estimatorOptions: EstimatorOptions(shots: 4096)
        )
        let mitigated = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(noise: noise, seed: 42),
            estimatorOptions: EstimatorOptions(
                shots: 4096,
                resilience: ResilienceOptions(
                    pec: PECOptions(channel: .globalDepolarizing, circuitSamples: 512)
                )
            )
        )

        let pec = try XCTUnwrap(mitigated.pec)
        XCTAssertEqual(pec.siteCount, 1)
        XCTAssertGreaterThan(pec.gammaPerSite, 1)
        XCTAssertEqual(pec.shotMultiplier, pec.gammaTotal * pec.gammaTotal, accuracy: 1e-5)
        XCTAssertNotEqual(raw.metadata.pipelineHash, mitigated.metadata.pipelineHash)

        let rawErr = abs(raw.value - ideal)
        let mitErr = abs(mitigated.value - ideal)
        XCTAssertLessThan(mitErr, rawErr)
    }

    func testInactiveZNEDoesNotBlockPEC() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let noise = NoiseModel(depolarizingProbability: 0.08)
        let inactive = ZNEOptions(scaleFactors: [1])

        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(noise: noise, seed: 8),
            estimatorOptions: EstimatorOptions(
                shots: 256,
                resilience: ResilienceOptions(
                    zne: inactive,
                    pec: PECOptions(channel: .globalDepolarizing, circuitSamples: 64)
                )
            )
        )
        XCTAssertNil(result.zne)
        XCTAssertNotNil(result.pec)
    }

    func testEstimatorPECInheritsRunOptionsReadoutMitigation() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let noise = NoiseModel(depolarizingProbability: 0.05)
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.0, p10: 0.0)

        let withoutInherit = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(noise: noise, seed: 4),
            estimatorOptions: EstimatorOptions(
                shots: 64,
                resilience: ResilienceOptions(
                    pec: PECOptions(channel: .globalDepolarizing, circuitSamples: 16)
                )
            )
        )
        let withInherit = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(
                noise: noise,
                seed: 4,
                resilience: ResilienceOptions(readoutMitigation: matrix)
            ),
            estimatorOptions: EstimatorOptions(
                shots: 64,
                resilience: ResilienceOptions(
                    pec: PECOptions(channel: .globalDepolarizing, circuitSamples: 16)
                )
            )
        )

        // Identity confusion matrix: values match, but fingerprint must include readout.
        XCTAssertEqual(withoutInherit.value, withInherit.value, accuracy: 1e-6)
        XCTAssertNotEqual(
            withoutInherit.metadata.pipelineHash,
            withInherit.metadata.pipelineHash
        )
        XCTAssertNotNil(withInherit.pec)
    }

    func testPECRejectsCircuitSamplesExceedingShots() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let noise = NoiseModel(depolarizingProbability: 0.05)

        XCTAssertThrowsError(
            try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: CPUStatevectorBackend(),
                options: QuantumRunOptions(noise: noise, seed: 3),
                estimatorOptions: EstimatorOptions(
                    shots: 32,
                    resilience: ResilienceOptions(
                        pec: PECOptions(circuitSamples: 100)
                    )
                )
            )
        ) { error in
            guard case PECError.circuitSamplesExceedShots(let samples, let shots) = error else {
                return XCTFail("expected circuitSamplesExceedShots, got \(error)")
            }
            XCTAssertEqual(samples, 100)
            XCTAssertEqual(shots, 32)
        }
    }
}
