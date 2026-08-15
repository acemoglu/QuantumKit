import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Thin VQE / QAOA workflows

    func testVQEEvaluatesRXAnsatzOnZHamiltonian() throws {
        // RX(θ)|0⟩: ⟨Z⟩ = cos(θ). One energy evaluation at θ = π/3.
        let theta = QFloat(Double.pi / 3)
        var ansatz = try QuantumCircuit(qubitCount: 1)
        try ansatz.rx(theta: Parameter("theta"), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))

        let result = try VQE().evaluate(
            ansatz: ansatz,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: CPUStatevectorBackend()
        )

        XCTAssertEqual(result.energy, cos(theta), accuracy: 1e-10)
        XCTAssertEqual(result.estimator.value, result.energy, accuracy: 1e-12)
        XCTAssertNil(result.gradient)
    }

    func testVQEGradientOnScaledRXMatchesAnalytic() throws {
        // RX(2θ)|0⟩: ⟨Z⟩ = cos(2θ), ∂/∂θ = −2 sin(2θ)
        let theta = QFloat(Double.pi / 8)
        var ansatz = try QuantumCircuit(qubitCount: 1)
        try ansatz.rx(theta: Parameter("theta").scaled(by: 2), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))

        let result = try VQE().evaluate(
            ansatz: ansatz,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: CPUStatevectorBackend(),
            computeGradient: true
        )

        XCTAssertEqual(result.energy, cos(2 * theta), accuracy: 1e-8)
        XCTAssertEqual(
            result.gradient?.gradient(for: "theta") ?? 0,
            -2 * sin(2 * theta),
            accuracy: 1e-6
        )
    }

    func testQAOABuildsP1WithTwoParameters() throws {
        let graph = try IsingGraph.maxCut(qubitCount: 2, edges: [(0, 1)])
        let qaoa = try QAOA.build(problem: graph, layers: 1)

        XCTAssertEqual(qaoa.layers, 1)
        XCTAssertEqual(qaoa.parameterCount, 2)
        XCTAssertEqual(qaoa.gammaNames, ["gamma0"])
        XCTAssertEqual(qaoa.betaNames, ["beta0"])
        XCTAssertEqual(qaoa.circuit.referencedParameters, ["beta0", "gamma0"])
        XCTAssertEqual(qaoa.circuit.qubitCount, 2)
        XCTAssertEqual(qaoa.circuit.gates.count, 2 + 1 + 2)

        let cost = try graph.costHamiltonian()
        XCTAssertEqual(cost.terms.count, 1)
        XCTAssertEqual(cost.terms[0].paulis, [0: .z, 1: .z])

        // p=1 MaxCut on one edge, γ=β=0: H⊗H → |++⟩, ⟨ZZ⟩ = 0.
        let atZero = try VQE().evaluate(
            qaoa: qaoa,
            cost: cost,
            gammas: [0],
            betas: [0],
            backend: CPUStatevectorBackend()
        )
        XCTAssertEqual(atZero.energy, 0, accuracy: 1e-10)

        // Special point for this gate convention (RZZ(2γ), RX(2β)): γ=π/4, β=π/8 ⇒ ⟨ZZ⟩=1.
        let gamma = QFloat(Double.pi / 4)
        let beta = QFloat(Double.pi / 8)
        let energy = try VQE().evaluate(
            qaoa: qaoa,
            cost: cost,
            gammas: [gamma],
            betas: [beta],
            backend: CPUStatevectorBackend()
        )
        let bound = try qaoa.circuit.bind(
            parameters: try QAOA.bindings(for: qaoa, gammas: [gamma], betas: [beta])
        )
        let independent = try Estimator().run(
            circuit: bound,
            hamiltonian: cost,
            backend: CPUStatevectorBackend()
        )
        XCTAssertEqual(energy.energy, independent.value, accuracy: 1e-10)
        XCTAssertEqual(energy.energy, 1, accuracy: 1e-8)
    }

    func testQAOAEqualWeightGradientIsFiniteAndHomogeneous() throws {
        let graph = try IsingGraph.maxCut(qubitCount: 2, edges: [(0, 1)])
        let qaoa = try QAOA.build(problem: graph, layers: 1)
        let cost = try graph.costHamiltonian()

        XCTAssertEqual(try ParameterShift.homogeneousScale(for: "beta0", in: qaoa.circuit), 2, accuracy: 1e-12)
        XCTAssertEqual(try ParameterShift.homogeneousScale(for: "gamma0", in: qaoa.circuit), 2, accuracy: 1e-12)

        let result = try VQE().evaluate(
            qaoa: qaoa,
            cost: cost,
            gammas: [0.2],
            betas: [0.3],
            backend: CPUStatevectorBackend(),
            computeGradient: true
        )
        XCTAssertNotNil(result.gradient)
        XCTAssertEqual(result.gradient?.parameterGradients.count, 2)
        XCTAssertTrue(result.gradient?.gradient(for: "beta0")?.isFinite ?? false)
        XCTAssertTrue(result.gradient?.gradient(for: "gamma0")?.isFinite ?? false)
    }

    func testQAOANegativeFieldWeightGradientMatchesFiniteDifference() throws {
        // Fields-only Ising with h=-1 ⇒ RZ(2hγ) scale s_γ=-2 (homogeneous). Unlike a lone
        // ZZ edge on |++⟩ (γ-independent global phase), local Z fields make ∂E/∂γ nonzero,
        // so abs-based δ would flip the gamma gradient sign.
        let graph = try IsingGraph(
            qubitCount: 2,
            edges: [],
            fields: [0: -1, 1: -1]
        )
        let qaoa = try QAOA.build(problem: graph, layers: 1)
        let cost = try graph.costHamiltonian()
        let backend = CPUStatevectorBackend()

        XCTAssertEqual(
            try ParameterShift.homogeneousScale(for: "gamma0", in: qaoa.circuit),
            -2,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            try ParameterShift.homogeneousScale(for: "beta0", in: qaoa.circuit),
            2,
            accuracy: 1e-12
        )

        let gamma: QFloat = 0.35
        let beta: QFloat = 0.4
        let result = try VQE().evaluate(
            qaoa: qaoa,
            cost: cost,
            gammas: [gamma],
            betas: [beta],
            backend: backend,
            computeGradient: true
        )

        let eps: QFloat = 1e-3
        func energy(gamma g: QFloat, beta b: QFloat) throws -> QFloat {
            try VQE().evaluate(
                qaoa: qaoa,
                cost: cost,
                gammas: [g],
                betas: [b],
                backend: backend
            ).energy
        }
        let dGammaFD = try (energy(gamma: gamma + eps, beta: beta) - energy(gamma: gamma - eps, beta: beta))
            / (2 * eps)
        let dBetaFD = try (energy(gamma: gamma, beta: beta + eps) - energy(gamma: gamma, beta: beta - eps))
            / (2 * eps)

        let dGamma = try XCTUnwrap(result.gradient?.gradient(for: "gamma0"))
        let dBeta = try XCTUnwrap(result.gradient?.gradient(for: "beta0"))
        XCTAssertEqual(dGamma, dGammaFD, accuracy: 5e-3)
        XCTAssertEqual(dBeta, dBetaFD, accuracy: 5e-3)
        // Nontrivial gamma slope — otherwise the regression would be vacuous.
        XCTAssertGreaterThan(abs(dGamma), 0.05)
        // Abs-based polarity would produce ≈ -dGammaFD here.
        XCTAssertEqual(signOf(dGamma), signOf(dGammaFD))
    }

    private func signOf(_ value: QFloat) -> Int {
        if value > 0 { return 1 }
        if value < 0 { return -1 }
        return 0
    }

    func testQAOAHeterogeneousWeightsThrowOnParameterShift() throws {
        let graph = try IsingGraph(
            qubitCount: 2,
            edges: [.init(qubitA: 0, qubitB: 1, weight: 1)],
            fields: [0: 0.5]
        )
        let qaoa = try QAOA.build(problem: graph, layers: 1)
        // gamma0: RZ(γ) scale 1 and RZZ(2γ) scale 2.
        XCTAssertThrowsError(
            try ParameterShift.homogeneousScale(for: "gamma0", in: qaoa.circuit)
        ) { error in
            guard case GradientCalculatorError.heterogeneousParameterScale = error else {
                return XCTFail("Expected heterogeneousParameterScale, got \(error)")
            }
        }
    }

    func testQAOAP2ParameterCountIsFour() throws {
        let graph = try IsingGraph(
            qubitCount: 3,
            edges: [
                .init(qubitA: 0, qubitB: 1, weight: 1),
                .init(qubitA: 1, qubitB: 2, weight: 0.5),
            ],
            fields: [0: 0.1]
        )
        let qaoa = try QAOA.build(problem: graph, layers: 2)
        XCTAssertEqual(qaoa.parameterCount, 4)
        XCTAssertEqual(qaoa.gammaNames, ["gamma0", "gamma1"])
        XCTAssertEqual(qaoa.betaNames, ["beta0", "beta1"])
    }
}
