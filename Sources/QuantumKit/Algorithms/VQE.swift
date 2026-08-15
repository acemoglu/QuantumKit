import Foundation

/// Result of a single VQE energy (and optional gradient) evaluation.
public struct VQEResult: Sendable, Equatable {
    /// ⟨H⟩ at the bound parameters.
    public let energy: QFloat
    public let parameters: [String: QFloat]
    public let estimator: EstimatorResult
    /// Present when `computeGradient` was requested.
    public let gradient: GradientResult?

    public init(
        energy: QFloat,
        parameters: [String: QFloat],
        estimator: EstimatorResult,
        gradient: GradientResult? = nil
    ) {
        self.energy = energy
        self.parameters = parameters
        self.estimator = estimator
        self.gradient = gradient
    }
}

/// Thin VQE helper: bind ansatz parameters → ``Estimator`` energy, optionally ``GradientCalculator``.
///
/// Not an optimizer — one evaluation / one gradient step of scaffolding only.
public struct VQE: Sendable {
    private let estimator: Estimator
    private let gradientCalculator: GradientCalculator

    public init(
        estimator: Estimator = Estimator(),
        gradientCalculator: GradientCalculator = GradientCalculator()
    ) {
        self.estimator = estimator
        self.gradientCalculator = gradientCalculator
    }

    /// Evaluate ⟨ψ(θ)|H|ψ(θ)⟩ for a parametric ansatz.
    ///
    /// - Parameters:
    ///   - computeGradient: when `true`, also runs ``GradientCalculator`` parameter-shift
    ///     (CPU + Metal SV/DM) or adjoint (Metal SV). Gate angles of the form `s·θ` are
    ///     supported when `s` is homogeneous; QAOA with equal edge weights qualifies.
    ///     Heterogeneous scales (e.g. distinct `Jᵢⱼ` on the same `γ`) throw
    ///     ``GradientCalculatorError/heterogeneousParameterScale``.
    public func evaluate(
        ansatz: QuantumCircuit,
        hamiltonian: Hamiltonian,
        parameters: [String: QFloat],
        backend: any QuantumBackend,
        options: QuantumRunOptions = QuantumRunOptions(),
        estimatorOptions: EstimatorOptions = .exact,
        computeGradient: Bool = false,
        gradientOptions: GradientOptions = GradientOptions()
    ) throws -> VQEResult {
        for name in ansatz.referencedParameters {
            guard parameters[name] != nil else {
                throw VariationalAlgorithmError.missingParameterBinding(name)
            }
        }

        let bound = try ansatz.bind(parameters: parameters)
        let estimate = try estimator.run(
            circuit: bound,
            hamiltonian: hamiltonian,
            backend: backend,
            options: options,
            estimatorOptions: estimatorOptions
        )

        var gradient: GradientResult?
        if computeGradient {
            gradient = try gradientCalculator.run(
                circuit: ansatz,
                hamiltonian: hamiltonian,
                parameters: parameters,
                backend: backend,
                options: options,
                gradientOptions: gradientOptions
            )
        }

        return VQEResult(
            energy: estimate.value,
            parameters: parameters,
            estimator: estimate,
            gradient: gradient
        )
    }

    /// Convenience: evaluate energy for a QAOA ansatz with `gammas` / `betas`.
    public func evaluate(
        qaoa: QAOACircuit,
        cost: Hamiltonian,
        gammas: [QFloat],
        betas: [QFloat],
        backend: any QuantumBackend,
        options: QuantumRunOptions = QuantumRunOptions(),
        estimatorOptions: EstimatorOptions = .exact,
        computeGradient: Bool = false,
        gradientOptions: GradientOptions = GradientOptions()
    ) throws -> VQEResult {
        let parameters = try QAOA.bindings(for: qaoa, gammas: gammas, betas: betas)
        return try evaluate(
            ansatz: qaoa.circuit,
            hamiltonian: cost,
            parameters: parameters,
            backend: backend,
            options: options,
            estimatorOptions: estimatorOptions,
            computeGradient: computeGradient,
            gradientOptions: gradientOptions
        )
    }
}
