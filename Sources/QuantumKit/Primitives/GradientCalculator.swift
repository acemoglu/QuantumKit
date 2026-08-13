import Foundation

/// Computes Hamiltonian expectation gradients via parameter-shift or adjoint differentiation.
///
/// Parameter-shift (default): for each symbolic parameter θ,
/// `∂⟨H⟩/∂θ = ½ (⟨H⟩(θ + π/2) − ⟨H⟩(θ − π/2))`.
///
/// Adjoint: reverse-mode sweep with O(1) circuit evolutions in the parameter count
/// (``StatevectorBackend`` only; `RX`/`RY`/`RZ` parameters).
public struct GradientCalculator: Sendable {
    private let estimator: Estimator

    public init(estimator: Estimator = Estimator()) {
        self.estimator = estimator
    }

    public func run(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        parameters: [String: QFloat],
        backend: any QuantumBackend,
        options: QuantumRunOptions = QuantumRunOptions(),
        gradientOptions: GradientOptions = GradientOptions()
    ) throws -> GradientResult {
        switch gradientOptions.method {
        case .adjoint:
            guard let statevectorBackend = backend as? StatevectorBackend else {
                throw GradientCalculatorError.adjointRequiresStatevectorBackend
            }
            return try AdjointDifferentiator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                parameters: parameters,
                backend: statevectorBackend,
                options: options
            )
        case .parameterShift:
            return try runParameterShift(
                circuit: circuit,
                hamiltonian: hamiltonian,
                parameters: parameters,
                backend: backend,
                options: options,
                gradientOptions: gradientOptions
            )
        }
    }

    private func runParameterShift(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        parameters: [String: QFloat],
        backend: any QuantumBackend,
        options: QuantumRunOptions,
        gradientOptions: GradientOptions
    ) throws -> GradientResult {
        let started = DispatchTime.now()

        let parameterNames = circuit.referencedParameters.sorted()
        guard !parameterNames.isEmpty else {
            throw GradientCalculatorError.noDifferentiableParameters
        }

        for name in parameterNames {
            guard parameters[name] != nil else {
                throw GradientCalculatorError.missingParameterBinding(name)
            }
        }

        let baseCircuit = try circuit.bind(parameters: parameters)
        let baseResult = try estimator.run(
            circuit: baseCircuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: options
        )

        var shiftJobs: [(parameter: String, polarity: ShiftPolarity, circuit: QuantumCircuit)] = []
        shiftJobs.reserveCapacity(parameterNames.count * 2)

        for name in parameterNames {
            let plusBindings = shiftedBindings(parameters: parameters, parameter: name, by: ParameterShift.shift)
            let minusBindings = shiftedBindings(parameters: parameters, parameter: name, by: -ParameterShift.shift)
            shiftJobs.append((name, .plus, try circuit.bind(parameters: plusBindings)))
            shiftJobs.append((name, .minus, try circuit.bind(parameters: minusBindings)))
        }

        let shiftedExpectations: [QFloat]
        let method: QuantumSimulationMethod

        if let statevectorBackend = backend as? StatevectorBackend {
            method = .statevector
            shiftedExpectations = try BatchExpectationExecutor.evaluate(
                circuits: shiftJobs.map(\.circuit),
                hamiltonian: hamiltonian,
                engine: statevectorBackend.engine,
                options: options,
                batchSize: gradientOptions.batchSize
            )
        } else if let densityBackend = backend as? DensityMatrixBackend {
            method = .densityMatrix
            shiftedExpectations = try BatchExpectationExecutor.evaluate(
                circuits: shiftJobs.map(\.circuit),
                hamiltonian: hamiltonian,
                engine: densityBackend.engine,
                options: options
            )
        } else {
            throw GradientCalculatorError.unsupportedBackend
        }

        var gradientsByName: [String: QFloat] = [:]
        gradientsByName.reserveCapacity(parameterNames.count)

        for index in shiftJobs.indices {
            let job = shiftJobs[index]
            let expectation = shiftedExpectations[index]
            switch job.polarity {
            case .plus:
                let minusIndex = index + 1
                let minusExpectation = shiftedExpectations[minusIndex]
                gradientsByName[job.parameter] = ParameterShift.gradient(
                    plusExpectation: expectation,
                    minusExpectation: minusExpectation
                )
            case .minus:
                continue
            }
        }

        let parameterGradients = parameterNames.map { name in
            ParameterGradient(
                parameter: QuantumParameter(name),
                gradient: gradientsByName[name] ?? 0
            )
        }

        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = QuantumResultMetadata(
            method: method,
            seed: options.seed,
            deviceName: MetalRuntime.deviceName,
            wallClockNanoseconds: elapsed,
            qubitCount: circuit.qubitCount,
            gateCount: circuit.gates.count,
            noiseSnapshot: options.noise
        )

        return GradientResult(
            expectationValue: baseResult.value,
            parameterGradients: parameterGradients,
            circuitEvaluations: 1 + shiftJobs.count,
            metadata: metadata
        )
    }

    private enum ShiftPolarity {
        case plus
        case minus
    }

    private func shiftedBindings(
        parameters: [String: QFloat],
        parameter name: String,
        by offset: QFloat
    ) -> [String: QFloat] {
        var bindings = parameters
        bindings[name] = (parameters[name] ?? 0) + offset
        return bindings
    }
}
