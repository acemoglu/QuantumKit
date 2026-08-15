import Foundation

/// Computes Hamiltonian expectation gradients via parameter-shift or adjoint differentiation.
///
/// Parameter-shift (default): for each symbolic parameter θ with homogeneous gate-angle
/// scale `s` (`φ = s·θ` in RX/RY/RZ/RXX/RYY/RZZ),
/// `∂⟨H⟩/∂θ = (s/2) (⟨H⟩(θ + π/(2s)) − ⟨H⟩(θ − π/(2s)))` (signed `s`, including `s < 0`).
///
/// Adjoint: reverse-mode sweep with O(1) circuit evolutions in the parameter count
/// (``StatevectorBackend`` only; `RX`/`RY`/`RZ` parameters).
///
/// Under ``SimulationProfilingOptions/detailed``, parameter-shift installs one outer recorder and
/// times a single `gradient` phase. Nested exact ``Estimator`` calls reuse that recorder for
/// user-circuit gate samples but do not emit an `estimate` phase or mid-run ``finishProfile``.
/// Nested **shot** Estimator suppresses per-gate samples (basis-changed measure circuits).
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
            return try SimulationProfiling.usingRecorder(for: options) {
                try runParameterShift(
                    circuit: circuit,
                    hamiltonian: hamiltonian,
                    parameters: parameters,
                    backend: backend,
                    options: options,
                    gradientOptions: gradientOptions
                )
            }
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

        var scales: [String: QFloat] = [:]
        for name in parameterNames {
            scales[name] = try ParameterShift.homogeneousScale(for: name, in: circuit)
        }

        let payload = try SimulationProfiling.timePhase("gradient") { () -> (QFloat, [ParameterGradient], Int, QuantumSimulationMethod, Bool) in
            let baseCircuit = try circuit.bind(parameters: parameters)

            var shiftJobs: [(parameter: String, polarity: ShiftPolarity, scale: QFloat, circuit: QuantumCircuit)] = []
            shiftJobs.reserveCapacity(parameterNames.count * 2)

            for name in parameterNames {
                let scale = scales[name] ?? 1
                let delta = ParameterShift.parameterShiftAmount(scale: scale)
                let plusBindings = shiftedBindings(parameters: parameters, parameter: name, by: delta)
                let minusBindings = shiftedBindings(parameters: parameters, parameter: name, by: -delta)
                shiftJobs.append((name, .plus, scale, try circuit.bind(parameters: plusBindings)))
                shiftJobs.append((name, .minus, scale, try circuit.bind(parameters: minusBindings)))
            }

            let allCircuits = [baseCircuit] + shiftJobs.map(\.circuit)
            let evaluated = try ShiftedExpectationEvaluator.evaluate(
                circuits: allCircuits,
                hamiltonian: hamiltonian,
                backend: backend,
                options: options,
                batchSize: gradientOptions.batchSize
            )
            let baseValue = evaluated.values[0]
            let shiftedExpectations = Array(evaluated.values.dropFirst())

            var gradientsByName: [String: QFloat] = [:]
            gradientsByName.reserveCapacity(parameterNames.count)

            for index in shiftJobs.indices {
                let job = shiftJobs[index]
                let expectation = shiftedExpectations[index]
                switch job.polarity {
                case .plus:
                    let minusExpectation = shiftedExpectations[index + 1]
                    gradientsByName[job.parameter] = ParameterShift.gradient(
                        plusExpectation: expectation,
                        minusExpectation: minusExpectation,
                        scale: job.scale
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

            return (
                baseValue,
                parameterGradients,
                allCircuits.count,
                evaluated.method,
                evaluated.isCPU
            )
        }

        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = QuantumResultMetadata(
            method: payload.3,
            seed: options.seed,
            deviceName: payload.4 ? "CPU" : MetalRuntime.deviceName,
            wallClockNanoseconds: elapsed,
            qubitCount: circuit.qubitCount,
            gateCount: circuit.gates.count,
            noiseSnapshot: options.noise,
            profile: SimulationProfiling.finishProfile(
                options: options,
                circuit: circuit,
                method: payload.3,
                isCPU: payload.4,
                elapsed: elapsed
            )
        )

        return GradientResult(
            expectationValue: payload.0,
            parameterGradients: payload.1,
            circuitEvaluations: payload.2,
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
