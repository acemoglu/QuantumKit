import Foundation

/// Jacobian and Hessian of an observable expectation ⟨H⟩(θ) for variational circuits.
///
/// ## Methods
/// - **Jacobian** (`∂⟨H⟩/∂θᵢ`):
///   - ``DifferentiationMethod/parameterShift`` (default): 2 shifts per parameter
///     (`±π/2`) plus one base evaluation → **1 + 2P** circuit evolutions.
///   - ``DifferentiationMethod/adjoint``: O(1) evolutions in P — Metal
///     ``StatevectorBackend`` only (same constraints as ``AdjointDifferentiator``).
/// - **Hessian** (`∂²⟨H⟩/∂θᵢ∂θⱼ`): **parameter-shift only** (no adjoint Hessian).
///   - Diagonal: ±π shifts → `¼(E₊π − 2E₀ + E₋π)` (**2P** plus shared base).
///   - Off-diagonal `i < j`: four ±π/2 corner shifts → `¼(E₊₊ − E₊₋ − E₋₊ + E₋₋)`
///     (**4 per unordered pair**).
///   - Total cost: **1 + 2P + 2P(P−1) = 1 + 2P²** expectation evaluations.
///
/// ## Supported backends
/// - Parameter-shift Jacobian / Hessian: ``StatevectorBackend`` (Metal),
///   ``CPUStatevectorBackend``, and ``DensityMatrixBackend`` (Metal) for Jacobian/Hessian
///   shift evaluations.
/// - Adjoint Jacobian: ``StatevectorBackend`` only.
///
/// Does not modify ``GradientResult`` / ``GradientCalculator``; Jacobian may wrap them
/// for Metal adjoint / shift paths.
public struct ObservableDifferentiator: Sendable {
    private let gradientCalculator: GradientCalculator

    public init(gradientCalculator: GradientCalculator = GradientCalculator()) {
        self.gradientCalculator = gradientCalculator
    }

    /// First-order Jacobian `∂⟨H⟩/∂θᵢ` (name-sorted).
    public func jacobian(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        parameters: [String: QFloat],
        backend: any QuantumBackend,
        options: QuantumRunOptions = QuantumRunOptions(),
        differentiatorOptions: ObservableDifferentiatorOptions = .default
    ) throws -> JacobianResult {
        switch differentiatorOptions.jacobianMethod {
        case .adjoint:
            let gradient = try gradientCalculator.run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                parameters: parameters,
                backend: backend,
                options: options,
                gradientOptions: GradientOptions(
                    batchSize: differentiatorOptions.batchSize,
                    method: .adjoint
                )
            )
            return JacobianResult(from: gradient, method: .adjoint)

        case .parameterShift:
            if backend is StatevectorBackend || backend is DensityMatrixBackend {
                let gradient = try gradientCalculator.run(
                    circuit: circuit,
                    hamiltonian: hamiltonian,
                    parameters: parameters,
                    backend: backend,
                    options: options,
                    gradientOptions: GradientOptions(
                        batchSize: differentiatorOptions.batchSize,
                        method: .parameterShift
                    )
                )
                return JacobianResult(from: gradient, method: .parameterShift)
            }
            return try SimulationProfiling.usingRecorder(for: options) {
                try parameterShiftJacobian(
                    circuit: circuit,
                    hamiltonian: hamiltonian,
                    parameters: parameters,
                    backend: backend,
                    options: options,
                    batchSize: differentiatorOptions.batchSize
                )
            }
        }
    }

    /// Second-order Hessian via parameter-shift (see type-level cost docs).
    public func hessian(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        parameters: [String: QFloat],
        backend: any QuantumBackend,
        options: QuantumRunOptions = QuantumRunOptions(),
        differentiatorOptions: ObservableDifferentiatorOptions = .default
    ) throws -> HessianResult {
        try SimulationProfiling.usingRecorder(for: options) {
            try hessianInstrumented(
                circuit: circuit,
                hamiltonian: hamiltonian,
                parameters: parameters,
                backend: backend,
                options: options,
                differentiatorOptions: differentiatorOptions
            )
        }
    }

    private func hessianInstrumented(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        parameters: [String: QFloat],
        backend: any QuantumBackend,
        options: QuantumRunOptions,
        differentiatorOptions: ObservableDifferentiatorOptions
    ) throws -> HessianResult {
        let started = DispatchTime.now()
        let parameterNames = try validatedParameterNames(circuit: circuit, parameters: parameters)
        let pCount = parameterNames.count

        var scales: [QFloat] = []
        scales.reserveCapacity(pCount)
        for name in parameterNames {
            scales.append(try ParameterShift.homogeneousScale(for: name, in: circuit))
        }

        var jobs: [QuantumCircuit] = []
        jobs.reserveCapacity(1 + 2 * pCount * pCount)

        // Index 0: base
        jobs.append(try circuit.bind(parameters: parameters))

        // Diagonal ±π on physical angle → ±π/s on parameter (signed scale)
        for (name, scale) in zip(parameterNames, scales) {
            let delta = ParameterShift.parameterPiShiftAmount(scale: scale)
            jobs.append(try circuit.bind(parameters: shifted(parameters, name, by: delta)))
            jobs.append(try circuit.bind(parameters: shifted(parameters, name, by: -delta)))
        }

        // Off-diagonal corners for i < j (±π/2 on each physical angle)
        var pairSpecs: [(i: Int, j: Int)] = []
        if pCount >= 2 {
            for i in 0..<pCount {
                for j in (i + 1)..<pCount {
                    pairSpecs.append((i, j))
                    let ni = parameterNames[i]
                    let nj = parameterNames[j]
                    let di = ParameterShift.parameterShiftAmount(scale: scales[i])
                    let dj = ParameterShift.parameterShiftAmount(scale: scales[j])
                    jobs.append(try circuit.bind(parameters: shifted(parameters, [(ni, di), (nj, dj)])))
                    jobs.append(try circuit.bind(parameters: shifted(parameters, [(ni, di), (nj, -dj)])))
                    jobs.append(try circuit.bind(parameters: shifted(parameters, [(ni, -di), (nj, dj)])))
                    jobs.append(try circuit.bind(parameters: shifted(parameters, [(ni, -di), (nj, -dj)])))
                }
            }
        }

        let evaluated = try ShiftedExpectationEvaluator.evaluate(
            circuits: jobs,
            hamiltonian: hamiltonian,
            backend: backend,
            options: options,
            batchSize: differentiatorOptions.batchSize
        )
        let expectations = evaluated.values
        let center = expectations[0]

        var matrix = Array(
            repeating: Array(repeating: QFloat(0), count: pCount),
            count: pCount
        )

        for i in 0..<pCount {
            let plusPi = expectations[1 + 2 * i]
            let minusPi = expectations[1 + 2 * i + 1]
            matrix[i][i] = ParameterShift.secondDerivative(
                plusPiExpectation: plusPi,
                centerExpectation: center,
                minusPiExpectation: minusPi,
                scale: scales[i]
            )
        }

        var cursor = 1 + 2 * pCount
        for (i, j) in pairSpecs {
            let pp = expectations[cursor]
            let pm = expectations[cursor + 1]
            let mp = expectations[cursor + 2]
            let mm = expectations[cursor + 3]
            cursor += 4
            let value = ParameterShift.mixedPartial(
                plusPlus: pp,
                plusMinus: pm,
                minusPlus: mp,
                minusMinus: mm,
                scaleI: scales[i],
                scaleJ: scales[j]
            )
            matrix[i][j] = value
            matrix[j][i] = value
        }

        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = QuantumResultMetadata(
            method: evaluated.method,
            seed: options.seed,
            deviceName: evaluated.isCPU ? "CPU" : MetalRuntime.deviceName,
            wallClockNanoseconds: elapsed,
            qubitCount: circuit.qubitCount,
            gateCount: circuit.gates.count,
            noiseSnapshot: options.noise,
            profile: SimulationProfiling.finishProfile(
                options: options,
                circuit: circuit,
                method: evaluated.method,
                isCPU: evaluated.isCPU,
                elapsed: elapsed
            )
        )

        return HessianResult(
            expectationValue: center,
            parameterNames: parameterNames,
            matrix: matrix,
            circuitEvaluations: jobs.count,
            metadata: metadata
        )
    }

    // MARK: - Parameter-shift Jacobian (CPU / non-Metal path)

    private func parameterShiftJacobian(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        parameters: [String: QFloat],
        backend: any QuantumBackend,
        options: QuantumRunOptions,
        batchSize: Int
    ) throws -> JacobianResult {
        let started = DispatchTime.now()
        let parameterNames = try validatedParameterNames(circuit: circuit, parameters: parameters)

        var scales: [QFloat] = []
        for name in parameterNames {
            scales.append(try ParameterShift.homogeneousScale(for: name, in: circuit))
        }

        var jobs: [QuantumCircuit] = []
        jobs.append(try circuit.bind(parameters: parameters))
        for (name, scale) in zip(parameterNames, scales) {
            let delta = ParameterShift.parameterShiftAmount(scale: scale)
            jobs.append(try circuit.bind(parameters: shifted(parameters, name, by: delta)))
            jobs.append(try circuit.bind(parameters: shifted(parameters, name, by: -delta)))
        }

        let evaluated = try ShiftedExpectationEvaluator.evaluate(
            circuits: jobs,
            hamiltonian: hamiltonian,
            backend: backend,
            options: options,
            batchSize: batchSize
        )

        var values: [QFloat] = []
        values.reserveCapacity(parameterNames.count)
        for i in parameterNames.indices {
            let plus = evaluated.values[1 + 2 * i]
            let minus = evaluated.values[1 + 2 * i + 1]
            values.append(
                ParameterShift.gradient(
                    plusExpectation: plus,
                    minusExpectation: minus,
                    scale: scales[i]
                )
            )
        }

        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = QuantumResultMetadata(
            method: evaluated.method,
            seed: options.seed,
            deviceName: evaluated.isCPU ? "CPU" : MetalRuntime.deviceName,
            wallClockNanoseconds: elapsed,
            qubitCount: circuit.qubitCount,
            gateCount: circuit.gates.count,
            noiseSnapshot: options.noise,
            profile: SimulationProfiling.finishProfile(
                options: options,
                circuit: circuit,
                method: evaluated.method,
                isCPU: evaluated.isCPU,
                elapsed: elapsed
            )
        )

        return JacobianResult(
            expectationValue: evaluated.values[0],
            parameterNames: parameterNames,
            values: values,
            circuitEvaluations: jobs.count,
            method: .parameterShift,
            metadata: metadata
        )
    }

    private func validatedParameterNames(
        circuit: QuantumCircuit,
        parameters: [String: QFloat]
    ) throws -> [String] {
        let parameterNames = circuit.referencedParameters.sorted()
        guard !parameterNames.isEmpty else {
            throw GradientCalculatorError.noDifferentiableParameters
        }
        for name in parameterNames {
            guard parameters[name] != nil else {
                throw GradientCalculatorError.missingParameterBinding(name)
            }
        }
        return parameterNames
    }

    private func shifted(
        _ parameters: [String: QFloat],
        _ name: String,
        by offset: QFloat
    ) -> [String: QFloat] {
        var bindings = parameters
        bindings[name] = (parameters[name] ?? 0) + offset
        return bindings
    }

    private func shifted(
        _ parameters: [String: QFloat],
        _ offsets: [(String, QFloat)]
    ) -> [String: QFloat] {
        var bindings = parameters
        for (name, offset) in offsets {
            bindings[name] = (parameters[name] ?? 0) + offset
        }
        return bindings
    }
}

/// Shared multi-circuit ⟨H⟩ evaluation for shift ensembles.
enum ShiftedExpectationEvaluator {
    struct Outcome {
        let values: [QFloat]
        let method: QuantumSimulationMethod
        let isCPU: Bool
    }

    static func evaluate(
        circuits: [QuantumCircuit],
        hamiltonian: Hamiltonian,
        backend: any QuantumBackend,
        options: QuantumRunOptions,
        batchSize: Int
    ) throws -> Outcome {
        if let statevectorBackend = backend as? StatevectorBackend {
            let values = try BatchExpectationExecutor.evaluate(
                circuits: circuits,
                hamiltonian: hamiltonian,
                engine: statevectorBackend.engine,
                options: options,
                batchSize: batchSize
            )
            return Outcome(values: values, method: .statevector, isCPU: false)
        }
        if let densityBackend = backend as? DensityMatrixBackend {
            let values = try BatchExpectationExecutor.evaluate(
                circuits: circuits,
                hamiltonian: hamiltonian,
                engine: densityBackend.engine,
                options: options
            )
            return Outcome(values: values, method: .densityMatrix, isCPU: false)
        }
        if let cpuSV = backend as? CPUStatevectorBackend {
            let estimator = Estimator()
            var values: [QFloat] = []
            values.reserveCapacity(circuits.count)
            for circuit in circuits {
                let result = try estimator.run(
                    circuit: circuit,
                    hamiltonian: hamiltonian,
                    backend: cpuSV,
                    options: options
                )
                values.append(result.value)
            }
            return Outcome(values: values, method: .statevector, isCPU: true)
        }
        if let cpuDM = backend as? CPUDensityMatrixBackend {
            let estimator = Estimator()
            var values: [QFloat] = []
            values.reserveCapacity(circuits.count)
            for circuit in circuits {
                let result = try estimator.run(
                    circuit: circuit,
                    hamiltonian: hamiltonian,
                    backend: cpuDM,
                    options: options
                )
                values.append(result.value)
            }
            return Outcome(values: values, method: .densityMatrix, isCPU: true)
        }
        throw GradientCalculatorError.unsupportedBackend
    }
}
