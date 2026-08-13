import Foundation

/// Sensitivity of an angle expression to a single symbolic parameter: `∂expr/∂param`.
struct ParameterSensitivity: Equatable {
    let name: String
    let scale: QFloat
}

extension QFloatExpr {
    /// Linear dependence on exactly one parameter. Returns `nil` for literals or multi-parameter forms.
    func singleParameterSensitivity() -> ParameterSensitivity? {
        switch self {
        case .literal:
            return nil
        case .parameter(let parameter):
            return ParameterSensitivity(name: parameter.name, scale: 1)
        case .negated(let inner):
            guard let sensitivity = inner.singleParameterSensitivity() else { return nil }
            return ParameterSensitivity(name: sensitivity.name, scale: -sensitivity.scale)
        case .scaled(let inner, let factor):
            guard let sensitivity = inner.singleParameterSensitivity() else { return nil }
            return ParameterSensitivity(name: sensitivity.name, scale: sensitivity.scale * factor)
        }
    }
}

extension Gate {
    /// Trainable rotation axes supported by adjoint differentiation (`RX`/`RY`/`RZ`).
    enum AdjointGenerator: Equatable {
        case x(target: Int)
        case y(target: Int)
        case z(target: Int)

        var pauli: Pauli {
            switch self {
            case .x: return .x
            case .y: return .y
            case .z: return .z
            }
        }

        var target: Int {
            switch self {
            case .x(let target), .y(let target), .z(let target):
                return target
            }
        }
    }

    /// Parameterized generator contribution for reverse-mode differentiation, if any.
    func adjointTrainable() -> (generator: AdjointGenerator, sensitivity: ParameterSensitivity)? {
        switch self {
        case .rx(let theta, let target):
            guard let sensitivity = theta.singleParameterSensitivity() else { return nil }
            return (.x(target: target), sensitivity)
        case .ry(let theta, let target):
            guard let sensitivity = theta.singleParameterSensitivity() else { return nil }
            return (.y(target: target), sensitivity)
        case .rz(let theta, let target):
            guard let sensitivity = theta.singleParameterSensitivity() else { return nil }
            return (.z(target: target), sensitivity)
        default:
            return nil
        }
    }
}

/// Applies a single-qubit Pauli to a dense amplitude vector (qubit 0 = LSB).
enum AmplitudeAlgebra {
    static func innerProduct(
        _ bra: [ComplexAmplitude],
        _ ket: [ComplexAmplitude]
    ) -> ComplexAmplitude {
        precondition(bra.count == ket.count)
        var real: Double = 0
        var imag: Double = 0
        for index in bra.indices {
            let a = bra[index]
            let b = ket[index]
            // conj(a) * b
            real += Double(a.real) * Double(b.real) + Double(a.imaginary) * Double(b.imaginary)
            imag += Double(a.real) * Double(b.imaginary) - Double(a.imaginary) * Double(b.real)
        }
        return ComplexAmplitude(real: QFloat(real), imaginary: QFloat(imag))
    }

    static func applyPauli(
        _ pauli: Pauli,
        target: Int,
        to amplitudes: inout [ComplexAmplitude]
    ) {
        let n = amplitudes.count
        let bit = 1 << target
        switch pauli {
        case .i:
            return
        case .x:
            var index = 0
            while index < n {
                if index & bit == 0 {
                    let other = index | bit
                    let tmp = amplitudes[index]
                    amplitudes[index] = amplitudes[other]
                    amplitudes[other] = tmp
                }
                index += 1
            }
        case .y:
            var index = 0
            while index < n {
                if index & bit == 0 {
                    let other = index | bit
                    // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                    let a0 = amplitudes[index]
                    let a1 = amplitudes[other]
                    amplitudes[index] = ComplexAmplitude(real: a1.imaginary, imaginary: -a1.real) // -i * a1
                    amplitudes[other] = ComplexAmplitude(real: -a0.imaginary, imaginary: a0.real) // +i * a0
                }
                index += 1
            }
        case .z:
            for index in 0..<n where index & bit != 0 {
                let a = amplitudes[index]
                amplitudes[index] = ComplexAmplitude(real: -a.real, imaginary: -a.imaginary)
            }
        }
    }

    static func applyPauliString(
        _ paulis: [Int: Pauli],
        to amplitudes: [ComplexAmplitude]
    ) -> [ComplexAmplitude] {
        var result = amplitudes
        for qubit in paulis.keys.sorted() {
            guard let pauli = paulis[qubit], pauli != .i else { continue }
            applyPauli(pauli, target: qubit, to: &result)
        }
        return result
    }

    static func applyHamiltonian(
        _ hamiltonian: Hamiltonian,
        to amplitudes: [ComplexAmplitude]
    ) -> [ComplexAmplitude] {
        var accumulator = Array(
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: amplitudes.count
        )
        for term in hamiltonian.terms {
            let acted = applyPauliString(term.paulis, to: amplitudes)
            for index in accumulator.indices {
                let a = acted[index]
                accumulator[index] = ComplexAmplitude(
                    real: accumulator[index].real + term.coefficient * a.real,
                    imaginary: accumulator[index].imaginary + term.coefficient * a.imaginary
                )
            }
        }
        return accumulator
    }
}

/// Reverse-mode (adjoint) differentiation of ⟨H⟩ for circuits with `RX`/`RY`/`RZ` parameters.
///
/// Uses one forward statevector evolution plus a reverse sweep — O(1) circuit evolutions
/// in the parameter count, versus O(P) for parameter-shift.
public struct AdjointDifferentiator: Sendable {
    public init() {}

    public func run(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        parameters: [String: QFloat],
        backend: StatevectorBackend,
        options: QuantumRunOptions = QuantumRunOptions()
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

        if options.noise != nil {
            throw GradientCalculatorError.adjointRequiresUnitaryNoiseFreeCircuit
        }
        guard circuit.isUnitaryOnly else {
            throw GradientCalculatorError.adjointRequiresUnitaryNoiseFreeCircuit
        }

        // Keep symbolic gates for generator extraction; bind angles for execution.
        let symbolicGates = circuit.gates
        let boundCircuit = try circuit.bind(parameters: parameters)
        for gate in symbolicGates {
            if gate.referencedParameters.count > 1 {
                throw GradientCalculatorError.adjointUnsupportedGate(gate)
            }
            if !gate.referencedParameters.isEmpty, gate.adjointTrainable() == nil {
                throw GradientCalculatorError.adjointUnsupportedGate(gate)
            }
        }

        let engine = backend.engine
        let psiState = try StateVector(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try engine.executeRNG(boundCircuit, on: psiState, rng: &rng, noise: nil)

        var psi = QuantumMeasurement.amplitudes(state: psiState)
        let expectationValue = try hamiltonian.expectation(state: psiState, engine: engine)
        var lambda = AmplitudeAlgebra.applyHamiltonian(hamiltonian, to: psi)

        var gradientsByName: [String: QFloat] = [:]
        for name in parameterNames {
            gradientsByName[name] = 0
        }

        for (symbolicGate, boundGate) in zip(symbolicGates, boundCircuit.gates).reversed() {
            // Move |ψ⟩ back through U†.
            try engine.executeUnitaryGate(boundGate.adjoint, on: psiState)
            psi = QuantumMeasurement.amplitudes(state: psiState)

            if let trainable = symbolicGate.adjointTrainable() {
                var generated = psi
                AmplitudeAlgebra.applyPauli(
                    trainable.generator.pauli,
                    target: trainable.generator.target,
                    to: &generated
                )
                // U = exp(-i θ G / 2) ⇒ ∂⟨H⟩/∂θ = Im(⟨λ|G|ψ⟩) with |ψ⟩ before U and |λ⟩ after U.
                let bracket = AmplitudeAlgebra.innerProduct(lambda, generated)
                let contribution = bracket.imaginary * trainable.sensitivity.scale
                gradientsByName[trainable.sensitivity.name, default: 0] += contribution
            }

            // Move |λ⟩ back through U†. λ = H|ψ⟩ is generally unnormalized.
            let lambdaState = try StateVector(qubitCount: circuit.qubitCount)
            try lambdaState.replaceAmplitudesUnchecked(lambda)
            try engine.executeUnitaryGate(boundGate.adjoint, on: lambdaState)
            lambda = QuantumMeasurement.amplitudes(state: lambdaState)
        }

        let parameterGradients = parameterNames.map { name in
            ParameterGradient(
                parameter: QuantumParameter(name),
                gradient: gradientsByName[name] ?? 0
            )
        }

        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = QuantumResultMetadata(
            method: .statevector,
            seed: options.seed,
            deviceName: MetalRuntime.deviceName,
            wallClockNanoseconds: elapsed,
            qubitCount: circuit.qubitCount,
            gateCount: circuit.gates.count,
            noiseSnapshot: options.noise
        )

        return GradientResult(
            expectationValue: expectationValue,
            parameterGradients: parameterGradients,
            circuitEvaluations: 1,
            metadata: metadata
        )
    }
}
