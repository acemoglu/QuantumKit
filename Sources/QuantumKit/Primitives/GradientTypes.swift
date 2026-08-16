import Foundation

public enum GradientCalculatorError: Error, Equatable {
    case unsupportedBackend
    case noDifferentiableParameters
    case missingParameterBinding(String)
    case adjointRequiresUnitaryNoiseFreeCircuit
    case adjointUnsupportedGate(Gate)
    case adjointRequiresStatevectorBackend
    /// Second-order (Hessian) differentiation is parameter-shift only.
    case hessianRequiresParameterShift
    /// Parameter appears with unequal linear scales (e.g. `2γ` and `0.5γ` in different gates).
    case heterogeneousParameterScale(parameter: String, scales: [QFloat])
    /// Parameter is referenced but not as a linear angle in a supported rotation gate.
    case unsupportedParameterShiftGate(parameter: String)
}

/// Selects the algorithm used by ``GradientCalculator`` / ``ObservableDifferentiator`` Jacobian.
public enum DifferentiationMethod: String, Sendable, Equatable, Codable {
    /// Analytic parameter-shift rule (2 evaluations per parameter for first order).
    case parameterShift
    /// Reverse-mode adjoint differentiation (O(1) evolutions in parameter count).
    case adjoint
}

/// Standard parameter-shift amount for Pauli-generator rotations (RX, RY, RZ, RXX, RYY, RZZ).
///
/// For a gate angle `φ = s·θ` (``QFloatExpr/scaled``), shift **θ** by `±π/(2s)` (**signed**
/// `s`, including `s < 0`) so the physical angle moves by `±π/2`, then multiply the
/// macroscopic `(E₊ − E₋)/2` by `s`. Homogeneous `s` across all uses of `θ` is required;
/// heterogeneous scales throw
/// ``GradientCalculatorError/heterogeneousParameterScale(parameter:scales:)``.
public enum ParameterShift {
    public static let shift: QFloat = QFloat(Double.pi / 2.0)

    /// Applies the macroscopic parameter-shift rule: `½ (E₊ − E₋)`.
    public static func gradient(plusExpectation: QFloat, minusExpectation: QFloat) -> QFloat {
        (plusExpectation - minusExpectation) * 0.5
    }

    /// Scale-aware first derivative: `s · ½ (E₊ − E₋)` after shifting θ by `±π/(2s)`.
    public static func gradient(
        plusExpectation: QFloat,
        minusExpectation: QFloat,
        scale: QFloat
    ) -> QFloat {
        scale * gradient(plusExpectation: plusExpectation, minusExpectation: minusExpectation)
    }

    /// Unique linear scale `s` such that every supported rotation angle is `s·θ` (+ literals).
    public static func homogeneousScale(
        for parameter: String,
        in circuit: QuantumCircuit
    ) throws -> QFloat {
        var scales: [QFloat] = []
        for gate in circuit.gates where gate.referencedParameters.contains(parameter) {
            let gateScales = rotationScales(for: parameter, in: gate)
            guard !gateScales.isEmpty else {
                throw GradientCalculatorError.unsupportedParameterShiftGate(parameter: parameter)
            }
            scales.append(contentsOf: gateScales)
        }
        guard !scales.isEmpty else {
            throw GradientCalculatorError.unsupportedParameterShiftGate(parameter: parameter)
        }
        let reference = scales[0]
        for scale in scales.dropFirst() where abs(scale - reference) > 1e-6 {
            throw GradientCalculatorError.heterogeneousParameterScale(
                parameter: parameter,
                scales: scales
            )
        }
        guard abs(reference) > 1e-12 else {
            throw GradientCalculatorError.heterogeneousParameterScale(
                parameter: parameter,
                scales: scales
            )
        }
        return reference
    }

    /// Parameter-space shift that moves the physical gate angle by `±π/2`.
    ///
    /// Uses **signed** `s`: `δ = (π/2)/s`. For `s < 0`, `θ + δ` decreases φ, which is
    /// required so `s · ½(E(θ+δ) − E(θ−δ))` matches `∂⟨H⟩/∂θ`.
    public static func parameterShiftAmount(scale: QFloat) -> QFloat {
        shift / scale
    }

    /// Parameter-space shift that moves the physical gate angle by `±π` (Hessian diagonal).
    /// Signed: `δ = π/s` so `θ ± δ` maps to `φ ± π` for any nonzero `s`.
    public static func parameterPiShiftAmount(scale: QFloat) -> QFloat {
        (shift * 2) / scale
    }

    /// Diagonal second derivative via ±π shifts on the physical angle φ:
    /// `∂²⟨H⟩/∂φ² = ¼ (E(φ+π) − 2E(φ) + E(φ−π))`.
    public static func secondDerivative(
        plusPiExpectation: QFloat,
        centerExpectation: QFloat,
        minusPiExpectation: QFloat
    ) -> QFloat {
        (plusPiExpectation - 2 * centerExpectation + minusPiExpectation) * 0.25
    }

    /// `∂²⟨H⟩/∂θ² = s² · ∂²⟨H⟩/∂φ²` for φ = sθ.
    public static func secondDerivative(
        plusPiExpectation: QFloat,
        centerExpectation: QFloat,
        minusPiExpectation: QFloat,
        scale: QFloat
    ) -> QFloat {
        scale * scale * secondDerivative(
            plusPiExpectation: plusPiExpectation,
            centerExpectation: centerExpectation,
            minusPiExpectation: minusPiExpectation
        )
    }

    /// Mixed partial via four ±π/2 shifts on physical angles.
    public static func mixedPartial(
        plusPlus: QFloat,
        plusMinus: QFloat,
        minusPlus: QFloat,
        minusMinus: QFloat
    ) -> QFloat {
        (plusPlus - plusMinus - minusPlus + minusMinus) * 0.25
    }

    public static func mixedPartial(
        plusPlus: QFloat,
        plusMinus: QFloat,
        minusPlus: QFloat,
        minusMinus: QFloat,
        scaleI: QFloat,
        scaleJ: QFloat
    ) -> QFloat {
        scaleI * scaleJ * mixedPartial(
            plusPlus: plusPlus,
            plusMinus: plusMinus,
            minusPlus: minusPlus,
            minusMinus: minusMinus
        )
    }

    private static func rotationScales(for parameter: String, in gate: Gate) -> [QFloat] {
        switch gate {
        case .p(let theta, _),
             .rx(let theta, _),
             .ry(let theta, _),
             .rz(let theta, _),
             .crx(let theta, _, _),
             .cry(let theta, _, _),
             .crz(let theta, _, _),
             .cp(let theta, _, _),
             .rxx(let theta, _, _),
             .ryy(let theta, _, _),
             .rzz(let theta, _, _):
            return scaleIfPresent(parameter, in: theta)
        case .u(let theta, let phi, let lambda, _):
            return scaleIfPresent(parameter, in: theta)
                + scaleIfPresent(parameter, in: phi)
                + scaleIfPresent(parameter, in: lambda)
        case .c_if(_, _, let inner):
            return rotationScales(for: parameter, in: inner)
        case .while_c(_, _, let body, _):
            return body.flatMap { rotationScales(for: parameter, in: $0) }
        default:
            return []
        }
    }

    private static func scaleIfPresent(_ parameter: String, in expr: QFloatExpr) -> [QFloat] {
        guard expr.referencedParameters.contains(parameter) else { return [] }
        guard let sensitivity = expr.singleParameterSensitivity(),
              sensitivity.name == parameter
        else {
            return []
        }
        return [sensitivity.scale]
    }
}

/// Gradient of an observable expectation with respect to one symbolic circuit parameter.
public struct ParameterGradient: Sendable, Equatable {
    public let parameter: QuantumParameter
    public let gradient: QFloat

    public init(parameter: QuantumParameter, gradient: QFloat) {
        self.parameter = parameter
        self.gradient = gradient
    }

    public var name: String { parameter.name }
}

/// Result of a parameter-shift or adjoint gradient evaluation.
public struct GradientResult: Sendable, Equatable {
    /// ⟨H⟩ at the supplied parameter bindings.
    public let expectationValue: QFloat
    /// Per-parameter gradients in deterministic name-sorted order.
    public let parameterGradients: [ParameterGradient]
    /// Total number of distinct circuit evolutions performed (base + shift pairs, or 1 for adjoint).
    public let circuitEvaluations: Int
    public let metadata: QuantumResultMetadata

    public init(
        expectationValue: QFloat,
        parameterGradients: [ParameterGradient],
        circuitEvaluations: Int,
        metadata: QuantumResultMetadata
    ) {
        self.expectationValue = expectationValue
        self.parameterGradients = parameterGradients
        self.circuitEvaluations = circuitEvaluations
        self.metadata = metadata
    }

    /// Looks up the gradient for a parameter by name.
    public func gradient(for name: String) -> QFloat? {
        parameterGradients.first { $0.name == name }?.gradient
    }
}

/// Observable Jacobian `∂⟨H⟩/∂θᵢ` in name-sorted parameter order.
public struct JacobianResult: Sendable, Equatable {
    public let expectationValue: QFloat
    /// Sorted symbolic parameter names (same order as ``values``).
    public let parameterNames: [String]
    /// `values[i] = ∂⟨H⟩/∂θᵢ` for `parameterNames[i]`.
    public let values: [QFloat]
    public let circuitEvaluations: Int
    public let method: DifferentiationMethod
    public let metadata: QuantumResultMetadata

    public init(
        expectationValue: QFloat,
        parameterNames: [String],
        values: [QFloat],
        circuitEvaluations: Int,
        method: DifferentiationMethod,
        metadata: QuantumResultMetadata
    ) {
        self.expectationValue = expectationValue
        self.parameterNames = parameterNames
        self.values = values
        self.circuitEvaluations = circuitEvaluations
        self.method = method
        self.metadata = metadata
    }

    public init(from gradient: GradientResult, method: DifferentiationMethod) {
        self.expectationValue = gradient.expectationValue
        self.parameterNames = gradient.parameterGradients.map(\.name)
        self.values = gradient.parameterGradients.map(\.gradient)
        self.circuitEvaluations = gradient.circuitEvaluations
        self.method = method
        self.metadata = gradient.metadata
    }

    public func value(for name: String) -> QFloat? {
        guard let index = parameterNames.firstIndex(of: name) else { return nil }
        return values[index]
    }
}

/// Observable Hessian `∂²⟨H⟩/∂θᵢ∂θⱼ` (parameter-shift), name-sorted axes.
public struct HessianResult: Sendable, Equatable {
    public let expectationValue: QFloat
    public let parameterNames: [String]
    /// Dense matrix with `matrix[i][j] = ∂²⟨H⟩/∂θᵢ∂θⱼ` (symmetric by construction).
    public let matrix: [[QFloat]]
    public let circuitEvaluations: Int
    public let metadata: QuantumResultMetadata

    public init(
        expectationValue: QFloat,
        parameterNames: [String],
        matrix: [[QFloat]],
        circuitEvaluations: Int,
        metadata: QuantumResultMetadata
    ) {
        self.expectationValue = expectationValue
        self.parameterNames = parameterNames
        self.matrix = matrix
        self.circuitEvaluations = circuitEvaluations
        self.metadata = metadata
    }

    public func value(row: String, column: String) -> QFloat? {
        guard
            let i = parameterNames.firstIndex(of: row),
            let j = parameterNames.firstIndex(of: column)
        else { return nil }
        return matrix[i][j]
    }
}

/// Options controlling gradient evaluation.
public struct GradientOptions: Sendable, Equatable {
    /// Maximum number of shifted circuits evaluated per GPU batch when batching is available.
    public var batchSize: Int
    /// Differentiation algorithm (default: parameter-shift).
    public var method: DifferentiationMethod

    public init(batchSize: Int = 32, method: DifferentiationMethod = .parameterShift) {
        self.batchSize = max(batchSize, 1)
        self.method = method
    }
}

/// Options for ``ObservableDifferentiator`` (Jacobian / Hessian).
public struct ObservableDifferentiatorOptions: Sendable, Equatable {
    public var batchSize: Int
    /// Jacobian only: ``parameterShift`` (CPU + Metal SV/DM) or ``adjoint`` (Metal SV).
    /// Hessian always uses parameter-shift regardless of this field when calling ``hessian``.
    public var jacobianMethod: DifferentiationMethod

    public init(
        batchSize: Int = 32,
        jacobianMethod: DifferentiationMethod = .parameterShift
    ) {
        self.batchSize = max(batchSize, 1)
        self.jacobianMethod = jacobianMethod
    }

    public static let `default` = ObservableDifferentiatorOptions()
}
