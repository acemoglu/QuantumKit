import Foundation

public enum GradientCalculatorError: Error, Equatable {
    case unsupportedBackend
    case noDifferentiableParameters
    case missingParameterBinding(String)
    case adjointRequiresUnitaryNoiseFreeCircuit
    case adjointUnsupportedGate(Gate)
    case adjointRequiresStatevectorBackend
}

/// Selects the algorithm used by ``GradientCalculator``.
public enum DifferentiationMethod: String, Sendable, Equatable, Codable {
    /// Analytic parameter-shift rule (2 evaluations per parameter).
    case parameterShift
    /// Reverse-mode adjoint differentiation (O(1) evolutions in parameter count).
    case adjoint
}

/// Standard parameter-shift amount for Pauli-generator rotations (RX, RY, RZ).
public enum ParameterShift {
    public static let shift: QFloat = QFloat(Double.pi / 2.0)

    /// Applies the macroscopic parameter-shift rule: `½ (E₊ − E₋)`.
    public static func gradient(plusExpectation: QFloat, minusExpectation: QFloat) -> QFloat {
        (plusExpectation - minusExpectation) * 0.5
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
