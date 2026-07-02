import Foundation

public enum GradientCalculatorError: Error, Equatable {
    case unsupportedBackend
    case noDifferentiableParameters
    case missingParameterBinding(String)
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

/// Result of a parameter-shift gradient evaluation.
public struct GradientResult: Sendable, Equatable {
    /// ⟨H⟩ at the supplied parameter bindings.
    public let expectationValue: QFloat
    /// Per-parameter gradients in deterministic name-sorted order.
    public let parameterGradients: [ParameterGradient]
    /// Total number of distinct circuit evolutions performed (base + shift pairs).
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

/// Options controlling batched expectation evaluation during gradient computation.
public struct GradientOptions: Sendable, Equatable {
    /// Maximum number of shifted circuits evaluated per GPU batch when batching is available.
    public var batchSize: Int

    public init(batchSize: Int = 32) {
        self.batchSize = max(batchSize, 1)
    }
}
