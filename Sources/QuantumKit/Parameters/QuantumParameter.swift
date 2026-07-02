import Foundation

/// A named symbolic angle used in variational circuits (e.g. VQE, QAOA).
public struct QuantumParameter: Hashable, Sendable, Codable, CustomStringConvertible {
    public let name: String

    public init(_ name: String) {
        self.name = name
    }

    public var description: String { name }
}

/// Creates a symbolic gate angle referencing the named parameter.
///
/// ```swift
/// try circuit.rx(theta: Parameter("theta"), 0)
/// ```
public func Parameter(_ name: String) -> QFloatExpr {
    .parameter(QuantumParameter(name))
}
