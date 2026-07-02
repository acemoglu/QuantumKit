import Foundation

/// A gate angle that is either a concrete ``QFloat`` or a named symbolic parameter.
public enum QFloatExpr: Equatable, Sendable {
    case literal(QFloat)
    case parameter(QuantumParameter)
    indirect case negated(QFloatExpr)
    indirect case scaled(QFloatExpr, QFloat)
}

extension QFloatExpr {

    public init(_ value: QFloat) {
        self = .literal(value)
    }

    /// `true` when every leaf is a resolved literal (no symbolic parameters remain).
    public var isFullyBound: Bool {
        switch self {
        case .literal:
            return true
        case .parameter:
            return false
        case .negated(let inner), .scaled(let inner, _):
            return inner.isFullyBound
        }
    }

    /// All symbolic parameter names referenced by this expression.
    public var referencedParameters: Set<String> {
        switch self {
        case .literal:
            return []
        case .parameter(let parameter):
            return [parameter.name]
        case .negated(let inner), .scaled(let inner, _):
            return inner.referencedParameters
        }
    }

    /// The concrete value when this expression is fully literal; otherwise `nil`.
    public var literalValue: QFloat? {
        switch self {
        case .literal(let value):
            return value
        case .parameter:
            return nil
        case .negated(let inner):
            guard let value = inner.literalValue else { return nil }
            return -value
        case .scaled(let inner, let factor):
            guard let value = inner.literalValue else { return nil }
            return value * factor
        }
    }

    public func scaled(by factor: QFloat) -> QFloatExpr {
        switch self {
        case .literal(let value):
            return .literal(value * factor)
        case .parameter, .negated, .scaled:
            return .scaled(self, factor)
        }
    }

    public static func / (lhs: QFloatExpr, rhs: QFloat) -> QFloatExpr {
        lhs.scaled(by: 1 / rhs)
    }

    public static prefix func - (expr: QFloatExpr) -> QFloatExpr {
        switch expr {
        case .literal(let value):
            return .literal(-value)
        case .parameter, .negated, .scaled:
            return .negated(expr)
        }
    }

    /// Evaluates this expression using the supplied bindings. Throws when a referenced
    /// parameter is missing from `parameters`.
    public func evaluate(using parameters: [String: QFloat]) throws -> QFloat {
        switch self {
        case .literal(let value):
            return value
        case .parameter(let parameter):
            guard let value = parameters[parameter.name] else {
                throw ParameterBindingError.missingBinding(for: parameter.name)
            }
            return value
        case .negated(let inner):
            return -(try inner.evaluate(using: parameters))
        case .scaled(let inner, let factor):
            return try inner.evaluate(using: parameters) * factor
        }
    }

    /// Replaces symbolic parameters with literals, preserving structure for unbound names.
    public func bound(using parameters: [String: QFloat]) throws -> QFloatExpr {
        switch self {
        case .literal:
            return self
        case .parameter(let parameter):
            guard let value = parameters[parameter.name] else {
                throw ParameterBindingError.missingBinding(for: parameter.name)
            }
            return .literal(value)
        case .negated(let inner):
            return try .negated(inner.bound(using: parameters))
        case .scaled(let inner, let factor):
            let resolved = try inner.bound(using: parameters)
            return try .literal(resolved.evaluate(using: parameters) * factor)
        }
    }

    /// Returns a fully literal expression or throws when symbolic parameters remain.
    public func requireLiteral() throws -> QFloat {
        guard let value = literalValue else {
            throw ParameterBindingError.unboundParameters(referencedParameters)
        }
        return value
    }

    /// Returns a concrete ``QFloat`` for GPU dispatch. Throws when symbolic parameters remain.
    public func gpuAngle() throws -> QFloat {
        try requireLiteral()
    }
}

extension QFloatExpr: Codable {

    private enum CodingKeys: String, CodingKey {
        case literal
        case parameter
        case negated
        case scaled
        case factor
    }

    public init(from decoder: any Decoder) throws {
        if let singleValue = try? decoder.singleValueContainer(),
           let value = try? singleValue.decode(QFloat.self) {
            self = .literal(value)
            return
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let value = try container.decodeIfPresent(QFloat.self, forKey: .literal) {
            self = .literal(value)
        } else if let parameter = try container.decodeIfPresent(QuantumParameter.self, forKey: .parameter) {
            self = .parameter(parameter)
        } else if let inner = try container.decodeIfPresent(QFloatExpr.self, forKey: .negated) {
            self = .negated(inner)
        } else if let inner = try container.decodeIfPresent(QFloatExpr.self, forKey: .scaled),
                  let factor = try container.decodeIfPresent(QFloat.self, forKey: .factor) {
            self = .scaled(inner, factor)
        } else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Invalid QFloatExpr payload")
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        switch self {
        case .literal(let value):
            var container = encoder.singleValueContainer()
            try container.encode(value)
        case .parameter(let parameter):
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(parameter, forKey: .parameter)
        case .negated(let inner):
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(inner, forKey: .negated)
        case .scaled(let inner, let factor):
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(inner, forKey: .scaled)
            try container.encode(factor, forKey: .factor)
        }
    }
}
