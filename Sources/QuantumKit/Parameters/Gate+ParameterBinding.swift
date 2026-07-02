import Foundation

extension Gate {

    /// `true` when any parametrized angle still references a symbolic parameter.
    public var containsUnboundParameters: Bool {
        switch self {
        case .p(let theta, _), .rx(let theta, _), .ry(let theta, _), .rz(let theta, _):
            return !theta.isFullyBound
        case .u(let theta, let phi, let lambda, _):
            return !theta.isFullyBound || !phi.isFullyBound || !lambda.isFullyBound
        case .crx(let theta, _, _), .cry(let theta, _, _), .crz(let theta, _, _), .cp(let theta, _, _):
            return !theta.isFullyBound
        case .c_if(_, _, let conditionedGate):
            return conditionedGate.containsUnboundParameters
        default:
            return false
        }
    }

    /// All symbolic parameter names referenced by this gate.
    public var referencedParameters: Set<String> {
        switch self {
        case .p(let theta, _), .rx(let theta, _), .ry(let theta, _), .rz(let theta, _):
            return theta.referencedParameters
        case .u(let theta, let phi, let lambda, _):
            return theta.referencedParameters.union(phi.referencedParameters).union(lambda.referencedParameters)
        case .crx(let theta, _, _), .cry(let theta, _, _), .crz(let theta, _, _), .cp(let theta, _, _):
            return theta.referencedParameters
        case .c_if(_, _, let conditionedGate):
            return conditionedGate.referencedParameters
        default:
            return []
        }
    }

    /// Returns a copy of this gate with symbolic parameters replaced by concrete values.
    public func bound(using parameters: [String: QFloat]) throws -> Gate {
        switch self {
        case .p(let theta, let target):
            return .p(theta: try theta.bound(using: parameters), target: target)
        case .u(let theta, let phi, let lambda, let target):
            return .u(
                theta: try theta.bound(using: parameters),
                phi: try phi.bound(using: parameters),
                lambda: try lambda.bound(using: parameters),
                target: target
            )
        case .rx(let theta, let target):
            return .rx(theta: try theta.bound(using: parameters), target: target)
        case .ry(let theta, let target):
            return .ry(theta: try theta.bound(using: parameters), target: target)
        case .rz(let theta, let target):
            return .rz(theta: try theta.bound(using: parameters), target: target)
        case .crx(let theta, let control, let target):
            return .crx(theta: try theta.bound(using: parameters), control: control, target: target)
        case .cry(let theta, let control, let target):
            return .cry(theta: try theta.bound(using: parameters), control: control, target: target)
        case .crz(let theta, let control, let target):
            return .crz(theta: try theta.bound(using: parameters), control: control, target: target)
        case .cp(let theta, let control, let target):
            return .cp(theta: try theta.bound(using: parameters), control: control, target: target)
        case .c_if(let classicalRegister, let expectedValue, let conditionedGate):
            return .c_if(
                classicalRegister: classicalRegister,
                expectedValue: expectedValue,
                gate: try conditionedGate.bound(using: parameters)
            )
        default:
            return self
        }
    }
}
