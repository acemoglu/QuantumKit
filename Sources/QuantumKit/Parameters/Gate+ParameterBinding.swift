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
        case .rxx(let theta, _, _), .ryy(let theta, _, _), .rzz(let theta, _, _):
            return !theta.isFullyBound
        case .c_if(_, _, let conditionedGate):
            return conditionedGate.containsUnboundParameters
        case .while_c(_, _, let body, _):
            return body.contains { $0.containsUnboundParameters }
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
        case .rxx(let theta, _, _), .ryy(let theta, _, _), .rzz(let theta, _, _):
            return theta.referencedParameters
        case .c_if(_, _, let conditionedGate):
            return conditionedGate.referencedParameters
        case .while_c(_, _, let body, _):
            return body.reduce(into: Set<String>()) { $0.formUnion($1.referencedParameters) }
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
        case .rxx(let theta, let q1, let q2):
            return .rxx(theta: try theta.bound(using: parameters), q1: q1, q2: q2)
        case .ryy(let theta, let q1, let q2):
            return .ryy(theta: try theta.bound(using: parameters), q1: q1, q2: q2)
        case .rzz(let theta, let q1, let q2):
            return .rzz(theta: try theta.bound(using: parameters), q1: q1, q2: q2)
        case .c_if(let classicalRegister, let expectedValue, let conditionedGate):
            return .c_if(
                classicalRegister: classicalRegister,
                expectedValue: expectedValue,
                gate: try conditionedGate.bound(using: parameters)
            )
        case .while_c(let classicalRegister, let expectedValue, let body, let maxIterations):
            return .while_c(
                classicalRegister: classicalRegister,
                expectedValue: expectedValue,
                body: try body.map { try $0.bound(using: parameters) },
                maxIterations: maxIterations
            )
        default:
            return self
        }
    }
}
