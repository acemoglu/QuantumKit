import Foundation

public enum TranspilerError: Error, Equatable {
    case unsupportedGate(Gate)
    case unsupportedGateForBasis(gate: Gate, basis: BasisGateSet)
    case mismatchedQubitCounts(original: Int, transpiled: Int)
}

enum GateDecomposition {

    private static let pi = QFloatExpr(QFloat(Double.pi))
    private static let halfPi = pi / 2
    private static let quarterPi = pi / 4

    /// One decomposition step: replaces `gate` with an equivalent shorter-list using simpler gates.
    static func expand(_ gate: Gate) throws -> [Gate] {
        switch gate {
        case .h(let target):
            return rzSxRz(target: target, pre: halfPi, post: halfPi)

        case .x(let target):
            return rzSxRz(target: target, pre: -halfPi, post: halfPi)

        case .y(let target):
            return [
                .rz(theta: halfPi, target: target),
                .sx(target: target),
                .rz(theta: halfPi, target: target),
                .sx(target: target),
                .rz(theta: halfPi, target: target),
            ]

        case .z(let target):
            return [.rz(theta: pi, target: target)]

        case .s(let target):
            return [.u(theta: QFloatExpr(0), phi: QFloatExpr(0), lambda: halfPi, target: target)]

        case .t(let target):
            return [.u(theta: QFloatExpr(0), phi: QFloatExpr(0), lambda: quarterPi, target: target)]

        case .sdg(let target):
            return [.u(theta: QFloatExpr(0), phi: QFloatExpr(0), lambda: -halfPi, target: target)]

        case .tdg(let target):
            return [.u(theta: QFloatExpr(0), phi: QFloatExpr(0), lambda: -quarterPi, target: target)]

        case .sxdg(let target):
            return rzSxRz(target: target, pre: -halfPi, post: -halfPi)

        case .p(let theta, let target):
            return [.u(theta: QFloatExpr(0), phi: QFloatExpr(0), lambda: theta, target: target)]

        case .rx(let theta, let target):
            return [
                .rz(theta: -halfPi, target: target),
                .sx(target: target),
                .rz(theta: theta, target: target),
                .sxdg(target: target),
                .rz(theta: halfPi, target: target),
            ]

        case .ry(let theta, let target):
            return [
                .rz(theta: halfPi, target: target),
                .sx(target: target),
                .rz(theta: theta, target: target),
                .sxdg(target: target),
                .rz(theta: -halfPi, target: target),
            ]

        case .rz, .sx, .cx:
            return [gate]

        case .u(let theta, let phi, let lambda, let target):
            return [
                .rz(theta: lambda, target: target),
                .sx(target: target),
                .rz(theta: theta, target: target),
                .sxdg(target: target),
                .rz(theta: phi, target: target),
            ]

        case .cz(let control, let target):
            return [
                .h(target: target),
                .cx(control: control, target: target),
                .h(target: target),
            ]

        case .swap(let q1, let q2):
            return [
                .cx(control: q1, target: q2),
                .cx(control: q2, target: q1),
                .cx(control: q1, target: q2),
            ]

        case .ccx(let control1, let control2, let target):
            return ccx(control1: control1, control2: control2, target: target)

        case .mcx(let controls, let target):
            switch controls.count {
            case 1:
                return [.cx(control: controls[0], target: target)]
            case 2:
                return ccx(control1: controls[0], control2: controls[1], target: target)
            default:
                throw TranspilerError.unsupportedGate(gate)
            }

        case .mcz(let controls, let target):
            return [
                .mcx(controls: controls, target: target),
                .x(target: target),
                .mcx(controls: controls, target: target),
                .x(target: target),
            ]

        case .crx(let theta, let control, let target):
            return [
                .rz(theta: -halfPi, target: target),
                .cx(control: control, target: target),
                .rz(theta: theta, target: target),
                .cx(control: control, target: target),
                .rz(theta: halfPi, target: target),
            ]

        case .cry(let theta, let control, let target):
            return [
                .ry(theta: theta / 2, target: target),
                .cx(control: control, target: target),
                .ry(theta: -theta / 2, target: target),
                .cx(control: control, target: target),
            ]

        case .crz(let theta, let control, let target):
            return [
                .rz(theta: theta / 2, target: target),
                .cx(control: control, target: target),
                .rz(theta: -theta / 2, target: target),
                .cx(control: control, target: target),
            ]

        case .cp(let theta, let control, let target):
            return [
                .crz(theta: theta, control: control, target: target),
            ]

        case .measure, .reset, .c_if, .unitary1, .initialize, .customUnitary:
            throw TranspilerError.unsupportedGate(gate)
        }
    }

    static func expandRecursively(
        _ gate: Gate,
        into basis: BasisGateSet
    ) throws -> [Gate] {
        if basis.contains(gate) {
            return [gate]
        }

        var expanded: [Gate] = []
        for replacement in try expand(gate) {
            expanded.append(contentsOf: try expandRecursively(replacement, into: basis))
        }

        if expanded.isEmpty {
            throw TranspilerError.unsupportedGateForBasis(gate: gate, basis: basis)
        }
        return expanded
    }

    private static func rzSxRz(target: Int, pre: QFloatExpr, post: QFloatExpr) -> [Gate] {
        [
            .rz(theta: pre, target: target),
            .sx(target: target),
            .rz(theta: post, target: target),
        ]
    }

    /// Standard Toffoli decomposition (H / T / CX ladder).
    private static func ccx(control1: Int, control2: Int, target: Int) -> [Gate] {
        [
            .h(target: target),
            .cx(control: control2, target: target),
            .tdg(target: target),
            .cx(control: control1, target: target),
            .t(target: target),
            .cx(control: control2, target: target),
            .tdg(target: target),
            .cx(control: control1, target: target),
            .t(target: target),
            .t(target: control2),
            .h(target: target),
            .cx(control: control1, target: control2),
            .t(target: control2),
            .tdg(target: control1),
            .cx(control: control1, target: control2),
        ]
    }
}
