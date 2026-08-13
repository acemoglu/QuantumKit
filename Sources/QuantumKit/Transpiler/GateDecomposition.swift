import Foundation

public enum TranspilerError: Error, Equatable {
    case unsupportedGate(Gate)
    case unsupportedGateForBasis(gate: Gate, basis: BasisGateSet)
    case mismatchedQubitCounts(original: Int, transpiled: Int)
    case invalidCouplingMap(reason: String)
    case invalidLayout(reason: String)
    case circuitWiderThanDevice(circuitQubits: Int, deviceQubits: Int)
    case qubitsNotConnected(Int, Int)
    case routingRequiresTwoQubitGates(Gate)
    case unboundParameters([String])
    case circuitExceedsMaxDepth(depth: Int, maxDepth: Int)
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

        case .id:
            return []

        case .barrier, .delay:
            // Structural / timing ops — not expanded into unitaries.
            return [gate]

        case .dcx(let q1, let q2):
            return [
                .cx(control: q1, target: q2),
                .cx(control: q2, target: q1),
            ]

        case .rzz(let theta, let q1, let q2):
            return [
                .cx(control: q1, target: q2),
                .rz(theta: theta, target: q2),
                .cx(control: q1, target: q2),
            ]

        case .rxx(let theta, let q1, let q2):
            return [
                .h(target: q1),
                .h(target: q2),
                .rzz(theta: theta, q1: q1, q2: q2),
                .h(target: q1),
                .h(target: q2),
            ]

        case .ryy(let theta, let q1, let q2):
            return [
                .rx(theta: halfPi, target: q1),
                .rx(theta: halfPi, target: q2),
                .rzz(theta: theta, q1: q1, q2: q2),
                .rx(theta: -halfPi, target: q1),
                .rx(theta: -halfPi, target: q2),
            ]

        case .iswap(let q1, let q2):
            // S⊗S · H⊗I · CX · CX · I⊗H  (up to global phase conventions)
            return [
                .s(target: q1),
                .s(target: q2),
                .h(target: q1),
                .cx(control: q1, target: q2),
                .cx(control: q2, target: q1),
                .h(target: q2),
            ]

        case .ecr(let control, let target):
            // ECR ≅ RZX(π/4) · X(control) · RZX(-π/4)
            let quarterPi = pi / 4
            return [
                .h(target: target),
                .cx(control: control, target: target),
                .rz(theta: quarterPi, target: target),
                .cx(control: control, target: target),
                .h(target: target),
                .x(target: control),
                .h(target: target),
                .cx(control: control, target: target),
                .rz(theta: -quarterPi, target: target),
                .cx(control: control, target: target),
                .h(target: target),
            ]

        case .cswap(let control, let q1, let q2):
            return [
                .cx(control: q2, target: q1),
                .ccx(control1: control, control2: q1, target: q2),
                .cx(control: q2, target: q1),
            ]

        case .ccx(let control1, let control2, let target):
            return ccx(control1: control1, control2: control2, target: target)

        case .mcx(let controls, let target):
            return try mcx(controls: controls, target: target)

        case .mcz(let controls, let target):
            return [
                .h(target: target),
                .mcx(controls: controls, target: target),
                .h(target: target),
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

        // Structural ops pass through even when outside the unitary basis.
        switch gate {
        case .barrier, .delay:
            return [gate]
        case .id:
            return []
        default:
            break
        }

        var expanded: [Gate] = []
        for replacement in try expand(gate) {
            expanded.append(contentsOf: try expandRecursively(replacement, into: basis))
        }

        if expanded.isEmpty, case .id = gate {
            return []
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

    /// Ancilla-free MCX via H · multi-controlled-phase(π) · H.
    private static func mcx(controls: [Int], target: Int) throws -> [Gate] {
        switch controls.count {
        case 0:
            return [.x(target: target)]
        case 1:
            return [.cx(control: controls[0], target: target)]
        case 2:
            return ccx(control1: controls[0], control2: controls[1], target: target)
        default:
            let pi = QFloatExpr(QFloat(Double.pi))
            return try [.h(target: target)]
                + mcp(theta: pi, controls: controls, target: target)
                + [.h(target: target)]
        }
    }

    /// Multi-controlled phase Cⁿ(P(θ)) via recursive demultiplexing (Barenco et al.).
    private static func mcp(theta: QFloatExpr, controls: [Int], target: Int) throws -> [Gate] {
        switch controls.count {
        case 0:
            return [.p(theta: theta, target: target)]
        case 1:
            return [.cp(theta: theta, control: controls[0], target: target)]
        default:
            let last = controls[controls.count - 1]
            let rest = Array(controls.dropLast())
            let half = theta / 2
            return try mcp(theta: half, controls: rest, target: target)
                + [.cx(control: last, target: target)]
                + mcp(theta: -half, controls: rest, target: target)
                + [.cx(control: last, target: target)]
                + mcp(theta: half, controls: rest, target: last)
        }
    }

    /// Whether `gate` must be expanded into primitive kernels before GPU encoding.
    static func needsExecutionExpansion(_ gate: Gate) -> Bool {
        switch gate {
        case .iswap, .ecr, .rxx, .ryy, .rzz, .dcx, .cswap, .id:
            return true
        default:
            return false
        }
    }
}
