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

    /// Context for strategy-aware expansions (MCX / controlled synthesis + ancillas).
    struct Context {
        var controlledSynthesis: ControlledGateSynthesisStrategy = .ancillaFree
        var ancillaAllocator: AncillaAllocator?

        static let `default` = Context()
    }

    /// One decomposition step: replaces `gate` with an equivalent shorter-list using simpler gates.
    static func expand(_ gate: Gate) throws -> [Gate] {
        var context = Context.default
        return try expand(gate, context: &context)
    }

    /// Strategy-aware expansion. Mutations to ``Context/ancillaAllocator`` are written back
    /// through `context` so callers can reuse the same allocator across gates.
    static func expand(_ gate: Gate, context: inout Context) throws -> [Gate] {
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
            return [
                .s(target: q1),
                .s(target: q2),
                .h(target: q1),
                .cx(control: q1, target: q2),
                .cx(control: q2, target: q1),
                .h(target: q2),
            ]

        case .ecr(let control, let target):
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
            return try mcx(controls: controls, target: target, context: &context)

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
        var context = Context.default
        return try expandRecursively(gate, into: basis, context: &context)
    }

    static func expandRecursively(
        _ gate: Gate,
        into basis: BasisGateSet,
        context: inout Context
    ) throws -> [Gate] {
        if basis.contains(gate) {
            return [gate]
        }

        switch gate {
        case .barrier, .delay:
            return [gate]
        case .id:
            return []
        default:
            break
        }

        var expanded: [Gate] = []
        for replacement in try expand(gate, context: &context) {
            expanded.append(contentsOf: try expandRecursively(replacement, into: basis, context: &context))
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
    static func ccx(control1: Int, control2: Int, target: Int) -> [Gate] {
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

    private static func mcx(
        controls: [Int],
        target: Int,
        context: inout Context
    ) throws -> [Gate] {
        switch controls.count {
        case 0:
            return [.x(target: target)]
        case 1:
            return [.cx(control: controls[0], target: target)]
        case 2:
            return ccx(control1: controls[0], control2: controls[1], target: target)
        default:
            switch context.controlledSynthesis {
            case .ancillaFree:
                let pi = QFloatExpr(QFloat(Double.pi))
                return try [.h(target: target)]
                    + mcp(theta: pi, controls: controls, target: target)
                    + [.h(target: target)]
            case .vChainAncilla:
                guard var allocator = context.ancillaAllocator else {
                    throw AncillaAllocationError.ancillaAllocationDisabled(strategy: .vChainAncilla)
                }
                let gates = try expandMCXWithAllocator(
                    controls: controls,
                    target: target,
                    strategy: .vChainAncilla,
                    allocator: &allocator
                )
                context.ancillaAllocator = allocator
                return gates
            }
        }
    }

    /// Acquires / releases ancillas for one V-chain expansion.
    static func expandMCXWithAllocator(
        controls: [Int],
        target: Int,
        strategy: ControlledGateSynthesisStrategy,
        allocator: inout AncillaAllocator,
        reuseAncillas: Bool = true
    ) throws -> [Gate] {
        switch strategy {
        case .ancillaFree:
            var ctx = Context.default
            ctx.controlledSynthesis = .ancillaFree
            return try mcx(controls: controls, target: target, context: &ctx)
        case .vChainAncilla:
            switch controls.count {
            case 0:
                return [.x(target: target)]
            case 1:
                return [.cx(control: controls[0], target: target)]
            case 2:
                return ccx(control1: controls[0], control2: controls[1], target: target)
            default:
                let required = controls.count - 2
                let ancillas = allocator.acquire(required)
                defer {
                    if reuseAncillas {
                        allocator.release(ancillas)
                    }
                }
                let n = controls.count
                var gates: [Gate] = []
                // Emit structured CCX so routing/unroll can expand; unitary checks use exact embed.
                gates.append(.ccx(control1: controls[0], control2: controls[1], target: ancillas[0]))
                if required > 1 {
                    for i in 1..<required {
                        gates.append(.ccx(
                            control1: ancillas[i - 1],
                            control2: controls[i + 1],
                            target: ancillas[i]
                        ))
                    }
                }
                gates.append(.ccx(
                    control1: ancillas[required - 1],
                    control2: controls[n - 1],
                    target: target
                ))
                if required > 1 {
                    for i in stride(from: required - 1, through: 1, by: -1) {
                        gates.append(.ccx(
                            control1: ancillas[i - 1],
                            control2: controls[i + 1],
                            target: ancillas[i]
                        ))
                    }
                }
                gates.append(.ccx(control1: controls[0], control2: controls[1], target: ancillas[0]))
                return gates
            }
        }
    }

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

    static func needsExecutionExpansion(_ gate: Gate) -> Bool {
        switch gate {
        case .iswap, .ecr, .rxx, .ryy, .rzz, .dcx, .cswap, .id:
            return true
        default:
            return false
        }
    }
}
