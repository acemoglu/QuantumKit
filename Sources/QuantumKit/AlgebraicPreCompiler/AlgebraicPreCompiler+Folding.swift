import Foundation

extension AlgebraicPreCompiler {

    // MARK: - Adjacent folding

    static func adjacentFoldPass(_ gates: [Gate]) -> [Gate] {
        var folded: [Gate] = []
        folded.reserveCapacity(gates.count)

        for gate in gates {
            guard let previous = folded.last else {
                folded.append(gate)
                continue
            }

            if let cancellation = cancelPair(previous, gate) {
                folded.removeLast()
                if let replacement = cancellation {
                    folded.append(replacement)
                }
                continue
            }

            if let merged = mergePair(previous, gate) {
                folded.removeLast()
                if let replacement = merged {
                    folded.append(replacement)
                }
                continue
            }

            folded.append(gate)
        }

        return folded
    }

    // MARK: - Cancellation (self-inverse pairs)

    static func cancelPair(_ lhs: Gate, _ rhs: Gate) -> Gate?? {
        switch (lhs, rhs) {
        case (.h(let a), .h(let b)) where a == b:
            return .some(nil)
        case (.x(let a), .x(let b)) where a == b:
            return .some(nil)
        case (.y(let a), .y(let b)) where a == b:
            return .some(nil)
        case (.z(let a), .z(let b)) where a == b:
            return .some(nil)
        case (.cx(let ac, let at), .cx(let bc, let bt)) where ac == bc && at == bt:
            return .some(nil)
        case (.cz(let ac, let at), .cz(let bc, let bt))
            where (ac == bc && at == bt) || (ac == bt && at == bc):
            return .some(nil)
        case (.swap(let a1, let a2), .swap(let b1, let b2))
            where (a1 == b1 && a2 == b2) || (a1 == b2 && a2 == b1):
            return .some(nil)
        case (.ccx(let a1, let a2, let a3), .ccx(let b1, let b2, let b3))
            where a1 == b1 && a2 == b2 && a3 == b3:
            return .some(nil)
        case (.mcx(let ac, let at), .mcx(let bc, let bt)) where at == bt && ac == bc:
            return .some(nil)
        case (.mcz(let ac, let at), .mcz(let bc, let bt)) where at == bt && ac == bc:
            return .some(nil)
        default:
            return nil
        }
    }

    // MARK: - Merge

    /// Returns `Gate??` where the outer optional encodes mergeability:
    /// `nil` = not mergeable, `.some(nil)` = pair merges to identity (remove both),
    /// `.some(gate)` = pair merges into a single `gate`.
    static func mergePair(_ lhs: Gate, _ rhs: Gate) -> Gate?? {
        if let merged = mergeRotationPair(lhs, rhs) {
            return .some(merged)
        }

        if let merged = mergeZAxisPair(lhs, rhs) {
            return .some(merged)
        }

        return nil
    }

    static func mergeRotationPair(_ lhs: Gate, _ rhs: Gate) -> Gate?? {
        switch (lhs, rhs) {
        case (.rx(let t1, let q1), .rx(let t2, let q2)) where q1 == q2:
            guard let a1 = t1.literalValue, let a2 = t2.literalValue else { return nil }
            return .some(canonicalRotation(axis: .x, angle: a1 + a2, target: q1))
        case (.ry(let t1, let q1), .ry(let t2, let q2)) where q1 == q2:
            guard let a1 = t1.literalValue, let a2 = t2.literalValue else { return nil }
            return .some(canonicalRotation(axis: .y, angle: a1 + a2, target: q1))
        case (.rz(let t1, let q1), .rz(let t2, let q2)) where q1 == q2:
            guard let a1 = t1.literalValue, let a2 = t2.literalValue else { return nil }
            return .some(canonicalZRotation(angle: a1 + a2, target: q1))
        default:
            return nil
        }
    }

    static func mergeZAxisPair(_ lhs: Gate, _ rhs: Gate) -> Gate?? {
        guard let left = zAxisAngle(lhs), let right = zAxisAngle(rhs), left.qubit == right.qubit else {
            return nil
        }
        guard let leftAngle = left.angle.literalValue, let rightAngle = right.angle.literalValue else {
            return nil
        }
        return .some(canonicalZRotation(angle: leftAngle + rightAngle, target: left.qubit))
    }

    enum RotationAxis {
        case x, y
    }

    static func zAxisAngle(_ gate: Gate) -> (qubit: Int, angle: QFloatExpr)? {
        switch gate {
        case .z(let target):
            return (target, QFloatExpr(QFloat(Double.pi)))
        case .s(let target):
            return (target, QFloatExpr(QFloat(Double.pi / 2.0)))
        case .t(let target):
            return (target, QFloatExpr(QFloat(Double.pi / 4.0)))
        case .sdg(let target):
            return (target, QFloatExpr(QFloat(-Double.pi / 2.0)))
        case .tdg(let target):
            return (target, QFloatExpr(QFloat(-Double.pi / 4.0)))
        case .p(let theta, let target):
            return (target, theta)
        case .rz(let theta, let target):
            return (target, theta)
        default:
            return nil
        }
    }

    static func canonicalRotation(axis: RotationAxis, angle: QFloatExpr, target: Int) -> Gate? {
        guard let literal = angle.literalValue else { return nil }
        return canonicalRotation(axis: axis, angle: literal, target: target)
    }

    static func canonicalZRotation(angle: QFloatExpr, target: Int) -> Gate? {
        guard let literal = angle.literalValue else { return nil }
        return canonicalZRotation(angle: literal, target: target)
    }

    static func canonicalRotation(axis: RotationAxis, angle: QFloat, target: Int) -> Gate? {
        let normalized = normalizeAngle(angle)
        guard abs(normalized) > angleTolerance else { return nil }

        switch axis {
        case .x:
            return .rx(theta: QFloatExpr(normalized), target: target)
        case .y:
            return .ry(theta: QFloatExpr(normalized), target: target)
        }
    }

    static func canonicalZRotation(angle: QFloat, target: Int) -> Gate? {
        let normalized = normalizeAngle(angle)
        guard abs(normalized) > angleTolerance else { return nil }

        if anglesEqual(normalized, QFloat(Double.pi / 4.0)) {
            return .t(target: target)
        }
        if anglesEqual(normalized, QFloat(Double.pi / 2.0)) {
            return .s(target: target)
        }
        if anglesEqual(normalized, QFloat(-Double.pi / 4.0)) {
            return .tdg(target: target)
        }
        if anglesEqual(normalized, QFloat(-Double.pi / 2.0)) {
            return .sdg(target: target)
        }
        if anglesEqual(abs(normalized), QFloat(Double.pi)) {
            return .z(target: target)
        }

        return .rz(theta: QFloatExpr(normalized), target: target)
    }

    static func normalizeAngle(_ angle: QFloat) -> QFloat {
        var value = angle.truncatingRemainder(dividingBy: twoPi)
        while value <= -QFloat.pi + angleTolerance {
            value += twoPi
        }
        while value > QFloat.pi - angleTolerance {
            value -= twoPi
        }
        return value
    }

    static func anglesEqual(_ lhs: QFloat, _ rhs: QFloat) -> Bool {
        abs(lhs - rhs) <= angleTolerance
    }
}
