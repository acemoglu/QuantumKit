import Foundation

/// Algebraic gate simplification applied before GPU execution.
///
/// Reduces gate count without changing the unitary. Cancels self-inverse pairs (H·H, X·X, CX·CX),
/// merges rotations and Z-axis phases (S·S → Z, adjacent RX/RY/RZ), and reorders commuting gates
/// so same-qubit operations that were separated can fold together (e.g. `H(0), X(1), H(0)` → `X(1)`).

public struct AlgebraicPreCompiler: Sendable {

    public struct Result: Sendable, Equatable {
        public let originalGateCount: Int
        public let optimizedGateCount: Int
        public let gates: [Gate]

        public var removedGateCount: Int {
            originalGateCount - optimizedGateCount
        }
    }

    private static let angleTolerance: QFloat = 1e-5
    private static let twoPi = QFloat(2 * Double.pi)

    /// Runs commutation sliding and adjacent folding until the gate list stabilizes.
    public static func optimize(gates: [Gate]) -> Result {
        let originalCount = gates.count
        var current = gates

        while true {
            let next = adjacentFoldPass(commutationSlidePass(current))
            if next == current {
                return Result(
                    originalGateCount: originalCount,
                    optimizedGateCount: next.count,
                    gates: next
                )
            }
            current = next
        }
    }

    /// Returns an optimized copy of `circuit` with the same qubit count.
    public static func optimize(_ circuit: QuantumCircuit) throws -> QuantumCircuit {
        let output = optimize(gates: circuit.gates)
        var optimized = try QuantumCircuit(qubitCount: circuit.qubitCount)
        for gate in output.gates {
            try optimized.apply(gate)
        }
        return optimized
    }

    // MARK: - Commutation sliding

    static func slideGates(_ gates: [Gate]) -> [Gate] {
        commutationSlidePass(gates)
    }

    static func foldGates(_ gates: [Gate]) -> [Gate] {
        adjacentFoldPass(gates)
    }

    private static func commutationSlidePass(_ gates: [Gate]) -> [Gate] {
        guard gates.count > 1 else { return gates }

        var result = gates
        var changed = true

        while changed {
            changed = false

            for index in 1..<result.count {
                guard commutes(result[index], result[index - 1]) else { continue }
                guard reorderingAdjacentPairReducesSameQubitDistance(
                    in: result,
                    leftIndex: index - 1,
                    rightIndex: index
                ) else { continue }

                result.swapAt(index, index - 1)
                changed = true
            }
        }

        return result
    }

    /// True when swapping `leftIndex` and `rightIndex` moves same-qubit gates closer together.
    private static func reorderingAdjacentPairReducesSameQubitDistance(
        in gates: [Gate],
        leftIndex: Int,
        rightIndex: Int
    ) -> Bool {
        let before = sameQubitPairDistanceSum(gates)
        var reordered = gates
        reordered.swapAt(leftIndex, rightIndex)
        let after = sameQubitPairDistanceSum(reordered)
        return after < before
    }

    private static func sameQubitPairDistanceSum(_ gates: [Gate]) -> Int {
        var indicesByQubit: [Int: [Int]] = [:]

        for (index, gate) in gates.enumerated() {
            guard let single = gate.asSingleQubitGate else { continue }
            indicesByQubit[single.qubit, default: []].append(index)
        }

        var distanceSum = 0
        for indices in indicesByQubit.values where indices.count > 1 {
            let sorted = indices.sorted()
            for pair in zip(sorted, sorted.dropFirst()) {
                distanceSum += pair.1 - pair.0
            }
        }

        return distanceSum
    }

    /// Whether two unitary gates can be reordered: `lhs * rhs == rhs * lhs`.
    public static func commutes(_ lhs: Gate, _ rhs: Gate) -> Bool {
        if isBarrier(lhs) || isBarrier(rhs) {
            return false
        }

        let leftQubits = Set(lhs.affectedQubits)
        let rightQubits = Set(rhs.affectedQubits)

        if leftQubits.isDisjoint(with: rightQubits) {
            return true
        }

        if commutesOnOverlappingQubits(lhs, rhs) {
            return true
        }

        return false
    }

    private static func isBarrier(_ gate: Gate) -> Bool {
        switch gate {
        case .measure, .reset:
            return true
        default:
            return false
        }
    }

    private static func commutesOnOverlappingQubits(_ lhs: Gate, _ rhs: Gate) -> Bool {
        if let single = lhs.asSingleQubitGate, commutesSingleQubitWithGate(single, rhs) {
            return true
        }
        if let single = rhs.asSingleQubitGate, commutesSingleQubitWithGate(single, lhs) {
            return true
        }

        switch (lhs, rhs) {
        case (.cx(let c1, let t1), .cx(let c2, let t2)):
            return commutesCNOTPair(control1: c1, target1: t1, control2: c2, target2: t2)
        default:
            return false
        }
    }

    private static func commutesSingleQubitWithGate(_ single: SingleQubitGate, _ other: Gate) -> Bool {
        switch other {
        case .cx(let control, let target):
            return singleQubitCommutesWithCX(single, control: control, target: target)
        case .ccx(let control1, let control2, let target):
            let qubits = Set([control1, control2, target])
            guard qubits.contains(single.qubit) else { return true }
            return single.isZAxis && qubits.contains(single.qubit)
        default:
            if let otherSingle = other.asSingleQubitGate {
                return commutesSingleQubitPair(single, otherSingle)
            }
            return false
        }
    }

    private static func commutesSingleQubitPair(_ lhs: SingleQubitGate, _ rhs: SingleQubitGate) -> Bool {
        guard lhs.qubit == rhs.qubit else { return false }

        if lhs.isZAxis && rhs.isZAxis {
            return true
        }

        switch (lhs.kind, rhs.kind) {
        case (.x, .x), (.y, .y), (.h, .h):
            return true
        case (.rx, .rx), (.ry, .ry), (.rz, .rz):
            return true
        default:
            return false
        }
    }

    private static func singleQubitCommutesWithCX(
        _ single: SingleQubitGate,
        control: Int,
        target: Int
    ) -> Bool {
        let q = single.qubit
        guard q == control || q == target else { return true }

        if single.isZAxis {
            return true
        }

        if case .x = single.kind, q == target {
            return true
        }

        return false
    }

    private static func commutesCNOTPair(
        control1: Int,
        target1: Int,
        control2: Int,
        target2: Int
    ) -> Bool {
        if control1 == control2 {
            return true
        }

        if target1 == target2 {
            return control1 == control2
        }

        if control1 == target2 && target1 == control2 {
            return false
        }

        if control1 == target2 || control2 == target1 {
            return false
        }

        return Set([control1, target1]).isDisjoint(with: [control2, target2])
    }

    // MARK: - Adjacent folding

    private static func adjacentFoldPass(_ gates: [Gate]) -> [Gate] {
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

    private static func cancelPair(_ lhs: Gate, _ rhs: Gate) -> Gate?? {
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
        case (.ccx(let a1, let a2, let a3), .ccx(let b1, let b2, let b3))
            where a1 == b1 && a2 == b2 && a3 == b3:
            return .some(nil)
        default:
            return nil
        }
    }

    // MARK: - Merge

    private static func mergePair(_ lhs: Gate, _ rhs: Gate) -> Gate?? {
        if let merged = mergeRotationPair(lhs, rhs) {
            return .some(merged)
        }

        if let merged = mergeZAxisPair(lhs, rhs) {
            return .some(merged)
        }

        return nil
    }

    private static func mergeRotationPair(_ lhs: Gate, _ rhs: Gate) -> Gate? {
        switch (lhs, rhs) {
        case (.rx(let t1, let q1), .rx(let t2, let q2)) where q1 == q2:
            return canonicalRotation(axis: .x, angle: t1 + t2, target: q1)
        case (.ry(let t1, let q1), .ry(let t2, let q2)) where q1 == q2:
            return canonicalRotation(axis: .y, angle: t1 + t2, target: q1)
        case (.rz(let t1, let q1), .rz(let t2, let q2)) where q1 == q2:
            return canonicalZRotation(angle: t1 + t2, target: q1)
        default:
            return nil
        }
    }

    private static func mergeZAxisPair(_ lhs: Gate, _ rhs: Gate) -> Gate? {
        guard let left = zAxisAngle(lhs), let right = zAxisAngle(rhs), left.qubit == right.qubit else {
            return nil
        }
        return canonicalZRotation(angle: left.angle + right.angle, target: left.qubit)
    }

    private enum RotationAxis {
        case x, y
    }

    private static func zAxisAngle(_ gate: Gate) -> (qubit: Int, angle: QFloat)? {
        switch gate {
        case .z(let target):
            return (target, QFloat(Double.pi))
        case .s(let target):
            return (target, QFloat(Double.pi / 2.0))
        case .t(let target):
            return (target, QFloat(Double.pi / 4.0))
        case .rz(let theta, let target):
            return (target, theta)
        default:
            return nil
        }
    }

    private static func canonicalRotation(axis: RotationAxis, angle: QFloat, target: Int) -> Gate? {
        let normalized = normalizeAngle(angle)
        guard abs(normalized) > angleTolerance else { return nil }

        switch axis {
        case .x:
            return .rx(theta: normalized, target: target)
        case .y:
            return .ry(theta: normalized, target: target)
        }
    }

    private static func canonicalZRotation(angle: QFloat, target: Int) -> Gate? {
        let normalized = normalizeAngle(angle)
        guard abs(normalized) > angleTolerance else { return nil }

        if anglesEqual(normalized, QFloat(Double.pi / 4.0)) {
            return .t(target: target)
        }
        if anglesEqual(normalized, QFloat(Double.pi / 2.0)) {
            return .s(target: target)
        }
        if anglesEqual(abs(normalized), QFloat(Double.pi)) {
            return .z(target: target)
        }

        return .rz(theta: normalized, target: target)
    }

    private static func normalizeAngle(_ angle: QFloat) -> QFloat {
        var value = angle.truncatingRemainder(dividingBy: twoPi)
        while value <= -QFloat.pi + angleTolerance {
            value += twoPi
        }
        while value > QFloat.pi - angleTolerance {
            value -= twoPi
        }
        return value
    }

    private static func anglesEqual(_ lhs: QFloat, _ rhs: QFloat) -> Bool {
        abs(lhs - rhs) <= angleTolerance
    }
}

// MARK: - Single-qubit gate view

private struct SingleQubitGate {
    enum Kind {
        case h, x, y, z, s, t, rx, ry, rz
    }

    let qubit: Int
    let kind: Kind

    var isZAxis: Bool {
        switch kind {
        case .z, .s, .t, .rz:
            return true
        default:
            return false
        }
    }
}

extension Gate {

    fileprivate var asSingleQubitGate: SingleQubitGate? {
        switch self {
        case .h(let target):
            return SingleQubitGate(qubit: target, kind: .h)
        case .x(let target):
            return SingleQubitGate(qubit: target, kind: .x)
        case .y(let target):
            return SingleQubitGate(qubit: target, kind: .y)
        case .z(let target):
            return SingleQubitGate(qubit: target, kind: .z)
        case .s(let target):
            return SingleQubitGate(qubit: target, kind: .s)
        case .t(let target):
            return SingleQubitGate(qubit: target, kind: .t)
        case .rx(_, let target):
            return SingleQubitGate(qubit: target, kind: .rx)
        case .ry(_, let target):
            return SingleQubitGate(qubit: target, kind: .ry)
        case .rz(_, let target):
            return SingleQubitGate(qubit: target, kind: .rz)
        default:
            return nil
        }
    }
}

extension QuantumCircuit {

    /// Returns an algebraically simplified copy of this circuit.
    public func algebraicallyOptimized() throws -> QuantumCircuit {
        try AlgebraicPreCompiler.optimize(self)
    }
}
