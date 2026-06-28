import Foundation

extension AlgebraicPreCompiler {

    static func commutationSlidePass(_ gates: [Gate]) -> [Gate] {
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

    /// True when swapping the adjacent gates at `leftIndex` and `rightIndex` (`rightIndex ==
    /// leftIndex + 1`) moves same-qubit gates closer together, i.e. strictly lowers the total
    /// same-qubit span Σ_q (maxPosition_q − minPosition_q).
    ///
    /// Rather than rebuilding the whole index map twice (the old O(n) before/after sum, run for
    /// every candidate → O(n³) overall), this computes only the net change. A swap moves at most the
    /// two gates at `leftIndex`/`rightIndex`; multi-qubit gates do not contribute to the span, and a
    /// single-qubit gate shifted by one slot changes its qubit's span by ±1 only when that gate sits
    /// at the qubit's extreme (min/max) position. So the decision needs just each moved qubit's
    /// current min/max positions.
    static func reorderingAdjacentPairReducesSameQubitDistance(
        in gates: [Gate],
        leftIndex: Int,
        rightIndex: Int
    ) -> Bool {
        let leftQubit = gates[leftIndex].asSingleQubitGate?.qubit
        let rightQubit = gates[rightIndex].asSingleQubitGate?.qubit

        // Same qubit (or neither single-qubit): the position set of every qubit is unchanged.
        guard leftQubit != rightQubit else { return false }

        var delta = 0
        if let qubit = leftQubit {
            // This gate slides one slot to the right (leftIndex → rightIndex).
            delta += spanDelta(movingRight: true, extreme: positionExtreme(of: qubit, at: leftIndex, in: gates))
        }
        if let qubit = rightQubit {
            // This gate slides one slot to the left (rightIndex → leftIndex).
            delta += spanDelta(movingRight: false, extreme: positionExtreme(of: qubit, at: rightIndex, in: gates))
        }

        return delta < 0
    }

    enum PositionExtreme {
        case soleOrInterior
        case minimum
        case maximum
    }

    /// Classifies `position` among the positions of `qubit`'s single-qubit gates: whether it is that
    /// qubit's minimum, maximum, or neither (a lone gate or an interior one, both of which leave the
    /// span unchanged when shifted by one slot).
    static func positionExtreme(of qubit: Int, at position: Int, in gates: [Gate]) -> PositionExtreme {
        var minPosition = Int.max
        var maxPosition = Int.min
        var count = 0

        for (index, gate) in gates.enumerated() where gate.asSingleQubitGate?.qubit == qubit {
            count += 1
            if index < minPosition { minPosition = index }
            if index > maxPosition { maxPosition = index }
        }

        guard count > 1 else { return .soleOrInterior }
        if position == minPosition { return .minimum }
        if position == maxPosition { return .maximum }
        return .soleOrInterior
    }

    /// Net change to a qubit's span (max − min) when one of its gates moves a single slot.
    /// Moving the minimum right, or the maximum left, shrinks the span (−1); moving the minimum
    /// left, or the maximum right, grows it (+1); an interior or lone gate leaves it unchanged.
    static func spanDelta(movingRight: Bool, extreme: PositionExtreme) -> Int {
        switch extreme {
        case .soleOrInterior:
            return 0
        case .minimum:
            return movingRight ? -1 : 1
        case .maximum:
            return movingRight ? 1 : -1
        }
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

    static func isBarrier(_ gate: Gate) -> Bool {
        switch gate {
        case .measure, .reset, .c_if:
            // `c_if` depends on classical state produced by an earlier measurement, so gates must
            // never slide across it: the optimizer treats it as an ordering barrier.
            return true
        default:
            return false
        }
    }

    static func commutesOnOverlappingQubits(_ lhs: Gate, _ rhs: Gate) -> Bool {
        if let single = lhs.asSingleQubitGate, commutesSingleQubitWithGate(single, rhs) {
            return true
        }
        if let single = rhs.asSingleQubitGate, commutesSingleQubitWithGate(single, lhs) {
            return true
        }

        switch (lhs, rhs) {
        case (.cx(let c1, let t1), .cx(let c2, let t2)):
            return commutesCNOTPair(control1: c1, target1: t1, control2: c2, target2: t2)
        case (.cz, .cz):
            // CZ gates are diagonal in the computational basis, so any pair commutes.
            return true
        default:
            return false
        }
    }

    static func commutesSingleQubitWithGate(_ single: SingleQubitGate, _ other: Gate) -> Bool {
        switch other {
        case .cx(let control, let target):
            return singleQubitCommutesWithCX(single, control: control, target: target)
        case .cz(let control, let target):
            // CZ is diagonal: only Z-axis single-qubit gates on its qubits commute.
            let qubits = Set([control, target])
            guard qubits.contains(single.qubit) else { return true }
            return single.isZAxis
        case .ccx(let control1, let control2, let target):
            return singleQubitCommutesWithCCX(single, control1: control1, control2: control2, target: target)
        default:
            if let otherSingle = other.asSingleQubitGate {
                return commutesSingleQubitPair(single, otherSingle)
            }
            return false
        }
    }

    static func commutesSingleQubitPair(_ lhs: SingleQubitGate, _ rhs: SingleQubitGate) -> Bool {
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

    static func singleQubitCommutesWithCX(
        _ single: SingleQubitGate,
        control: Int,
        target: Int
    ) -> Bool {
        let q = single.qubit
        guard q == control || q == target else { return true }

        // CX = |0⟩⟨0|⊗I + |1⟩⟨1|⊗X. Diagonal (Z-axis) gates commute through the *control*
        // (they share its |0⟩/|1⟩ eigenbasis), but on the *target* a Z-axis phase sees the bit
        // get flipped by the CX, so Z(target) and CX do NOT commute.
        if single.isZAxis && q == control {
            return true
        }

        // X commutes through the *target* (X·X = X·X under the controlled flip), but X on the
        // control flips which branch fires, so X(control) and CX do NOT commute.
        if case .x = single.kind, q == target {
            return true
        }

        return false
    }

    static func singleQubitCommutesWithCCX(
        _ single: SingleQubitGate,
        control1: Int,
        control2: Int,
        target: Int
    ) -> Bool {
        let q = single.qubit
        guard q == control1 || q == control2 || q == target else { return true }

        // CCX = (I − |11⟩⟨11|)⊗I + |11⟩⟨11|⊗X on (control1, control2, target). Exactly as for CX,
        // diagonal (Z-axis) gates commute through either *control* (they share the controls'
        // |0⟩/|1⟩ eigenbasis), but on the *target* a Z-axis phase sees the bit get flipped by the
        // CCX, so Z(target) and CCX do NOT commute.
        if single.isZAxis && (q == control1 || q == control2) {
            return true
        }

        // X commutes through the *target* (X·X under the doubly-controlled flip), but X on either
        // control flips which branch fires, so X(control) and CCX do NOT commute.
        if case .x = single.kind, q == target {
            return true
        }

        return false
    }

    static func commutesCNOTPair(
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
}
