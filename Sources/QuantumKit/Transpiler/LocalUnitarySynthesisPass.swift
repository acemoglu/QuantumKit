import Foundation

/// Merges adjacent single-qubit rotations on the same wire and cancels trivial 2q pairs.
///
/// Safe local resynthesis only — not Solovay–Kitaev / approximate template matching.
/// Consecutive `RZ` angles are summed; `SX·SX → X`; identical back-to-back `CX`/`CZ` cancel.
public struct LocalUnitarySynthesisPass: CompilerPass, Sendable {
    private let angleTolerance: QFloat

    public init(angleTolerance: QFloat = 1e-9) {
        self.angleTolerance = angleTolerance
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        var gates = circuit.gates
        var changed = true
        while changed {
            changed = false
            let next = foldOnce(gates)
            if next != gates {
                gates = next
                changed = true
            }
        }

        var output = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for gate in gates {
            try output.apply(gate)
        }
        return output
    }

    private func foldOnce(_ gates: [Gate]) -> [Gate] {
        var result: [Gate] = []
        var index = 0
        while index < gates.count {
            let gate = gates[index]
            if index + 1 < gates.count, let merged = merge(gate, gates[index + 1]) {
                result.append(contentsOf: merged)
                index += 2
                continue
            }
            result.append(gate)
            index += 1
        }
        return result.filter { gate in
            if case .id = gate { return false }
            if case .rz(let theta, _) = gate, let value = try? theta.requireLiteral(), abs(value) <= angleTolerance {
                return false
            }
            return true
        }
    }

    private func merge(_ a: Gate, _ b: Gate) -> [Gate]? {
        switch (a, b) {
        case (.rz(let t1, let q1), .rz(let t2, let q2)) where q1 == q2:
            guard let a1 = t1.literalValue, let a2 = t2.literalValue else { return nil }
            return [.rz(theta: QFloatExpr(a1 + a2), target: q1)]
        case (.p(let t1, let q1), .p(let t2, let q2)) where q1 == q2:
            guard let a1 = t1.literalValue, let a2 = t2.literalValue else { return nil }
            return [.p(theta: QFloatExpr(a1 + a2), target: q1)]
        case (.sx(let q1), .sx(let q2)) where q1 == q2:
            return [.x(target: q1)]
        case (.sxdg(let q1), .sxdg(let q2)) where q1 == q2:
            return [.x(target: q1)]
        case (.x(let q1), .x(let q2)) where q1 == q2:
            return []
        case (.h(let q1), .h(let q2)) where q1 == q2:
            return []
        case (.cx(let c1, let t1), .cx(let c2, let t2)) where c1 == c2 && t1 == t2:
            return []
        case (.cz(let c1, let t1), .cz(let c2, let t2))
            where (c1 == c2 && t1 == t2) || (c1 == t2 && t1 == c2):
            return []
        case (.swap(let a1, let a2), .swap(let b1, let b2))
            where Set([a1, a2]) == Set([b1, b2]):
            return []
        default:
            return nil
        }
    }
}
