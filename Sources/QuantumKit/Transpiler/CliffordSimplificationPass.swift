import Foundation

/// Cancels and folds adjacent Clifford generators (H, S/Sdg, Pauli, CX/CZ/SWAP).
///
/// Not template matching / Solovay–Kitaev — only local Clifford identities:
/// self-inverse pairs, `S·S → Z`, `Sdg·Sdg → Z`. A single left-to-right commutation
/// slide runs so disjoint-qubit gates can meet foldable partners without oscillating.
public struct CliffordSimplificationPass: CompilerPass, Sendable {
    public init() {}

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        var gates = foldToFixedPoint(slideOnce(circuit.gates))
        gates = foldToFixedPoint(gates)

        var output = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for gate in gates {
            try output.apply(gate)
        }
        return output
    }

    private func foldToFixedPoint(_ gates: [Gate]) -> [Gate] {
        var current = gates
        while true {
            let next = foldCliffords(current)
            if next == current { return next }
            current = next
        }
    }

    private func isClifford(_ gate: Gate) -> Bool {
        switch gate {
        case .h, .x, .y, .z, .s, .sdg, .cx, .cz, .swap, .id:
            return true
        default:
            return false
        }
    }

    private func qubits(of gate: Gate) -> Set<Int> {
        Set(gate.affectedQubits)
    }

    private func commute(_ a: Gate, _ b: Gate) -> Bool {
        qubits(of: a).isDisjoint(with: qubits(of: b))
    }

    /// One left-to-right pass: move each Clifford right only while the next gate
    /// is a non-Clifford that commutes (so later Cliffords on the same wires can meet).
    private func slideOnce(_ gates: [Gate]) -> [Gate] {
        var result = gates
        var i = 0
        while i < result.count {
            guard isClifford(result[i]) else {
                i += 1
                continue
            }
            var k = i
            while k + 1 < result.count,
                  commute(result[k], result[k + 1]),
                  !isClifford(result[k + 1]) {
                result.swapAt(k, k + 1)
                k += 1
            }
            i = max(i + 1, k)
        }
        return result
    }

    private func foldCliffords(_ gates: [Gate]) -> [Gate] {
        var result: [Gate] = []
        for gate in gates {
            if let last = result.last, let folded = foldPair(last, gate) {
                result.removeLast()
                result.append(contentsOf: folded)
            } else {
                result.append(gate)
            }
        }
        return result.filter {
            if case .id = $0 { return false }
            return true
        }
    }

    private func foldPair(_ a: Gate, _ b: Gate) -> [Gate]? {
        switch (a, b) {
        case (.h(let t1), .h(let t2)) where t1 == t2:
            return []
        case (.x(let t1), .x(let t2)) where t1 == t2:
            return []
        case (.y(let t1), .y(let t2)) where t1 == t2:
            return []
        case (.z(let t1), .z(let t2)) where t1 == t2:
            return []
        case (.s(let t1), .sdg(let t2)) where t1 == t2,
             (.sdg(let t1), .s(let t2)) where t1 == t2:
            return []
        case (.s(let t1), .s(let t2)) where t1 == t2:
            return [.z(target: t1)]
        case (.sdg(let t1), .sdg(let t2)) where t1 == t2:
            return [.z(target: t1)]
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
