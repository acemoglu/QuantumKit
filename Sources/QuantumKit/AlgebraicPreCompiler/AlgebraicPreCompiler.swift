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

    static let angleTolerance: QFloat = 1e-5
    static let twoPi = QFloat(2 * Double.pi)

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
}
