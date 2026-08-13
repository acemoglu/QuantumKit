import Foundation

/// Expands gates that act on more than two qubits into 1q/2q sequences.
///
/// Routing only understands pairwise connectivity, so Toffoli / MCX / MCZ must be
/// unrolled before ``BasicSwapRoutingPass``. Measurement, reset, and already 1q/2q
/// gates are left unchanged.
public struct UnrollMultiQubitPass: CompilerPass, Sendable {
    public init() {}

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        var output = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )

        for gate in circuit.gates {
            for replacement in try Self.expandUntilTwoQubit(gate) {
                try output.apply(replacement)
            }
        }

        return output
    }

    static func expandUntilTwoQubit(_ gate: Gate) throws -> [Gate] {
        switch gate {
        case .measure, .reset, .initialize, .unitary1, .barrier, .delay, .id:
            return [gate]

        case .c_if(let classicalRegister, let expectedValue, let inner):
            let expandedInner = try expandUntilTwoQubit(inner)
            if expandedInner.count == 1, expandedInner[0] == inner {
                return [gate]
            }
            return expandedInner.map { replacement in
                .c_if(
                    classicalRegister: classicalRegister,
                    expectedValue: expectedValue,
                    gate: replacement
                )
            }

        case .customUnitary(_, let qubits) where qubits.count > 2:
            throw TranspilerError.routingRequiresTwoQubitGates(gate)

        case .ccx, .mcx, .mcz, .cswap:
            break

        default:
            if gate.affectedQubits.count <= 2 {
                return [gate]
            }
            throw TranspilerError.routingRequiresTwoQubitGates(gate)
        }

        var result: [Gate] = []
        for piece in try GateDecomposition.expand(gate) {
            result.append(contentsOf: try expandUntilTwoQubit(piece))
        }
        return result
    }
}
