import Foundation

/// DAG-native pass: removes idle identity ops and empty barriers.
///
/// Drops ``Gate/id`` and ``Gate/barrier`` with an empty qubit list, splicing dependency
/// edges so remaining ops stay ordered. Non-empty barriers and ``Gate/delay`` are kept
/// (scheduling / ordering markers). Metadata on removed nodes is discarded; survivors
/// keep theirs (see ``InstructionMetadata``).
public struct IdleIdentityRemovalPass: DAGCompilerPass, CompilerPass {
    public init() {}

    public func run(on dag: DAGCircuit) throws -> DAGCircuit {
        var working = dag
        let removable = working.nodes.values
            .filter { DAGCircuit.isIdleIdentity($0.gate) }
            .map(\.id)
        for id in removable {
            try working.removeNode(id)
        }
        return working
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        try DAGCompilerPassAdapter(pass: self).run(on: circuit)
    }
}
