import Foundation

/// DAG-native pass: removes idle identity ops.
///
/// Drops ``Gate/id``, splicing dependency edges so remaining ops stay ordered.
/// ``Gate/barrier`` (including empty = full-width) and ``Gate/delay`` are kept as
/// scheduling / ordering markers. Metadata on removed nodes is discarded; survivors
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
