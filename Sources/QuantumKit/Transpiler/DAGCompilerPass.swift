import Foundation

/// A compiler pass that transforms a ``DAGCircuit``.
///
/// Existing ``CompilerPass`` implementations remain gate-list-only. Wrap a DAG pass with
/// ``DAGCompilerPassAdapter`` (or ``PassManager/dag(_:)``) to run it in the gate-list
/// ``PassManager`` pipeline; execution backends still receive a flattened ``QuantumCircuit``.
public protocol DAGCompilerPass: Sendable {
    func run(on dag: DAGCircuit) throws -> DAGCircuit
}

/// Adapts a ``DAGCompilerPass`` into a ``CompilerPass`` via lossless circuit ↔ DAG conversion.
public struct DAGCompilerPassAdapter: CompilerPass {
    public let pass: any DAGCompilerPass

    public init(pass: any DAGCompilerPass) {
        self.pass = pass
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        var dag = try DAGCircuit(circuit: circuit)
        dag = try pass.run(on: dag)
        return try dag.toQuantumCircuit()
    }
}

extension PassManager {
    /// Builds a manager that runs DAG-native passes (each adapted through circuit ↔ DAG).
    public static func dag(_ passes: [any DAGCompilerPass]) -> PassManager {
        PassManager(passes: passes.map { DAGCompilerPassAdapter(pass: $0) })
    }

    /// Runs DAG passes on an existing DAG and returns the transformed DAG (no flatten).
    public static func runDAG(
        _ passes: [any DAGCompilerPass],
        on dag: DAGCircuit
    ) throws -> DAGCircuit {
        var current = dag
        for pass in passes {
            current = try pass.run(on: current)
        }
        return current
    }
}
