import Foundation

/// High-level circuit transpilation API.
public struct Transpiler: Sendable {

    public init() {}

    /// Transpiles `circuit` into `targetBasis` using the default basis-translation pass.
    public static func transpile(
        _ circuit: QuantumCircuit,
        targetBasis: BasisGateSet
    ) throws -> QuantumCircuit {
        try transpile(circuit, passes: [BasisTranslatorPass(targetBasis: targetBasis)])
    }

    /// Transpiles `circuit` into the listed basis gate kinds.
    public static func transpile(
        _ circuit: QuantumCircuit,
        targetBasis: BasisGateKind...
    ) throws -> QuantumCircuit {
        try transpile(circuit, targetBasis: BasisGateSet(kinds: Set(targetBasis)))
    }

    /// Runs an explicit compilation pipeline.
    public static func transpile(
        _ circuit: QuantumCircuit,
        passes: [any CompilerPass]
    ) throws -> QuantumCircuit {
        try PassManager(passes: passes).run(on: circuit)
    }
}
