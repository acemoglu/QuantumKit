import Foundation

/// Wraps ``AlgebraicPreCompiler/optimize(_:)`` as a ``CompilerPass``.
public struct AlgebraicOptimizationPass: CompilerPass, Sendable {
    public init() {}

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        var optimized = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for gate in result.gates {
            try optimized.apply(gate)
        }
        return optimized
    }
}
