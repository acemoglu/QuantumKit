import Foundation

/// Sequentially applies an ordered list of compiler passes to a circuit.
public struct PassManager: Sendable {
    public let passes: [any CompilerPass]

    public init(passes: [any CompilerPass]) {
        self.passes = passes
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        var current = circuit
        for pass in passes {
            current = try pass.run(on: current)
        }
        return current
    }
}
