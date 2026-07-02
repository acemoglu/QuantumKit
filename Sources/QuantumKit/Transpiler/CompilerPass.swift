import Foundation

/// A single transformation applied to a ``QuantumCircuit`` during compilation.
public protocol CompilerPass: Sendable {
    func run(on circuit: QuantumCircuit) throws -> QuantumCircuit
}
