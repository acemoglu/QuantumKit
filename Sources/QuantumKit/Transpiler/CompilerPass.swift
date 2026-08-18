import Foundation

/// A single transformation applied to a ``QuantumCircuit`` during compilation.
///
/// ## Extension point
///
/// Implement ``CompilerPass`` (or ``DAGCompilerPass`` + ``DAGCompilerPassAdapter``) and
/// hand instances to ``PassManager``. For named discovery without changing existing
/// pipelines, register a ``CompilerPassFactory`` on ``CompilerPassRegistry`` and resolve
/// with ``CompilerPassRegistry/makePassManager(ids:)``.
///
/// Related extension surfaces (no marketplace / dylib loading):
/// - Noise: ``QuantumChannel`` + ``NoiseModel/adding(_:for:)``
/// - Simulation: ``QuantumBackend`` + ``QuantumBackendFactory``
public protocol CompilerPass: Sendable {
    func run(on circuit: QuantumCircuit) throws -> QuantumCircuit
}
