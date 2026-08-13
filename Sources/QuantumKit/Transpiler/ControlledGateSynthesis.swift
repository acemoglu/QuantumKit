import Foundation

/// Strategy for expanding multi-controlled X / related controlled gates during unroll.
///
/// Default ``ancillaFree`` preserves the historical Barenco demultiplexing path.
public enum ControlledGateSynthesisStrategy: String, Sendable, Equatable, Codable, CaseIterable {
    /// Ancilla-free recursive multi-controlled phase demultiplexing (default).
    case ancillaFree
    /// Linear-cost V-chain using compiler-allocated scratch qubits (clean |0⟩ ancillas).
    /// Relative gate-count cost vs ``ancillaFree`` improves for larger control counts when
    /// ``TranspileOptions/enableAncillaAllocation`` is on.
    case vChainAncilla
}
