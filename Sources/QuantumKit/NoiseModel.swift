import Foundation

/// Stochastic noise applied after each unitary gate during execution.
public struct NoiseModel: Sendable, Equatable {

    /// Per-qubit depolarizing probability `p` in `[0, 1]`.
    /// After a gate, each affected qubit independently gets a random Pauli (X/Y/Z) with probability `p`.
    public var depolarizingProbability: QFloat

    public init(depolarizingProbability: QFloat = 0) {
        self.depolarizingProbability = min(max(depolarizingProbability, 0), 1)
    }

    public var appliesDepolarizing: Bool {
        depolarizingProbability > 0
    }
}
