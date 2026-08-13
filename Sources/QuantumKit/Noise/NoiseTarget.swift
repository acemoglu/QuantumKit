import Foundation

/// Identifies where a ``QuantumChannel`` is applied during circuit execution.
public enum NoiseTarget: Sendable, Equatable, Hashable, Codable {
  /// Apply after any gate touching `qubit`, on that qubit only.
  case qubit(Int)
  /// Apply after any gate of `kind`, on all qubits the gate touches.
  case gate(GateKind)
  /// Apply after a `kind` gate that acts on `qubit`, on that qubit only.
  case gateOnQubit(gate: GateKind, qubit: Int)
  /// Apply after a `kind` gate on exactly these qubits (order-independent).
  case gateOnQubits(gate: GateKind, qubits: [Int])
  /// Apply after every gate that touches `qubit`, on that qubit only.
  case allGatesOnQubit(Int)
  /// Apply after the instruction at this absolute circuit index (C7).
  case circuitIndex(Int)
  /// Spectator crosstalk (C5): after a gate touching `driven`, apply on `spectator`.
  case crosstalk(driven: Int, spectator: Int)
  /// Correlated pair (C5): after a gate touching either qubit, apply on both (order-independent).
  case qubitPair(Int, Int)
}

extension NoiseTarget {

    func matches(gate: Gate, affectedQubits: [Int], gateIndex: Int) -> Bool {
        switch self {
        case .qubit(let qubit):
            return affectedQubits.contains(qubit)

        case .gate(let kind):
            return gate.kind == kind

        case .gateOnQubit(let kind, let qubit):
            return gate.kind == kind && affectedQubits.contains(qubit)

        case .gateOnQubits(let kind, let qubits):
            return gate.kind == kind && Set(affectedQubits) == Set(qubits)

        case .allGatesOnQubit(let qubit):
            return affectedQubits.contains(qubit)

        case .circuitIndex(let index):
            return gateIndex == index

        case .crosstalk(let driven, _):
            return affectedQubits.contains(driven)

        case .qubitPair(let a, let b):
            return affectedQubits.contains(a) || affectedQubits.contains(b)
        }
    }

    /// Qubits that should receive the channel when `matches` is true.
    func applicationQubits(gate: Gate, affectedQubits: [Int]) -> [Int] {
        switch self {
        case .qubit(let qubit), .gateOnQubit(_, let qubit), .allGatesOnQubit(let qubit):
            return affectedQubits.contains(qubit) ? [qubit] : []

        case .gate, .gateOnQubits, .circuitIndex:
            return affectedQubits

        case .crosstalk(_, let spectator):
            return [spectator]

        case .qubitPair(let a, let b):
            return [min(a, b), max(a, b)]
        }
    }
}
