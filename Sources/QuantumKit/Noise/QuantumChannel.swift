import Foundation

/// Axis for single-qubit coherent rotation error channels (C6).
public enum CoherentRotationAxis: String, Sendable, Equatable, Codable, CaseIterable {
    case x, y, z
}

/// A physical noise channel attachable to specific gates or qubits.
public enum QuantumChannel: Sendable, Equatable, Codable {
    case depolarizing(probability: QFloat)
    case amplitudeDamping(probability: QFloat)
    case phaseDamping(probability: QFloat)
    case pauliXFlip(probability: QFloat)
    case pauliYFlip(probability: QFloat)
    case pauliZFlip(probability: QFloat)
    /// Deterministic coherent over-rotation: apply `R_axis(angle)` on matched qubits.
    case coherentOverRotation(axis: CoherentRotationAxis, angle: QFloat)
    /// Stochastic coherent error: with probability `probability`, apply `R_axis(angle)`;
    /// otherwise identity. Exact DM realization is `(1-p)ρ + p UρU†`.
    case coherentUnitaryError(axis: CoherentRotationAxis, angle: QFloat, probability: QFloat)
}

extension QuantumChannel {

    public var probability: QFloat {
        switch self {
        case .depolarizing(let p), .amplitudeDamping(let p), .phaseDamping(let p),
             .pauliXFlip(let p), .pauliYFlip(let p), .pauliZFlip(let p):
            return p
        case .coherentOverRotation:
            return 1
        case .coherentUnitaryError(_, _, let p):
            return p
        }
    }

    public var isGateChannel: Bool {
        switch self {
        case .depolarizing, .amplitudeDamping, .phaseDamping,
             .pauliXFlip, .pauliYFlip, .pauliZFlip:
            return probability > 0
        case .coherentOverRotation(_, let angle):
            return abs(angle) > 0
        case .coherentUnitaryError(_, let angle, let p):
            return p > 0 && abs(angle) > 0
        }
    }

    /// Amplitude damping probability from T1 and gate duration.
    public static func amplitudeDamping(t1: QFloat, gateTime: QFloat) -> QuantumChannel {
        guard t1 > 0, gateTime > 0 else { return .amplitudeDamping(probability: 0) }
        return .amplitudeDamping(probability: min(max(1 - exp(-gateTime / t1), 0), 1))
    }

    /// Pure-dephasing strength derived from T1/T2 and gate duration (IBM-style).
    public static func phaseDamping(t1: QFloat, t2: QFloat, gateTime: QFloat) -> QuantumChannel {
        guard t2 > 0, gateTime > 0 else { return .phaseDamping(probability: 0) }

        let inverseT2 = 1.0 / Double(t2)
        let inversePureDephasing = t1 > 0
            ? inverseT2 - 1.0 / (2.0 * Double(t1))
            : inverseT2
        guard inversePureDephasing > 0 else { return .phaseDamping(probability: 0) }

        let lambda = 1.0 - exp(-2.0 * Double(gateTime) * inversePureDephasing)
        return .phaseDamping(probability: QFloat(min(max(lambda, 0), 1)))
    }

    /// Imperfect |0⟩ / computational preparation: bit-flip with the given probability (C9).
    public static func preparationBitFlip(probability: QFloat) -> QuantumChannel {
        .pauliXFlip(probability: probability)
    }
}
