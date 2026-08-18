import Foundation

/// Axis for single-qubit coherent rotation error channels.
public enum CoherentRotationAxis: String, Sendable, Equatable, Codable, CaseIterable {
    case x, y, z
}

public enum QuantumChannelError: Error, Equatable {
    case emptyKrausSet
    case invalidKrausDimension(operatorIndex: Int, count: Int)
    case pauliProbabilitiesExceedOne(sum: QFloat)
    /// Process matrix must be length 16 (row-major 4×4) for 1-qubit import.
    case invalidProcessMatrixDimension(count: Int, expected: Int)
    /// Only 1-qubit Choi / superoperator import is supported (no multi-qubit Kraus import kernel).
    case multiQubitProcessMatrixUnsupported(qubitCount: Int)
    /// Choi failed the practical hermiticity check `|H - H†|∞ > tol`.
    case choiNotHermitian(maxDeviation: QFloat)
    /// Choi had an eigenvalue below `-eigenvalueTolerance` (not positive semidefinite within tol).
    case choiNotPositiveSemidefinite(minimumEigenvalue: QFloat)
}

/// A physical noise channel attachable to specific gates or qubits.
///
/// ## Extension point
///
/// Build channels with the static helpers / cases below, then attach via
/// ``NoiseModel/adding(_:for:)`` (or composition helpers). Custom *channel builder*
/// registries are deferred — named discovery shipped for ``CompilerPass`` via
/// ``CompilerPassRegistry``.
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
    /// Thermal relaxation during idle time. Strength is computed from the matched
    /// ``Gate/delay`` duration (C8); ignored on non-delay instructions.
    case idleThermalRelaxation(t1: QFloat, t2: QFloat)
    /// Correlated two-qubit Pauli error (C5): with probability `probability` apply `P⊗P`
    /// on the application pair; otherwise identity. Requires exactly two application qubits.
    case correlatedPauli(axis: CoherentRotationAxis, probability: QFloat)
    /// Coherent always-on ZZ crosstalk (C5): apply ``Gate/rzz`` with the given angle
    /// on the application pair. Requires exactly two application qubits.
    case correlatedZZ(angle: QFloat)
    /// Arbitrary single-qubit Kraus channel (C2).
    /// Each operator is a length-4 row-major 2×2 complex matrix.
    /// Callers are responsible for supplying a valid (e.g. CPTP) set — this case does not validate completeness.
    case kraus1Q(operators: [[ComplexAmplitude]])
    /// General single-qubit Pauli channel (C2):
    /// `(1-px-py-pz)ρ + px XρX + py YρY + pz ZρZ` with `px+py+pz ≤ 1`.
    case pauliChannel(px: QFloat, py: QFloat, pz: QFloat)
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
        case .idleThermalRelaxation(let t1, let t2):
            return (t1 > 0 || t2 > 0) ? 1 : 0
        case .correlatedPauli(_, let p):
            return p
        case .correlatedZZ(let angle):
            return abs(angle) > 0 ? 1 : 0
        case .kraus1Q(let operators):
            return operators.isEmpty ? 0 : 1
        case .pauliChannel(let px, let py, let pz):
            return px + py + pz
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
        case .idleThermalRelaxation(let t1, let t2):
            return t1 > 0 || t2 > 0
        case .correlatedPauli(_, let p):
            return p > 0
        case .correlatedZZ(let angle):
            return abs(angle) > 0
        case .kraus1Q(let operators):
            return !operators.isEmpty
        case .pauliChannel(let px, let py, let pz):
            return px + py + pz > 0
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

    /// Validates shape and wraps a 1Q Kraus set (each operator: 4 complex entries, row-major).
    /// Does not prove the set is CPTP.
    public static func fromKraus1Q(_ operators: [[ComplexAmplitude]]) throws -> QuantumChannel {
        guard !operators.isEmpty else { throw QuantumChannelError.emptyKrausSet }
        for (index, op) in operators.enumerated() where op.count != 4 {
            throw QuantumChannelError.invalidKrausDimension(operatorIndex: index, count: op.count)
        }
        return .kraus1Q(operators: operators)
    }

    /// Pauli channel with probabilities `px, py, pz` (sum must be ≤ 1).
    public static func makePauliChannel(px: QFloat, py: QFloat, pz: QFloat) throws -> QuantumChannel {
        let cx = min(max(px, 0), 1)
        let cy = min(max(py, 0), 1)
        let cz = min(max(pz, 0), 1)
        let sum = cx + cy + cz
        guard sum <= 1 + 1e-6 else {
            throw QuantumChannelError.pauliProbabilitiesExceedOne(sum: sum)
        }
        return .pauliChannel(px: cx, py: cy, pz: min(cz, max(0, 1 - cx - cy)))
    }

    /// Pauli–Lindblad rates → Pauli channel over duration `t` (C2):
    /// `p_a = (1 - exp(-2 λ_a t)) / 2`.
    public static func fromPauliLindblad(
        lambdaX: QFloat = 0,
        lambdaY: QFloat = 0,
        lambdaZ: QFloat = 0,
        duration: QFloat
    ) throws -> QuantumChannel {
        func rateToProbability(_ lambda: QFloat) -> QFloat {
            guard lambda > 0, duration > 0 else { return 0 }
            return min(max((1 - exp(-2 * lambda * duration)) / 2, 0), 1)
        }
        return try makePauliChannel(
            px: rateToProbability(lambdaX),
            py: rateToProbability(lambdaY),
            pz: rateToProbability(lambdaZ)
        )
    }
}
