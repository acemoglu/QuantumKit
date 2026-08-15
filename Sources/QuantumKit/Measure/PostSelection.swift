import Foundation

/// Predicate on **final qubit** bits of a shot outcome histogram.
///
/// Evaluation uses ``QubitBitOrdering/engineLSB`` packing: qubit `k` is bit `k` of the
/// integer ``ShotCounts/counts`` key (same as Metal/CPU histogram indices).
///
/// ## Not classical-register conditioning
/// ``ShotCounts`` from ``Sampler`` / backends are terminal **qubit** Z-basis outcomes
/// (after any mid-circuit projective measures that collapsed the state). This API does
/// **not** read ``ClassicalMemory`` / creg values. After `measure(q→creg)` followed by
/// gates on `q`, the creg bit and final qubit bit can diverge — filtering `.bit(q, …)`
/// follows the **qubit**, not the stored classical bit.
public indirect enum ClassicalBitPredicate: Sendable, Equatable {
    /// Require final qubit `qubit`'s bit equal `value` (`0` or `1`).
    case bitEquals(qubit: Int, value: Int)
    /// All conjuncts must hold. An empty list is vacuously `true`.
    case and([ClassicalBitPredicate])
    /// At least one disjunct must hold. An empty list is vacuously `false`
    /// (always yields an empty keep-set when validated against a finite histogram).
    case or([ClassicalBitPredicate])
    case not(ClassicalBitPredicate)

    /// Convenience: ancilla / data bit equals zero.
    public static func bit(_ qubit: Int, equals value: Int) -> ClassicalBitPredicate {
        .bitEquals(qubit: qubit, value: value)
    }

    /// Whether `outcome` (``engineLSB`` index) satisfies this predicate.
    public func evaluates(toTrueForOutcome outcome: Int) -> Bool {
        switch self {
        case .bitEquals(let qubit, let value):
            return ((outcome >> qubit) & 1) == value
        case .and(let parts):
            return parts.allSatisfy { $0.evaluates(toTrueForOutcome: outcome) }
        case .or(let parts):
            return parts.contains { $0.evaluates(toTrueForOutcome: outcome) }
        case .not(let inner):
            return !inner.evaluates(toTrueForOutcome: outcome)
        }
    }

    /// Validates qubit ranges and bit values against `qubitCount`.
    public func validate(qubitCount: Int) throws {
        switch self {
        case .bitEquals(let qubit, let value):
            guard value == 0 || value == 1 else {
                throw PostSelectionError.invalidBitValue(value)
            }
            guard qubit >= 0, qubit < qubitCount else {
                throw PostSelectionError.qubitOutOfRange(qubit: qubit, qubitCount: qubitCount)
            }
        case .and(let parts), .or(let parts):
            for part in parts {
                try part.validate(qubitCount: qubitCount)
            }
        case .not(let inner):
            try inner.validate(qubitCount: qubitCount)
        }
    }
}

public enum PostSelectionError: Error, Equatable, Sendable {
    case invalidBitValue(Int)
    case qubitOutOfRange(qubit: Int, qubitCount: Int)
    /// No shots satisfied the predicate (`discardedShots` is the original shot budget).
    case emptyKeepSet(discardedShots: Int)
    /// ``SamplerResult`` had no ``SamplerResult/shotCounts`` (exact Born path).
    case missingShotCounts
}

/// Behavior when filtering removes every shot.
public enum EmptyKeepSetPolicy: Sendable, Equatable {
    /// Throw ``PostSelectionError/emptyKeepSet(discardedShots:)``.
    case throwError
    /// Return `acceptedShots == 0`, empty counts, `acceptanceFraction == 0`.
    case emptyResult
}

/// Filtered histogram plus acceptance metadata.
public struct PostSelectionResult: Sendable, Equatable {
    /// Renormalized histogram: ``ShotCounts/shots`` equals ``acceptedShots``.
    public let shotCounts: ShotCounts
    public let qubitCount: Int
    public let acceptedShots: Int
    public let discardedShots: Int
    /// `acceptedShots / originalShots` (0 when the original budget was 0).
    public let acceptanceFraction: QFloat
    /// Empirical frequencies on the **accepted** ensemble (``bitstringMSB`` keys).
    public let quasiProbabilities: [String: QFloat]

    public init(
        shotCounts: ShotCounts,
        qubitCount: Int,
        acceptedShots: Int,
        discardedShots: Int,
        acceptanceFraction: QFloat,
        quasiProbabilities: [String: QFloat]
    ) {
        self.shotCounts = shotCounts
        self.qubitCount = qubitCount
        self.acceptedShots = acceptedShots
        self.discardedShots = discardedShots
        self.acceptanceFraction = acceptanceFraction
        self.quasiProbabilities = quasiProbabilities
    }
}

/// Post-selection / conditioning on shot histograms and ``SamplerResult``s.
///
/// Does not change circuit execution or ``ShotExecutionPolicy`` (mid-circuit projective
/// measure / `c_if` remain ``mustSerial``). This API only filters already-collected
/// **qubit-outcome** counts — see ``ClassicalBitPredicate`` for the creg vs qubit caveat.
public enum PostSelection {

    /// Keep outcomes matching `predicate`; renormalize counts and probabilities.
    public static func filter(
        _ counts: ShotCounts,
        qubitCount: Int,
        where predicate: ClassicalBitPredicate,
        emptyKeepSet: EmptyKeepSetPolicy = .throwError
    ) throws -> PostSelectionResult {
        try predicate.validate(qubitCount: qubitCount)

        var kept: [Int: Int] = [:]
        var accepted = 0
        for (outcome, count) in counts.counts {
            guard count > 0 else { continue }
            if predicate.evaluates(toTrueForOutcome: outcome) {
                kept[outcome, default: 0] += count
                accepted += count
            }
        }

        let originalShots = counts.shots
        let discarded = max(0, originalShots - accepted)
        let fraction: QFloat = originalShots > 0
            ? QFloat(accepted) / QFloat(originalShots)
            : 0

        if accepted == 0 {
            switch emptyKeepSet {
            case .throwError:
                throw PostSelectionError.emptyKeepSet(discardedShots: originalShots)
            case .emptyResult:
                return PostSelectionResult(
                    shotCounts: ShotCounts(shots: 0, counts: [:]),
                    qubitCount: qubitCount,
                    acceptedShots: 0,
                    discardedShots: discarded,
                    acceptanceFraction: fraction,
                    quasiProbabilities: [:]
                )
            }
        }

        let filtered = ShotCounts(shots: accepted, counts: kept)
        return PostSelectionResult(
            shotCounts: filtered,
            qubitCount: qubitCount,
            acceptedShots: accepted,
            discardedShots: discarded,
            acceptanceFraction: fraction,
            quasiProbabilities: filtered.probabilities(qubitCount: qubitCount)
        )
    }

    /// Filter a shot-based ``SamplerResult`` (uses ``SamplerResult/qubitCount``).
    public static func filter(
        _ result: SamplerResult,
        where predicate: ClassicalBitPredicate,
        emptyKeepSet: EmptyKeepSetPolicy = .throwError
    ) throws -> PostSelectionResult {
        guard let counts = result.shotCounts else {
            throw PostSelectionError.missingShotCounts
        }
        return try filter(
            counts,
            qubitCount: result.qubitCount,
            where: predicate,
            emptyKeepSet: emptyKeepSet
        )
    }

    /// Pauli expectation from a (possibly post-selected) computational-basis histogram.
    ///
    /// Same parity convention as shot-based ``Estimator``: non-`Z` factors must already have
    /// been rotated into the Z basis before sampling. Empty / identity support → `1`.
    public static func pauliExpectation(
        from counts: ShotCounts,
        term: PauliTerm
    ) -> QFloat {
        let qubits = term.paulis.keys.filter { term.paulis[$0] != .i }.sorted()
        guard !qubits.isEmpty, counts.shots > 0 else { return 1 }

        var sum: QFloat = 0
        for (outcome, count) in counts.counts {
            var parity = 0
            for qubit in qubits {
                parity ^= (outcome >> qubit) & 1
            }
            sum += (parity == 0 ? QFloat(1) : QFloat(-1)) * QFloat(count)
        }
        return sum / QFloat(counts.shots)
    }

    /// Σ cᵢ ⟨Pᵢ⟩ on the accepted ensemble of a ``PostSelectionResult``.
    public static func expectation(
        from result: PostSelectionResult,
        hamiltonian: Hamiltonian
    ) throws -> QFloat {
        guard result.acceptedShots > 0 else {
            throw PostSelectionError.emptyKeepSet(discardedShots: result.discardedShots)
        }
        var total: QFloat = 0
        for term in hamiltonian.terms {
            let support = term.paulis.keys.filter { term.paulis[$0] != .i }
            for qubit in support where qubit < 0 || qubit >= result.qubitCount {
                throw PostSelectionError.qubitOutOfRange(qubit: qubit, qubitCount: result.qubitCount)
            }
            total += term.coefficient * pauliExpectation(from: result.shotCounts, term: term)
        }
        return total
    }
}
