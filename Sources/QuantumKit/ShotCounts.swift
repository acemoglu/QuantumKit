import Foundation

/// Histogram of measurement outcomes from repeated shots.
///
/// Integer keys in ``counts`` use ``QubitBitOrdering/engineLSB`` (qubit `k` = index bit `k`).
/// ``bitstringCounts`` defaults to ``QubitBitOrdering/bitstringMSB`` for display keys.
public struct ShotCounts: Sendable, Equatable {

    public let shots: Int
    /// Engine-native outcome index → count (``QubitBitOrdering/engineLSB``).
    public let counts: [Int: Int]

    public init(shots: Int, counts: [Int: Int]) {
        self.shots = shots
        self.counts = counts
    }

    /// Outcome counts keyed by bitstring (``QubitBitOrdering/bitstringMSB``).
    ///
    /// Example: index `1` (qubit 0 excited) → `"01"` on 2 qubits.
    public func bitstringCounts(qubitCount: Int) -> [String: Int] {
        bitstringCounts(qubitCount: qubitCount, ordering: .bitstringDefault)
    }

    /// Outcome counts keyed by bitstring under an explicit ``QubitBitOrdering``.
    public func bitstringCounts(qubitCount: Int, ordering: QubitBitOrdering) -> [String: Int] {
        var result: [String: Int] = [:]
        result.reserveCapacity(counts.count)
        for (index, count) in counts {
            let key = (try? ordering.bitstring(forIndex: index, qubitCount: qubitCount))
                ?? QubitBitOrdering.bitstringDefaultFallback(index: index, qubitCount: qubitCount)
            result[key, default: 0] += count
        }
        return result
    }

    /// Outcome counts keyed by bitstring in historical packed-index order.
    ///
    /// **Unsafe / historical:** this overload does **not** permute by the listed qubit
    /// identities. It formats the low `qubits.count` bits of each integer key as
    /// ``QubitBitOrdering/bitstringMSB`` (left = high bit of that packed index).
    /// Colliding keys are summed. Prefer ``bitstringCounts(qubitCount:ordering:)`` when
    /// the policy must be named. For `qubits == 0..<n` this matches ``bitstringMSB`` on a
    /// full-register ``engineLSB`` index.
    public func bitstringCounts(qubits: [Int]) -> [String: Int] {
        var result: [String: Int] = [:]
        result.reserveCapacity(counts.count)
        for (index, count) in counts {
            let key = Self.bitstring(for: index, qubits: qubits)
            result[key, default: 0] += count
        }
        return result
    }

    /// Outcome counts keyed by hex strings (`"0x0"`, `"0x1"`, …) for the packed
    /// ``QubitBitOrdering/engineLSB`` index.
    public func hexCounts(qubitCount: Int) -> [String: Int] {
        var result: [String: Int] = [:]
        result.reserveCapacity(counts.count)
        let width = max((qubitCount + 3) / 4, 1)
        for (index, count) in counts {
            result[String(format: "0x%0\(width)x", index)] = count
        }
        return result
    }

    /// Empirical frequencies from the histogram (``bitstringMSB`` keys by default).
    public func probabilities(qubitCount: Int) -> [String: QFloat] {
        probabilities(qubitCount: qubitCount, ordering: .bitstringDefault)
    }

    public func probabilities(qubitCount: Int, ordering: QubitBitOrdering) -> [String: QFloat] {
        guard shots > 0 else { return [:] }
        let bitstrings = bitstringCounts(qubitCount: qubitCount, ordering: ordering)
        var result: [String: QFloat] = [:]
        result.reserveCapacity(bitstrings.count)
        for (key, count) in bitstrings {
            result[key] = QFloat(count) / QFloat(shots)
        }
        return result
    }

    private static func bitstring(for index: Int, qubits: [Int]) -> String {
        // Historical subset helper: width = qubits.count, MSB-first over low `width` bits.
        (try? QubitBitOrdering.bitstringMSB.bitstring(forIndex: index & ((1 << qubits.count) - 1), qubitCount: qubits.count))
            ?? ""
    }
}

extension QubitBitOrdering {
    /// Non-throwing fallback for histogram formatting when index is somehow out of range.
    fileprivate static func bitstringDefaultFallback(index: Int, qubitCount: Int) -> String {
        let masked = index & ((1 << qubitCount) - 1)
        return (try? bitstringMSB.bitstring(forIndex: masked, qubitCount: qubitCount))
            ?? String(repeating: "0", count: qubitCount)
    }
}
