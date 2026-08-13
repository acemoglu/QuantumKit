import Foundation

/// Histogram of measurement outcomes from repeated shots.
public struct ShotCounts: Sendable, Equatable {

    public let shots: Int
    public let counts: [Int: Int]

    public init(shots: Int, counts: [Int: Int]) {
        self.shots = shots
        self.counts = counts
    }

    /// Outcome counts keyed by bitstring (MSB-first, e.g. `"01"` for a 2-qubit result).
    public func bitstringCounts(qubitCount: Int) -> [String: Int] {
        bitstringCounts(qubits: Array(0..<qubitCount))
    }

    /// Outcome counts keyed by bitstring in the order of `qubits` (left = first qubit).
    public func bitstringCounts(qubits: [Int]) -> [String: Int] {
        var result: [String: Int] = [:]
        result.reserveCapacity(counts.count)
        for (index, count) in counts {
            result[Self.bitstring(for: index, qubits: qubits)] = count
        }
        return result
    }

    /// Outcome counts keyed by hex strings (`"0x0"`, `"0x1"`, …) for the packed index.
    public func hexCounts(qubitCount: Int) -> [String: Int] {
        var result: [String: Int] = [:]
        result.reserveCapacity(counts.count)
        let width = max((qubitCount + 3) / 4, 1)
        for (index, count) in counts {
            result[String(format: "0x%0\(width)x", index)] = count
        }
        return result
    }

    /// Empirical frequencies from the histogram.
    public func probabilities(qubitCount: Int) -> [String: QFloat] {
        guard shots > 0 else { return [:] }
        let bitstrings = bitstringCounts(qubitCount: qubitCount)
        var result: [String: QFloat] = [:]
        result.reserveCapacity(bitstrings.count)
        for (key, count) in bitstrings {
            result[key] = QFloat(count) / QFloat(shots)
        }
        return result
    }

    private static func bitstring(for index: Int, qubits: [Int]) -> String {
        (0..<qubits.count)
            .reversed()
            .map { position in
                ((index >> position) & 1) == 1 ? "1" : "0"
            }
            .joined()
    }
}
