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

    private static func bitstring(for index: Int, qubits: [Int]) -> String {
        (0..<qubits.count)
            .reversed()
            .map { position in
                ((index >> position) & 1) == 1 ? "1" : "0"
            }
            .joined()
    }
}
