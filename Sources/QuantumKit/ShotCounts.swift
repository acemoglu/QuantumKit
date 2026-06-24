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
        var result: [String: Int] = [:]
        result.reserveCapacity(counts.count)
        for (index, count) in counts {
            result[Self.bitstring(for: index, qubitCount: qubitCount)] = count
        }
        return result
    }

    private static func bitstring(for index: Int, qubitCount: Int) -> String {
        (0..<qubitCount)
            .map { bit in
                ((index >> (qubitCount - 1 - bit)) & 1) == 1 ? "1" : "0"
            }
            .joined()
    }
}
