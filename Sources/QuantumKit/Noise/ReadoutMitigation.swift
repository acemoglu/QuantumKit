import Foundation

public enum ReadoutMitigationError: Error, Equatable, Sendable {
    case qubitCountMismatch(histogram: Int, matrix: Int)
    case singularConfusionMatrix
    /// Inverse (after clamping negatives) produced no positive prepared mass.
    case nonPositivePreparedMass
}

/// Host-side readout assignment mitigation for ``ShotCounts``.
///
/// Given confusion `C[prepared][measured] = P(measured|prepared)`, observed counts satisfy
/// `n_m = Σ_p N_p C[p][m]`. This solves for prepared counts `N` via `A N = n` with
/// `A[m][p] = C[p][m]`, then clamps negatives and renormalizes to the original shot budget.
///
/// Not a full mitigation suite — no twirling, ZNE, or PEC (those live on ``ResilienceOptions``).
public enum ReadoutMitigation {

    /// Apply inverse readout correction. Returns a new histogram with the same ``ShotCounts/shots``
    /// and `Σ counts == shots` (when `shots > 0`).
    public static func apply(
        to counts: ShotCounts,
        matrix: ReadoutConfusionMatrix,
        qubitCount: Int
    ) throws -> ShotCounts {
        guard matrix.qubitCount == qubitCount else {
            throw ReadoutMitigationError.qubitCountMismatch(
                histogram: qubitCount,
                matrix: matrix.qubitCount
            )
        }

        let dim = 1 << qubitCount
        var measured = Array(repeating: 0.0, count: dim)
        for (outcome, count) in counts.counts {
            let index = outcome & (dim - 1)
            measured[index] += Double(count)
        }

        // A[m][p] = C[p][m]
        var a = Array(repeating: Array(repeating: 0.0, count: dim), count: dim)
        for p in 0..<dim {
            for m in 0..<dim {
                a[m][p] = Double(matrix.probabilities[p][m])
            }
        }

        let prepared = try solveLinearSystem(a, measured)
        let clipped = prepared.map { max($0, 0.0) }
        let clippedSum = clipped.reduce(0, +)
        let targetShots = max(counts.shots, 0)

        guard targetShots > 0 else {
            return ShotCounts(shots: 0, counts: [:])
        }

        if clippedSum <= 0 {
            throw ReadoutMitigationError.nonPositivePreparedMass
        }

        let scale = Double(targetShots) / clippedSum
        let weights = clipped.map { $0 * scale }
        let integerCounts = largestRemainderIntegers(weights, total: targetShots)
        return ShotCounts(shots: targetShots, counts: integerCounts)
    }

    /// Largest-remainder method: floors that sum ≤ total, then award leftover to largest fractions.
    private static func largestRemainderIntegers(_ weights: [Double], total: Int) -> [Int: Int] {
        precondition(total >= 0)
        guard total > 0, !weights.isEmpty else { return [:] }

        var floors = weights.map { Int(floor($0)) }
        var assigned = floors.reduce(0, +)
        var remainder = total - assigned

        if remainder < 0 {
            // Numerical overshoot: trim from the largest bins.
            var order = floors.indices.sorted { floors[$0] > floors[$1] }
            var idx = 0
            while remainder < 0, idx < order.count {
                let i = order[idx]
                let take = min(floors[i], -remainder)
                floors[i] -= take
                remainder += take
                idx += 1
            }
            assigned = floors.reduce(0, +)
            remainder = total - assigned
        }

        let fractions: [(Int, Double)] = weights.enumerated().map { index, value in
            (index, value - floor(value))
        }
        let byFraction = fractions.sorted { lhs, rhs in
            if lhs.1 != rhs.1 { return lhs.1 > rhs.1 }
            return lhs.0 < rhs.0
        }
        var cursor = 0
        while remainder > 0, cursor < byFraction.count {
            floors[byFraction[cursor].0] += 1
            remainder -= 1
            cursor += 1
        }
        // If still short (empty fractions), dump into bin 0.
        if remainder > 0 {
            floors[0] += remainder
        }

        var result: [Int: Int] = [:]
        for (index, count) in floors.enumerated() where count > 0 {
            result[index] = count
        }
        return result
    }

    /// Gaussian elimination with partial pivoting. Solves `A x = b`.
    private static func solveLinearSystem(_ matrix: [[Double]], _ rhs: [Double]) throws -> [Double] {
        let n = rhs.count
        var aug = matrix
        var b = rhs

        for col in 0..<n {
            var pivotRow = col
            var pivotVal = abs(aug[col][col])
            for row in (col + 1)..<n {
                let candidate = abs(aug[row][col])
                if candidate > pivotVal {
                    pivotVal = candidate
                    pivotRow = row
                }
            }
            guard pivotVal > 1e-14 else {
                throw ReadoutMitigationError.singularConfusionMatrix
            }
            if pivotRow != col {
                aug.swapAt(pivotRow, col)
                b.swapAt(pivotRow, col)
            }

            let pivot = aug[col][col]
            for row in (col + 1)..<n {
                let factor = aug[row][col] / pivot
                if factor == 0 { continue }
                for k in col..<n {
                    aug[row][k] -= factor * aug[col][k]
                }
                b[row] -= factor * b[col]
            }
        }

        var x = Array(repeating: 0.0, count: n)
        for row in stride(from: n - 1, through: 0, by: -1) {
            var sum = b[row]
            for k in (row + 1)..<n {
                sum -= aug[row][k] * x[k]
            }
            let diag = aug[row][row]
            guard abs(diag) > 1e-14 else {
                throw ReadoutMitigationError.singularConfusionMatrix
            }
            x[row] = sum / diag
        }
        return x
    }
}
