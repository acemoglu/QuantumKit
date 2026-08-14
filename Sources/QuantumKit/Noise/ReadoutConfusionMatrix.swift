import Foundation

public enum ReadoutConfusionError: Error, Equatable {
    case invalidDimension(expected: Int, actual: Int)
    case emptyMatrix
    case nonStochasticRow(row: Int)
    case probabilityOutOfRange
}

/// Full readout assignment / confusion matrix: `P(measured | prepared)`.
///
/// Rows are prepared computational-basis states; columns are measured outcomes.
/// Each row is a probability distribution (sums to ~1). Independent per-bit flip
/// models remain available via ``NoiseModel/readoutFlip0To1``; when this matrix is
/// set on a ``NoiseModel``, it takes precedence for multi-bit outcomes of matching width.
public struct ReadoutConfusionMatrix: Sendable, Equatable, Codable {
    public let qubitCount: Int
    /// Row-stochastic matrix of size `2^n × 2^n`.
    public let probabilities: [[QFloat]]

    public var dimension: Int { 1 << qubitCount }

    public init(qubitCount: Int, probabilities: [[QFloat]]) throws {
        guard qubitCount > 0 else {
            throw ReadoutConfusionError.emptyMatrix
        }
        let dimension = 1 << qubitCount
        guard probabilities.count == dimension else {
            throw ReadoutConfusionError.invalidDimension(expected: dimension, actual: probabilities.count)
        }
        for (rowIndex, row) in probabilities.enumerated() {
            guard row.count == dimension else {
                throw ReadoutConfusionError.invalidDimension(expected: dimension, actual: row.count)
            }
            var sum: Double = 0
            for value in row {
                guard value >= 0, value <= 1 else {
                    throw ReadoutConfusionError.probabilityOutOfRange
                }
                sum += Double(value)
            }
            guard abs(sum - 1.0) < 1e-3 else {
                throw ReadoutConfusionError.nonStochasticRow(row: rowIndex)
            }
        }
        self.qubitCount = qubitCount
        self.probabilities = probabilities
    }

    /// Single-qubit assignment from asymmetric bit-flip rates.
    /// `P(1|0) = p01`, `P(0|1) = p10`.
    public static func singleQubit(p01: QFloat, p10: QFloat) throws -> ReadoutConfusionMatrix {
        let p0g0 = 1 - NoiseModel.clampPublic(p01)
        let p1g0 = NoiseModel.clampPublic(p01)
        let p0g1 = NoiseModel.clampPublic(p10)
        let p1g1 = 1 - NoiseModel.clampPublic(p10)
        return try ReadoutConfusionMatrix(
            qubitCount: 1,
            probabilities: [
                [p0g0, p1g0],
                [p0g1, p1g1],
            ]
        )
    }

    /// Tensor product of identical single-qubit confusion on `qubitCount` qubits.
    /// Uses ``QubitBitOrdering/engineLSB`` (same as ``product(of:)``).
    public static func independentBits(qubitCount: Int, p01: QFloat, p10: QFloat) throws -> ReadoutConfusionMatrix {
        guard qubitCount > 0 else { throw ReadoutConfusionError.emptyMatrix }
        return try product(of: Array(repeating: (p01: p01, p10: p10), count: qubitCount))
    }

    /// Heterogeneous independent readout (C3): qubit 0 is LSB
    /// (``QubitBitOrdering/engineLSB``).
    ///
    /// Builds `… ⊗ M₁ ⊗ M₀` so computational bit `i` uses `qubits[i]`.
    public static func product(of qubits: [(p01: QFloat, p10: QFloat)]) throws -> ReadoutConfusionMatrix {
        guard let first = qubits.first else {
            throw ReadoutConfusionError.emptyMatrix
        }
        var result = try singleQubit(p01: first.p01, p10: first.p10)
        for entry in qubits.dropFirst() {
            // New qubit becomes the high-order factor; existing register stays as low bits.
            result = try singleQubit(p01: entry.p01, p10: entry.p10).tensor(result)
        }
        return result
    }

    /// Kronecker product with another confusion matrix (larger system = `self ⊗ other`).
    public func tensor(_ other: ReadoutConfusionMatrix) throws -> ReadoutConfusionMatrix {
        let newQubits = qubitCount + other.qubitCount
        let dim = 1 << newQubits
        var matrix = Array(repeating: Array(repeating: QFloat(0), count: dim), count: dim)

        let dimA = dimension
        let dimB = other.dimension
        for preparedA in 0..<dimA {
            for preparedB in 0..<dimB {
                let prepared = preparedA * dimB + preparedB
                for measuredA in 0..<dimA {
                    for measuredB in 0..<dimB {
                        let measured = measuredA * dimB + measuredB
                        matrix[prepared][measured] =
                            probabilities[preparedA][measuredA] * other.probabilities[preparedB][measuredB]
                    }
                }
            }
        }
        return try ReadoutConfusionMatrix(qubitCount: newQubits, probabilities: matrix)
    }

    /// Samples a measured outcome given a prepared computational-basis index.
    public func sampleMeasured(prepared: Int, rng: inout QuantumRNG) -> Int {
        let rowIndex = max(0, min(prepared, dimension - 1))
        let row = probabilities[rowIndex]
        let draw = rng.nextUnitFloat()
        var cumulative: QFloat = 0
        for (measured, probability) in row.enumerated() {
            cumulative += probability
            if draw < cumulative {
                return measured
            }
        }
        return dimension - 1
    }
}

extension NoiseModel {
    /// Public clamp helper for confusion-matrix factories.
    static func clampPublic(_ value: QFloat) -> QFloat {
        min(max(value, 0), 1)
    }
}
