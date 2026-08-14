import Foundation

/// Formal qubit / bitstring ordering policy for QuantumKit.
///
/// ## HARD RULE — engine state index (do **not** flip)
///
/// Computational-basis **amplitude and histogram index** bit `k` is always **qubit `k`**.
/// Qubit **0** is the **LSB** of every Metal/CPU statevector / density-matrix layout,
/// ``ShotCounts/counts`` integer keys, and ``ReadoutConfusionMatrix`` row/column indices.
/// This matches ``engineLSB``. Changing it would invalidate kernels and noise tensors.
///
/// ## Display bitstrings
///
/// Public bitstring histograms (``ShotCounts/bitstringCounts``, ``SamplerResult/quasiProbabilities``,
/// ``BackendRunResult/bitstringCounts``) use ``bitstringMSB`` by default: the **leftmost**
/// character is qubit `n-1` (MSB of the index), the **rightmost** is qubit `0`.
/// Convert explicitly with ``bitstring(forIndex:qubitCount:)`` / ``index(fromBitstring:qubitCount:)``
/// — do not assume call sites share a convention without naming this type.
public enum QubitBitOrdering: String, Sendable, Equatable, Codable, Hashable, CaseIterable {
    /// Engine-native packing: qubit `k` ↔ index bit `k` (qubit 0 = LSB).
    ///
    /// When formatting a bitstring under this policy, characters run **left → right =
    /// qubit 0 → qubit n-1** (LSB written first).
    case engineLSB

    /// MSB-first wire / display bitstring: **left → right = qubit n-1 → qubit 0**.
    /// Example (2 qubits): `"10"` ↔ index `2` (qubit 1 set, qubit 0 clear).
    case bitstringMSB

    /// Ordering of Metal/CPU amplitudes and integer shot keys.
    public static let engineDefault: QubitBitOrdering = .engineLSB

    /// Default for public bitstring dictionaries (Sampler / ShotCounts / backend results).
    public static let bitstringDefault: QubitBitOrdering = .bitstringMSB
}

public enum QubitBitOrderingError: Error, Equatable, Sendable {
    case invalidBitstringLength(expected: Int, actual: Int)
    case invalidBitCharacter(Character)
    case indexOutOfRange(index: Int, qubitCount: Int)
    case invalidBitValue(Int)
}

extension QubitBitOrdering {

    /// Formats `index` as a bitstring under this policy.
    public func bitstring(forIndex index: Int, qubitCount: Int) throws -> String {
        guard qubitCount > 0 else { return "" }
        let maxIndex = 1 << qubitCount
        guard index >= 0, index < maxIndex else {
            throw QubitBitOrderingError.indexOutOfRange(index: index, qubitCount: qubitCount)
        }
        switch self {
        case .engineLSB:
            // Left = qubit 0 … right = qubit n-1
            return (0..<qubitCount).map { q in
                ((index >> q) & 1) == 1 ? "1" : "0"
            }.joined()
        case .bitstringMSB:
            // Left = qubit n-1 … right = qubit 0
            return (0..<qubitCount).reversed().map { q in
                ((index >> q) & 1) == 1 ? "1" : "0"
            }.joined()
        }
    }

    /// Parses a bitstring under this policy into an engine-native index (``engineLSB`` packing).
    public func index(fromBitstring bitstring: String, qubitCount: Int) throws -> Int {
        guard bitstring.count == qubitCount else {
            throw QubitBitOrderingError.invalidBitstringLength(
                expected: qubitCount,
                actual: bitstring.count
            )
        }
        var value = 0
        switch self {
        case .engineLSB:
            for (q, char) in bitstring.enumerated() {
                value |= try Self.bitValue(char) << q
            }
        case .bitstringMSB:
            for (position, char) in bitstring.enumerated() {
                let q = qubitCount - 1 - position
                value |= try Self.bitValue(char) << q
            }
        }
        return value
    }

    /// Left-to-right bit array under this policy (`0`/`1` ints).
    public func bits(forIndex index: Int, qubitCount: Int) throws -> [Int] {
        let string = try bitstring(forIndex: index, qubitCount: qubitCount)
        return try string.map { try Self.bitValue($0) }
    }

    /// Engine-native index from a left-to-right bit array under this policy.
    public func index(fromBits bits: [Int], qubitCount: Int? = nil) throws -> Int {
        let n = qubitCount ?? bits.count
        guard bits.count == n else {
            throw QubitBitOrderingError.invalidBitstringLength(expected: n, actual: bits.count)
        }
        let chars: String = try bits.map { bit in
            guard bit == 0 || bit == 1 else {
                throw QubitBitOrderingError.invalidBitValue(bit)
            }
            return bit == 1 ? "1" : "0"
        }.joined()
        return try index(fromBitstring: chars, qubitCount: n)
    }

    /// Converts a full-width bitstring between policies (same underlying index).
    public static func convertBitstring(
        _ bitstring: String,
        from: QubitBitOrdering,
        to: QubitBitOrdering
    ) throws -> String {
        if from == to { return bitstring }
        let index = try from.index(fromBitstring: bitstring, qubitCount: bitstring.count)
        return try to.bitstring(forIndex: index, qubitCount: bitstring.count)
    }

    private static func bitValue(_ char: Character) throws -> Int {
        switch char {
        case "0": return 0
        case "1": return 1
        default: throw QubitBitOrderingError.invalidBitCharacter(char)
        }
    }
}
