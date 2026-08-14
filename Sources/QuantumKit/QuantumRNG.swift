import Foundation
import Security

/// Random source for measurement sampling.
public enum QuantumRNG: Sendable {

    /// Hardware entropy via `SecRandomCopyBytes`.
    case hardware

    /// Deterministic PRNG for reproducible shots in tests and benchmarks.
    case seeded(UInt64)

    public mutating func nextUnitFloat() -> Float {
        switch self {
        case .hardware:
            return TRNGCollapse.generateHardwareFloat()
        case .seeded(let state):
            let nextState = Self.splitMix64(state)
            self = .seeded(nextState)
            return Self.unitFloat(from: UInt32(truncatingIfNeeded: nextState >> 32))
        }
    }

    /// A `Double` uniformly distributed in the half-open range `[0, 1)` with full 53-bit resolution.
    ///
    /// Used for measurement dice rolls: a 24-bit `Float` can only address ~16M distinct thresholds,
    /// which is too coarse for the CDF of a state with more than ~24 qubits. The extra precision is
    /// what makes the compensated (double-single) CDF on the GPU actually reachable.
    public mutating func nextUnitDouble() -> Double {
        switch self {
        case .hardware:
            return TRNGCollapse.generateHardwareDouble()
        case .seeded(let state):
            let nextState = Self.splitMix64(state)
            self = .seeded(nextState)
            return Self.unitDouble(from: nextState)
        }
    }

    /// Next 64-bit value from the PRNG (or hardware entropy). Used by deterministic transpiler seeding.
    public mutating func nextUInt64() -> UInt64 {
        switch self {
        case .hardware:
            var bytes = [UInt8](repeating: 0, count: 8)
            _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
            return bytes.withUnsafeBytes { $0.load(as: UInt64.self) }
        case .seeded(let state):
            let nextState = Self.splitMix64(state)
            self = .seeded(nextState)
            return nextState
        }
    }

    /// Uniform integer in `0..<upperBound` (`upperBound` must be > 0).
    public mutating func nextInt(upperBound: Int) -> Int {
        precondition(upperBound > 0)
        return Int(nextUInt64() % UInt64(upperBound))
    }

    /// Independent PRNG stream for shot `shotIndex` under a run-level `seed`.
    ///
    /// Used for every ``ShotExecutionPolicy/canBatch`` CPU shot (concurrent or serial,
    /// including `batchSize == 1`): workers must not share one ``QuantumRNG``, and
    /// `batchSize` must not change the seeded schedule. `shotIndex` is the global shot
    /// ordinal (`0..<shots`), not a batch-local slot. Mixing is `splitMix64`-based and
    /// injective in `shotIndex` for a fixed `seed` (bijective mix, then XOR with a
    /// seed-derived constant).
    ///
    /// Stream root comes from ``QuantumRunOptions/seed``, or — when that is `nil` — the
    /// initial state of a ``seeded`` backend RNG (``CPUShotSampler`` does not advance it).
    ///
    /// **Not** the same draw sequence as a single ``seeded`` stream advanced across shots
    /// (legacy CPU / Metal unitary-batch measurement schedule). Same `seed` on Metal
    /// therefore does **not** reproduce CPU ``canBatch`` histograms.
    ///
    /// Sequential ``seeded`` consumption remains only on ``ShotExecutionPolicy/mustSerial``
    /// paths (projective mid-circuit measure / `c_if`).
    public static func independentShotStream(seed: UInt64, shotIndex: Int) -> QuantumRNG {
        precondition(shotIndex >= 0)
        let mixed = splitMix64(seed) ^ splitMix64(UInt64(shotIndex) &+ 1)
        return .seeded(splitMix64(mixed))
    }

    /// Maps 32 random bits to a uniformly distributed `Float` in the half-open range `[0, 1)`.
    ///
    /// `Float32` has a 24-bit significand, so the top 24 bits are scaled by `2⁻²⁴`. Every result is
    /// an exact multiple of `2⁻²⁴`, evenly spaced, and strictly less than `1.0` — unlike dividing by
    /// `Float(UInt32.max) + 1`, which rounds to `2³²` and can yield exactly `1.0`.
    static func unitFloat(from bits: UInt32) -> Float {
        Float(bits >> 8) * (1.0 / 16_777_216.0)
    }

    /// Maps 64 random bits to a uniformly distributed `Double` in `[0, 1)` using the top 53 bits
    /// (the `Double` significand width), scaled by `2⁻⁵³`. Always strictly less than `1.0`.
    static func unitDouble(from bits: UInt64) -> Double {
        Double(bits >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    private static func splitMix64(_ value: UInt64) -> UInt64 {
        var z = value &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
