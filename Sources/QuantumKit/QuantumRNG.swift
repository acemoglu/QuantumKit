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
            let bits = UInt32(truncatingIfNeeded: nextState >> 11)
            return Float(bits) / (Float(UInt32.max) + 1.0)
        }
    }

    private static func splitMix64(_ value: UInt64) -> UInt64 {
        var z = value &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
