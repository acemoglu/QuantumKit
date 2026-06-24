import Foundation

/// Stochastic noise applied after each unitary gate during execution.
public struct NoiseModel: Sendable, Equatable {

    /// Per-qubit depolarizing probability `p` in `[0, 1]`.
    /// After a gate, each affected qubit independently gets a random Pauli (X/Y/Z) with probability `p`.
    public var depolarizingProbability: QFloat

    /// Fixed per-gate amplitude damping probability. Mutually exclusive with ``t1``/``gateTime``;
    /// use one model or the other.
    public var amplitudeDampingProbability: QFloat

    /// T1 relaxation time (same units as ``gateTime``). With ``gateTime``, per-gate
    /// damping uses `p = 1 - exp(-gateTime / t1)`.
    public var t1: QFloat

    /// Duration attributed to each gate for T1 damping (same units as ``t1``).
    public var gateTime: QFloat

    /// Per-gate phase damping / dephasing probability `p` in `[0, 1]`.
    public var phaseDampingProbability: QFloat

    /// Readout bit-flip probability 0 → 1.
    public var readoutFlip0To1: QFloat

    /// Readout bit-flip probability 1 → 0.
    public var readoutFlip1To0: QFloat

    public init(
        depolarizingProbability: QFloat = 0,
        amplitudeDampingProbability: QFloat = 0,
        t1: QFloat = 0,
        gateTime: QFloat = 0,
        phaseDampingProbability: QFloat = 0,
        readoutErrorProbability: QFloat = 0,
        readoutFlip0To1: QFloat = 0,
        readoutFlip1To0: QFloat = 0
    ) {
        self.depolarizingProbability = Self.clamp(depolarizingProbability)
        self.phaseDampingProbability = Self.clamp(phaseDampingProbability)
        self.t1 = max(t1, 0)
        self.gateTime = max(gateTime, 0)

        if t1 > 0 && gateTime > 0 {
            self.amplitudeDampingProbability = 0
        } else {
            self.amplitudeDampingProbability = Self.clamp(amplitudeDampingProbability)
        }

        if readoutErrorProbability > 0 {
            let halfP = Self.clamp(readoutErrorProbability) / 2
            self.readoutFlip0To1 = halfP
            self.readoutFlip1To0 = halfP
        } else {
            self.readoutFlip0To1 = Self.clamp(readoutFlip0To1)
            self.readoutFlip1To0 = Self.clamp(readoutFlip1To0)
        }
    }

    public var appliesDepolarizing: Bool {
        depolarizingProbability > 0
    }

    /// Per-gate T1 damping probability from either the time model or the fixed parameter.
    public var effectiveAmplitudeDampingProbability: QFloat {
        if usesT1TimeModel {
            return 1 - exp(-gateTime / t1)
        }
        return amplitudeDampingProbability
    }

    public var appliesAmplitudeDamping: Bool {
        effectiveAmplitudeDampingProbability > 0
    }

    public var appliesPhaseDamping: Bool {
        phaseDampingProbability > 0
    }

    public var appliesReadoutError: Bool {
        readoutFlip0To1 > 0 || readoutFlip1To0 > 0
    }

    /// Symmetric readout alias: `p01 + p10` (setting via init maps `p01 = p10 = p/2`).
    public var readoutErrorProbability: QFloat {
        readoutFlip0To1 + readoutFlip1To0
    }

    public var usesT1TimeModel: Bool {
        t1 > 0 && gateTime > 0
    }

    /// Gate-time noise channels that require per-gate execution (disables batching).
    public var hasGateNoise: Bool {
        appliesDepolarizing || appliesAmplitudeDamping || appliesPhaseDamping
    }

    public var hasAnyChannel: Bool {
        hasGateNoise || appliesReadoutError
    }

    /// Flips a single classical bit using asymmetric readout error rates.
    public func flipReadoutBit(_ bit: Int, rng: inout QuantumRNG) -> Int {
        guard appliesReadoutError else { return bit }
        if bit == 0 {
            guard readoutFlip0To1 > 0, rng.nextUnitFloat() < readoutFlip0To1 else { return 0 }
            return 1
        }
        guard readoutFlip1To0 > 0, rng.nextUnitFloat() < readoutFlip1To0 else { return 1 }
        return 0
    }

    /// Flips each bit in a packed outcome index independently.
    public func flipReadoutOutcome(_ outcome: Int, measuredQubitCount: Int, rng: inout QuantumRNG) -> Int {
        guard appliesReadoutError else { return outcome }
        var result = outcome
        for position in 0..<measuredQubitCount {
            let bit = (outcome >> position) & 1
            let flipped = flipReadoutBit(bit, rng: &rng)
            if flipped != bit {
                result ^= 1 << position
            }
        }
        return result
    }

    /// Flips each classical bit in a measurement bit array independently.
    public func flipReadoutBits(_ bits: [Int], rng: inout QuantumRNG) -> [Int] {
        guard appliesReadoutError else { return bits }
        return bits.map { flipReadoutBit($0, rng: &rng) }
    }

    private static func clamp(_ value: QFloat) -> QFloat {
        min(max(value, 0), 1)
    }
}
