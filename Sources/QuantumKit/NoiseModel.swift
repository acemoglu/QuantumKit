import Foundation

/// Stochastic noise applied after each unitary gate during execution.
public struct NoiseModel: Sendable, Equatable {

    /// Per-qubit depolarizing probability `p` in `[0, 1]`.
    /// After a gate, each affected qubit independently gets a random Pauli (X/Y/Z) with probability `p`.
    public var depolarizingProbability: QFloat

    /// Per-qubit amplitude damping (T1) probability `p` in `[0, 1]`.
    /// After a gate, each affected qubit is reset to |0⟩ with probability `p`.
    public var amplitudeDampingProbability: QFloat

    /// Per-qubit phase damping / dephasing (T2) probability `p` in `[0, 1]`.
    /// After a gate, each affected qubit independently gets a Z gate with probability `p`.
    public var phaseDampingProbability: QFloat

    /// Symmetric readout bit-flip probability `p` in `[0, 1]` with `p01 = p10 = p/2`.
    /// Applied only to reported classical measurement outcomes; the quantum state is unchanged.
    public var readoutErrorProbability: QFloat

    public init(
        depolarizingProbability: QFloat = 0,
        amplitudeDampingProbability: QFloat = 0,
        phaseDampingProbability: QFloat = 0,
        readoutErrorProbability: QFloat = 0
    ) {
        self.depolarizingProbability = Self.clamp(depolarizingProbability)
        self.amplitudeDampingProbability = Self.clamp(amplitudeDampingProbability)
        self.phaseDampingProbability = Self.clamp(phaseDampingProbability)
        self.readoutErrorProbability = Self.clamp(readoutErrorProbability)
    }

    public var appliesDepolarizing: Bool {
        depolarizingProbability > 0
    }

    public var appliesAmplitudeDamping: Bool {
        amplitudeDampingProbability > 0
    }

    public var appliesPhaseDamping: Bool {
        phaseDampingProbability > 0
    }

    public var appliesReadoutError: Bool {
        readoutErrorProbability > 0
    }

    /// Gate-time noise channels that require per-gate execution (disables batching).
    public var hasGateNoise: Bool {
        appliesDepolarizing || appliesAmplitudeDamping || appliesPhaseDamping
    }

    public var hasAnyChannel: Bool {
        hasGateNoise || appliesReadoutError
    }

    /// Flips a single classical bit with symmetric readout error (`p01 = p10 = p/2`).
    public func flipReadoutBit(_ bit: Int, rng: inout QuantumRNG) -> Int {
        guard appliesReadoutError else { return bit }
        let halfP = readoutErrorProbability / 2
        guard halfP > 0, rng.nextUnitFloat() < halfP else { return bit }
        return 1 - bit
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
