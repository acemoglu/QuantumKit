import Foundation

/// Aaronson–Gottesman CHP tableau for stabilizer states (host CPU).
///
/// Layout: `2n` generator rows (destabilizers `0..<n`, stabilizers `n..<2n`) plus one
/// scratch row at index `2n`. Each row stores X bits, Z bits, and a phase bit `r`
/// (`r == 1` means overall factor `-1`).
public struct StabilizerTableau: Sendable {
    /// Soft width cap — tableau storage is O(n²), not 2ⁿ.
    public static let maxQubitCount = 1024

    public let qubitCount: Int
    /// Flat `(2n+1) × n` X bits, row-major.
    private var x: [UInt8]
    /// Flat `(2n+1) × n` Z bits, row-major.
    private var z: [UInt8]
    /// Phase bits for `2n+1` rows (`0` or `1`).
    private var r: [UInt8]

    private var rowCount: Int { 2 * qubitCount + 1 }

    public init(qubitCount: Int) throws {
        guard qubitCount > 0 else {
            throw StabilizerError.invalidQubitCount(qubitCount)
        }
        guard qubitCount <= Self.maxQubitCount else {
            throw StabilizerError.qubitCountExceedsLimit(
                max: Self.maxQubitCount,
                requested: qubitCount
            )
        }
        self.qubitCount = qubitCount
        let rows = 2 * qubitCount + 1
        let cells = rows * qubitCount
        self.x = [UInt8](repeating: 0, count: cells)
        self.z = [UInt8](repeating: 0, count: cells)
        self.r = [UInt8](repeating: 0, count: rows)
        // |0…0⟩: destabilizer i = X_i, stabilizer i = Z_i.
        for i in 0..<qubitCount {
            setX(row: i, qubit: i, 1)
            setZ(row: qubitCount + i, qubit: i, 1)
        }
    }

    public mutating func resetToZero() {
        x = [UInt8](repeating: 0, count: x.count)
        z = [UInt8](repeating: 0, count: z.count)
        r = [UInt8](repeating: 0, count: r.count)
        for i in 0..<qubitCount {
            setX(row: i, qubit: i, 1)
            setZ(row: qubitCount + i, qubit: i, 1)
        }
    }

    // MARK: - Generators

    mutating func applyH(qubit a: Int) {
        for i in 0..<rowCount {
            let xi = xBit(row: i, qubit: a)
            let zi = zBit(row: i, qubit: a)
            r[i] ^= xi & zi
            setX(row: i, qubit: a, zi)
            setZ(row: i, qubit: a, xi)
        }
    }

    mutating func applyS(qubit a: Int) {
        for i in 0..<rowCount {
            let xi = xBit(row: i, qubit: a)
            let zi = zBit(row: i, qubit: a)
            r[i] ^= xi & zi
            setZ(row: i, qubit: a, zi ^ xi)
        }
    }

    mutating func applySdg(qubit a: Int) {
        // S† = S³; three S applications, or direct: phase flip when x∧¬z then z ^= x.
        for i in 0..<rowCount {
            let xi = xBit(row: i, qubit: a)
            let zi = zBit(row: i, qubit: a)
            r[i] ^= xi & (1 ^ zi)
            setZ(row: i, qubit: a, zi ^ xi)
        }
    }

    mutating func applyX(qubit a: Int) {
        for i in 0..<rowCount {
            r[i] ^= zBit(row: i, qubit: a)
        }
    }

    mutating func applyZ(qubit a: Int) {
        for i in 0..<rowCount {
            r[i] ^= xBit(row: i, qubit: a)
        }
    }

    mutating func applyY(qubit a: Int) {
        // Y = iXZ globally; tableau phase: flip when x≠z (X and Z both contribute).
        for i in 0..<rowCount {
            r[i] ^= xBit(row: i, qubit: a) ^ zBit(row: i, qubit: a)
        }
    }

    mutating func applyCX(control a: Int, target b: Int) {
        for i in 0..<rowCount {
            let xa = xBit(row: i, qubit: a)
            let xb = xBit(row: i, qubit: b)
            let za = zBit(row: i, qubit: a)
            let zb = zBit(row: i, qubit: b)
            r[i] ^= xa & zb & (xb ^ za ^ 1)
            setX(row: i, qubit: b, xb ^ xa)
            setZ(row: i, qubit: a, za ^ zb)
        }
    }

    mutating func applyCZ(control a: Int, target b: Int) {
        applyH(qubit: b)
        applyCX(control: a, target: b)
        applyH(qubit: b)
    }

    mutating func applySWAP(q1: Int, q2: Int) {
        guard q1 != q2 else { return }
        applyCX(control: q1, target: q2)
        applyCX(control: q2, target: q1)
        applyCX(control: q1, target: q2)
    }

    /// √X = H S H (global phase ignored).
    mutating func applySX(qubit a: Int) {
        applyH(qubit: a)
        applyS(qubit: a)
        applyH(qubit: a)
    }

    /// √X† = H S† H.
    mutating func applySXdg(qubit a: Int) {
        applyH(qubit: a)
        applySdg(qubit: a)
        applyH(qubit: a)
    }

    // MARK: - Measurement

    /// Computational-basis measure of qubit `a`. Returns `0` or `1`.
    mutating func measure(qubit a: Int, rng: inout QuantumRNG) -> Int {
        let n = qubitCount
        // First stabilizer (rows n..<2n) with X on a → random outcome.
        var p: Int?
        for i in n..<(2 * n) where xBit(row: i, qubit: a) == 1 {
            p = i
            break
        }

        if let p {
            // Random measurement: clear X_a from every other row via rowsum with p.
            for i in 0..<rowCount where i != p {
                if xBit(row: i, qubit: a) == 1 {
                    rowsum(h: i, i: p)
                }
            }
            // Move stabilizer p into the matching destabilizer slot.
            let dest = p - n
            copyRow(from: p, to: dest)
            // New stabilizer: ±Z_a.
            clearRow(p)
            setZ(row: p, qubit: a, 1)
            let outcome = rng.nextUnitDouble() < 0.5 ? 0 : 1
            r[p] = UInt8(outcome)
            return outcome
        }

        // Deterministic: accumulate destabilizers that anticommute with Z_a into scratch.
        let scratch = 2 * n
        clearRow(scratch)
        for i in 0..<n where xBit(row: i, qubit: a) == 1 {
            rowsum(h: scratch, i: i + n)
        }
        return Int(r[scratch])
    }

    /// Measure listed qubits (engineLSB packing: qubit `qubits[k]` → bit `k`).
    mutating func measure(qubits: [Int], rng: inout QuantumRNG) -> Int {
        var outcome = 0
        for (position, qubit) in qubits.enumerated() {
            let bit = measure(qubit: qubit, rng: &rng)
            outcome |= bit << position
        }
        return outcome
    }

    /// Projective reset to |0⟩ (measure, then X if outcome was 1).
    mutating func reset(qubit a: Int, rng: inout QuantumRNG) {
        if measure(qubit: a, rng: &rng) == 1 {
            applyX(qubit: a)
        }
    }

    // MARK: - Row algebra

    /// Replace row `h` with row `h` · row `i` (Pauli product), updating phase.
    private mutating func rowsum(h: Int, i: Int) {
        // Phase contribution from multiplying Paulis qubit-wise (mod 4 → {0,2} → r bit).
        var g = 0
        for q in 0..<qubitCount {
            let x1 = Int(xBit(row: i, qubit: q))
            let z1 = Int(zBit(row: i, qubit: q))
            let x2 = Int(xBit(row: h, qubit: q))
            let z2 = Int(zBit(row: h, qubit: q))
            g += Self.pauliPhaseExponent(x1: x1, z1: z1, x2: x2, z2: z2)
        }
        let phaseSum = (2 * Int(r[h] ^ r[i]) + g) & 3
        // Valid stabilizer rowsums land on 0 or 2 (mod 4).
        r[h] = UInt8((phaseSum == 2) ? 1 : 0)
        for q in 0..<qubitCount {
            setX(row: h, qubit: q, xBit(row: h, qubit: q) ^ xBit(row: i, qubit: q))
            setZ(row: h, qubit: q, zBit(row: h, qubit: q) ^ zBit(row: i, qubit: q))
        }
    }

    /// Contribution of left Pauli (x1,z1) times right Pauli (x2,z2) to the i-power sum.
    private static func pauliPhaseExponent(x1: Int, z1: Int, x2: Int, z2: Int) -> Int {
        if x1 == 0 && z1 == 0 { return 0 } // I
        if x1 == 1 && z1 == 1 { // Y
            return z2 - x2
        }
        if x1 == 1 { // X
            return z2 * (2 * x2 - 1)
        }
        // Z
        return x2 * (1 - 2 * z2)
    }

    private mutating func copyRow(from src: Int, to dst: Int) {
        r[dst] = r[src]
        let n = qubitCount
        let srcBase = src * n
        let dstBase = dst * n
        for q in 0..<n {
            x[dstBase + q] = x[srcBase + q]
            z[dstBase + q] = z[srcBase + q]
        }
    }

    private mutating func clearRow(_ row: Int) {
        r[row] = 0
        let base = row * qubitCount
        for q in 0..<qubitCount {
            x[base + q] = 0
            z[base + q] = 0
        }
    }

    private func xBit(row: Int, qubit: Int) -> UInt8 {
        x[row * qubitCount + qubit]
    }

    private func zBit(row: Int, qubit: Int) -> UInt8 {
        z[row * qubitCount + qubit]
    }

    private mutating func setX(row: Int, qubit: Int, _ value: UInt8) {
        x[row * qubitCount + qubit] = value
    }

    private mutating func setZ(row: Int, qubit: Int, _ value: UInt8) {
        z[row * qubitCount + qubit] = value
    }
}
