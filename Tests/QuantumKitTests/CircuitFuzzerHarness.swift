import Foundation
@testable import QuantumKit

/// I7-lite seeded random-circuit fuzzer (CPU SV only; no Aer).
///
/// **Gate set (safe / documented):**
/// - 1Q: `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`
/// - 2Q: `cx`, `cz`, `swap` (adjacent or random pair)
/// - Rare (≈5%): mid-circuit `measure` — unitary-only trials **reject** these and
///   regenerate or skip; CPU SV supports measure, but I7 lite asserts unitary Born.
///
/// Bounds: `n ∈ [minQubitCount, maxQubitCount]`, `depth ≤ maxDepth`.
enum CircuitFuzzerHarness {

    static let minQubitCount = 2
    static let maxQubitCount = 6
    /// Layers / random ops per trial (CI-safe).
    static let maxDepth = 20
    /// Default seeded trial count for the fuzz suite.
    static let defaultTrialCount = 48

    /// Documented safe unitary opcodes (no measure / reset / c_if / initialize).
    enum UnitaryOpcode: CaseIterable {
        case h, x, y, z, s, t, rx, ry, rz
        case cx, cz, swap
    }

    struct GeneratedCircuit: Sendable {
        let circuit: QuantumCircuit
        let qubitCount: Int
        let depth: Int
        let seed: UInt64
        /// `true` when a mid-circuit measure was inserted (not unitary-only).
        let containsMidMeasure: Bool
    }

    /// Build one seeded random circuit. Set `allowMidMeasure` to optionally emit measure.
    static func makeCircuit(
        seed: UInt64,
        allowMidMeasure: Bool = true
    ) throws -> GeneratedCircuit {
        var rng = QuantumRNG.seeded(seed)
        let n = minQubitCount + rng.nextInt(upperBound: maxQubitCount - minQubitCount + 1)
        let depth = 1 + rng.nextInt(upperBound: maxDepth)
        var circuit = try QuantumCircuit(qubitCount: n)
        var sawMeasure = false

        for _ in 0..<depth {
            if allowMidMeasure, n >= 1, rng.nextInt(upperBound: 20) == 0 {
                // ~5% per layer: mid-measure (unitary path rejects / skips).
                let q = rng.nextInt(upperBound: n)
                try circuit.measure(q)
                sawMeasure = true
                continue
            }
            try appendRandomUnitary(to: &circuit, rng: &rng)
        }

        return GeneratedCircuit(
            circuit: circuit,
            qubitCount: n,
            depth: depth,
            seed: seed,
            containsMidMeasure: sawMeasure
        )
    }

    /// Keep drawing until a unitary-only circuit is produced (bounded retries).
    static func makeUnitaryCircuit(seed: UInt64, maxAttempts: Int = 8) throws -> GeneratedCircuit {
        var attemptSeed = seed
        for _ in 0..<maxAttempts {
            let generated = try makeCircuit(seed: attemptSeed, allowMidMeasure: true)
            if !generated.containsMidMeasure {
                return generated
            }
            attemptSeed &+= 0x9E37_79B9_7F4A_7C15
        }
        // Fallback: force unitary-only generation.
        return try makeCircuit(seed: seed &+ 1, allowMidMeasure: false)
    }

    private static func appendRandomUnitary(
        to circuit: inout QuantumCircuit,
        rng: inout QuantumRNG
    ) throws {
        let n = circuit.qubitCount
        let opcodes = UnitaryOpcode.allCases
        let op = opcodes[rng.nextInt(upperBound: opcodes.count)]
        let q0 = rng.nextInt(upperBound: n)
        let q1 = n >= 2 ? (q0 + 1 + rng.nextInt(upperBound: n - 1)) % n : q0

        switch op {
        case .h: try circuit.h(q0)
        case .x: try circuit.x(q0)
        case .y: try circuit.y(q0)
        case .z: try circuit.z(q0)
        case .s: try circuit.s(q0)
        case .t: try circuit.t(q0)
        case .rx: try circuit.rx(theta: QFloat(rng.nextUnitDouble() * 2.0 * Double.pi), q0)
        case .ry: try circuit.ry(theta: QFloat(rng.nextUnitDouble() * 2.0 * Double.pi), q0)
        case .rz: try circuit.rz(theta: QFloat(rng.nextUnitDouble() * 2.0 * Double.pi), q0)
        case .cx:
            guard n >= 2 else { try circuit.h(q0); return }
            try circuit.cx(q0, q1)
        case .cz:
            guard n >= 2 else { try circuit.z(q0); return }
            try circuit.cz(q0, q1)
        case .swap:
            guard n >= 2 else { try circuit.x(q0); return }
            try circuit.swap(q0, q1)
        }
    }

}
