import Foundation

/// Exact multi-gate template rewrite over a **fixed catalog** (A6 lite).
///
/// **Opt-in only** — not part of the default transpiler pipeline. Enable via
/// ``TranspileOptions/enableTemplateMatching`` or run explicitly through
/// ``PassManager``.
///
/// ## Scope (and non-goals)
/// - Matches **adjacent** gate subsequences only (no commutation sliding).
/// - Rewrites are **exact** algebraic identities — gate count strictly decreases.
/// - Does **not** implement Solovay–Kitaev, approximate synthesis, KAK, or
///   general unitary template libraries (**J1** — deferred).
/// - Self-inverse pairs already owned by ``CliffordSimplificationPass`` /
///   ``LocalUnitarySynthesisPass`` (`H·H`, `CX·CX`, …) are **not** duplicated here.
///
/// ## Catalog
/// | ID | Pattern (adjacent) | Replacement |
/// |----|--------------------|-------------|
/// | `hxh` | `H·X·H` (same wire) | `Z` |
/// | `hzh` | `H·Z·H` (same wire) | `X` |
/// | `sxsdg` | `S·X·Sdg` (same wire) | `Y` |
/// | `sdgxsd` | `Sdg·X·S` (same wire) | `Y` |
/// | `hcxh` | `H(t)·CX(c,t)·H(t)` | `CZ(c,t)` |
/// | `hczh` | `H(t)·CZ(c,t)·H(t)` | `CX(c,t)` |
/// | `cx3swap` | `CX(a,b)·CX(b,a)·CX(a,b)` **or** `CX(b,a)·CX(a,b)·CX(b,a)` | `SWAP(a,b)` |
/// | `cx_xt_cx` | `CX(c,t)·X(t)·CX(c,t)` | `X(t)` |
/// | `cx_xc_cx` | `CX(c,t)·X(c)·CX(c,t)` | `X(c)·X(t)` |
/// | `cx_zt_cx` | `CX(c,t)·Z(t)·CX(c,t)` | `Z(c)·Z(t)` |
/// | `cx_zc_cx` | `CX(c,t)·Z(c)·CX(c,t)` | `Z(c)` |
///
/// Matching is greedy left-to-right on length-3 windows, repeated to a fixed point.
public struct TemplateMatchingPass: CompilerPass, Sendable {
    public init() {}

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        var gates = circuit.gates
        while true {
            let next = rewriteOnce(gates)
            if next == gates { break }
            gates = next
        }

        var output = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for gate in gates {
            try output.apply(gate)
        }
        return output
    }

    /// Documented catalog size (for tests / diagnostics).
    public static var catalogEntryCount: Int { 11 }

    // MARK: - Rewrite

    private func rewriteOnce(_ gates: [Gate]) -> [Gate] {
        var result: [Gate] = []
        result.reserveCapacity(gates.count)
        var i = 0
        while i < gates.count {
            if i + 2 < gates.count, let replacement = matchLength3(gates[i], gates[i + 1], gates[i + 2]) {
                result.append(contentsOf: replacement)
                i += 3
                continue
            }
            result.append(gates[i])
            i += 1
        }
        return result
    }

    private func matchLength3(_ a: Gate, _ b: Gate, _ c: Gate) -> [Gate]? {
        // H·X·H → Z
        if case (.h(let q0), .x(let q1), .h(let q2)) = (a, b, c), q0 == q1, q1 == q2 {
            return [.z(target: q0)]
        }
        // H·Z·H → X
        if case (.h(let q0), .z(let q1), .h(let q2)) = (a, b, c), q0 == q1, q1 == q2 {
            return [.x(target: q0)]
        }
        // S·X·Sdg → Y  and  Sdg·X·S → Y
        if case (.s(let q0), .x(let q1), .sdg(let q2)) = (a, b, c), q0 == q1, q1 == q2 {
            return [.y(target: q0)]
        }
        if case (.sdg(let q0), .x(let q1), .s(let q2)) = (a, b, c), q0 == q1, q1 == q2 {
            return [.y(target: q0)]
        }
        // H(t)·CX(c,t)·H(t) → CZ(c,t)
        if case (.h(let t0), .cx(let control, let target), .h(let t1)) = (a, b, c),
           t0 == target, t1 == target {
            return [.cz(control: control, target: target)]
        }
        // H(t)·CZ(c,t)·H(t) → CX(c,t)
        if case (.h(let t0), .cz(let control, let target), .h(let t1)) = (a, b, c),
           t0 == target, t1 == target {
            return [.cx(control: control, target: target)]
        }
        // CX(a,b)·CX(b,a)·CX(a,b) → SWAP(a,b)  (also matches the reverse cyclic order)
        if case (.cx(let a1, let b1), .cx(let a2, let b2), .cx(let a3, let b3)) = (a, b, c),
           a1 == b2, b1 == a2, a1 == a3, b1 == b3 {
            return [.swap(q1: a1, q2: b1)]
        }
        // CX(c,t)·X(t)·CX(c,t) → X(t)
        if case (.cx(let c0, let t0), .x(let xq), .cx(let c1, let t1)) = (a, b, c),
           c0 == c1, t0 == t1, xq == t0 {
            return [.x(target: t0)]
        }
        // CX(c,t)·X(c)·CX(c,t) → X(c)·X(t)
        if case (.cx(let c0, let t0), .x(let xq), .cx(let c1, let t1)) = (a, b, c),
           c0 == c1, t0 == t1, xq == c0 {
            return [.x(target: c0), .x(target: t0)]
        }
        // CX(c,t)·Z(t)·CX(c,t) → Z(c)·Z(t)
        if case (.cx(let c0, let t0), .z(let zq), .cx(let c1, let t1)) = (a, b, c),
           c0 == c1, t0 == t1, zq == t0 {
            return [.z(target: c0), .z(target: t0)]
        }
        // CX(c,t)·Z(c)·CX(c,t) → Z(c)
        if case (.cx(let c0, let t0), .z(let zq), .cx(let c1, let t1)) = (a, b, c),
           c0 == c1, t0 == t1, zq == c0 {
            return [.z(target: c0)]
        }
        return nil
    }
}
