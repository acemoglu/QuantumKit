import Foundation

/*
 Pauli twirling — Clifford 1Q/2Q layer ensembles
 =========================================================

 Exact model
 ----------------------
 Opt-in on shot ``Estimator`` only. For each **Clifford** 1Q/2Q unitary gate site,
 sample a uniform random Pauli string `P` on the gate support and rewrite the site as

     P  →  U  →  P'     with   P' = U P U†  (Pauli frame; global phase dropped)

 so the ideal unitary is unchanged (randomized compiling / layer Pauli twirling).
 The ensemble average is an unbiased estimator of the untwirled ideal expectation when
 noise is absent; under gate noise the average converts coherent errors on those layers
 toward a stochastic Pauli channel (standard RC motivation).

Sites: `H,X,Y,Z,S,S†,SX,SX†,CX,CZ,SWAP,iSWAP,DCX` (DCX = CX(q1,q2) then CX(q2,q1),
matching the engine). Non-Clifford unitaries, id/barrier/delay/measure/reset/`c_if`,
and 3Q+ gates are left untouched (not twirled).

 Overhead
 --------
 `ensembleSize` distinct circuits (default = Estimator shot count when `nil`). Shot budget
 is split as `shotsPerMember = max(1, shots / ensembleSize)`; effective shots =
 `ensembleSize * shotsPerMember`. There is no quasiprobability `γ` (unlike PEC) — variance
 shrinks with ensemble averaging, but each member sees fewer shots. Extra gates: up to two
 Pauli layers per twirled site.

 Deferred
 --------
 Measurement / Clifford-frame twirling of Estimator Pauli terms alone; non-Clifford layer
 twirling; 3Q+ Clifford sites (CCX/CSWAP/…); Sampler ensembles; stacking with active ZNE
 or PEC; noise-free virtual frames that avoid engine noise on inserted Paulis.
 */

/// Opt-in Pauli twirling knobs. Presence on ``ResilienceOptions/pauliTwirling`` enables.
public struct PauliTwirlingOptions: Sendable, Equatable {
    /// Number of distinct twirled circuits.
    ///
    /// `nil` (``default``) uses the Estimator shot count, so each member gets
    /// `shotsPerMember = 1` and variance is dominated by the circuit ensemble — prefer an
    /// explicit smaller `ensembleSize` when you want more shots per member.
    public var ensembleSize: Int?

    public init(ensembleSize: Int? = nil) {
        self.ensembleSize = ensembleSize
    }

    public static let `default` = PauliTwirlingOptions()

    public var isActive: Bool { true }
}

/// Metadata attached to ``EstimatorResult`` when Pauli twirling ran.
public struct PauliTwirlingMetadata: Sendable, Equatable {
    public let siteCount: Int
    public let ensembleSize: Int
    public let shotsPerMember: Int
    /// Circuit-ensemble count (shot budget split across members; no PEC-style `γ`).
    public let ensembleOverhead: Int

    public init(siteCount: Int, ensembleSize: Int, shotsPerMember: Int) {
        self.siteCount = siteCount
        self.ensembleSize = ensembleSize
        self.shotsPerMember = shotsPerMember
        self.ensembleOverhead = ensembleSize
    }
}

public enum PauliTwirlingError: Error, Equatable {
    case incompatibleWithZNE
    case incompatibleWithPEC
    case emptyCircuitNoTwirlSites
    case invalidEnsembleSize(Int)
    /// Explicit ``PauliTwirlingOptions/ensembleSize`` must not exceed the Estimator shot budget.
    case ensembleExceedsShots(ensemble: Int, shots: Int)
}

/// Host-side Pauli twirling helpers.
public enum PauliTwirling {

    /// Gate indices that are Clifford 1Q/2Q unitary twirl sites.
    public static func twirlSites(in circuit: QuantumCircuit) throws -> [Int] {
        var sites: [Int] = []
        for (index, gate) in circuit.gates.enumerated() {
            if isTwirlSite(gate) {
                sites.append(index)
            }
        }
        guard !sites.isEmpty else { throw PauliTwirlingError.emptyCircuitNoTwirlSites }
        return sites
    }

    /// Sample a uniform n-qubit Pauli string (`4^n` outcomes, including identity).
    public static func samplePauliString(qubitCount: Int, rng: inout QuantumRNG) -> [Pauli] {
        precondition(qubitCount >= 1)
        var result: [Pauli] = []
        result.reserveCapacity(qubitCount)
        let letters: [Pauli] = [.i, .x, .y, .z]
        for _ in 0..<qubitCount {
            let index = rng.nextInt(upperBound: 4)
            result.append(letters[index])
        }
        return result
    }

    /// `P' = U P U†` as a Pauli string on the gate support (phase discarded).
    public static func conjugatePauliString(gate: Gate, paulis: [Pauli]) -> [Pauli] {
        let qubits = gate.affectedQubits
        precondition(paulis.count == qubits.count)
        switch gate {
        case .h:
            return [conjugate1Q(.h, paulis[0])]
        case .x:
            return [conjugate1Q(.x, paulis[0])]
        case .y:
            return [conjugate1Q(.y, paulis[0])]
        case .z:
            return [conjugate1Q(.z, paulis[0])]
        case .s:
            return [conjugate1Q(.s, paulis[0])]
        case .sdg:
            return [conjugate1Q(.sdg, paulis[0])]
        case .sx:
            return [conjugate1Q(.sx, paulis[0])]
        case .sxdg:
            return [conjugate1Q(.sxdg, paulis[0])]
        case .cx(let control, let target):
            return conjugateCX(control: control, target: target, paulis: paulis, qubits: qubits)
        case .cz:
            return conjugateCZ(paulis: paulis)
        case .swap:
            return [paulis[1], paulis[0]]
        case .iswap:
            return conjugateISWAP(paulis: paulis)
        case .dcx(let q1, let q2):
            // Engine/Stabilizer: DCX applies CX(q1,q2) then CX(q2,q1).
            // U = U₂U₁ ⇒ U P U† = U₂ (U₁ P U₁†) U₂† — conjugate in application order.
            let afterFirst = conjugateCX(
                control: q1,
                target: q2,
                paulis: paulis,
                qubits: qubits
            )
            return conjugateCX(
                control: q2,
                target: q1,
                paulis: afterFirst,
                qubits: qubits
            )
        default:
            preconditionFailure("conjugatePauliString called on non-twirl gate \(gate)")
        }
    }

    /// Rewrite `circuit` by inserting `P` before and conjugated `P'` after each twirl site.
    public static func circuitByTwirlingSites(
        _ circuit: QuantumCircuit,
        siteGateIndices: [Int],
        leftPaulis: [[Pauli]]
    ) throws -> QuantumCircuit {
        precondition(siteGateIndices.count == leftPaulis.count)
        var leftBySite: [Int: [Pauli]] = [:]
        for (site, paulis) in zip(siteGateIndices, leftPaulis) {
            leftBySite[site] = paulis
        }

        var out = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for (index, gate) in circuit.gates.enumerated() {
            if let left = leftBySite[index] {
                let qubits = gate.affectedQubits
                try applyPauliString(left, on: qubits, to: &out)
                try out.apply(gate)
                let right = conjugatePauliString(gate: gate, paulis: left)
                try applyPauliString(right, on: qubits, to: &out)
            } else {
                try out.apply(gate)
            }
        }
        return out
    }

    public static func fingerprintToken(for options: PauliTwirlingOptions) -> String {
        "twirl:pauliLayer:clifford12q:samples:\(options.ensembleSize.map(String.init) ?? "auto")"
    }

    // MARK: - Internals

    public static func isTwirlSite(_ gate: Gate) -> Bool {
        switch gate {
        case .h, .x, .y, .z, .s, .sdg, .sx, .sxdg,
             .cx, .cz, .swap, .iswap, .dcx:
            return true
        default:
            return false
        }
    }

    private enum Conjugate1Q {
        case h, x, y, z, s, sdg, sx, sxdg
    }

    private static func conjugate1Q(_ kind: Conjugate1Q, _ p: Pauli) -> Pauli {
        switch kind {
        case .h:
            switch p {
            case .i: return .i
            case .x: return .z
            case .y: return .y
            case .z: return .x
            }
        case .x:
            // X P X: Y→−Y, Z→−Z (phase dropped)
            return p
        case .y:
            return p
        case .z:
            return p
        case .s:
            // S: X→Y, Y→−X, Z→Z (phase dropped ⇒ Y→X)
            switch p {
            case .i: return .i
            case .x: return .y
            case .y: return .x
            case .z: return .z
            }
        case .sdg:
            // S†: X→−Y, Y→X, Z→Z (phase dropped ⇒ X→Y)
            switch p {
            case .i: return .i
            case .x: return .y
            case .y: return .x
            case .z: return .z
            }
        case .sx:
            // SX: X→X, Y→Z, Z→−Y (phase dropped ⇒ Z→Y)
            switch p {
            case .i: return .i
            case .x: return .x
            case .y: return .z
            case .z: return .y
            }
        case .sxdg:
            // SX†: X→X, Y→−Z, Z→Y (phase dropped ⇒ Y→Z)
            switch p {
            case .i: return .i
            case .x: return .x
            case .y: return .z
            case .z: return .y
            }
        }
    }

    private static func conjugateCX(
        control: Int,
        target: Int,
        paulis: [Pauli],
        qubits: [Int]
    ) -> [Pauli] {
        var map: [Int: Pauli] = [:]
        for (q, p) in zip(qubits, paulis) where p != .i {
            map[q] = p
        }
        var pc = map[control] ?? .i
        var pt = map[target] ?? .i

        // Expand generators: conjugating a tensor product = product of conjugations.
        // CX: Xc→Xc Xt, Zt→Zc Zt, Yc→Yc Xt, Yt→Zc Yt; Xt, Zc fixed.
        let (xc, zc) = pauliXZ(pc)
        let (xt, zt) = pauliXZ(pt)
        // After CX: X on control → Xc Xt, Z on target → Zc Zt.
        let outXc = xc
        let outZc = zc != zt
        let outXt = xc != xt
        let outZt = zt
        pc = pauliFromXZ(x: outXc, z: outZc)
        pt = pauliFromXZ(x: outXt, z: outZt)

        return qubits.map { q in
            if q == control { return pc }
            if q == target { return pt }
            return .i
        }
    }

    private static func conjugateCZ(paulis: [Pauli]) -> [Pauli] {
        // CZ: Xa → Xa Zb, Xb → Za Xb; Z fixed. Using symplectic on two qubits.
        guard paulis.count == 2 else { return paulis }
        let (xa, za) = pauliXZ(paulis[0])
        let (xb, zb) = pauliXZ(paulis[1])
        let outXa = xa
        let outZa = za != xb
        let outXb = xb
        let outZb = zb != xa
        return [
            pauliFromXZ(x: outXa, z: outZa),
            pauliFromXZ(x: outXb, z: outZb),
        ]
    }

    /// `iSWAP P iSWAP†` on the Pauli group (global phase discarded).
    /// Not the same as SWAP: e.g. `X⊗I → Z⊗Y`, `I⊗X → Y⊗Z`.
    private static func conjugateISWAP(paulis: [Pauli]) -> [Pauli] {
        guard paulis.count == 2 else { return paulis }
        switch (paulis[0], paulis[1]) {
        case (.i, .i): return [.i, .i]
        case (.i, .x): return [.y, .z]
        case (.i, .y): return [.x, .z]
        case (.i, .z): return [.z, .i]
        case (.x, .i): return [.z, .y]
        case (.x, .x): return [.x, .x]
        case (.x, .y): return [.y, .x]
        case (.x, .z): return [.i, .y]
        case (.y, .i): return [.z, .x]
        case (.y, .x): return [.x, .y]
        case (.y, .y): return [.y, .y]
        case (.y, .z): return [.i, .x]
        case (.z, .i): return [.i, .z]
        case (.z, .x): return [.y, .i]
        case (.z, .y): return [.x, .i]
        case (.z, .z): return [.z, .z]
        }
    }

    private static func pauliXZ(_ p: Pauli) -> (Bool, Bool) {
        switch p {
        case .i: return (false, false)
        case .x: return (true, false)
        case .z: return (false, true)
        case .y: return (true, true)
        }
    }

    private static func pauliFromXZ(x: Bool, z: Bool) -> Pauli {
        switch (x, z) {
        case (false, false): return .i
        case (true, false): return .x
        case (false, true): return .z
        case (true, true): return .y
        }
    }

    private static func applyPauliString(
        _ paulis: [Pauli],
        on qubits: [Int],
        to circuit: inout QuantumCircuit
    ) throws {
        for (qubit, pauli) in zip(qubits, paulis) {
            switch pauli {
            case .i: break
            case .x: try circuit.x(qubit)
            case .y: try circuit.y(qubit)
            case .z: try circuit.z(qubit)
            }
        }
    }
}
