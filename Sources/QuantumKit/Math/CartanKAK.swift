import Foundation
import Accelerate

// MARK: - Errors

/// Failures from ``CartanKAK`` (host-only 2-qubit Cartan / KAK decomposition).
public enum CartanKAKError: Error, Equatable, Sendable {
    /// Input was not a length-16 row-major 4×4 unitary within tolerance.
    case notUnitary(reason: String)
    /// Magic-basis / Takagi step failed to diagonalize `UᵀU` to tolerance.
    case diagonalizationFailed
    /// Local SU(2) Kronecker factorization of an SO(4) magic factor failed.
    case localFactorizationFailed
    /// Round-trip product of factors missed the input beyond the documented bound.
    case roundTripFailed(frobenius: Double, fidelity: Double)
}

extension CartanKAKError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .notUnitary(let reason):
            return "Cartan/KAK input is not unitary: \(reason)"
        case .diagonalizationFailed:
            return "Cartan/KAK failed to diagonalize the magic-basis Gram matrix"
        case .localFactorizationFailed:
            return "Cartan/KAK failed to factor an SO(4) block into SU(2)⊗SU(2)"
        case .roundTripFailed(let frobenius, let fidelity):
            return "Cartan/KAK round-trip failed (‖U−V‖_F=\(frobenius), fidelity=\(fidelity))"
        }
    }
}

// MARK: - Result types

/// Exact Cartan / KAK-style factorization of a 2-qubit unitary.
///
/// ## Convention
/// Any `U ∈ U(4)` is written (up to the stored global phase `g`) as
///
/// ```
/// U ≃ g · (A₁ ⊗ A₀) · exp(i (x XX + y YY + z ZZ)) · (B₁ ⊗ B₀)
/// ```
///
/// with `Aᵢ, Bᵢ ∈ SU(2)`, Cartan coordinates in the Weyl chamber
/// `π/4 ≥ x ≥ y ≥ |z|`, and qubit **0 = LSB** of the row-major 4×4 packing
/// (same as ``QuantumCircuit/customUnitary(matrix:qubits:)`` on `[0, 1]`).
///
/// ## U(4) vs SU(4)
/// The input may be any unitary in U(4). A global phase is peeled so the
/// interaction lives in SU(4): `U ← U · det(U)^{-1/4}` (principal branch), and
/// that phase is returned as ``globalPhase``. Born probabilities are unchanged
/// by `g`; circuit equivalence uses the same phase-alignment rule as
/// ``CircuitEquivalenceVerifier``.
///
/// ## CX count
/// ``gates`` realizes the factorization with local ``Gate/u`` factors and at most
/// **3** ``Gate/cx`` (Vatan–Williams / Cirq full-CZ interaction, rewritten as CX).
/// Degenerate Cartan vectors use 0–2 CX when the interaction is exactly local /
/// controlled / XX+YY within tolerance.
public struct CartanKAKDecomposition: Sendable, Equatable {
    /// Global phase `g` such that `g · (locals) · Cartan · (locals) ≈ U`.
    public let globalPhase: ComplexAmplitude
    /// Cartan / Weyl coordinates `(x, y, z)` for `exp(i(x XX + y YY + z ZZ))`.
    public let interaction: (x: Double, y: Double, z: Double)
    /// Pre-interaction SU(2) on qubit 0 (`B₀`), row-major 2×2.
    public let before0: [ComplexAmplitude]
    /// Pre-interaction SU(2) on qubit 1 (`B₁`), row-major 2×2.
    public let before1: [ComplexAmplitude]
    /// Post-interaction SU(2) on qubit 0 (`A₀`), row-major 2×2.
    public let after0: [ComplexAmplitude]
    /// Post-interaction SU(2) on qubit 1 (`A₁`), row-major 2×2.
    public let after1: [ComplexAmplitude]
    /// Circuit factors on ``qubits``: local `u` + at most 3 `cx` (left-to-right apply order).
    public let gates: [Gate]
    /// Qubit pair `(q0, q1)` used in ``gates`` (`q0` = LSB of the 4×4).
    public let qubits: (Int, Int)

    public var cxCount: Int {
        gates.reduce(0) { count, gate in
            if case .cx = gate { return count + 1 }
            return count
        }
    }

    public static func == (lhs: CartanKAKDecomposition, rhs: CartanKAKDecomposition) -> Bool {
        lhs.globalPhase == rhs.globalPhase
            && lhs.interaction.x == rhs.interaction.x
            && lhs.interaction.y == rhs.interaction.y
            && lhs.interaction.z == rhs.interaction.z
            && lhs.before0 == rhs.before0
            && lhs.before1 == rhs.before1
            && lhs.after0 == rhs.after0
            && lhs.after1 == rhs.after1
            && lhs.gates == rhs.gates
            && lhs.qubits == rhs.qubits
    }
}

// MARK: - Public API

/// Host-only **2-qubit Cartan / KAK** decomposition (no Metal / backend / pass wiring).
///
/// ## Numerical method
/// 1. Validate row-major 4×4 unitarity (`U†U ≈ I`) with ``unitarityTolerance``.
/// 2. **Primary (Cirq):** magic-basis change → simultaneous real/imag **bidiagonalization**
///    (4×4 SVD via `AᵀA` + LAPACK `dsyev`) with special-orthogonal factors → SO(4)→SU(2)⊗SU(2)
///    via magic + Kronecker factorization → Cartan angles from `KAK_GAMMA · arg(diag)` →
///    Weyl-chamber canonicalization.
/// 3. **Fallback (Qiskit / Kraus–Cirac):** used when SO(4)→SU(2)² fails on degenerate cases
///    (e.g. CX). Project to SU(4) by `det^{-1/4}`, form Gram `M₂ = Uₘᵀ Uₘ` in the Qiskit magic
///    basis, Takagi-factor `M₂` by probing real combinations of `Re/Im` + `dsyev`, then chamber
///    flips with local Pauli corrections.
/// 4. Emit `exp(i(xXX+yYY+zZZ))` as ≤3 CX + 1Q rotations (Cirq full-CZ template → CX).
///
/// ## U(4) vs SU(4)
/// Input may be any U(4) unitary (row-major 16 ``ComplexAmplitude``). Global phase is tracked in
/// ``CartanKAKDecomposition/globalPhase``; circuit factors match `U` up to that phase (same rule
/// as ``CircuitEquivalenceVerifier``). The Qiskit fallback explicitly strips `det^{-1/4}` into
/// that phase; the Cirq path recovers phase from diagonal magic phases.
///
/// ## Tolerances (defaults)
/// - ``unitarityTolerance``: same value as ``UnitaryValidation/unitarityTolerance`` (`1e-4`);
///   ``decompose`` rejects non-unitary input at that scale.
/// - ``absoluteTolerance``: `1e-8` (zeros / chamber edges / CX skipping)
/// - Round-trip (**binding**): after best global-phase alignment,
///   `‖U − V‖_F ≤ roundTripFrobeniusTolerance` (**default `1e-5`**).
///   ``fidelityFloor`` (`F̄ = (4 + |Tr(U†V)|²) / 20 ≥ 0.9999`) is an additional check that
///   follows from the Frobenius bound for unitaries; both must hold when verification is on.
///
/// ## CX count guarantee
/// ``CartanKAKDecomposition/gates`` contains at most **3** ``Gate/cx`` (Vatan–Williams bound),
/// with 0–2 CX when the Cartan vector is degenerate within ``absoluteTolerance``.
public enum CartanKAK {

    /// Unitarity entry tolerance — kept equal to ``UnitaryValidation/unitarityTolerance``.
    public static let unitarityTolerance: Double = UnitaryValidation.unitarityTolerance
    public static let absoluteTolerance: Double = 1e-8
    public static let roundTripFrobeniusTolerance: Double = 1e-5
    public static let fidelityFloor: Double = 0.9999

    /// Decompose a row-major 4×4 unitary into local ``Gate/u`` factors + ≤3 CX
    /// (interaction may also emit ``Gate/rx`` / ``Gate/ry`` / ``Gate/rz``). Never emits
    /// ``Gate/unitary1`` or ``Gate/customUnitary``.
    ///
    /// - Parameters:
    ///   - unitary: 16 ``ComplexAmplitude`` entries, row-major, qubit 0 = LSB.
    ///   - qubits: Logical pair `(q0, q1)` for emitted gates (`q0` = LSB role).
    ///   - absoluteTolerance: Angle / zero cutoff for chamber + CX reduction.
    ///   - verifyRoundTrip: When `true` (default), require phase-aligned
    ///     `‖U − V‖_F ≤ roundTripFrobeniusTolerance` and `F̄ ≥ fidelityFloor`.
    public static func decompose(
        _ unitary: [ComplexAmplitude],
        qubits: (Int, Int) = (0, 1),
        absoluteTolerance: Double = absoluteTolerance,
        verifyRoundTrip: Bool = true
    ) throws -> CartanKAKDecomposition {
        guard unitary.count == 16 else {
            throw CartanKAKError.notUnitary(reason: "expected 16 entries, got \(unitary.count)")
        }
        guard qubits.0 >= 0, qubits.1 >= 0, qubits.0 != qubits.1 else {
            throw CartanKAKError.notUnitary(reason: "qubits must be distinct non-negative indices")
        }

        try validateInputUnitary(unitary)

        // Cirq KAK works on U(4) directly; global phase is recovered from diag phases.
        let u = unitary.map { C64(re: Double($0.real), im: Double($0.imaginary)) }
        let weyl = try weylDecomposition(su4: u, atol: absoluteTolerance)
        let gates = try emitGates(
            weyl: weyl,
            qubits: qubits,
            atol: absoluteTolerance
        )

        let result = CartanKAKDecomposition(
            globalPhase: ComplexAmplitude(
                real: QFloat(cos(weyl.extraPhase)),
                imaginary: QFloat(sin(weyl.extraPhase))
            ),
            interaction: (weyl.x, weyl.y, weyl.z),
            before0: weyl.k2r.map { $0.asAmplitude() },
            before1: weyl.k2l.map { $0.asAmplitude() },
            after0: weyl.k1r.map { $0.asAmplitude() },
            after1: weyl.k1l.map { $0.asAmplitude() },
            gates: gates,
            qubits: qubits
        )

        if verifyRoundTrip {
            let gates01 = result.gates.map { remapGate($0, from: qubits, to: (0, 1)) }
            let rebuilt = try matrix(ofGates: gates01)
            let frob = phaseAlignedFrobenius(target: unitary, candidate: rebuilt)
            let fid = averageGateFidelity(target: unitary, candidate: rebuilt)
            // Frobenius is binding; fidelityFloor is required as well (implied by a tight Frob bound).
            if frob > roundTripFrobeniusTolerance || fid < fidelityFloor {
                throw CartanKAKError.roundTripFailed(frobenius: frob, fidelity: fid)
            }
        }
        return result
    }

    /// Rejects non-unitary input using ``unitarityTolerance`` via ``UnitaryValidation``.
    private static func validateInputUnitary(_ unitary: [ComplexAmplitude]) throws {
        guard unitarityTolerance == UnitaryValidation.unitarityTolerance else {
            throw CartanKAKError.notUnitary(
                reason: "CartanKAK.unitarityTolerance drifted from UnitaryValidation"
            )
        }
        try UnitaryValidation.validateUnitary(matrix: unitary, dimension: 4)
    }

    /// Average gate fidelity `F̄ = (4 + |Tr(U†V)|²) / 20` after best global-phase alignment of `V` to `U`.
    public static func averageGateFidelity(
        target: [ComplexAmplitude],
        candidate: [ComplexAmplitude]
    ) -> Double {
        let u = target.map { C64(re: Double($0.real), im: Double($0.imaginary)) }
        let v = candidate.map { C64(re: Double($0.real), im: Double($0.imaginary)) }
        let aligned = phaseAlign(v, to: u)
        return averageGateFidelityC64(u, aligned)
    }

    /// Frobenius distance `‖U − V‖_F` after best global-phase alignment of `V` to `U`.
    public static func phaseAlignedFrobenius(
        target: [ComplexAmplitude],
        candidate: [ComplexAmplitude]
    ) -> Double {
        let u = target.map { C64(re: Double($0.real), im: Double($0.imaginary)) }
        let v = candidate.map { C64(re: Double($0.real), im: Double($0.imaginary)) }
        let aligned = phaseAlign(v, to: u)
        return frobeniusDistance(u, aligned)
    }

    /// Rebuild the 4×4 unitary implied by ``gates`` on qubits `(0, 1)` (host multiply; no `customUnitary`).
    public static func matrix(ofGates gates: [Gate]) throws -> [ComplexAmplitude] {
        var circuit = try QuantumCircuit(qubitCount: 2)
        for gate in gates {
            try circuit.apply(gate)
        }
        let built = try CircuitUnitary.build(circuit: circuit)
        return built.elements.map {
            ComplexAmplitude(real: QFloat($0.re), imaginary: QFloat($0.im))
        }
    }
}

// MARK: - Weyl / magic internals

private struct WeylPieces {
    var x: Double
    var y: Double
    var z: Double
    var k1l: [C64] // 2×2 on qubit 1
    var k1r: [C64] // 2×2 on qubit 0
    var k2l: [C64]
    var k2r: [C64]
    var extraPhase: Double
}

private extension CartanKAK {

    /// Cirq magic basis (normalized).
    static var cirqMagic: [C64] {
        let s = sqrt(0.5)
        return [
            C64(re: s, im: 0), .zero, .zero, C64(re: 0, im: s),
            .zero, C64(re: 0, im: s), C64(re: s, im: 0), .zero,
            .zero, C64(re: 0, im: s), C64(re: -s, im: 0), .zero,
            C64(re: s, im: 0), .zero, .zero, C64(re: 0, im: -s),
        ]
    }

    static var cirqMagicDag: [C64] { adjoint4(cirqMagic) }

    /// KAK_GAMMA maps diag phases → (w, x, y, z).
    static let kakGamma: [[Double]] = [
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, -0.25, -0.25],
        [-0.25, 0.25, -0.25, 0.25],
        [0.25, -0.25, -0.25, 0.25],
    ]

    static func weylDecomposition(su4: [C64], atol: Double) throws -> WeylPieces {
        // Cirq: bidiagonalize in magic basis, then SO(4) → SU(2)×SU(2).
        let magicU = multiply(multiply(cirqMagicDag, su4), cirqMagic)
        let (left, diagEntries, right) = try bidiagonalizeUnitarySpecialOrthogonals(magicU, atol: atol)

        let (a1, a0): ([C64], [C64])
        let (b1, b0): ([C64], [C64])
        do {
            (a1, a0) = try so4ToMagicSU2s(transposeRealMatrix(left), atol: atol)
            (b1, b0) = try so4ToMagicSU2s(transposeRealMatrix(right), atol: atol)
        } catch {
            // Degenerate singular-value cases (e.g. CX): fall back to Qiskit M₂ / Takagi locals.
            return try weylDecompositionQiskitFallback(su4, atol: atol)
        }

        let angles = diagEntries.map { atan2($0.im, $0.re) }
        var wxyz = [0.0, 0.0, 0.0, 0.0]
        for i in 0..<4 {
            for j in 0..<4 {
                wxyz[i] += kakGamma[i][j] * angles[j]
            }
        }
        let g0 = wxyz[0]
        var x = wxyz[1]
        var y = wxyz[2]
        var z = wxyz[3]

        let cannon = canonicalizeVector(x: x, y: y, z: z, atol: atol)
        x = cannon.x
        y = cannon.y
        z = cannon.z

        // Cirq: b1 = inner.before[0] @ b1, etc.  before=(right[1], right[0]) in canonicalize
        // after=(left[1], left[0])
        let outB1 = multiply2(cannon.before1, b1)
        let outB0 = multiply2(cannon.before0, b0)
        let outA1 = multiply2(a1, cannon.after1)
        let outA0 = multiply2(a0, cannon.after0)

        return WeylPieces(
            x: x, y: y, z: z,
            k1l: outA1, k1r: outA0,
            k2l: outB1, k2r: outB0,
            extraPhase: g0 + cannon.phase
        )
    }

    /// Qiskit-style magic Gram `M₂ = UₘᵀUₘ` path — used when Cirq SO(4)→SU(2)² fails (e.g. CX).
    static func weylDecompositionQiskitFallback(_ su4: [C64], atol: Double) throws -> WeylPieces {
        let det = determinant4(su4)
        let detPhase = atan2(det.im, det.re) / 4.0
        let strip = C64(re: cos(-detPhase), im: sin(-detPhase))
        let u = su4.map { $0 * strip }

        let up = qiskitMagicTransform(u, reverse: true)
        let m2 = multiply(transpose(up), up)
        let p0 = try diagonalizeComplexSymmetricFallback(m2, atol: atol)

        var d = [C64](repeating: .zero, count: 4)
        for i in 0..<4 {
            var sum = C64.zero
            for r in 0..<4 {
                for c in 0..<4 {
                    sum = sum + C64(re: p0[r * 4 + i], im: 0) * m2[r * 4 + c] * C64(re: p0[c * 4 + i], im: 0)
                }
            }
            d[i] = sum
        }

        var angles = d.map { -atan2($0.im, $0.re) / 2.0 }
        angles[3] = -angles[0] - angles[1] - angles[2]
        var cs = (0..<3).map { ((angles[$0] + angles[3]) / 2.0).modTwoPi }
        var pMat = p0

        let pi2 = Double.pi / 2
        let pi4 = Double.pi / 4
        let cstemp = cs.map { v -> Double in
            let m = v.mod(pi2)
            return min(m, pi2 - m)
        }
        let order = argsort(cstemp)
        let perm = [order[1], order[2], order[0]]
        cs = perm.map { cs[$0] }
        for i in 0..<3 { angles[i] = angles[perm[i]] }
        angles[3] = -angles[0] - angles[1] - angles[2]
        var pPerm = pMat
        for r in 0..<4 {
            for (newCol, oldCol) in perm.enumerated() {
                pPerm[r * 4 + newCol] = pMat[r * 4 + oldCol]
            }
            pPerm[r * 4 + 3] = pMat[r * 4 + 3]
        }
        pMat = pPerm
        if detReal4(pMat) < 0 {
            for r in 0..<4 { pMat[r * 4 + 3] = -pMat[r * 4 + 3] }
        }

        let expD = angles.map { C64(re: cos($0), im: sin($0)) }
        let k2Magic = qiskitMagicTransform(transposeReal(pMat), reverse: false)
        var diag = [C64](repeating: .zero, count: 16)
        for i in 0..<4 { diag[i * 4 + i] = expD[i] }
        let k1Magic = qiskitMagicTransform(multiply(multiply(up, realAsComplex(pMat)), diag), reverse: false)

        var (k1l, k1r, phaseL) = try qiskitFactorProduct(k1Magic)
        var (k2l, k2r, phaseR) = try qiskitFactorProduct(k2Magic)
        var extraPhase = detPhase + phaseL + phaseR

        let ipx = [C64.zero, C64(re: 0, im: 1), C64(re: 0, im: 1), C64.zero]
        let ipy = [C64.zero, C64.one, C64(re: -1, im: 0), C64.zero]
        let ipz = [C64(re: 0, im: 1), C64.zero, C64.zero, C64(re: 0, im: -1)]

        if cs[0] > pi2 {
            cs[0] -= 3 * pi2
            k1l = multiply2(k1l, ipy); k1r = multiply2(k1r, ipy); extraPhase += pi2
        }
        if cs[1] > pi2 {
            cs[1] -= 3 * pi2
            k1l = multiply2(k1l, ipx); k1r = multiply2(k1r, ipx); extraPhase += pi2
        }
        var conjs = 0
        if cs[0] > pi4 {
            cs[0] = pi2 - cs[0]
            k1l = multiply2(k1l, ipy); k2r = multiply2(ipy, k2r)
            conjs += 1; extraPhase -= pi2
        }
        if cs[1] > pi4 {
            cs[1] = pi2 - cs[1]
            k1l = multiply2(k1l, ipx); k2r = multiply2(ipx, k2r)
            conjs += 1; extraPhase += pi2
            if conjs == 1 { extraPhase -= Double.pi }
        }
        if cs[2] > pi2 {
            cs[2] -= 3 * pi2
            k1l = multiply2(k1l, ipz); k1r = multiply2(k1r, ipz)
            extraPhase += pi2
            if conjs == 1 { extraPhase -= Double.pi }
        }
        if conjs == 1 {
            cs[2] = pi2 - cs[2]
            k1l = multiply2(k1l, ipz); k2r = multiply2(ipz, k2r)
            extraPhase += pi2
        }
        if cs[2] > pi4 {
            cs[2] -= pi2
            k1l = multiply2(k1l, ipz); k1r = multiply2(k1r, ipz)
            extraPhase -= pi2
        }

        return WeylPieces(
            x: cs[1], y: cs[0], z: cs[2],
            k1l: k1l, k1r: k1r, k2l: k2l, k2r: k2r,
            extraPhase: extraPhase
        )
    }

    static func qiskitMagicTransform(_ u: [C64], reverse: Bool) -> [C64] {
        let b: [C64] = [
            .one, C64(re: 0, im: 1), .zero, .zero,
            .zero, .zero, C64(re: 0, im: 1), .one,
            .zero, .zero, C64(re: 0, im: 1), C64(re: -1, im: 0),
            .one, C64(re: 0, im: -1), .zero, .zero,
        ]
        let bDagScaled: [C64] = [
            C64(re: 0.5, im: 0), .zero, .zero, C64(re: 0.5, im: 0),
            C64(re: 0, im: -0.5), .zero, .zero, C64(re: 0, im: 0.5),
            .zero, C64(re: 0, im: -0.5), C64(re: 0, im: -0.5), .zero,
            .zero, C64(re: 0.5, im: 0), C64(re: -0.5, im: 0), .zero,
        ]
        if reverse { return multiply(multiply(bDagScaled, u), b) }
        return multiply(multiply(b, u), bDagScaled)
    }

    static func qiskitFactorProduct(_ matrix: [C64]) throws -> (left: [C64], right: [C64], phase: Double) {
        var r = [matrix[0], matrix[1], matrix[4], matrix[5]]
        var detR = r[0] * r[3] - r[1] * r[2]
        if detR.abs < 0.1 {
            r = [matrix[8], matrix[9], matrix[12], matrix[13]]
            detR = r[0] * r[3] - r[1] * r[2]
        }
        guard detR.abs >= 0.1 else { throw CartanKAKError.localFactorizationFailed }
        r = r.map { $0 * detR.inverseSqrtPrincipal }
        let temp = multiply(matrix, kronecker(identity2, adjoint2(r)))
        var l = [temp[0], temp[2], temp[8], temp[10]]
        let detL = l[0] * l[3] - l[1] * l[2]
        guard detL.abs >= 0.9 else { throw CartanKAKError.localFactorizationFailed }
        let phase = atan2(detL.im, detL.re) / 2.0
        l = l.map { $0 * detL.inverseSqrtPrincipal }
        let tr = trace(multiply(adjoint4(kronecker(l, r)), matrix))
        guard abs(tr.abs - 4.0) < 1e-5 else { throw CartanKAKError.localFactorizationFailed }
        return (l, r, phase)
    }

    static func diagonalizeComplexSymmetricFallback(_ m2: [C64], atol: Double) throws -> [Double] {
        var probes: [(Double, Double)] = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2)]
        var state: UInt64 = 2020
        for _ in 0..<60 {
            state = state &* 6364136223846793005 &+ 1
            let u1 = Double(state >> 11) / Double(1 << 53)
            state = state &* 6364136223846793005 &+ 1
            let u2 = Double(state >> 11) / Double(1 << 53)
            let r = sqrt(-2 * log(max(u1, 1e-16)))
            let θ = 2 * Double.pi * u2
            probes.append((r * cos(θ), r * sin(θ)))
        }
        var bestP: [Double]?
        var bestErr = Double.infinity
        for (α, β) in probes {
            var real = [Double](repeating: 0, count: 16)
            for i in 0..<16 { real[i] = α * m2[i].re + β * m2[i].im }
            for r in 0..<4 {
                for c in (r + 1)..<4 {
                    let avg = 0.5 * (real[r * 4 + c] + real[c * 4 + r])
                    real[r * 4 + c] = avg
                    real[c * 4 + r] = avg
                }
            }
            guard let p = try? syev4(real) else { continue }
            var d = [C64](repeating: .zero, count: 4)
            for i in 0..<4 {
                var sum = C64.zero
                for r in 0..<4 {
                    for c in 0..<4 {
                        sum = sum + C64(re: p[r * 4 + i], im: 0) * m2[r * 4 + c] * C64(re: p[c * 4 + i], im: 0)
                    }
                }
                d[i] = sum
            }
            var recon = [C64](repeating: .zero, count: 16)
            for r in 0..<4 {
                for c in 0..<4 {
                    var sum = C64.zero
                    for k in 0..<4 {
                        sum = sum + C64(re: p[r * 4 + k], im: 0) * d[k] * C64(re: p[c * 4 + k], im: 0)
                    }
                    recon[r * 4 + c] = sum
                }
            }
            let err = frobeniusDistance(recon, m2)
            if err < bestErr { bestErr = err; bestP = p }
            if err <= max(atol * 100, 1e-8) { return p }
        }
        if let bestP, bestErr <= 1e-5 { return bestP }
        throw CartanKAKError.diagonalizationFailed
    }

    /// Cirq `kak_canonicalize_vector`.
    static func canonicalizeVector(
        x: Double, y: Double, z: Double, atol: Double
    ) -> (x: Double, y: Double, z: Double, phase: Double, before0: [C64], before1: [C64], after0: [C64], after1: [C64]) {
        var phase = 0.0
        var left = [identity2, identity2]
        var right = [identity2, identity2]
        var v = [x, y, z]

        let flipX: [C64] = [C64.zero, C64(re: 0, im: 1), C64(re: 0, im: 1), C64.zero] // iX
        let flipY: [C64] = [C64.zero, C64.one, C64(re: -1, im: 0), C64.zero] // [[0,1],[-1,0]]
        let flipZ: [C64] = [C64(re: 0, im: 1), C64.zero, C64.zero, C64(re: 0, im: -1)] // iZ
        let flipsArr = [flipX, flipY, flipZ]

        let s = sqrt(0.5)
        let swapsArr = [
            [C64(re: 0, im: s), C64(re: s, im: 0), C64(re: -s, im: 0), C64(re: 0, im: -s)],
            [C64(re: 0, im: s), C64(re: 0, im: s), C64(re: 0, im: s), C64(re: 0, im: -s)],
            [C64.zero, C64(re: s, im: s), C64(re: -s, im: s), C64.zero],
        ]

        func powFlip(_ f: [C64], _ step: Int) -> [C64] {
            var r = identity2
            let n = ((step % 4) + 4) % 4
            for _ in 0..<n { r = multiply2(f, r) }
            return r
        }

        func shift(_ k: Int, _ step: Int) {
            v[k] += Double(step) * Double.pi / 2
            phase += Double(step) * Double.pi / 2 // 1j**step → phase += step * π/2
            // Actually 1j**step: step=1 → i (phase π/2), step=2 → -1 (π), etc. Yes.
            right[0] = multiply2(powFlip(flipsArr[k], step), right[0])
            right[1] = multiply2(powFlip(flipsArr[k], step), right[1])
        }

        func negate(_ k1: Int, _ k2: Int) {
            v[k1] *= -1
            v[k2] *= -1
            phase += Double.pi // *= -1
            let sIdx = 3 - k1 - k2
            left[1] = multiply2(left[1], flipsArr[sIdx])
            right[1] = multiply2(flipsArr[sIdx], right[1])
        }

        func swap(_ k1: Int, _ k2: Int) {
            v.swapAt(k1, k2)
            let sIdx = 3 - k1 - k2
            let s = swapsArr[sIdx]
            left[0] = multiply2(left[0], s)
            left[1] = multiply2(left[1], s)
            right[0] = multiply2(s, right[0])
            right[1] = multiply2(s, right[1])
        }

        func canonicalShift(_ k: Int) {
            while v[k] <= -Double.pi / 4 { shift(k, +1) }
            while v[k] > Double.pi / 4 { shift(k, -1) }
        }

        func sort() {
            if abs(v[0]) < abs(v[1]) { swap(0, 1) }
            if abs(v[1]) < abs(v[2]) { swap(1, 2) }
            if abs(v[0]) < abs(v[1]) { swap(0, 1) }
        }

        canonicalShift(0)
        canonicalShift(1)
        canonicalShift(2)
        sort()
        if v[0] < 0 { negate(0, 2) }
        if v[1] < 0 { negate(1, 2) }
        canonicalShift(2)
        if v[0] > Double.pi / 4 - atol && v[2] < 0 {
            shift(0, -1)
            negate(0, 2)
        }

        // Cirq returns after=(left[1], left[0]), before=(right[1], right[0])
        return (
            x: v[0], y: v[1], z: v[2],
            phase: phase,
            before0: right[0], before1: right[1],
            after0: left[0], after1: left[1]
        )
    }

    static func so4ToMagicSU2s(_ mat: [Double], atol: Double) throws -> (a: [C64], b: [C64]) {
        // ab = MAGIC @ mat @ MAGIC†
        let ab = multiply(multiply(cirqMagic, realAsComplex(mat)), cirqMagicDag)
        let (_, a, b) = try kronFactor4x4to2x2s(ab, atol: atol)
        return (a, b)
    }

    static func kronFactor4x4to2x2s(_ matrix: [C64], atol: Double) throws -> (g: C64, a: [C64], b: [C64]) {
        var bestA = 0, bestB = 0, bestMag = -1.0
        for i in 0..<4 {
            for j in 0..<4 {
                let m = matrix[i * 4 + j].abs
                if m > bestMag {
                    bestMag = m
                    bestA = i
                    bestB = j
                }
            }
        }
        guard bestMag > 1e-14 else { throw CartanKAKError.localFactorizationFailed }

        var f1 = [C64](repeating: .zero, count: 4)
        var f2 = [C64](repeating: .zero, count: 4)
        for i in 0..<2 {
            for j in 0..<2 {
                f1[((bestA >> 1) ^ i) * 2 + ((bestB >> 1) ^ j)] =
                    matrix[(bestA ^ (i << 1)) * 4 + (bestB ^ (j << 1))]
                f2[((bestA & 1) ^ i) * 2 + ((bestB & 1) ^ j)] =
                    matrix[(bestA ^ i) * 4 + (bestB ^ j)]
            }
        }
        let det1 = f1[0] * f1[3] - f1[1] * f1[2]
        let det2 = f2[0] * f2[3] - f2[1] * f2[2]
        if det1.abs > 1e-30 { f1 = f1.map { $0 * det1.inverseSqrtPrincipal } }
        if det2.abs > 1e-30 { f2 = f2.map { $0 * det2.inverseSqrtPrincipal } }

        let f1ref = f1[(bestA >> 1) * 2 + (bestB >> 1)]
        let f2ref = f2[(bestA & 1) * 2 + (bestB & 1)]
        let denom = f1ref * f2ref
        guard denom.abs > 1e-30 else { throw CartanKAKError.localFactorizationFailed }
        var g = matrix[bestA * 4 + bestB] * C64(
            re: denom.re / denom.squaredNorm,
            im: -denom.im / denom.squaredNorm
        )
        if g.re < 0 {
            f1 = f1.map { C64(re: -$0.re, im: -$0.im) }
            g = C64(re: -g.re, im: -g.im)
        }
        let recon = kronecker(f1, f2).map { $0 * g }
        guard frobeniusDistance(recon, matrix) <= max(atol * 1e4, 1e-4) else {
            throw CartanKAKError.localFactorizationFailed
        }
        return (g, f1, f2)
    }

    static func transposeRealMatrix(_ a: [Double]) -> [Double] {
        var t = [Double](repeating: 0, count: 16)
        for r in 0..<4 {
            for c in 0..<4 {
                t[c * 4 + r] = a[r * 4 + c]
            }
        }
        return t
    }

    /// Cirq `bidiagonalize_unitary_with_special_orthogonals`.
    static func bidiagonalizeUnitarySpecialOrthogonals(
        _ mat: [C64],
        atol: Double
    ) throws -> (left: [Double], diag: [C64], right: [Double]) {
        var mat1 = [Double](repeating: 0, count: 16)
        var mat2 = [Double](repeating: 0, count: 16)
        for i in 0..<16 {
            mat1[i] = mat[i].re
            mat2[i] = mat[i].im
        }
        let (left0, right0) = try bidiagonalizeRealPair(mat1, mat2, atol: atol)

        // Force det(left)=det(right)=+1 without breaking diagonalization (Cirq).
        var left = left0
        var right = right0
        if detReal4(left) < 0 {
            for c in 0..<4 { left[0 * 4 + c] = -left[0 * 4 + c] } // flip row 0
        }
        if detReal4(right) < 0 {
            for r in 0..<4 { right[r * 4 + 0] = -right[r * 4 + 0] } // flip col 0
        }

        let leftC = realAsComplex(left)
        let rightC = realAsComplex(right)
        let mid = multiply(multiply(leftC, mat), rightC)
        var diag = [C64](repeating: .zero, count: 4)
        for i in 0..<4 { diag[i] = mid[i * 4 + i] }
        return (left, diag, right)
    }

    /// Cirq `bidiagonalize_real_matrix_pair_with_symmetric_products` (4×4).
    static func bidiagonalizeRealPair(
        _ mat1: [Double],
        _ mat2: [Double],
        atol: Double
    ) throws -> (left: [Double], right: [Double]) {
        let (baseLeft, baseS, baseVH) = try svdReal4(mat1)
        // a = U S Vh ⇒ baseLeft=U, baseVH=Vh
        var rank = 4
        while rank > 0 && abs(baseS[rank - 1]) <= atol { rank -= 1 }

        // semi = Uᵀ mat2 Vhᵀ = Uᵀ mat2 V
        let baseV = transposeRealMatrix(baseVH)
        let uT = transposeRealMatrix(baseLeft)
        let semi = multiplyReal(multiplyReal(uT, mat2), baseV)

        // Overlap block: simultaneous diagonalization with sorted singular values
        let overlapSize = rank
        var overlap = [Double](repeating: 0, count: overlapSize * overlapSize)
        for r in 0..<overlapSize {
            for c in 0..<overlapSize {
                overlap[r * overlapSize + c] = semi[r * 4 + c]
            }
        }
        var baseDiag = [Double](repeating: 0, count: overlapSize * overlapSize)
        for i in 0..<overlapSize { baseDiag[i * overlapSize + i] = baseS[i] }

        let overlapAdjust: [Double]
        if overlapSize == 0 {
            overlapAdjust = []
        } else if overlapSize == 4 {
            overlapAdjust = try diagonalizeRealSymmetricAndSortedDiagonal(overlap, baseDiag, atol: atol)
        } else {
            overlapAdjust = try diagonalizeRealSymmetricAndSortedDiagonal(overlap, baseDiag, atol: atol)
        }

        // Extra block SVD
        let extraDim = 4 - rank
        var extraLeft = identityReal(extraDim)
        var extraRight = identityReal(extraDim)
        if extraDim > 0 {
            var extra = [Double](repeating: 0, count: extraDim * extraDim)
            for r in 0..<extraDim {
                for c in 0..<extraDim {
                    extra[r * extraDim + c] = semi[(rank + r) * 4 + (rank + c)]
                }
            }
            if extraDim == 1 {
                extraLeft = [1]
                extraRight = [1]
            } else {
                let (eu, _, evh) = try svdRealN(extra, n: extraDim)
                extraLeft = eu
                extraRight = evh
            }
        }

        // left_adjust = block_diag(overlap_adjust, extra_left)
        // right_adjust = block_diag(overlap_adjust.T, extra_right)
        let leftAdjust = blockDiag(overlapAdjust, overlapSize, extraLeft, extraDim)
        let rightAdjust = blockDiag(transposeN(overlapAdjust, overlapSize), overlapSize, extraRight, extraDim)

        // left = left_adjust.T @ U.T
        // right = Vh.T @ right_adjust.T = V @ right_adjust.T
        let left = multiplyReal(transposeRealMatrix(leftAdjust), uT)
        let right = multiplyReal(baseV, transposeRealMatrix(rightAdjust))
        return (left, right)
    }

    static func diagonalizeRealSymmetricAndSortedDiagonal(
        _ symmetric: [Double],
        _ diagonal: [Double],
        atol: Double
    ) throws -> [Double] {
        let n = Int(sqrt(Double(symmetric.count)))
        // Group equal singular values
        var ranges: [(Int, Int)] = []
        var start = 0
        while start < n {
            var past = start + 1
            while past < n && abs(diagonal[start * n + start] - diagonal[past * n + past]) <= atol {
                past += 1
            }
            ranges.append((start, past))
            start = past
        }
        var p = [Double](repeating: 0, count: n * n)
        for (a, b) in ranges {
            let m = b - a
            if m == 1 {
                p[a * n + a] = 1
                continue
            }
            var block = [Double](repeating: 0, count: m * m)
            for r in 0..<m {
                for c in 0..<m {
                    block[r * m + c] = symmetric[(a + r) * n + (a + c)]
                }
            }
            let q = try syevN(block, n: m)
            for r in 0..<m {
                for c in 0..<m {
                    p[(a + r) * n + (a + c)] = q[r * m + c]
                }
            }
        }
        return p
    }

    static func svdReal4(_ a: [Double]) throws -> (u: [Double], s: [Double], vh: [Double]) {
        try svdRealN(a, n: 4)
    }

    static func svdRealN(_ a: [Double], n: Int) throws -> (u: [Double], s: [Double], vh: [Double]) {
        // Via AᵀA eigendecomposition.
        let at = transposeN(a, n)
        let ata = multiplyRealN(at, a, n)
        let (evals, evecs) = try syevNWithValues(ata, n: n) // ascending
        // Descending singular values
        var s = [Double](repeating: 0, count: n)
        var vCols = [[Double]](repeating: [], count: n)
        for i in 0..<n {
            let src = n - 1 - i
            s[i] = sqrt(max(0, evals[src]))
            var col = [Double](repeating: 0, count: n)
            for r in 0..<n {
                col[r] = evecs[r * n + src]
            }
            vCols[i] = col
        }
        var vh = [Double](repeating: 0, count: n * n)
        for j in 0..<n {
            for c in 0..<n {
                vh[j * n + c] = vCols[j][c] // Vh = Vᵀ with rows = eigenvector cols transposed... 
                // V columns are eigenvectors; Vh[j,c] = V[c,j] = evec_j[c]
            }
        }
        // Fix: Vh row j = vCols[j]^T, so vh[j,c] = vCols[j][c]. Yes.

        var u = [Double](repeating: 0, count: n * n)
        for j in 0..<n {
            let inv = s[j] > 1e-15 ? 1.0 / s[j] : 0.0
            for r in 0..<n {
                var sum = 0.0
                for c in 0..<n {
                    sum += a[r * n + c] * vCols[j][c]
                }
                u[r * n + j] = sum * inv
            }
        }
        // Orthonormalize null space if needed (σ≈0): keep as zeros / complete basis
        return (u, s, vh)
    }

    static func syevN(_ symmetricRowMajor: [Double], n: Int) throws -> [Double] {
        let (_, vecs) = try syevNWithValues(symmetricRowMajor, n: n)
        return vecs
    }

    static func syevNWithValues(_ symmetricRowMajor: [Double], n: Int) throws -> (values: [Double], vectors: [Double]) {
        var jobz = Int8(UInt8(ascii: "V"))
        var uplo = Int8(UInt8(ascii: "L"))
        var nn: __CLPK_integer = __CLPK_integer(n)
        var lda = nn
        var a = [Double](repeating: 0, count: n * n)
        for col in 0..<n {
            for row in 0..<n {
                a[col * n + row] = symmetricRowMajor[row * n + col]
            }
        }
        var w = [Double](repeating: 0, count: n)
        var workQuery = [Double](repeating: 0, count: 1)
        var lwork: __CLPK_integer = -1
        var info: __CLPK_integer = 0
        dsyev_(&jobz, &uplo, &nn, &a, &lda, &w, &workQuery, &lwork, &info)
        guard info == 0 else { throw CartanKAKError.diagonalizationFailed }
        lwork = max(1, __CLPK_integer(workQuery[0]))
        var work = [Double](repeating: 0, count: Int(lwork))
        info = 0
        dsyev_(&jobz, &uplo, &nn, &a, &lda, &w, &work, &lwork, &info)
        guard info == 0 else { throw CartanKAKError.diagonalizationFailed }
        var p = [Double](repeating: 0, count: n * n)
        for col in 0..<n {
            for row in 0..<n {
                p[row * n + col] = a[col * n + row]
            }
        }
        return (w, p)
    }

    static func syev4(_ symmetricRowMajor: [Double]) throws -> [Double] {
        try syevN(symmetricRowMajor, n: 4)
    }

    static func multiplyReal(_ a: [Double], _ b: [Double]) -> [Double] {
        multiplyRealN(a, b, 4)
    }

    static func multiplyRealN(_ a: [Double], _ b: [Double], _ n: Int) -> [Double] {
        var out = [Double](repeating: 0, count: n * n)
        for r in 0..<n {
            for c in 0..<n {
                var sum = 0.0
                for k in 0..<n {
                    sum += a[r * n + k] * b[k * n + c]
                }
                out[r * n + c] = sum
            }
        }
        return out
    }

    static func transposeN(_ a: [Double], _ n: Int) -> [Double] {
        var t = [Double](repeating: 0, count: n * n)
        for r in 0..<n {
            for c in 0..<n {
                t[c * n + r] = a[r * n + c]
            }
        }
        return t
    }

    static func identityReal(_ n: Int) -> [Double] {
        var m = [Double](repeating: 0, count: max(n * n, 0))
        for i in 0..<n { m[i * n + i] = 1 }
        return m
    }

    static func blockDiag(_ a: [Double], _ na: Int, _ b: [Double], _ nb: Int) -> [Double] {
        let n = na + nb
        var out = [Double](repeating: 0, count: n * n)
        for r in 0..<na {
            for c in 0..<na {
                out[r * n + c] = a[r * na + c]
            }
        }
        for r in 0..<nb {
            for c in 0..<nb {
                out[(na + r) * n + (na + c)] = b[r * nb + c]
            }
        }
        return out
    }

    static func emitGates(
        weyl: WeylPieces,
        qubits: (Int, Int),
        atol: Double
    ) throws -> [Gate] {
        let q0 = qubits.0
        let q1 = qubits.1
        var gates: [Gate] = []

        if let g0 = try uGate(from: weyl.k2r, target: q0, atol: atol) { gates.append(g0) }
        if let g1 = try uGate(from: weyl.k2l, target: q1, atol: atol) { gates.append(g1) }
        gates.append(contentsOf: interactionGates(x: weyl.x, y: weyl.y, z: weyl.z, q0: q0, q1: q1, atol: atol))
        if let g0 = try uGate(from: weyl.k1r, target: q0, atol: atol) { gates.append(g0) }
        if let g1 = try uGate(from: weyl.k1l, target: q1, atol: atol) { gates.append(g1) }
        return gates
    }

    /// Cirq `_non_local_part` with full CZs rewritten as CX (≤3).
    static func interactionGates(
        x: Double, y: Double, z: Double,
        q0: Int, q1: Int,
        atol: Double
    ) -> [Gate] {
        let ax = abs(x), ay = abs(y), az = abs(z)
        if ax < atol && ay < atol && az < atol {
            return []
        }
        if ay < atol && az < atol {
            return xxOnlyCX(x: x, q0: q0, q1: q1)
        }
        if az < atol {
            return xxYyCX(x: x, y: y, q0: q0, q1: q1)
        }
        return xxYyZzCX(x: x, y: y, z: z, q0: q0, q1: q1)
    }

    /// Cirq `_xx_interaction_via_full_czs` with `H·CZ·H → CX`.
    static func xxOnlyCX(x: Double, q0: Int, q1: Int) -> [Gate] {
        let a = x * -2 / Double.pi
        return [
            .cx(control: q0, target: q1),
            .rx(theta: QFloatExpr(QFloat(a * Double.pi)), target: q0),
            .cx(control: q0, target: q1),
        ]
    }

    /// Cirq `_xx_yy_interaction_via_full_czs` with `H·CZ·H → CX`.
    static func xxYyCX(x: Double, y: Double, q0: Int, q1: Int) -> [Gate] {
        let a = x * -2 / Double.pi
        let b = y * -2 / Double.pi
        return [
            .rx(theta: QFloatExpr(QFloat(Double.pi / 2)), target: q0),
            .cx(control: q0, target: q1),
            .rx(theta: QFloatExpr(QFloat(a * Double.pi)), target: q0),
            .ry(theta: QFloatExpr(QFloat(b * Double.pi)), target: q1),
            .cx(control: q0, target: q1),
            .rx(theta: QFloatExpr(QFloat(-Double.pi / 2)), target: q0),
        ]
    }

    /// Cirq `_xx_yy_zz_interaction_via_full_czs` → CX.
    static func xxYyZzCX(x: Double, y: Double, z: Double, q0: Int, q1: Int) -> [Gate] {
        let a = x * -2 / Double.pi + 0.5
        let b = y * -2 / Double.pi + 0.5
        let c = z * -2 / Double.pi + 0.5
        return [
            .rx(theta: QFloatExpr(QFloat(Double.pi / 2)), target: q0),
            .cx(control: q0, target: q1),
            .rx(theta: QFloatExpr(QFloat(a * Double.pi)), target: q0),
            .ry(theta: QFloatExpr(QFloat(b * Double.pi)), target: q1),
            .cx(control: q1, target: q0),
            .rx(theta: QFloatExpr(QFloat(-Double.pi / 2)), target: q1),
            .rz(theta: QFloatExpr(QFloat(c * Double.pi)), target: q1),
            .cx(control: q0, target: q1),
        ]
    }

    static func uGate(from matrix2: [C64], target: Int, atol: Double) throws -> Gate? {
        let um = UnitaryMatrix(
            dimension: 2,
            elements: matrix2.map { UnitaryComplex(re: $0.re, im: $0.im) }
        )
        if um.isApproximatelyEqual(to: .identity(2), tolerance: atol) {
            return nil
        }
        // Always emit ``Gate/u`` — never ``Gate/unitary1`` (basis translate rejects it).
        // ``eulerUAngles`` returns nil only for non-2×2; locals here are always 2×2.
        guard let angles = GateFusionPass.eulerUAngles(from: um, tolerance: atol) else {
            throw CartanKAKError.localFactorizationFailed
        }
        return .u(
            theta: QFloatExpr(QFloat(angles.theta)),
            phi: QFloatExpr(QFloat(angles.phi)),
            lambda: QFloatExpr(QFloat(angles.lambda)),
            target: target
        )
    }

    static func remapGate(_ gate: Gate, from: (Int, Int), to: (Int, Int)) -> Gate {
        func mapQ(_ q: Int) -> Int {
            if q == from.0 { return to.0 }
            if q == from.1 { return to.1 }
            return q
        }
        switch gate {
        case .u(let theta, let phi, let lambda, let target):
            return .u(theta: theta, phi: phi, lambda: lambda, target: mapQ(target))
        case .unitary1(let matrix, let target):
            return .unitary1(matrix: matrix, target: mapQ(target))
        case .rx(let theta, let target):
            return .rx(theta: theta, target: mapQ(target))
        case .ry(let theta, let target):
            return .ry(theta: theta, target: mapQ(target))
        case .rz(let theta, let target):
            return .rz(theta: theta, target: mapQ(target))
        case .h(let target):
            return .h(target: mapQ(target))
        case .cx(let control, let target):
            return .cx(control: mapQ(control), target: mapQ(target))
        default:
            return gate
        }
    }
}

// MARK: - Host complex scalar / 4×4 helpers

private struct C64: Equatable {
    var re: Double
    var im: Double

    static let zero = C64(re: 0, im: 0)
    static let one = C64(re: 1, im: 0)

    static func + (lhs: C64, rhs: C64) -> C64 { C64(re: lhs.re + rhs.re, im: lhs.im + rhs.im) }
    static func - (lhs: C64, rhs: C64) -> C64 { C64(re: lhs.re - rhs.re, im: lhs.im - rhs.im) }
    static func * (lhs: C64, rhs: C64) -> C64 {
        C64(re: lhs.re * rhs.re - lhs.im * rhs.im, im: lhs.re * rhs.im + lhs.im * rhs.re)
    }
    static func * (lhs: Double, rhs: C64) -> C64 { C64(re: lhs * rhs.re, im: lhs * rhs.im) }

    var conjugate: C64 { C64(re: re, im: -im) }
    var abs: Double { hypot(re, im) }
    var squaredNorm: Double { re * re + im * im }

    /// Principal `1/√z` for SU(2) renormalization.
    var inverseSqrtPrincipal: C64 {
        let n = abs
        if n < 1e-30 { return .one }
        let halfArg = atan2(im, re) / 2.0
        let invSqrtN = 1.0 / sqrt(n)
        return C64(re: invSqrtN * cos(-halfArg), im: invSqrtN * sin(-halfArg))
    }

    func asAmplitude() -> ComplexAmplitude {
        ComplexAmplitude(real: QFloat(re), imaginary: QFloat(im))
    }
}

private let identity2: [C64] = [.one, .zero, .zero, .one]

private func multiply(_ a: [C64], _ b: [C64]) -> [C64] {
    var out = [C64](repeating: .zero, count: 16)
    for r in 0..<4 {
        for c in 0..<4 {
            var sum = C64.zero
            for k in 0..<4 {
                sum = sum + a[r * 4 + k] * b[k * 4 + c]
            }
            out[r * 4 + c] = sum
        }
    }
    return out
}

private func multiply2(_ a: [C64], _ b: [C64]) -> [C64] {
    var out = [C64](repeating: .zero, count: 4)
    for r in 0..<2 {
        for c in 0..<2 {
            var sum = C64.zero
            for k in 0..<2 {
                sum = sum + a[r * 2 + k] * b[k * 2 + c]
            }
            out[r * 2 + c] = sum
        }
    }
    return out
}

private func transpose(_ a: [C64]) -> [C64] {
    var t = [C64](repeating: .zero, count: 16)
    for r in 0..<4 {
        for c in 0..<4 {
            t[c * 4 + r] = a[r * 4 + c]
        }
    }
    return t
}

private func transposeReal(_ a: [Double]) -> [C64] {
    var t = [C64](repeating: .zero, count: 16)
    for r in 0..<4 {
        for c in 0..<4 {
            t[c * 4 + r] = C64(re: a[r * 4 + c], im: 0)
        }
    }
    return t
}

private func realAsComplex(_ a: [Double]) -> [C64] {
    a.map { C64(re: $0, im: 0) }
}

private func adjoint2(_ a: [C64]) -> [C64] {
    [a[0].conjugate, a[2].conjugate, a[1].conjugate, a[3].conjugate]
}

private func adjoint4(_ a: [C64]) -> [C64] {
    var t = [C64](repeating: .zero, count: 16)
    for r in 0..<4 {
        for c in 0..<4 {
            t[c * 4 + r] = a[r * 4 + c].conjugate
        }
    }
    return t
}

private func kronecker(_ a: [C64], _ b: [C64]) -> [C64] {
    // kron(A,B) with A on qubit 1 (MSB), B on qubit 0 (LSB)
    var out = [C64](repeating: .zero, count: 16)
    for i in 0..<2 {
        for j in 0..<2 {
            for k in 0..<2 {
                for l in 0..<2 {
                    out[(2 * i + k) * 4 + (2 * j + l)] = a[i * 2 + j] * b[k * 2 + l]
                }
            }
        }
    }
    return out
}

private func trace(_ a: [C64]) -> C64 {
    a[0] + a[5] + a[10] + a[15]
}

private func frobeniusDistance(_ a: [C64], _ b: [C64]) -> Double {
    var s = 0.0
    for i in a.indices {
        let d = a[i] - b[i]
        s += d.squaredNorm
    }
    return sqrt(s)
}

private func phaseAlign(_ v: [C64], to u: [C64]) -> [C64] {
    var bestIdx = 0
    var bestMag = 0.0
    for i in u.indices {
        let m = u[i].abs
        if m > bestMag {
            bestMag = m
            bestIdx = i
        }
    }
    guard bestMag > 1e-14, v[bestIdx].abs > 1e-14 else { return v }
    let ratio = u[bestIdx] * C64(
        re: v[bestIdx].re / v[bestIdx].squaredNorm,
        im: -v[bestIdx].im / v[bestIdx].squaredNorm
    )
    let n = ratio.abs
    guard n > 1e-30 else { return v }
    let ph = C64(re: ratio.re / n, im: ratio.im / n)
    return v.map { $0 * ph }
}

private func averageGateFidelityC64(_ u: [C64], _ v: [C64]) -> Double {
    let t = trace(multiply(adjoint4(u), v))
    let trAbsSq = t.squaredNorm
    return (4.0 + trAbsSq) / 20.0
}

private func determinant4(_ a: [C64]) -> C64 {
    var m = a
    var det = C64.one
    for k in 0..<4 {
        var pivot = k
        var best = m[k * 4 + k].squaredNorm
        for r in (k + 1)..<4 {
            let nrm = m[r * 4 + k].squaredNorm
            if nrm > best {
                best = nrm
                pivot = r
            }
        }
        if best < 1e-30 { return .zero }
        if pivot != k {
            for c in 0..<4 {
                m.swapAt(k * 4 + c, pivot * 4 + c)
            }
            det = C64(re: -det.re, im: -det.im)
        }
        let diag = m[k * 4 + k]
        det = det * diag
        let invDiag = C64(re: diag.re / diag.squaredNorm, im: -diag.im / diag.squaredNorm)
        for r in (k + 1)..<4 {
            let factor = m[r * 4 + k] * invDiag
            for c in k..<4 {
                m[r * 4 + c] = m[r * 4 + c] - (factor * m[k * 4 + c])
            }
        }
    }
    return det
}

private func detReal4(_ a: [Double]) -> Double {
    // Laplace on 4×4
    func minor(_ m: [Double], skipR: Int, skipC: Int) -> [Double] {
        var out = [Double]()
        out.reserveCapacity(9)
        for r in 0..<4 where r != skipR {
            for c in 0..<4 where c != skipC {
                out.append(m[r * 4 + c])
            }
        }
        return out
    }
    func det3(_ m: [Double]) -> Double {
        m[0] * (m[4] * m[8] - m[5] * m[7])
            - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    }
    var det = 0.0
    for c in 0..<4 {
        let sign = c % 2 == 0 ? 1.0 : -1.0
        det += sign * a[c] * det3(minor(a, skipR: 0, skipC: c))
    }
    return det
}

private func argsort(_ values: [Double]) -> [Int] {
    values.indices.sorted { values[$0] < values[$1] }
}

private extension Double {
    var modTwoPi: Double {
        var x = self.truncatingRemainder(dividingBy: 2 * Double.pi)
        if x < 0 { x += 2 * Double.pi }
        return x
    }

    func mod(_ m: Double) -> Double {
        var x = truncatingRemainder(dividingBy: m)
        if x < 0 { x += m }
        return x
    }
}
