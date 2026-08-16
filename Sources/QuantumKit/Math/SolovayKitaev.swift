import Foundation

// MARK: - Errors / result

/// Failures from iterative Solovay–Kitaev approximation.
public enum SolovayKitaevError: Error, Equatable, Sendable {
    case invalidEpsilon(Double)
    case invalidMaxRefinementIterations(Int)
    case invalidMatrix(reason: String)
    case unboundParameters
    /// Approximation finished after `refinementIterations` but phase-aligned Frobenius distance still exceeds `epsilon`.
    case approximationFailed(achievedDistance: Double, epsilon: Double, refinementIterations: Int)
}

extension SolovayKitaevError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidEpsilon(let epsilon):
            return "Solovay–Kitaev epsilon must be > 0 (got \(epsilon)). Choose a positive approximation tolerance."
        case .invalidMaxRefinementIterations(let depth):
            return "Solovay–Kitaev maxRefinementIterations must be ≥ 0 (got \(depth))."
        case .invalidMatrix(let reason):
            return "Solovay–Kitaev rejected the 1Q unitary matrix: \(reason)."
        case .unboundParameters:
            return "Solovay–Kitaev requires literal (bound) gate parameters; bind parameters before this pass."
        case .approximationFailed(let achieved, let epsilon, let iterations):
            return "Solovay–Kitaev could not reach epsilon=\(epsilon): achieved phase-aligned Frobenius distance \(achieved) after \(iterations) refinement iteration(s). Increase maxRefinementIterations or loosen epsilon."
        }
    }
}

/// Discrete sequence approximating a 1Q unitary under ``Discrete1QNet/phaseAlignedFrobenius(target:candidate:)``.
public struct SolovayKitaevApproximation: Sendable, Equatable {
    /// Left-to-right apply order on qubit `0` (retarget with ``retargeted(to:)``).
    public let gates: [Gate]
    /// Row-major 2×2 product of ``gates`` (U(2); not forced into SU(2)).
    public let matrix: [ComplexAmplitude]
    /// ``Discrete1QNet/phaseAlignedFrobenius(target:candidate:)`` vs the requested target.
    public let distance: Double
    /// Residual-GC refinement iterations used (`0` = basic approximation only).
    public let refinementIterations: Int

    public func retargeted(to qubit: Int) -> SolovayKitaevApproximation {
        SolovayKitaevApproximation(
            gates: gates.map { Discrete1QNet.retargetGate($0, to: qubit) },
            matrix: matrix,
            distance: distance,
            refinementIterations: refinementIterations
        )
    }
}

// MARK: - Cached synthesizer

/// Holds the discrete net plus a **group-commutator expansion** used as the SK basic
/// approximation library. Build once and reuse across many targets.
public final class SolovayKitaevSynthesizer: @unchecked Sendable {
    public let net: Discrete1QNet
    fileprivate let table: SolovayKitaev.BasicApproxTable

    public init(net: Discrete1QNet, commutatorExpansionRounds: Int = 4) {
        self.net = net
        self.table = SolovayKitaev.BasicApproxTable(
            net: net,
            commutatorExpansionRounds: commutatorExpansionRounds
        )
    }

    public func approximate(
        _ target: [ComplexAmplitude],
        epsilon: Double,
        maxRefinementIterations: Int = SolovayKitaev.defaultMaxRefinementIterations
    ) throws -> SolovayKitaevApproximation {
        try SolovayKitaev.approximate(
            target,
            epsilon: epsilon,
            table: table,
            maxRefinementIterations: maxRefinementIterations
        )
    }
}

// MARK: - Algorithm

/// Iterative Dawson–Nielsen–style Solovay–Kitaev over a ``Discrete1QNet``.
///
/// This is **not** a full recursive SK tree (A and B are *not* themselves expanded by
/// recursive SK). Each refinement step approximates the residual with a **basic** library
/// lookup / ZYZ polish of the balanced group-commutator factors.
///
/// ## Distance
/// Claims use the same **phase-aligned Frobenius** metric as ``Discrete1QNet``:
/// `d(U,V) = min_φ ‖U − e^{iφ} V‖_F = √(‖U‖_F² + ‖V‖_F² − 2 |Tr(U†V)|)`.
///
/// ## Method
/// 1. Build a basic library = net words + length-capped group-commutator expansions (fills the
///    near-identity gap of plain `{H,T}` BFS).
/// 2. Basic step: library NN, or ZYZ (`Rz`–`Ry`–`Rz`) with `T`/`T†` greedy `Rz` plus
///    length-aware library polish — whichever is closer without pathological length blow-up.
/// 3. Iterative refinement up to `maxRefinementIterations`: residual `Δ = V†U` so `U = VΔ`;
///    approximate `Δ ≈ [A,B]` (or `Δ` directly) with length-aware basic approx, then form
///    `V' = V[A,B]` (right-multiply / DN update). Keep the better of GC vs direct residual
///    (shorter wins on near ties).
///
/// Prefer ``SolovayKitaevSynthesizer`` when approximating many targets (caches the library).
public enum SolovayKitaev {

    public static let defaultEpsilon: Double = 0.01
    /// Max residual-GC refinement iterations (iterative DN updates — not a recursive SK tree depth).
    public static let defaultMaxRefinementIterations: Int = 12
    /// Soft cap on library / polish words so nested GC expansions cannot explode sequence length.
    public static let maxBasicWordLength: Int = 48

    /// Approximate `target` (row-major 2×2) to phase-aligned Frobenius distance `≤ epsilon`.
    ///
    /// Prefers ``SolovayKitaevSynthesizer`` when approximating many targets (avoids rebuilding
    /// the commutator library).
    public static func approximate(
        _ target: [ComplexAmplitude],
        epsilon: Double,
        net: Discrete1QNet,
        maxRefinementIterations: Int = defaultMaxRefinementIterations
    ) throws -> SolovayKitaevApproximation {
        let synthesizer = SolovayKitaevSynthesizer(net: net)
        return try synthesizer.approximate(
            target,
            epsilon: epsilon,
            maxRefinementIterations: maxRefinementIterations
        )
    }

    /// Matrix product of a qubit-0 Clifford+T word (left-to-right apply), matching ``Discrete1QNet`` order.
    public static func matrix(ofGates gates: [Gate]) -> [ComplexAmplitude] {
        var product = identity2
        for gate in gates {
            product = multiply2(matrix(forGenerator: gate), product)
        }
        return amplitudes(product)
    }

    fileprivate static func approximate(
        _ target: [ComplexAmplitude],
        epsilon: Double,
        table: BasicApproxTable,
        maxRefinementIterations: Int
    ) throws -> SolovayKitaevApproximation {
        guard epsilon > 0 else { throw SolovayKitaevError.invalidEpsilon(epsilon) }
        guard maxRefinementIterations >= 0 else {
            throw SolovayKitaevError.invalidMaxRefinementIterations(maxRefinementIterations)
        }
        guard target.count == 4 else {
            throw SolovayKitaevError.invalidMatrix(reason: "expected 4 amplitudes (2×2 row-major)")
        }

        let suTarget = projectToSU2(c2(target))
        var current = basicApproximate(suTarget, table: table)
        var iterationsUsed = 0
        var bestDistance = Discrete1QNet.phaseAlignedFrobenius(
            target: target,
            candidate: amplitudes(current.matrix)
        )

        if bestDistance <= epsilon {
            return SolovayKitaevApproximation(
                gates: current.gates,
                matrix: amplitudes(current.matrix),
                distance: bestDistance,
                refinementIterations: 0
            )
        }

        // Iterative DN update: Δ = V†U ⇒ U = VΔ; set V' = V[A,B] (or V W) with basic approx
        // of A,B / W. LTR gate order: Δ (or [A,B]) applied first, then V ⇒ matrix V·Δ.
        // Length-aware basic approx + library word caps keep sequences bounded; prefer shorter
        // on near ties.
        if maxRefinementIterations >= 1 {
            for iter in 1...maxRefinementIterations {
                let residual = multiply2(adjoint2(current.matrix), suTarget)
                let (aMat, bMat) = groupCommutatorFactors(residual)
                let aApprox = basicApproximate(aMat, table: table)
                let bApprox = basicApproximate(bMat, table: table)
                // [A,B] word then V: V' = V [A,B]
                let gcWord =
                    adjointWord(bApprox.gates)
                    + adjointWord(aApprox.gates)
                    + bApprox.gates
                    + aApprox.gates
                let gcMatrix = multiply2(
                    aApprox.matrix,
                    multiply2(
                        bApprox.matrix,
                        multiply2(adjoint2(aApprox.matrix), adjoint2(bApprox.matrix))
                    )
                )
                let gcCandidate = Approx(
                    gates: gcWord + current.gates,
                    matrix: multiply2(current.matrix, gcMatrix)
                )

                let w = basicApproximate(residual, table: table)
                let residualCandidate = Approx(
                    gates: w.gates + current.gates,
                    matrix: multiply2(current.matrix, w.matrix)
                )

                var stepBest = current
                var stepBestDistance = bestDistance
                for candidate in [gcCandidate, residualCandidate] {
                    let distance = Discrete1QNet.phaseAlignedFrobenius(
                        target: target,
                        candidate: amplitudes(candidate.matrix)
                    )
                    if isPreferable(
                        distance: distance,
                        length: candidate.gates.count,
                        overDistance: stepBestDistance,
                        overLength: stepBest.gates.count
                    ) {
                        stepBest = candidate
                        stepBestDistance = distance
                    }
                }

                if stepBestDistance < bestDistance - 1e-15
                    || (abs(stepBestDistance - bestDistance) <= 1e-15
                        && stepBest.gates.count < current.gates.count)
                {
                    current = stepBest
                    bestDistance = stepBestDistance
                    iterationsUsed = iter
                } else {
                    break
                }

                if bestDistance <= epsilon {
                    return SolovayKitaevApproximation(
                        gates: current.gates,
                        matrix: amplitudes(current.matrix),
                        distance: bestDistance,
                        refinementIterations: iterationsUsed
                    )
                }
            }
        }

        throw SolovayKitaevError.approximationFailed(
            achievedDistance: bestDistance,
            epsilon: epsilon,
            refinementIterations: iterationsUsed
        )
    }
}

// MARK: - Internals

fileprivate extension SolovayKitaev {

    struct Approx {
        var gates: [Gate]
        var matrix: [C2]
    }

    /// Prefer strictly better distance; on near ties prefer the shorter word.
    static func isPreferable(
        distance: Double,
        length: Int,
        overDistance: Double,
        overLength: Int
    ) -> Bool {
        if distance < overDistance - 1e-12 { return true }
        if distance > overDistance + 1e-12 { return false }
        return length < overLength
    }

    /// Net words plus length-capped group-commutator expansions (fills the near-identity gap).
    final class BasicApproxTable: @unchecked Sendable {
        let gates: [[Gate]]
        let matrices: [[C2]]

        init(net: Discrete1QNet, commutatorExpansionRounds: Int) {
            var gates: [[Gate]] = []
            var matrices: [[C2]] = []
            var seen = Set<PhaseKey>()
            let wordCap = max(maxBasicWordLength, net.maxWordLength)

            func insert(_ word: [Gate], _ matrix: [C2]) {
                guard word.count <= wordCap else { return }
                let key = PhaseKey(matrix)
                if seen.insert(key).inserted {
                    gates.append(word)
                    matrices.append(matrix)
                }
            }

            for element in net.elements {
                insert(element.gates, c2(element.matrix))
            }

            // Seed short words for commutator expansion.
            var poolIndices = matrices.indices.filter { gates[$0].count <= 3 }
            let poolCap = 48

            for round in 0..<max(0, commutatorExpansionRounds) {
                let ranked = poolIndices.sorted {
                    let d0 = SolovayKitaev.phaseAlignedFrobeniusC2(identity2, matrices[$0])
                    let d1 = SolovayKitaev.phaseAlignedFrobeniusC2(identity2, matrices[$1])
                    if abs(d0 - d1) > 1e-15 { return d0 < d1 }
                    return gates[$0].count < gates[$1].count
                }
                // First round: use all short seeds; later rounds: nearest-to-I pool.
                let pool = round == 0 ? ranked : Array(ranked.prefix(poolCap))
                var newIndices: [Int] = []
                for i in pool {
                    for j in pool {
                        let aGates = gates[i]
                        let bGates = gates[j]
                        // Word apply order for matrix A B A† B†: B†, A†, B, A.
                        let word =
                            SolovayKitaev.adjointWord(bGates)
                            + SolovayKitaev.adjointWord(aGates)
                            + bGates
                            + aGates
                        guard word.count <= wordCap else { continue }
                        let matrix = multiply2(
                            matrices[i],
                            multiply2(
                                matrices[j],
                                multiply2(adjoint2(matrices[i]), adjoint2(matrices[j]))
                            )
                        )
                        let before = seen.count
                        insert(word, matrix)
                        if seen.count > before {
                            newIndices.append(gates.count - 1)
                        }
                    }
                }
                poolIndices.append(contentsOf: newIndices)
            }

            self.gates = gates
            self.matrices = matrices
        }

        func nearest(to target: [C2]) -> Approx {
            var bestIndex = 0
            var bestDistance = SolovayKitaev.phaseAlignedFrobeniusC2(target, matrices[0])
            for index in 1..<matrices.count {
                let distance = SolovayKitaev.phaseAlignedFrobeniusC2(target, matrices[index])
                if SolovayKitaev.isPreferable(
                    distance: distance,
                    length: gates[index].count,
                    overDistance: bestDistance,
                    overLength: gates[bestIndex].count
                ) {
                    bestDistance = distance
                    bestIndex = index
                }
            }
            return Approx(gates: gates[bestIndex], matrix: matrices[bestIndex])
        }
    }

    /// Depth-0 library lookup, with ZYZ fallback when the net+GC library is still coarse.
    static func basicApproximate(_ target: [C2], table: BasicApproxTable) -> Approx {
        let direct = table.nearest(to: target)
        let directDistance = phaseAlignedFrobeniusC2(target, direct.matrix)
        let zyz = approximateViaZYZ(target, table: table)
        let zyzDistance = phaseAlignedFrobeniusC2(target, zyz.matrix)
        // Prefer closer; on near ties prefer shorter. Never reject a clearly closer ZYZ.
        if isPreferable(
            distance: zyzDistance,
            length: zyz.gates.count,
            overDistance: directDistance,
            overLength: direct.gates.count
        ) {
            return zyz
        }
        return direct
    }

    /// U ≈ Rz(α) Ry(β) Rz(γ) with Ry(β) = Rz(-π/2) H Rz(β) H Rz(π/2)
    /// (left-to-right apply). Merged: Rz(γ−π/2), H, Rz(β), H, Rz(α+π/2).
    static func approximateViaZYZ(_ target: [C2], table: BasicApproxTable) -> Approx {
        let (alpha, beta, gamma) = zyzAngles(target)
        let first = approximateRz(gamma - Double.pi / 2.0, table: table)
        let mid = approximateRz(beta, table: table)
        let last = approximateRz(alpha + Double.pi / 2.0, table: table)

        let h: [Gate] = [.h(target: 0)]
        let gates = first.gates + h + mid.gates + h + last.gates
        let matrix = multiply2(
            last.matrix,
            multiply2(
                hadamard2,
                multiply2(mid.matrix, multiply2(hadamard2, first.matrix))
            )
        )
        return Approx(gates: gates, matrix: matrix)
    }

    static func approximateRz(_ thetaIn: Double, table: BasicApproxTable) -> Approx {
        var theta = thetaIn.remainder(dividingBy: 2.0 * Double.pi)
        if theta > Double.pi { theta -= 2.0 * Double.pi }
        if theta <= -Double.pi { theta += 2.0 * Double.pi }

        var gates: [Gate] = []
        var matrix = identity2
        let step = Double.pi / 4.0
        while theta > step / 2.0 + 1e-12 {
            gates.append(.t(target: 0))
            matrix = multiply2(tGate2, matrix)
            theta -= step
        }
        while theta < -step / 2.0 - 1e-12 {
            gates.append(.tdg(target: 0))
            matrix = multiply2(tdgGate2, matrix)
            theta += step
        }

        let remainderTarget = rotation(axis: (0, 0, 1), angle: theta)
        var rem = table.nearest(to: remainderTarget)
        var remDistance = phaseAlignedFrobeniusC2(remainderTarget, rem.matrix)
        let polishGrowthCap = maxBasicWordLength
        for _ in 0..<8 {
            if remDistance < 1e-6 { break }
            let residual = multiply2(adjoint2(rem.matrix), remainderTarget)
            let (aMat, bMat) = groupCommutatorFactors(residual)
            let aApprox = table.nearest(to: aMat)
            let bApprox = table.nearest(to: bMat)
            // DN right-multiply: rem' = rem · [A,B] (LTR: GC word then rem)
            let gcWord =
                adjointWord(bApprox.gates)
                + adjointWord(aApprox.gates)
                + bApprox.gates
                + aApprox.gates
            let gcMatrix = multiply2(
                aApprox.matrix,
                multiply2(
                    bApprox.matrix,
                    multiply2(adjoint2(aApprox.matrix), adjoint2(bApprox.matrix))
                )
            )
            let gcCand = Approx(
                gates: gcWord + rem.gates,
                matrix: multiply2(rem.matrix, gcMatrix)
            )
            let w = table.nearest(to: residual)
            let resCand = Approx(
                gates: w.gates + rem.gates,
                matrix: multiply2(rem.matrix, w.matrix)
            )
            var improved = false
            for candidate in [resCand, gcCand] {
                let added = candidate.gates.count - rem.gates.count
                guard added <= polishGrowthCap else { continue }
                let distance = phaseAlignedFrobeniusC2(remainderTarget, candidate.matrix)
                if distance < remDistance - 1e-15
                    || (abs(distance - remDistance) <= 1e-15
                        && candidate.gates.count < rem.gates.count)
                {
                    rem = candidate
                    remDistance = distance
                    improved = true
                }
            }
            if !improved { break }
        }

        return Approx(
            gates: gates + rem.gates,
            matrix: multiply2(rem.matrix, matrix)
        )
    }

    /// ZYZ angles for SU(2): U ~ Rz(α) Ry(β) Rz(γ) (global phase ignored).
    static func zyzAngles(_ uIn: [C2]) -> (Double, Double, Double) {
        let u = projectToSU2(uIn)
        // Peel global phase so u00 is real when possible.
        let phase00 = atan2(u[0].im, u[0].re)
        let peel = C2(re: cos(-phase00), im: sin(-phase00))
        let a00 = peel * u[0]
        let a01 = peel * u[1]
        let a10 = peel * u[2]
        let a11 = peel * u[3]

        let cosHalf = max(-1.0, min(1.0, a00.re))
        let beta = 2.0 * acos(cosHalf)
        let sinHalf = sin(beta / 2.0)
        let alpha: Double
        let gamma: Double
        if abs(sinHalf) < 1e-12 {
            alpha = 0
            gamma = atan2(a11.im, a11.re)
        } else {
            alpha = atan2(a10.im, a10.re)
            gamma = atan2(-a01.im, -a01.re)
        }
        return (alpha, beta, gamma)
    }

    static func adjointWord(_ gates: [Gate]) -> [Gate] {
        gates.reversed().map(\.adjoint)
    }

    static func phaseAlignedFrobeniusC2(_ u: [C2], _ v: [C2]) -> Double {
        var normU = 0.0
        var normV = 0.0
        var overlap = C2.zero
        for i in 0..<4 {
            normU += u[i].re * u[i].re + u[i].im * u[i].im
            normV += v[i].re * v[i].re + v[i].im * v[i].im
            overlap = overlap + (u[i].conjugate * v[i])
        }
        return sqrt(max(0.0, normU + normV - 2.0 * overlap.abs))
    }

    /// Exact balanced GC: find `A,B` with `[A,B] = U` for `U ∈ SU(2)`.
    static func groupCommutatorFactors(_ u: [C2]) -> ([C2], [C2]) {
        let (theta, axis) = axisAngle(u)
        if theta < 1e-14 {
            return (identity2, identity2)
        }
        let phi = phiForCommutatorAngle(theta)
        let rx = rotation(axis: (1, 0, 0), angle: phi)
        let ry = rotation(axis: (0, 1, 0), angle: phi)
        let gc0 = multiply2(rx, multiply2(ry, multiply2(adjoint2(rx), adjoint2(ry))))
        let (_, gcAxis) = axisAngle(gc0)
        let s = alignRotation(from: gcAxis, to: axis)
        let sDag = adjoint2(s)
        let a = multiply2(s, multiply2(rx, sDag))
        let b = multiply2(s, multiply2(ry, sDag))
        return (a, b)
    }

    static func phiForCommutatorAngle(_ theta: Double) -> Double {
        let target = min(max(theta, 0), Double.pi)
        var lo = 1e-12
        var hi = 2.0
        for _ in 0..<80 {
            let mid = 0.5 * (lo + hi)
            let rx = rotation(axis: (1, 0, 0), angle: mid)
            let ry = rotation(axis: (0, 1, 0), angle: mid)
            let gc = multiply2(rx, multiply2(ry, multiply2(adjoint2(rx), adjoint2(ry))))
            let (got, _) = axisAngle(gc)
            if got < target {
                lo = mid
            } else {
                hi = mid
            }
        }
        return 0.5 * (lo + hi)
    }

    static func axisAngle(_ uIn: [C2]) -> (Double, (Double, Double, Double)) {
        let u = projectToSU2(uIn)
        let tr = (u[0] + u[3]).re
        let c = max(-1.0, min(1.0, tr / 2.0))
        var theta = 2.0 * acos(c)
        if theta < 1e-14 || abs(theta - 2.0 * Double.pi) < 1e-14 {
            return (0, (0, 0, 1))
        }
        let s = sin(theta / 2.0)
        let m01 = C2(re: 0, im: 1) * (u[1] * (1.0 / s))
        let m00 = C2(re: 0, im: 1) * ((u[0] - C2(re: c, im: 0)) * (1.0 / s))
        var nx = m01.re
        var ny = -m01.im
        var nz = m00.re
        let n = sqrt(nx * nx + ny * ny + nz * nz)
        if n < 1e-14 {
            return (0, (0, 0, 1))
        }
        nx /= n
        ny /= n
        nz /= n
        if theta > Double.pi {
            theta = 2.0 * Double.pi - theta
            nx = -nx
            ny = -ny
            nz = -nz
        }
        return (theta, (nx, ny, nz))
    }

    static func alignRotation(
        from: (Double, Double, Double),
        to: (Double, Double, Double)
    ) -> [C2] {
        let f = normalize3(from)
        let t = normalize3(to)
        let c = dot3(f, t)
        if c > 1.0 - 1e-14 {
            return identity2
        }
        if c < -1.0 + 1e-14 {
            let p: (Double, Double, Double)
            if abs(f.0) < 0.9 {
                p = cross3(f, (1, 0, 0))
            } else {
                p = cross3(f, (0, 1, 0))
            }
            return rotation(axis: p, angle: Double.pi)
        }
        return rotation(axis: cross3(f, t), angle: acos(max(-1, min(1, c))))
    }

    static func rotation(axis: (Double, Double, Double), angle: Double) -> [C2] {
        let n = normalize3(axis)
        let ns = add2(
            add2(scale2(n.0, pauliX), scale2(n.1, pauliY)),
            scale2(n.2, pauliZ)
        )
        return add2(
            scale2(cos(angle / 2.0), identity2),
            scale2Complex(C2(re: 0, im: -sin(angle / 2.0)), ns)
        )
    }

    static func matrix(forGenerator gate: Gate) -> [C2] {
        switch gate {
        case .h: return hadamard2
        case .t: return tGate2
        case .tdg: return tdgGate2
        case .s: return sGate2
        case .sdg: return sdgGate2
        default:
            preconditionFailure("SolovayKitaev word contains non-generator gate \(gate)")
        }
    }
}

// MARK: - Phase-collapse key

private struct PhaseKey: Hashable {
    let bits: [UInt64]

    init(_ matrix: [C2]) {
        // Align by arg(Tr) when possible so U ~ e^{iφ}V collide.
        var tr = C2.zero
        for i in [0, 3] { tr = tr + matrix[i] }
        let phase: C2
        if tr.abs > 1e-12 {
            phase = tr.unitPhase.conjugate
        } else if matrix[0].abs > 1e-12 {
            phase = matrix[0].unitPhase.conjugate
        } else {
            phase = .one
        }
        var bits: [UInt64] = []
        bits.reserveCapacity(8)
        for entry in matrix {
            let aligned = phase * entry
            bits.append(Self.quantized(aligned.re))
            bits.append(Self.quantized(aligned.im))
        }
        self.bits = bits
    }

    private static func quantized(_ value: Double) -> UInt64 {
        let scaled = (value * 1_000_000.0).rounded()
        return UInt64(bitPattern: Int64(clamping: Int(scaled)))
    }
}

// MARK: - 2×2 helpers (file-private)

private struct C2: Equatable {
    var re: Double
    var im: Double

    static let zero = C2(re: 0, im: 0)
    static let one = C2(re: 1, im: 0)

    static func + (lhs: C2, rhs: C2) -> C2 { C2(re: lhs.re + rhs.re, im: lhs.im + rhs.im) }
    static func - (lhs: C2, rhs: C2) -> C2 { C2(re: lhs.re - rhs.re, im: lhs.im - rhs.im) }
    static func * (lhs: C2, rhs: C2) -> C2 {
        C2(re: lhs.re * rhs.re - lhs.im * rhs.im, im: lhs.re * rhs.im + lhs.im * rhs.re)
    }
    static func * (lhs: Double, rhs: C2) -> C2 { C2(re: lhs * rhs.re, im: lhs * rhs.im) }
    static func * (lhs: C2, rhs: Double) -> C2 { C2(re: lhs.re * rhs, im: lhs.im * rhs) }

    var conjugate: C2 { C2(re: re, im: -im) }
    var abs: Double { hypot(re, im) }

    var unitPhase: C2 {
        let n = abs
        if n < 1e-30 { return .one }
        return C2(re: re / n, im: im / n)
    }

    func asAmplitude() -> ComplexAmplitude {
        ComplexAmplitude(real: QFloat(re), imaginary: QFloat(im))
    }
}

private let identity2: [C2] = [.one, .zero, .zero, .one]
private let pauliX: [C2] = [.zero, .one, .one, .zero]
private let pauliY: [C2] = [.zero, C2(re: 0, im: -1), C2(re: 0, im: 1), .zero]
private let pauliZ: [C2] = [.one, .zero, .zero, C2(re: -1, im: 0)]

private let hadamard2: [C2] = {
    let s = sqrt(0.5)
    return [
        C2(re: s, im: 0), C2(re: s, im: 0),
        C2(re: s, im: 0), C2(re: -s, im: 0)
    ]
}()

private let tGate2: [C2] = {
    let v = sqrt(0.5)
    return [.one, .zero, .zero, C2(re: v, im: v)]
}()

private let tdgGate2: [C2] = {
    let v = sqrt(0.5)
    return [.one, .zero, .zero, C2(re: v, im: -v)]
}()

private let sGate2: [C2] = [.one, .zero, .zero, C2(re: 0, im: 1)]
private let sdgGate2: [C2] = [.one, .zero, .zero, C2(re: 0, im: -1)]

private func multiply2(_ a: [C2], _ b: [C2]) -> [C2] {
    var out = [C2](repeating: .zero, count: 4)
    for r in 0..<2 {
        for c in 0..<2 {
            var sum = C2.zero
            for k in 0..<2 {
                sum = sum + a[r * 2 + k] * b[k * 2 + c]
            }
            out[r * 2 + c] = sum
        }
    }
    return out
}

private func adjoint2(_ a: [C2]) -> [C2] {
    [a[0].conjugate, a[2].conjugate, a[1].conjugate, a[3].conjugate]
}

private func projectToSU2(_ matrix: [C2]) -> [C2] {
    let det = matrix[0] * matrix[3] - matrix[1] * matrix[2]
    let phase = det.unitPhase
    let arg = atan2(phase.im, phase.re) / 2.0
    let scale = C2(re: cos(-arg), im: sin(-arg))
    return matrix.map { $0 * scale }
}

private func c2(_ amplitudes: [ComplexAmplitude]) -> [C2] {
    amplitudes.map { C2(re: Double($0.real), im: Double($0.imaginary)) }
}

private func amplitudes(_ matrix: [C2]) -> [ComplexAmplitude] {
    matrix.map { $0.asAmplitude() }
}

private func add2(_ a: [C2], _ b: [C2]) -> [C2] { zip(a, b).map { $0 + $1 } }
private func scale2(_ s: Double, _ m: [C2]) -> [C2] { m.map { s * $0 } }
private func scale2Complex(_ s: C2, _ m: [C2]) -> [C2] { m.map { s * $0 } }

private func normalize3(_ v: (Double, Double, Double)) -> (Double, Double, Double) {
    let n = sqrt(v.0 * v.0 + v.1 * v.1 + v.2 * v.2)
    if n < 1e-30 { return (0, 0, 1) }
    return (v.0 / n, v.1 / n, v.2 / n)
}

private func dot3(_ a: (Double, Double, Double), _ b: (Double, Double, Double)) -> Double {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

private func cross3(
    _ a: (Double, Double, Double),
    _ b: (Double, Double, Double)
) -> (Double, Double, Double) {
    (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0)
}
