import Foundation

// MARK: - Generating set

/// Fixed discrete generators for approximate **1-qubit** synthesis (ε₀-net foundation).
///
/// Recursive Solovay–Kitaev is **not** implemented here — only the basic net + nearest neighbor.
public enum Discrete1QGeneratingSet: String, Sendable, Equatable, CaseIterable {
    /// `{H, T}` — Dawson–Nielsen / classic Clifford+T generators.
    ///
    /// `S = T²` is **not** an independent generator (it appears as a length-2 word).
    /// When ``Discrete1QNet/build(generatingSet:includeInverses:maxWordLength:calibrationSampleCount:calibrationSeed:coveringRadiusSafetyFactor:)``
    /// is called with `includeInverses: true` (default), the alphabet is `{H, T, T†}`
    /// (`H† = H`).
    case ht
}

// MARK: - Net element / lookup result

/// One stored group element: a shortest ``Gate`` word and its 2×2 unitary.
public struct Discrete1QNetElement: Sendable, Equatable {
    /// Left-to-right apply order on a single qubit (default target `0` in stored gates).
    public let gates: [Gate]
    /// Row-major 2×2 unitary in U(2) (not forced into SU(2); see ``Discrete1QNet`` phase docs).
    public let matrix: [ComplexAmplitude]
    /// Number of generators in ``gates`` (0 for identity).
    public let wordLength: Int

    /// Copy with every 1Q gate retargeted to `qubit` (matrix unchanged).
    public func retargeted(to qubit: Int) -> Discrete1QNetElement {
        Discrete1QNetElement(
            gates: gates.map { Discrete1QNet.retargetGate($0, to: qubit) },
            matrix: matrix,
            wordLength: wordLength
        )
    }
}

/// Result of nearest-neighbor lookup in a ``Discrete1QNet``.
public struct Discrete1QNearestNeighbor: Sendable, Equatable {
    public let element: Discrete1QNetElement
    /// Phase-aligned Frobenius distance ``Discrete1QNet/phaseAlignedFrobenius(target:candidate:)``.
    public let distance: Double
    /// Index into ``Discrete1QNet/elements``.
    public let index: Int
}

// MARK: - ε₀-net

/// Exhaustive **1-qubit** ε₀-net over a fixed discrete generating set (SK precursor).
///
/// ## Generating set (default)
/// ``Discrete1QGeneratingSet/ht`` = `{H, T}`. With `includeInverses: true` the BFS alphabet
/// is `{H, T, T†}`. `S = T²` is available only as a composite word, not a native letter.
///
/// ## Phase convention (U(2) / PU(2))
/// Stored matrices are exact generator products in **U(2)** (global phases from `T` / `T†`
/// are kept). Lookup distance is **global-phase invariant**:
///
/// ```
/// d(U, V) = min_φ ‖U − e^{iφ} V‖_F = √(‖U‖_F² + ‖V‖_F² − 2 |Tr(U† V)|)
/// ```
///
/// For exact 2×2 unitaries this is `√(4 − 2 |Tr(U†V)|)`. Equivalently, the metric is on
/// **PU(2) ≅ SO(3)**; SU(2) vs U(2) targets are interchangeable under this distance.
///
/// This is the same *notion* of global-phase invariance used by
/// ``CircuitEquivalenceVerifier`` and ``CartanKAK/phaseAlignedFrobenius(target:candidate:)``,
/// but **not the same algorithm**: those helpers use entry-wise phase heuristics, while this
/// net uses the closed-form optimal Frobenius phase via `|Tr(U†V)|`.
///
/// ## Distance metric
/// **Operator / Frobenius**, phase-aligned as above — **not** diamond distance, trace distance
/// of states, or average-gate-fidelity. Related 1Q fidelity from the same overlap:
/// `F̄ = (2 + |Tr(U†V)|²) / 6` (not used for lookup).
///
/// ## Construction
/// Deterministic exhaustive BFS of words of length `≤ maxWordLength`. Duplicate matrices
/// (phase-aligned Frobenius `< duplicateTolerance`) keep the **shortest** word (first found).
/// Generation is **not** stochastic; only the optional covering-radius **calibration** uses
/// a seeded Haar sampler.
///
/// ## Covering radius
/// ``claimedCoveringRadius`` is an **honest sample-based upper estimate**, not a closed-form
/// group-covering theorem: the max nearest-neighbor distance over
/// `calibrationSampleCount` Haar SU(2) draws from `calibrationSeed`, inflated by
/// `coveringRadiusSafetyFactor`. New independent Haar samples are expected to fall at or
/// below this claim with high probability for the default net; it is not a rigorous ε₀ proof.
public struct Discrete1QNet: Sendable {

    /// Default BFS depth for ``build``.
    public static let defaultMaxWordLength: Int = 8
    /// Collapse tolerance when comparing net matrices up to global phase.
    public static let duplicateTolerance: Double = 1e-10
    /// Default Haar samples used to estimate ``claimedCoveringRadius``.
    public static let defaultCalibrationSampleCount: Int = 256
    /// Default PRNG seed for covering-radius calibration (deterministic).
    public static let defaultCalibrationSeed: UInt64 = 20260816
    /// Inflates the empirical max NN distance into ``claimedCoveringRadius``.
    public static let defaultCoveringRadiusSafetyFactor: Double = 1.25

    public let generatingSet: Discrete1QGeneratingSet
    public let includeInverses: Bool
    public let maxWordLength: Int
    /// Distinct net elements (identity first), shortest-word representatives.
    public let elements: [Discrete1QNetElement]
    /// Sample-based covering-radius claim (see type docs).
    public let claimedCoveringRadius: Double
    public let calibrationSampleCount: Int
    public let calibrationSeed: UInt64
    /// Empirical max NN distance before the safety factor.
    public let empiricalCoveringRadius: Double

    public var count: Int { elements.count }

    /// Build an exhaustive net and calibrate a covering-radius claim.
    ///
    /// - Parameters:
    ///   - generatingSet: Discrete alphabet family (default `{H, T}`).
    ///   - includeInverses: When `true`, include `T†` (`H` is already self-inverse).
    ///   - maxWordLength: Maximum generator word length (inclusive).
    ///   - calibrationSampleCount: Haar SU(2) draws for the covering-radius estimate.
    ///   - calibrationSeed: Seed for calibration draws (lookup itself is deterministic).
    ///   - coveringRadiusSafetyFactor: Multiplier applied to the empirical max NN distance.
    public static func build(
        generatingSet: Discrete1QGeneratingSet = .ht,
        includeInverses: Bool = true,
        maxWordLength: Int = defaultMaxWordLength,
        calibrationSampleCount: Int = defaultCalibrationSampleCount,
        calibrationSeed: UInt64 = defaultCalibrationSeed,
        coveringRadiusSafetyFactor: Double = defaultCoveringRadiusSafetyFactor
    ) -> Discrete1QNet {
        precondition(maxWordLength >= 0)
        precondition(calibrationSampleCount >= 0)
        precondition(coveringRadiusSafetyFactor >= 1.0)

        let alphabet = generators(for: generatingSet, includeInverses: includeInverses)
        var elements: [Discrete1QNetElement] = [
            Discrete1QNetElement(
                gates: [],
                matrix: identityMatrixAmplitudes,
                wordLength: 0
            )
        ]
        // Internal Double matrices parallel to `elements` for fast multiply / compare.
        var matrices: [[C2]] = [identity2]

        var queue: [(matrix: [C2], word: [Gate], length: Int)] = [
            (identity2, [], 0)
        ]
        var head = 0

        while head < queue.count {
            let current = queue[head]
            head += 1
            guard current.length < maxWordLength else { continue }

            for letter in alphabet {
                let extended = multiply2(letter.matrix, current.matrix)
                if isDuplicate(extended, among: matrices, tolerance: duplicateTolerance) {
                    continue
                }
                let word = current.word + [letter.gate]
                let length = current.length + 1
                matrices.append(extended)
                elements.append(
                    Discrete1QNetElement(
                        gates: word,
                        matrix: extended.map { $0.asAmplitude() },
                        wordLength: length
                    )
                )
                queue.append((extended, word, length))
            }
        }

        var empirical = 0.0
        if calibrationSampleCount > 0, !elements.isEmpty {
            var rng = QuantumRNG.seeded(calibrationSeed)
            for _ in 0..<calibrationSampleCount {
                let sample = sampleHaarSU2(rng: &rng)
                let d = nearestDistance(to: sample, in: elements)
                if d > empirical { empirical = d }
            }
        }

        let claimed = empirical * coveringRadiusSafetyFactor
        return Discrete1QNet(
            generatingSet: generatingSet,
            includeInverses: includeInverses,
            maxWordLength: maxWordLength,
            elements: elements,
            claimedCoveringRadius: claimed,
            calibrationSampleCount: calibrationSampleCount,
            calibrationSeed: calibrationSeed,
            empiricalCoveringRadius: empirical
        )
    }

    /// Brute-force nearest neighbor under ``phaseAlignedFrobenius(target:candidate:)``.
    public func nearestNeighbor(to target: [ComplexAmplitude]) -> Discrete1QNearestNeighbor {
        precondition(target.count == 4, "expected row-major 2×2 (4 amplitudes)")
        precondition(!elements.isEmpty)

        var bestIndex = 0
        var bestDistance = Self.phaseAlignedFrobenius(target: target, candidate: elements[0].matrix)
        for index in 1..<elements.count {
            let distance = Self.phaseAlignedFrobenius(target: target, candidate: elements[index].matrix)
            if distance < bestDistance {
                bestDistance = distance
                bestIndex = index
            }
        }
        return Discrete1QNearestNeighbor(
            element: elements[bestIndex],
            distance: bestDistance,
            index: bestIndex
        )
    }

    /// Optimal phase-aligned Frobenius distance for 2×2 matrices:
    /// `√(‖U‖_F² + ‖V‖_F² − 2 |Tr(U†V)|)`. For exact unitaries this equals `√(4 − 2 |Tr|)`.
    public static func phaseAlignedFrobenius(
        target: [ComplexAmplitude],
        candidate: [ComplexAmplitude]
    ) -> Double {
        precondition(target.count == 4 && candidate.count == 4)
        let u = target.map { C2(re: Double($0.real), im: Double($0.imaginary)) }
        let v = candidate.map { C2(re: Double($0.real), im: Double($0.imaginary)) }
        return phaseAlignedFrobenius(u, v)
    }

    /// Haar SU(2) via Ginibre → QR → `det^{-1/2}` (deterministic given ``QuantumRNG``).
    public static func sampleHaarSU2(rng: inout QuantumRNG) -> [ComplexAmplitude] {
        let u = haarUnitaryGinibreQR(dimension: 2, rng: &rng)
        let su = projectToSpecialUnitary(u, dimension: 2)
        return su.map { $0.asAmplitude() }
    }

    /// 1Q average gate fidelity from the same overlap as the net metric:
    /// `F̄ = (2 + |Tr(U†V)|²) / 6` (phase-invariant).
    public static func averageGateFidelity(
        target: [ComplexAmplitude],
        candidate: [ComplexAmplitude]
    ) -> Double {
        precondition(target.count == 4 && candidate.count == 4)
        let u = target.map { C2(re: Double($0.real), im: Double($0.imaginary)) }
        let v = candidate.map { C2(re: Double($0.real), im: Double($0.imaginary)) }
        let overlap = absTraceOverlap(u, v)
        return (2.0 + overlap * overlap) / 6.0
    }
}

// MARK: - Internals

private extension Discrete1QNet {

    struct GeneratorLetter {
        let gate: Gate
        let matrix: [C2]
    }

    static var identityMatrixAmplitudes: [ComplexAmplitude] {
        identity2.map { $0.asAmplitude() }
    }

    static func generators(
        for set: Discrete1QGeneratingSet,
        includeInverses: Bool
    ) -> [GeneratorLetter] {
        switch set {
        case .ht:
            var letters = [
                GeneratorLetter(gate: .h(target: 0), matrix: hadamard2),
                GeneratorLetter(gate: .t(target: 0), matrix: tGate2)
            ]
            if includeInverses {
                letters.append(GeneratorLetter(gate: .tdg(target: 0), matrix: tdgGate2))
            }
            return letters
        }
    }

    static func nearestDistance(to target: [ComplexAmplitude], in elements: [Discrete1QNetElement]) -> Double {
        var best = phaseAlignedFrobenius(target: target, candidate: elements[0].matrix)
        for index in 1..<elements.count {
            let d = phaseAlignedFrobenius(target: target, candidate: elements[index].matrix)
            if d < best { best = d }
        }
        return best
    }

    static func isDuplicate(_ matrix: [C2], among known: [[C2]], tolerance: Double) -> Bool {
        for other in known {
            if phaseAlignedFrobenius(matrix, other) < tolerance {
                return true
            }
        }
        return false
    }

    static func retargetGate(_ gate: Gate, to qubit: Int) -> Gate {
        switch gate {
        case .h: return .h(target: qubit)
        case .t: return .t(target: qubit)
        case .tdg: return .tdg(target: qubit)
        case .s: return .s(target: qubit)
        case .sdg: return .sdg(target: qubit)
        default:
            preconditionFailure("Discrete1QNet only retargets H/T/T†/S/S†")
        }
    }

    static func phaseAlignedFrobenius(_ u: [C2], _ v: [C2]) -> Double {
        var normU = 0.0
        var normV = 0.0
        var overlap = C2.zero
        for i in 0..<4 {
            normU += u[i].squaredNorm
            normV += v[i].squaredNorm
            overlap = overlap + (u[i].conjugate * v[i])
        }
        let d2 = max(0.0, normU + normV - 2.0 * overlap.abs)
        return sqrt(d2)
    }

    /// `|Tr(U† V)|` (Frobenius inner-product magnitude).
    static func absTraceOverlap(_ u: [C2], _ v: [C2]) -> Double {
        var tr = C2.zero
        for i in 0..<4 {
            tr = tr + (u[i].conjugate * v[i])
        }
        return tr.abs
    }
}

// MARK: - 2×2 complex helpers (file-private)

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

    var conjugate: C2 { C2(re: re, im: -im) }
    var abs: Double { hypot(re, im) }
    var squaredNorm: Double { re * re + im * im }

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

private let hadamard2: [C2] = {
    let s = sqrt(0.5)
    return [
        C2(re: s, im: 0), C2(re: s, im: 0),
        C2(re: s, im: 0), C2(re: -s, im: 0)
    ]
}()

private let tGate2: [C2] = {
    let v = sqrt(0.5)
    return [
        .one, .zero,
        .zero, C2(re: v, im: v)
    ]
}()

private let tdgGate2: [C2] = {
    let v = sqrt(0.5)
    return [
        .one, .zero,
        .zero, C2(re: v, im: -v)
    ]
}()

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

private func haarUnitaryGinibreQR(dimension: Int, rng: inout QuantumRNG) -> [C2] {
    var columns = [[C2]](repeating: [], count: dimension)
    for col in 0..<dimension {
        var column = [C2](repeating: .zero, count: dimension)
        for row in 0..<dimension {
            column[row] = complexNormal(rng: &rng)
        }
        columns[col] = column
    }

    for j in 0..<dimension {
        for i in 0..<j {
            let rij = hermitianDot(columns[i], columns[j])
            columns[j] = axpy(-1, rij, columns[i], onto: columns[j])
        }
        let norm = sqrt(max(hermitianDot(columns[j], columns[j]).re, 0))
        if norm > 1e-30 {
            columns[j] = columns[j].map { (1.0 / norm) * $0 }
        }
    }

    var matrix = [C2](repeating: .zero, count: dimension * dimension)
    for col in 0..<dimension {
        for row in 0..<dimension {
            matrix[row * dimension + col] = columns[col][row]
        }
    }
    return matrix
}

private func projectToSpecialUnitary(_ matrix: [C2], dimension: Int) -> [C2] {
    let det = determinant2(matrix)
    let phase = det.unitPhase
    let arg = atan2(phase.im, phase.re) / Double(dimension)
    let scale = C2(re: cos(-arg), im: sin(-arg))
    return matrix.map { $0 * scale }
}

private func determinant2(_ m: [C2]) -> C2 {
    // det = m00 m11 − m01 m10 (dim-2 Haar projection only).
    m[0] * m[3] - m[1] * m[2]
}

private func complexNormal(rng: inout QuantumRNG) -> C2 {
    let u1 = max(rng.nextUnitDouble(), 1e-16)
    let u2 = rng.nextUnitDouble()
    let r = sqrt(-log(u1))
    let theta = 2.0 * Double.pi * u2
    let scale = r * sqrt(0.5)
    return C2(re: scale * cos(theta), im: scale * sin(theta))
}

private func hermitianDot(_ a: [C2], _ b: [C2]) -> C2 {
    var sum = C2.zero
    for i in a.indices {
        sum = sum + (a[i].conjugate * b[i])
    }
    return sum
}

private func axpy(_ alpha: Double, _ z: C2, _ x: [C2], onto y: [C2]) -> [C2] {
    let scale = C2(re: alpha * z.re, im: alpha * z.im)
    return zip(x, y).map { scale * $0 + $1 }
}
