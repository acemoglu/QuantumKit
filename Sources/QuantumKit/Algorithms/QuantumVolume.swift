import Foundation

// MARK: - Configuration

/// How successive depth layers pair qubits for two-qubit SU(4)/U(4) gates.
public enum QuantumVolumePairing: Sendable, Equatable {
    /// Line-friendly square layering: even layers `(0,1), (2,3), …`; odd layers
    /// `(1,2), (3,4), …`. One qubit idles when `qubitCount` is odd.
    ///
    /// **Default in this module** (routing-friendly). This is **not** the Cross et al.
    /// model-circuit default, which redraws a random pairing each layer — use
    /// ``randomPermutation`` for that.
    case nearestNeighborAlternating

    /// Cross-style layer pairing: uniform random permutation of `0..<n`, then pairs
    /// `(π[0],π[1]), (π[2],π[3]), …` (leftover qubit idle when `n` is odd).
    case randomPermutation
}

/// Options for square (or equal-depth) Quantum Volume–style model circuits.
///
/// Default ``pairing`` is ``QuantumVolumePairing/nearestNeighborAlternating`` (not
/// Cross’s per-layer random pairing). Pass ``.randomPermutation`` for Cross-like
/// pairing; HOP math is unchanged either way.
public struct QuantumVolumeOptions: Sendable, Equatable {
    /// Circuit width (and default depth when `depth == nil`).
    public var qubitCount: Int
    /// Layer count. When `nil`, uses `qubitCount` (square model circuit).
    public var depth: Int?
    public var pairing: QuantumVolumePairing
    /// When `true`, inserts a `barrier` after each SU(4) layer (identity for unitary sim).
    public var insertLayerBarriers: Bool

    public init(
        qubitCount: Int,
        depth: Int? = nil,
        pairing: QuantumVolumePairing = .nearestNeighborAlternating,
        insertLayerBarriers: Bool = false
    ) {
        self.qubitCount = qubitCount
        self.depth = depth
        self.pairing = pairing
        self.insertLayerBarriers = insertLayerBarriers
    }

    public var resolvedDepth: Int { depth ?? qubitCount }
}

public enum QuantumVolumeError: Error, Equatable, Sendable {
    case invalidQubitCount(Int)
    case invalidDepth(Int)
    case emptyProbabilities
    case probabilityCountMismatch(expected: Int, actual: Int)
}

// MARK: - Heavy Output Probability

/// Ideal (noiseless) Heavy Output Probability computed from Born probabilities.
///
/// **Definition used here (Cross et al. Quantum Volume):**
/// let `p` be the ideal outcome distribution over all `2ⁿ` bitstrings.
/// Let `m = median(p)`. An outcome `x` is *heavy* iff `p(x) > m`.
/// `HOP = Σ_{x heavy} p(x)`.
///
/// This type stores **ideal HOP**: the sum is taken over the ideal distribution itself
/// (not a shot estimate vs the heavy set). That matches the analytic noiseless check
/// `HOP > 2/3`.
///
/// **Not** a full IBM Quantum Volume certificate: hardware/device QV also needs
/// experimental (shot) heavy-output estimates and confidence intervals. This module
/// validates the ideal / simulator-side HOP pipeline.
public struct HeavyOutputProbability: Sendable, Equatable {
    public let median: Double
    public let hop: Double
    /// Computational-basis indices (engine LSB: qubit 0 = LSB) that are heavy.
    public let heavyIndices: [Int]
    public let probabilityCount: Int

    /// Standard QV noiseless acceptance threshold (`HOP > 2/3`, strict).
    public static let acceptanceThreshold: Double = 2.0 / 3.0
}

// MARK: - Quantum Volume API

/// Quantum Volume–**style** model circuits and Heavy Output Probability (HOP).
///
/// MVP honesty: this is **not** an IBM QV certification suite. It implements Cross-style
/// ideal HOP (`> 2/3`) on CPU statevector model circuits. Full device QV (shot HOP + CI)
/// is out of scope here.
///
/// ## Pairing
/// Default ``QuantumVolumeOptions/pairing`` is ``nearestNeighborAlternating`` (line /
/// routing friendly). Cross et al. redraw a **random** pairing each layer — pass
/// ``QuantumVolumePairing/randomPermutation`` for that model. HOP definition is the same.
///
/// ## SU(4) / U(4) sampling
/// Each two-qubit gate is a Haar-random unitary on ℂ⁴ obtained by the **Ginibre QR**
/// construction (Mezzadri): draw a 4×4 complex Ginibre matrix (i.i.d. complex normal
/// entries), compute a QR factorization `A = QR`, then replace `Q` with
/// `Q · diag(rᵢᵢ/|rᵢᵢ|)` so the result is Haar on U(4). A global-phase correction
/// `det^{-1/4}` is applied so `det = 1` (SU(4)); Born probabilities are unchanged by
/// that overall phase on each gate.
///
/// Matrices are packed **row-major** into 16 ``ComplexAmplitude`` entries and applied
/// via ``QuantumCircuit/customUnitary(matrix:qubits:)``.
///
/// ## Ideal probabilities
/// Ideal `p(x)` come from host **CPU statevector** Born probabilities
/// (``CPUStateVector/probabilitiesDouble()``). Do **not** use ``CircuitUnitary`` as
/// the QV oracle (known CRX/CP mismatches vs engines; 2Q `customUnitary` unsupported there).
///
/// ## Transpile note
/// Full ``TranspileOptions`` ibmEagle basis translation **rejects** `customUnitary`.
/// A safe limited path is linear-coupling ``BasicSwapRoutingPass`` only (keeps 2Q
/// custom unitaries). Prefer nearest-neighbor layers so identity layout needs no SWAPs;
/// see ``QuantumVolume/routeOntoLinearCoupling(_:seed:)``.
public enum QuantumVolume {

    /// Builds a square (or equal-depth) model circuit of Haar SU(4) layers.
    public static func makeModelCircuit(
        options: QuantumVolumeOptions,
        seed: UInt64
    ) throws -> QuantumCircuit {
        guard options.qubitCount >= 2 else {
            throw QuantumVolumeError.invalidQubitCount(options.qubitCount)
        }
        let depth = options.resolvedDepth
        guard depth >= 1 else {
            throw QuantumVolumeError.invalidDepth(depth)
        }

        var rng = QuantumRNG.seeded(seed)
        var circuit = try QuantumCircuit(qubitCount: options.qubitCount)

        for layer in 0..<depth {
            let pairs = pairing(for: layer, qubitCount: options.qubitCount, mode: options.pairing, rng: &rng)
            for (q0, q1) in pairs {
                let matrix = sampleHaarSU4(rng: &rng)
                try circuit.customUnitary(matrix: matrix, qubits: [q0, q1])
            }
            if options.insertLayerBarriers {
                try circuit.barrier()
            }
        }
        return circuit
    }

    /// Ideal Born probabilities from CPU statevector (engine LSB indexing).
    public static func idealProbabilities(
        of circuit: QuantumCircuit
    ) throws -> [Double] {
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        _ = try engine.execute(circuit, on: state)
        return state.probabilitiesDouble()
    }

    /// Heavy Output Probability from an ideal probability vector.
    public static func heavyOutputProbability(
        probabilities: [Double]
    ) throws -> HeavyOutputProbability {
        guard !probabilities.isEmpty else {
            throw QuantumVolumeError.emptyProbabilities
        }

        let median = Self.median(of: probabilities)
        var heavy: [Int] = []
        heavy.reserveCapacity(probabilities.count / 2)
        var hop = 0.0
        for (index, p) in probabilities.enumerated() {
            if p > median {
                heavy.append(index)
                hop += p
            }
        }
        return HeavyOutputProbability(
            median: median,
            hop: hop,
            heavyIndices: heavy,
            probabilityCount: probabilities.count
        )
    }

    /// Ideal noiseless HOP for a model circuit via CPU statevector Born rule.
    public static func idealHeavyOutputProbability(
        of circuit: QuantumCircuit
    ) throws -> HeavyOutputProbability {
        try heavyOutputProbability(probabilities: idealProbabilities(of: circuit))
    }

    /// Generate a seeded model circuit and return its ideal HOP.
    public static func evaluateIdealHOP(
        options: QuantumVolumeOptions,
        seed: UInt64
    ) throws -> (circuit: QuantumCircuit, hop: HeavyOutputProbability) {
        let circuit = try makeModelCircuit(options: options, seed: seed)
        let hop = try idealHeavyOutputProbability(of: circuit)
        return (circuit, hop)
    }

    /// Device-aware **routing only** onto a linear coupling map (identity initial layout).
    ///
    /// Does **not** run ibmEagle ``BasisTranslatorPass`` — that path cannot expand
    /// ``Gate/customUnitary``. With ``QuantumVolumePairing/nearestNeighborAlternating``,
    /// every SU(4) already sits on a linear edge, so this typically inserts no SWAPs and
    /// preserves computational-basis Born probabilities bit-identically.
    public static func routeOntoLinearCoupling(
        _ circuit: QuantumCircuit,
        seed: UInt64? = nil
    ) throws -> QuantumCircuit {
        let map = try CouplingMap.linear(circuit.qubitCount)
        return try BasicSwapRoutingPass(
            couplingMap: map,
            initialLayout: try Layout.identity(qubitCount: circuit.qubitCount),
            seed: seed
        ).run(on: circuit)
    }

    // MARK: - Pairing

    /// Layer pairing rule (documented for reproducibility).
    public static func pairing(
        for layer: Int,
        qubitCount: Int,
        mode: QuantumVolumePairing,
        rng: inout QuantumRNG
    ) -> [(Int, Int)] {
        switch mode {
        case .nearestNeighborAlternating:
            return nearestNeighborPairs(layer: layer, qubitCount: qubitCount)
        case .randomPermutation:
            return randomPermutationPairs(qubitCount: qubitCount, rng: &rng)
        }
    }

    /// Even layer → `(0,1),(2,3),…`; odd layer → `(1,2),(3,4),…`.
    /// For `qubitCount == 2` every layer uses the single edge `(0,1)`.
    public static func nearestNeighborPairs(layer: Int, qubitCount: Int) -> [(Int, Int)] {
        if qubitCount == 2 {
            return [(0, 1)]
        }
        var pairs: [(Int, Int)] = []
        let start = layer % 2
        var q = start
        while q + 1 < qubitCount {
            pairs.append((q, q + 1))
            q += 2
        }
        return pairs
    }

    private static func randomPermutationPairs(
        qubitCount: Int,
        rng: inout QuantumRNG
    ) -> [(Int, Int)] {
        var order = Array(0..<qubitCount)
        // Fisher–Yates
        if qubitCount > 1 {
            for i in stride(from: qubitCount - 1, through: 1, by: -1) {
                let j = rng.nextInt(upperBound: i + 1)
                order.swapAt(i, j)
            }
        }
        var pairs: [(Int, Int)] = []
        var i = 0
        while i + 1 < qubitCount {
            pairs.append((order[i], order[i + 1]))
            i += 2
        }
        return pairs
    }

    // MARK: - Haar SU(4)

    /// Samples a Haar SU(4) matrix as 16 row-major ``ComplexAmplitude`` entries.
    public static func sampleHaarSU4(rng: inout QuantumRNG) -> [ComplexAmplitude] {
        let u = haarUnitaryGinibreQR(dimension: 4, rng: &rng)
        let su = projectToSpecialUnitary(u, dimension: 4)
        return su.map { ComplexAmplitude(real: QFloat($0.re), imaginary: QFloat($0.im)) }
    }

    // MARK: - Statistics helpers

    /// Median of an unsorted sample (average of two central values when count is even).
    public static func median(of values: [Double]) -> Double {
        precondition(!values.isEmpty)
        let sorted = values.sorted()
        let n = sorted.count
        if n % 2 == 1 {
            return sorted[n / 2]
        }
        return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

// MARK: - Complex Double linear algebra (Haar sampling)

private struct C64: Equatable {
    var re: Double
    var im: Double

    static let zero = C64(re: 0, im: 0)
    static let one = C64(re: 1, im: 0)

    static func + (lhs: C64, rhs: C64) -> C64 {
        C64(re: lhs.re + rhs.re, im: lhs.im + rhs.im)
    }

    static func - (lhs: C64, rhs: C64) -> C64 {
        C64(re: lhs.re - rhs.re, im: lhs.im - rhs.im)
    }

    static func * (lhs: C64, rhs: C64) -> C64 {
        C64(
            re: lhs.re * rhs.re - lhs.im * rhs.im,
            im: lhs.re * rhs.im + lhs.im * rhs.re
        )
    }

    static func * (lhs: Double, rhs: C64) -> C64 {
        C64(re: lhs * rhs.re, im: lhs * rhs.im)
    }

    var conjugate: C64 { C64(re: re, im: -im) }
    var squaredNorm: Double { re * re + im * im }
    var abs: Double { sqrt(squaredNorm) }

    /// Unit-modulus complex number in the same direction (or 1 if near zero).
    var unitPhase: C64 {
        let n = abs
        if n < 1e-30 { return .one }
        return C64(re: re / n, im: im / n)
    }
}

private extension QuantumVolume {

    /// Haar U(n) via Ginibre → QR (Mezzadri).
    ///
    /// Modified Gram–Schmidt produces `Q` with a real-positive `R` diagonal; for a
    /// complex Ginibre draw that `Q` is already Haar on U(n). Equivalent to the
    /// textbook `Q · diag(rᵢᵢ/|rᵢᵢ|)` correction when `R` is phased to be positive.
    static func haarUnitaryGinibreQR(dimension: Int, rng: inout QuantumRNG) -> [C64] {
        var columns = [[C64]](repeating: [], count: dimension)
        for col in 0..<dimension {
            var column = [C64](repeating: .zero, count: dimension)
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

        var matrix = [C64](repeating: .zero, count: dimension * dimension)
        for col in 0..<dimension {
            for row in 0..<dimension {
                matrix[row * dimension + col] = columns[col][row]
            }
        }
        return matrix
    }

    /// Multiply by `det^{-1/n}` so det ≈ 1 (SU(n)).
    static func projectToSpecialUnitary(_ matrix: [C64], dimension: Int) -> [C64] {
        let det = determinant(matrix, dimension: dimension)
        let phase = det.unitPhase
        // det^{-1/n} = exp(-i arg(det) / n)
        let arg = atan2(phase.im, phase.re) / Double(dimension)
        let scale = C64(re: cos(-arg), im: sin(-arg))
        return matrix.map { $0 * scale }
    }

    static func complexNormal(rng: inout QuantumRNG) -> C64 {
        // Box–Muller with r = √(-ln u₁), scale 1/√2:
        //   Re, Im i.i.d. ~ N(0, 1/4), so E[|z|²] = 1/2
        // (Textbook complex Ginibre often uses N(0, 1/2) each ⇒ E[|z|²]=1.)
        // Haar Q from Ginibre→QR is **scale-invariant**, so this still yields Haar U(n);
        // only the intermediate Ginibre radius differs from the unit-variance convention.
        let u1 = max(rng.nextUnitDouble(), 1e-16)
        let u2 = rng.nextUnitDouble()
        let r = sqrt(-log(u1))
        let theta = 2.0 * Double.pi * u2
        let scale = 1.0 / sqrt(2.0)
        return C64(re: scale * r * cos(theta), im: scale * r * sin(theta))
    }

    static func hermitianDot(_ a: [C64], _ b: [C64]) -> C64 {
        var sum = C64.zero
        for i in a.indices {
            sum = sum + (a[i].conjugate * b[i])
        }
        return sum
    }

    static func axpy(_ alpha: Double, _ z: C64, _ x: [C64], onto y: [C64]) -> [C64] {
        let coeff = (alpha * z)
        return zip(x, y).map { xi, yi in yi + (coeff * xi) }
    }

    static func determinant(_ matrix: [C64], dimension: Int) -> C64 {
        // Small-n Laplace expansion / elimination copy.
        var a = matrix
        var det = C64.one
        for k in 0..<dimension {
            var pivot = k
            var best = a[k * dimension + k].squaredNorm
            for r in (k + 1)..<dimension {
                let nrm = a[r * dimension + k].squaredNorm
                if nrm > best {
                    best = nrm
                    pivot = r
                }
            }
            if best < 1e-30 {
                return .zero
            }
            if pivot != k {
                for c in 0..<dimension {
                    a.swapAt(k * dimension + c, pivot * dimension + c)
                }
                det = C64(re: -det.re, im: -det.im)
            }
            let diag = a[k * dimension + k]
            det = det * diag
            let invDiag = C64(re: diag.re / diag.squaredNorm, im: -diag.im / diag.squaredNorm)
            for r in (k + 1)..<dimension {
                let factor = a[r * dimension + k] * invDiag
                for c in k..<dimension {
                    a[r * dimension + c] = a[r * dimension + c] - (factor * a[k * dimension + c])
                }
            }
        }
        return det
    }
}
