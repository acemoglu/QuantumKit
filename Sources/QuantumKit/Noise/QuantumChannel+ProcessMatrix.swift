import Foundation
import Accelerate

/*
 Process-matrix conventions — 1 qubit only
 ================================================

 Storage
 -------
 Both Choi and superoperator inputs are length-16 arrays: **row-major** 4×4 complex
 matrices (index `row * 4 + col`), matching Kraus / density-matrix layout elsewhere.

 Endian / basis
 --------------
 Single-qubit computational basis `{|0⟩, |1⟩}` with engine LSB packing (qubit 0 = LSB).
 The 4-dimensional process space uses the standard Kronecker order
 `|i⟩ ⊗ |a⟩` with index `i * 2 + a` (input bit most significant in the 2-bit code).

 Choi
 ----
 `J(Φ) = Σ_{i,j} |i⟩⟨j| ⊗ Φ(|i⟩⟨j|)`.
 Element relation: `J[i*2+a, j*2+b] = ⟨a| Φ(|i⟩⟨j|) |b⟩`.

 Superoperator (Liouville)
 -------------------------
 Column-vectorization: `vec([[ρ00, ρ01], [ρ10, ρ11]]) = [ρ00, ρ10, ρ01, ρ11]`,
 i.e. column-stack. Then `vec(Φ(ρ)) = S · vec(ρ)`.
 Reshuffle to Choi: `J[i*2+a, j*2+b] = S[a + 2*b, i + 2*j]`.

 Kraus recovery
 --------------
 Hermitian eigendecomposition of Choi: `J = Σ_k λ_k |v_k⟩⟨v_k|`.
 For each `λ_k > eigenvalueTolerance`, reshape `√λ_k · v_k` with
 `E_rowmajor = [v0, v2, v1, v3]` so that column-`vec(E) = v` (see above).
 Result is ``QuantumChannel/kraus1Q`` and reuses existing `applyLocalized` paths.

 Practical checks (NOT a CPTP proof)
 -----------------------------------
 - Shape: 16 complex entries (4×4). Length 256 is rejected as unsupported 2-qubit Choi.
 - Hermiticity: `max |J_ij - conj(J_ji)| ≤ hermiticityTolerance`, then symmetrize `(J+J†)/2`.
 - Complete positivity (numerical): all eigenvalues `≥ -eigenvalueTolerance`.
 - Does **not** verify trace preservation, unitality, or exact CPTP completeness.
 */

extension QuantumChannel {

    private static let processMatrixDim1Q = 4
    private static let processMatrixElements1Q = 16
    private static let processMatrixElements2Q = 256

    public static let defaultChoiHermiticityTolerance: QFloat = 1e-5
    public static let defaultChoiEigenvalueTolerance: QFloat = 1e-6

    /// Import a 1-qubit channel from its Choi matrix (row-major 4×4). Returns ``kraus1Q``.
    ///
    /// See file-level convention notes. Does not prove the map is CPTP — only shape,
    /// hermiticity within `hermiticityTolerance`, and eigenvalues `≥ -eigenvalueTolerance`.
    public static func fromChoi1Q(
        _ matrix: [ComplexAmplitude],
        hermiticityTolerance: QFloat = defaultChoiHermiticityTolerance,
        eigenvalueTolerance: QFloat = defaultChoiEigenvalueTolerance
    ) throws -> QuantumChannel {
        try validateProcessMatrixElementCount(matrix.count)
        var choi = matrix
        try enforceChoiHermiticity(&choi, tolerance: hermiticityTolerance)
        let operators = try krausOperatorsFromHermitianChoi(
            choi,
            eigenvalueTolerance: eigenvalueTolerance
        )
        return try fromKraus1Q(operators)
    }

    /// Import a 1-qubit channel from its Liouville superoperator (row-major 4×4).
    ///
    /// Uses column-vectorization; reshuffles to Choi then ``fromChoi1Q``. Same practical
    /// checks and limitations — does not prove trace preservation.
    public static func fromSuperoperator1Q(
        _ matrix: [ComplexAmplitude],
        hermiticityTolerance: QFloat = defaultChoiHermiticityTolerance,
        eigenvalueTolerance: QFloat = defaultChoiEigenvalueTolerance
    ) throws -> QuantumChannel {
        try validateProcessMatrixElementCount(matrix.count)
        let choi = choiFromSuperoperator1Q(matrix)
        return try fromChoi1Q(
            choi,
            hermiticityTolerance: hermiticityTolerance,
            eigenvalueTolerance: eigenvalueTolerance
        )
    }

    // MARK: - Validation / conversion

    private static func validateProcessMatrixElementCount(_ count: Int) throws {
        if count == processMatrixElements1Q { return }
        if count == processMatrixElements2Q {
            throw QuantumChannelError.multiQubitProcessMatrixUnsupported(qubitCount: 2)
        }
        // Infer a plausible qubit count when length is (2^n)^2 × (2^n)^2 = 2^{4n}.
        var n = 3
        var expected = 1 << (4 * n)
        while expected < count && n < 8 {
            n += 1
            expected = 1 << (4 * n)
        }
        if expected == count {
            throw QuantumChannelError.multiQubitProcessMatrixUnsupported(qubitCount: n)
        }
        throw QuantumChannelError.invalidProcessMatrixDimension(
            count: count,
            expected: processMatrixElements1Q
        )
    }

    private static func enforceChoiHermiticity(
        _ matrix: inout [ComplexAmplitude],
        tolerance: QFloat
    ) throws {
        let dim = processMatrixDim1Q
        var maxDeviation: QFloat = 0
        for row in 0..<dim {
            for col in row..<dim {
                let a = matrix[row * dim + col]
                let b = matrix[col * dim + row]
                let dRe = abs(a.real - b.real)
                let dIm = abs(a.imaginary + b.imaginary)
                maxDeviation = max(maxDeviation, max(dRe, dIm))
            }
        }
        guard maxDeviation <= tolerance else {
            throw QuantumChannelError.choiNotHermitian(maxDeviation: maxDeviation)
        }
        // Symmetrize for stable eigendecomposition.
        for row in 0..<dim {
            for col in row..<dim {
                let a = matrix[row * dim + col]
                let b = matrix[col * dim + row]
                let re = (a.real + b.real) * 0.5
                let im = (a.imaginary - b.imaginary) * 0.5
                matrix[row * dim + col] = ComplexAmplitude(real: re, imaginary: im)
                matrix[col * dim + row] = ComplexAmplitude(real: re, imaginary: -im)
            }
        }
        for diag in 0..<dim {
            let z = matrix[diag * dim + diag]
            matrix[diag * dim + diag] = ComplexAmplitude(real: z.real, imaginary: 0)
        }
    }

    /// `J[i*2+a, j*2+b] = S[a+2*b, i+2*j]` (column-vec superoperator → Choi).
    private static func choiFromSuperoperator1Q(
        _ superop: [ComplexAmplitude]
    ) -> [ComplexAmplitude] {
        let d = 2
        var choi = [ComplexAmplitude](
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: processMatrixElements1Q
        )
        for i in 0..<d {
            for a in 0..<d {
                for j in 0..<d {
                    for b in 0..<d {
                        let choiIndex = (i * d + a) * processMatrixDim1Q + (j * d + b)
                        let sopIndex = (a + d * b) * processMatrixDim1Q + (i + d * j)
                        choi[choiIndex] = superop[sopIndex]
                    }
                }
            }
        }
        return choi
    }

    private static func krausOperatorsFromHermitianChoi(
        _ choi: [ComplexAmplitude],
        eigenvalueTolerance: QFloat
    ) throws -> [[ComplexAmplitude]] {
        let (eigenvalues, eigenvectors) = try hermitianEigendecomposition4x4(choi)
        var minimumEigenvalue = eigenvalues[0]
        for λ in eigenvalues {
            minimumEigenvalue = min(minimumEigenvalue, λ)
        }
        guard minimumEigenvalue >= -eigenvalueTolerance else {
            throw QuantumChannelError.choiNotPositiveSemidefinite(
                minimumEigenvalue: minimumEigenvalue
            )
        }

        var operators: [[ComplexAmplitude]] = []
        operators.reserveCapacity(4)
        for k in 0..<4 {
            let λ = eigenvalues[k]
            if λ <= eigenvalueTolerance { continue }
            let scale = sqrt(λ)
            let v = eigenvectors[k]
            // column-vec(E) = √λ v  ⇒  row-major E = [v0, v2, v1, v3]
            let kraus: [ComplexAmplitude] = [
                ComplexAmplitude(real: scale * v[0].real, imaginary: scale * v[0].imaginary),
                ComplexAmplitude(real: scale * v[2].real, imaginary: scale * v[2].imaginary),
                ComplexAmplitude(real: scale * v[1].real, imaginary: scale * v[1].imaginary),
                ComplexAmplitude(real: scale * v[3].real, imaginary: scale * v[3].imaginary),
            ]
            operators.append(kraus)
        }
        guard !operators.isEmpty else { throw QuantumChannelError.emptyKrausSet }
        return operators
    }

    /// Column-major `cheev` eigendecomposition; returns eigenvalues and eigenvectors as
    /// length-4 column vectors (same ordering as Choi `|i⟩⊗|a⟩` basis).
    private static func hermitianEigendecomposition4x4(
        _ rowMajor: [ComplexAmplitude]
    ) throws -> (eigenvalues: [QFloat], eigenvectors: [[ComplexAmplitude]]) {
        let n = processMatrixDim1Q
        var jobz = Int8(UInt8(ascii: "V"))
        var uplo = Int8(UInt8(ascii: "L"))
        var dim: __CLPK_integer = __CLPK_integer(n)
        var lda: __CLPK_integer = dim

        // LAPACK expects column-major; lower triangle is read for UPLO=L.
        var a = [__CLPK_complex](repeating: __CLPK_complex(r: 0, i: 0), count: n * n)
        for col in 0..<n {
            for row in 0..<n {
                let z = rowMajor[row * n + col]
                a[col * n + row] = __CLPK_complex(r: z.real, i: z.imaginary)
            }
        }

        var w = [Float](repeating: 0, count: n)
        var workQuery = [__CLPK_complex](repeating: __CLPK_complex(r: 0, i: 0), count: 1)
        var lwork: __CLPK_integer = -1
        var rwork = [Float](repeating: 0, count: max(1, 3 * n - 2))
        var info: __CLPK_integer = 0

        cheev_(&jobz, &uplo, &dim, &a, &lda, &w, &workQuery, &lwork, &rwork, &info)
        guard info == 0 else {
            // Workspace query failure is unexpected for n=4; treat as non-Hermitian PSD path.
            throw QuantumChannelError.choiNotHermitian(maxDeviation: .infinity)
        }

        let optimal = max(__CLPK_integer(2 * n), __CLPK_integer(workQuery[0].r))
        lwork = optimal
        var work = [__CLPK_complex](
            repeating: __CLPK_complex(r: 0, i: 0),
            count: Int(optimal)
        )
        info = 0
        cheev_(&jobz, &uplo, &dim, &a, &lda, &w, &work, &lwork, &rwork, &info)
        guard info == 0 else {
            throw QuantumChannelError.choiNotPositiveSemidefinite(minimumEigenvalue: -.infinity)
        }

        var eigenvectors: [[ComplexAmplitude]] = []
        eigenvectors.reserveCapacity(n)
        for col in 0..<n {
            var vector: [ComplexAmplitude] = []
            vector.reserveCapacity(n)
            for row in 0..<n {
                let z = a[col * n + row]
                vector.append(ComplexAmplitude(real: z.r, imaginary: z.i))
            }
            eigenvectors.append(vector)
        }
        return (w.map { QFloat($0) }, eigenvectors)
    }
}
