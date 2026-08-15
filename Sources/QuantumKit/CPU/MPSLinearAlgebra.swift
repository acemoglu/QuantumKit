import Foundation
import Accelerate

/// Double-precision complex used inside the host MPS engine.
struct MPSComplex: Equatable, Sendable {
    var re: Double
    var im: Double

    static let zero = MPSComplex(re: 0, im: 0)
    static let one = MPSComplex(re: 1, im: 0)

    init(re: Double, im: Double) {
        self.re = re
        self.im = im
    }

    init(_ z: UnitaryComplex) {
        self.re = z.re
        self.im = z.im
    }

    init(_ z: ComplexAmplitude) {
        self.re = Double(z.real)
        self.im = Double(z.imaginary)
    }

    var magnitudeSquared: Double { re * re + im * im }

    static func + (lhs: MPSComplex, rhs: MPSComplex) -> MPSComplex {
        MPSComplex(re: lhs.re + rhs.re, im: lhs.im + rhs.im)
    }

    static func - (lhs: MPSComplex, rhs: MPSComplex) -> MPSComplex {
        MPSComplex(re: lhs.re - rhs.re, im: lhs.im - rhs.im)
    }

    static func * (lhs: MPSComplex, rhs: MPSComplex) -> MPSComplex {
        MPSComplex(
            re: lhs.re * rhs.re - lhs.im * rhs.im,
            im: lhs.re * rhs.im + lhs.im * rhs.re
        )
    }

    static func * (lhs: MPSComplex, rhs: Double) -> MPSComplex {
        MPSComplex(re: lhs.re * rhs, im: lhs.im * rhs)
    }

    var conjugate: MPSComplex { MPSComplex(re: re, im: -im) }

    func asAmplitude() -> ComplexAmplitude {
        ComplexAmplitude(real: QFloat(re), imaginary: QFloat(im))
    }
}

enum MPSLinearAlgebra {

    /// Thin SVD `A = U S V†` for a row-major `m × n` complex matrix.
    ///
    /// Uses `A†A` eigendecomposition via Accelerate `zheev` (avoids `zgesvd` layout pitfalls).
    static func svd(
        rows m: Int,
        cols n: Int,
        matrix: [MPSComplex]
    ) throws -> (u: [MPSComplex], s: [Double], vt: [MPSComplex]) {
        precondition(matrix.count == m * n)
        precondition(m > 0 && n > 0)

        // Prefer the smaller Gram matrix.
        if n <= m {
            return try svdViaAtA(rows: m, cols: n, matrix: matrix)
        } else {
            // A A† path: SVD of A† is related; compute then swap.
            let (uT, s, vtT) = try svdViaAtA(rows: n, cols: m, matrix: transpose(matrix, rows: m, cols: n))
            // A = U S V† ⇒ A† = V S U† ⇒ for A† we got uT≈V, vtT≈U†.
            // So U = vtT†, V† = uT†.
            let k = s.count
            var u = [MPSComplex](repeating: .zero, count: m * k)
            for row in 0..<m {
                for j in 0..<k {
                    u[row * k + j] = vtT[j * m + row].conjugate
                }
            }
            var vt = [MPSComplex](repeating: .zero, count: k * n)
            for j in 0..<k {
                for col in 0..<n {
                    vt[j * n + col] = uT[col * k + j].conjugate
                }
            }
            return (u, s, vt)
        }
    }

    private static func transpose(_ a: [MPSComplex], rows m: Int, cols n: Int) -> [MPSComplex] {
        var t = [MPSComplex](repeating: .zero, count: m * n)
        for r in 0..<m {
            for c in 0..<n {
                t[c * m + r] = a[r * n + c]
            }
        }
        return t
    }

    /// SVD via eigendecomposition of `A†A` (n × n), assuming n ≤ m.
    private static func svdViaAtA(
        rows m: Int,
        cols n: Int,
        matrix: [MPSComplex]
    ) throws -> (u: [MPSComplex], s: [Double], vt: [MPSComplex]) {
        // G = A† A (column-major for zheev).
        var g = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                var sum = MPSComplex.zero
                for r in 0..<m {
                    sum = sum + matrix[r * n + i].conjugate * matrix[r * n + j]
                }
                g[j * n + i] = __CLPK_doublecomplex(r: sum.re, i: sum.im)
            }
        }

        var jobz = Int8(UInt8(ascii: "V"))
        var uplo = Int8(UInt8(ascii: "L"))
        var nn: __CLPK_integer = __CLPK_integer(n)
        var lda: __CLPK_integer = nn
        var w = [Double](repeating: 0, count: n)
        var workQuery = [__CLPK_doublecomplex](repeating: __CLPK_doublecomplex(r: 0, i: 0), count: 1)
        var lwork: __CLPK_integer = -1
        var rwork = [Double](repeating: 0, count: max(1, 3 * n - 2))
        var info: __CLPK_integer = 0

        zheev_(&jobz, &uplo, &nn, &g, &lda, &w, &workQuery, &lwork, &rwork, &info)
        guard info == 0 else { throw MPSError.svdFailed(info: Int(info)) }
        lwork = max(__CLPK_integer(1), __CLPK_integer(workQuery[0].r))
        var work = [__CLPK_doublecomplex](
            repeating: __CLPK_doublecomplex(r: 0, i: 0),
            count: Int(lwork)
        )
        info = 0
        zheev_(&jobz, &uplo, &nn, &g, &lda, &w, &work, &lwork, &rwork, &info)
        guard info == 0 else { throw MPSError.svdFailed(info: Int(info)) }

        // zheev returns ascending eigenvalues; reverse for descending singular values.
        var s = [Double](repeating: 0, count: n)
        var vCols = [[MPSComplex]](repeating: [], count: n)
        for idx in 0..<n {
            let src = n - 1 - idx
            s[idx] = max(0, w[src]) // λ = σ²; clamp tiny negatives
            s[idx] = sqrt(s[idx])
            var col = [MPSComplex](repeating: .zero, count: n)
            for row in 0..<n {
                let z = g[src * n + row]
                col[row] = MPSComplex(re: z.r, im: z.i)
            }
            vCols[idx] = col
        }

        // Vt[j, c] = V[c, j]̅? V columns are eigenvectors; V†[j,c] = conj(V[c,j]).
        var vt = [MPSComplex](repeating: .zero, count: n * n)
        for j in 0..<n {
            for c in 0..<n {
                vt[j * n + c] = vCols[j][c].conjugate
            }
        }

        // U = A V S^{+}  (columns: u_j = A v_j / σ_j)
        var u = [MPSComplex](repeating: .zero, count: m * n)
        for j in 0..<n {
            let sigma = s[j]
            let inv = sigma > 1e-15 ? 1.0 / sigma : 0.0
            for r in 0..<m {
                var sum = MPSComplex.zero
                for c in 0..<n {
                    sum = sum + matrix[r * n + c] * vCols[j][c]
                }
                u[r * n + j] = MPSComplex(re: sum.re * inv, im: sum.im * inv)
            }
        }
        return (u, s, vt)
    }
}
