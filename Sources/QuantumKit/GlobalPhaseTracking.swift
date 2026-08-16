import Foundation

/// Cumulative global-phase bookkeeping for **statevector** unitary evolution (roadmap B16).
///
/// ## Convention
/// - **Units:** radians, unwrapped (not reduced modulo \(2\pi\)).
/// - **Definition:** for each applied **circuit** expanded unitary piece \(U\) of qubit-width \(w\),
///   add \(\arg(\det U) / 2^w\). For a product of full-space embeddings this equals
///   \(\arg(\det U_{\mathrm{full}}) / 2^n\).
/// - **1-qubit examples:** \(e^{i\phi} I\) contributes \(\phi\); \(S\) contributes \(\pi/4\);
///   \(T\) contributes \(\pi/8\); \(RZ(\theta)\) contributes \(0\) (\(\det = 1\)).
/// - **Amplitudes are not peeled:** engines leave buffer phases unchanged; \(\Phi\) is metadata only.
/// - **Born / shots are phase-blind:** probabilities and histograms ignore \(\Phi\).
/// - **Not a physical observable:** expectation values and measurement statistics do not depend on
///   global phase; do not treat \(\Phi\) as one.
///
/// ## Scope
/// - Tracked on CPU and Metal **statevector** paths (`CPUStateVector` / `StateVector`,
///   ``CircuitExecutionResult/cumulativeGlobalPhaseRadians``, ``QuantumResultMetadata``).
/// - Density-matrix, MPS, and stabilizer engines leave the field `nil` (mixed-state gauge
///   tracking is out of scope for B16).
/// - Only **circuit** unitary pieces update \(\Phi\). Measure / reset collapse, noise
///   unraveling Paulis, and stochastic reset/preparation error flips do **not**.
public enum GlobalPhaseTracking: Sendable {

    /// \(\arg(\det U) / 2^w\) for a row-major \(2^w \times 2^w\) complex matrix.
    public static func contribution(
        matrix: [ComplexAmplitude],
        qubitWidth: Int
    ) -> Double {
        precondition(qubitWidth >= 0)
        let dimension = 1 << qubitWidth
        precondition(matrix.count == dimension * dimension)
        let detArg = determinantArg(matrix: matrix, dimension: dimension)
        return detArg / Double(dimension)
    }

    /// Contribution of one expanded execution piece (see convention above).
    ///
    /// Composite gates that ``GateDecomposition/needsExecutionExpansion`` expands are summed
    /// over their recursive expansion so callers may pass either a circuit gate or a piece.
    public static func contribution(of gate: Gate) throws -> Double {
        if GateDecomposition.needsExecutionExpansion(gate) {
            var sum = 0.0
            for piece in try expandedExecutionPieces(gate) {
                sum += try contribution(of: piece)
            }
            return sum
        }

        switch gate {
        case .id, .barrier, .delay, .measure, .reset, .initialize, .c_if, .while_c:
            return 0

        case .h:
            return Double.pi / 2
        case .x:
            return Double.pi / 2
        case .y:
            return Double.pi / 2
        case .z:
            return Double.pi / 2
        case .s:
            return Double.pi / 4
        case .sdg:
            return -Double.pi / 4
        case .t:
            return Double.pi / 8
        case .tdg:
            return -Double.pi / 8
        case .sx:
            return Double.pi / 4
        case .sxdg:
            return -Double.pi / 4

        case .p(let theta, _):
            return Double(try theta.requireLiteral()) / 2.0

        case .rx, .ry, .rz:
            return 0

        case .u(_, let phi, let lambda, _):
            let phiV = Double(try phi.requireLiteral())
            let lambdaV = Double(try lambda.requireLiteral())
            return (phiV + lambdaV) / 2.0

        case .cx, .cz, .swap:
            return Double.pi / 4

        case .ccx:
            return Double.pi / 8

        case .mcx(let controls, _):
            let width = controls.count + 1
            return Double.pi / Double(1 << width)

        case .mcz(let controls, _):
            let width = controls.count + 1
            return Double.pi / Double(1 << width)

        case .crx, .cry, .crz:
            return 0

        case .cp(let theta, _, _):
            return Double(try theta.requireLiteral()) / 4.0

        case .unitary1(let matrix, _):
            return contribution(matrix: matrix, qubitWidth: 1)

        case .customUnitary(let matrix, let qubits):
            return contribution(matrix: matrix, qubitWidth: qubits.count)

        case .iswap, .ecr, .rxx, .ryy, .rzz, .dcx, .cswap:
            // Unreachable: `needsExecutionExpansion` expands these first.
            return 0
        }
    }

    /// Same expansion policy as ``QuantumEngine/expandForExecution`` (no Metal dependency).
    private static func expandedExecutionPieces(_ gate: Gate) throws -> [Gate] {
        guard GateDecomposition.needsExecutionExpansion(gate) else { return [gate] }
        var result: [Gate] = []
        for piece in try GateDecomposition.expand(gate) {
            result.append(contentsOf: try expandedExecutionPieces(piece))
        }
        return result
    }

    /// \(\arg(u_{00}u_{11} - u_{01}u_{10}) / 2\) for a 1-qubit matrix given as four complex pairs.
    public static func contribution1Q(
        u00: (Double, Double),
        u01: (Double, Double),
        u10: (Double, Double),
        u11: (Double, Double)
    ) -> Double {
        let detRe = u00.0 * u11.0 - u00.1 * u11.1 - (u01.0 * u10.0 - u01.1 * u10.1)
        let detIm = u00.0 * u11.1 + u00.1 * u11.0 - (u01.0 * u10.1 + u01.1 * u10.0)
        return atan2(detIm, detRe) / 2.0
    }

    /// Diagonal phase \(e^{i\phi}\) on the subspace where all bits in `mask` are set
    /// (Z / S / T / CZ / CP / MCZ-style kernels).
    public static func contributionDiagonalPhase(mask: Int, phaseRe: Double, phaseIm: Double) -> Double {
        let width = mask.nonzeroBitCount
        guard width > 0 else { return 0 }
        return atan2(phaseIm, phaseRe) / Double(1 << width)
    }
}

// MARK: - Determinant argument

private extension GlobalPhaseTracking {
    static func determinantArg(matrix: [ComplexAmplitude], dimension: Int) -> Double {
        if dimension == 1 {
            return atan2(Double(matrix[0].imaginary), Double(matrix[0].real))
        }
        if dimension == 2 {
            return contribution1Q(
                u00: (Double(matrix[0].real), Double(matrix[0].imaginary)),
                u01: (Double(matrix[1].real), Double(matrix[1].imaginary)),
                u10: (Double(matrix[2].real), Double(matrix[2].imaginary)),
                u11: (Double(matrix[3].real), Double(matrix[3].imaginary))
            ) * 2.0
        }

        var a = matrix.map { (re: Double($0.real), im: Double($0.imaginary)) }
        var detRe = 1.0
        var detIm = 0.0
        for k in 0..<dimension {
            var pivot = k
            var best = a[k * dimension + k].re * a[k * dimension + k].re
                + a[k * dimension + k].im * a[k * dimension + k].im
            for r in (k + 1)..<dimension {
                let z = a[r * dimension + k]
                let nrm = z.re * z.re + z.im * z.im
                if nrm > best {
                    best = nrm
                    pivot = r
                }
            }
            if best < 1e-30 {
                return 0
            }
            if pivot != k {
                for c in 0..<dimension {
                    a.swapAt(k * dimension + c, pivot * dimension + c)
                }
                detRe = -detRe
                detIm = -detIm
            }
            let diag = a[k * dimension + k]
            let newRe = detRe * diag.re - detIm * diag.im
            let newIm = detRe * diag.im + detIm * diag.re
            detRe = newRe
            detIm = newIm
            let invNorm = 1.0 / max(diag.re * diag.re + diag.im * diag.im, 1e-30)
            let invRe = diag.re * invNorm
            let invIm = -diag.im * invNorm
            for r in (k + 1)..<dimension {
                let factorRe = a[r * dimension + k].re * invRe - a[r * dimension + k].im * invIm
                let factorIm = a[r * dimension + k].re * invIm + a[r * dimension + k].im * invRe
                for c in k..<dimension {
                    let src = a[k * dimension + c]
                    a[r * dimension + c].re -= factorRe * src.re - factorIm * src.im
                    a[r * dimension + c].im -= factorRe * src.im + factorIm * src.re
                }
            }
        }
        return atan2(detIm, detRe)
    }
}
