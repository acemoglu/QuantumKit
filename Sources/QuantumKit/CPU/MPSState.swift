import Foundation

/// Truncation / width knobs for host MPS simulation (B18).
///
/// Singular values with `σ_i / σ_max < svdTruncationThreshold` are dropped, and at most
/// ``maxBondDimension`` (χ) values are kept after each adjacent two-qubit SVD (tensor path)
/// or when ``MPSState/compress()`` is called.
public struct MPSConfiguration: Sendable, Equatable {
    /// Maximum bond dimension χ.
    public var maxBondDimension: Int
    /// Relative singular-value cutoff (`σ_i / σ_max`). Default `1e-12`.
    public var svdTruncationThreshold: Double
    /// Soft width cap for this backend instance.
    public var maxQubitCount: Int
    /// Max width for full amplitude / Born-probability export via site contraction.
    /// Evolution is always local MPS (χ truncation on adjacent 2Q updates) at every width.
    public var maxAmplitudeExportQubits: Int

    public init(
        maxBondDimension: Int = 64,
        svdTruncationThreshold: Double = 1e-12,
        maxQubitCount: Int = 256,
        maxAmplitudeExportQubits: Int = 16
    ) {
        self.maxBondDimension = max(1, maxBondDimension)
        self.svdTruncationThreshold = max(0, svdTruncationThreshold)
        self.maxQubitCount = max(1, maxQubitCount)
        self.maxAmplitudeExportQubits = max(1, min(maxAmplitudeExportQubits, 28))
    }

    public static let `default` = MPSConfiguration()
}

public enum MPSError: Error, Equatable, CustomStringConvertible {
    case invalidQubitCount(Int)
    case qubitCountExceedsLimit(max: Int, requested: Int)
    case qubitCountMismatch(circuit: Int, state: Int)
    case unsupportedGate(Gate)
    case unsupportedMultiQubitGate(qubitCount: Int)
    case amplitudeExportTooWide(qubitCount: Int, max: Int)
    case noiseNotSupported
    case svdFailed(info: Int)
    case invalidBondDimension(Int)

    public var description: String {
        switch self {
        case .invalidQubitCount(let n):
            return "MPS requires qubitCount > 0 (got \(n))."
        case .qubitCountExceedsLimit(let max, let requested):
            return "MPS qubit count \(requested) exceeds limit \(max)."
        case .qubitCountMismatch(let circuit, let state):
            return "Circuit width \(circuit) does not match MPS width \(state)."
        case .unsupportedGate(let gate):
            return "MPS backend rejected unsupported gate: \(gate)."
        case .unsupportedMultiQubitGate(let count):
            return "MPS MVP supports at most 2-qubit unitaries after expansion (got \(count)-qubit)."
        case .amplitudeExportTooWide(let n, let max):
            return "MPS amplitude export requires n ≤ \(max) (got \(n))."
        case .noiseNotSupported:
            return "MPS backend does not support noise models."
        case .svdFailed(let info):
            return "MPS SVD failed (LAPACK info=\(info))."
        case .invalidBondDimension(let chi):
            return "MPS maxBondDimension must be >= 1 (got \(chi))."
        }
    }
}

/// One site tensor `A[α, s, β]` with physical dim 2, stored row-major as
/// `((α * 2) + s) * rightDim + β`.
struct MPSSite: Sendable {
    var leftDim: Int
    var rightDim: Int
    var data: [MPSComplex]

    init(leftDim: Int, rightDim: Int, data: [MPSComplex]) {
        precondition(data.count == leftDim * 2 * rightDim)
        self.leftDim = leftDim
        self.rightDim = rightDim
        self.data = data
    }

    static func productZero() -> MPSSite {
        MPSSite(leftDim: 1, rightDim: 1, data: [.one, .zero])
    }

    subscript(left: Int, phys: Int, right: Int) -> MPSComplex {
        get { data[((left * 2) + phys) * rightDim + right] }
        set { data[((left * 2) + phys) * rightDim + right] = newValue }
    }
}

/// Open-boundary MPS on a 1D qubit chain (host CPU).
///
/// Evolution is always local adjacent SVD updates with χ truncation. Amplitude export
/// (``amplitudes()`` / ``probabilities()``) contracts sites when
/// `n ≤ maxAmplitudeExportQubits`; wider registers use bond-local Z sampling only.
public struct MPSState: Sendable {
    public let qubitCount: Int
    public private(set) var configuration: MPSConfiguration
    public private(set) var lastTruncationError: Double

    var sites: [MPSSite]

    public var maxBondDimension: Int { configuration.maxBondDimension }

    /// Always `false`: gates never materialize a dense 2ⁿ statevector.
    public var usesDenseEvolution: Bool { false }

    public var bondDimensions: [Int] {
        sites.dropLast().map(\.rightDim)
    }

    public init(qubitCount: Int, configuration: MPSConfiguration = .default) throws {
        guard qubitCount > 0 else { throw MPSError.invalidQubitCount(qubitCount) }
        guard qubitCount <= configuration.maxQubitCount else {
            throw MPSError.qubitCountExceedsLimit(
                max: configuration.maxQubitCount,
                requested: qubitCount
            )
        }
        guard configuration.maxBondDimension >= 1 else {
            throw MPSError.invalidBondDimension(configuration.maxBondDimension)
        }
        self.qubitCount = qubitCount
        self.configuration = configuration
        self.lastTruncationError = 0
        self.sites = (0..<qubitCount).map { _ in MPSSite.productZero() }
    }

    /// Test / advanced hook: install site tensors directly.
    mutating func adoptTensorSites(_ newSites: [MPSSite]) {
        precondition(newSites.count == qubitCount)
        sites = newSites
    }

    public mutating func resetToZero() {
        lastTruncationError = 0
        sites = (0..<qubitCount).map { _ in MPSSite.productZero() }
    }

    // MARK: - Gates

    mutating func apply1Q(_ u: [MPSComplex], target: Int) {
        precondition(u.count == 4)
        var site = sites[target]
        var out = [MPSComplex](repeating: .zero, count: site.data.count)
        for a in 0..<site.leftDim {
            for b in 0..<site.rightDim {
                let in0 = site[a, 0, b]
                let in1 = site[a, 1, b]
                out[((a * 2) + 0) * site.rightDim + b] = u[0] * in0 + u[1] * in1
                out[((a * 2) + 1) * site.rightDim + b] = u[2] * in0 + u[3] * in1
            }
        }
        site.data = out
        sites[target] = site
    }

    mutating func applyAdjacent2Q(_ u: [MPSComplex], left: Int) throws {
        precondition(u.count == 16)
        precondition(left + 1 < qubitCount)
        try applyAdjacent2QLocal(u, left: left)
    }

    mutating func applyAdjacentCX(controlOnLeft: Bool, left: Int) throws {
        var u = [MPSComplex](repeating: .zero, count: 16)
        for s in 0..<2 {
            for t in 0..<2 {
                let inPacked = s + 2 * t
                let outS: Int
                let outT: Int
                if controlOnLeft {
                    outS = s
                    outT = t ^ s
                } else {
                    outS = s ^ t
                    outT = t
                }
                u[(outS + 2 * outT) * 4 + inPacked] = .one
            }
        }
        try applyAdjacent2Q(u, left: left)
    }

    mutating func applyAdjacentSwap(left: Int) throws {
        let u: [MPSComplex] = [
            .one, .zero, .zero, .zero,
            .zero, .zero, .one, .zero,
            .zero, .one, .zero, .zero,
            .zero, .zero, .zero, .one,
        ]
        try applyAdjacent2Q(u, left: left)
    }

    /// Re-truncate by contracting to amplitudes and rebuilding with the configured χ.
    /// Requires `n ≤ maxAmplitudeExportQubits`; no-op for wider registers (already truncated
    /// on each adjacent 2Q update).
    public mutating func compress() throws {
        guard qubitCount <= configuration.maxAmplitudeExportQubits else { return }
        let amps = try contractSites()
        try replaceFromAmplitudes(amps)
    }

    // MARK: - Export / sample

    public func amplitudes() throws -> [ComplexAmplitude] {
        guard qubitCount <= configuration.maxAmplitudeExportQubits else {
            throw MPSError.amplitudeExportTooWide(
                qubitCount: qubitCount,
                max: configuration.maxAmplitudeExportQubits
            )
        }
        return try contractSites().map { $0.asAmplitude() }
    }

    public func probabilities() throws -> [QFloat] {
        let amps = try amplitudes()
        return amps.map { QFloat($0.real * $0.real + $0.imaginary * $0.imaginary) }
    }

    public func sampleOutcome(rng: inout QuantumRNG) throws -> Int {
        // Sequential Z sampling with right environments (no 2ⁿ materialization).
        return sampleOutcomeFromSites(rng: &rng)
    }

    /// Left-to-right Z measurement using right-bond environments (works for any width).
    private func sampleOutcomeFromSites(rng: inout QuantumRNG) -> Int {
        // rightEnv[i] is the Hermitian environment on the left bond of site `i`
        // (equivalently the right bond of site `i-1`), stored row-major.
        var rightEnv = [[MPSComplex]](repeating: [], count: qubitCount + 1)
        rightEnv[qubitCount] = [.one]
        for i in stride(from: qubitCount - 1, through: 0, by: -1) {
            rightEnv[i] = transferEnvironment(through: sites[i], right: rightEnv[i + 1])
        }

        var leftVec: [MPSComplex] = [.one]
        var outcome = 0
        for i in 0..<qubitCount {
            let site = sites[i]
            let leftDim = site.leftDim
            let rightDim = site.rightDim
            let envR = rightEnv[i + 1]
            precondition(envR.count == rightDim * rightDim)

            var probs = [0.0, 0.0]
            var nextByPhys = [[MPSComplex]](
                repeating: [MPSComplex](repeating: .zero, count: rightDim),
                count: 2
            )
            for phys in 0..<2 {
                var next = [MPSComplex](repeating: .zero, count: rightDim)
                for beta in 0..<rightDim {
                    var sum = MPSComplex.zero
                    for alpha in 0..<leftDim {
                        let lv = alpha < leftVec.count ? leftVec[alpha] : .zero
                        sum = sum + lv * site[alpha, phys, beta]
                    }
                    next[beta] = sum
                }
                var p = 0.0
                for beta in 0..<rightDim {
                    for betaP in 0..<rightDim {
                        let term = next[beta].conjugate * envR[beta * rightDim + betaP] * next[betaP]
                        p += term.re
                    }
                }
                probs[phys] = max(p, 0)
                nextByPhys[phys] = next
            }

            let total = probs[0] + probs[1]
            let draw = rng.nextUnitDouble() * (total > 0 ? total : 1)
            let bit = (total == 0 || draw < probs[0]) ? 0 : 1
            outcome |= bit << i
            leftVec = nextByPhys[bit]
        }
        return outcome
    }

    /// `E_L[α,α'] = Σ_{s,β,β'} A[α,s,β] conj(A[α',s,β']) E_R[β,β']`.
    private func transferEnvironment(through site: MPSSite, right envRight: [MPSComplex]) -> [MPSComplex] {
        let leftDim = site.leftDim
        let rightDim = site.rightDim
        precondition(envRight.count == rightDim * rightDim)
        var envLeft = [MPSComplex](repeating: .zero, count: leftDim * leftDim)
        for alpha in 0..<leftDim {
            for alphaP in 0..<leftDim {
                var sum = MPSComplex.zero
                for phys in 0..<2 {
                    for beta in 0..<rightDim {
                        for betaP in 0..<rightDim {
                            sum = sum
                                + site[alpha, phys, beta]
                                * site[alphaP, phys, betaP].conjugate
                                * envRight[beta * rightDim + betaP]
                        }
                    }
                }
                envLeft[alpha * leftDim + alphaP] = sum
            }
        }
        return envLeft
    }

    private func contractSites() throws -> [MPSComplex] {
        let dim = 1 << qubitCount
        var out = [MPSComplex](repeating: .zero, count: dim)
        func walk(site: Int, leftVec: [MPSComplex], bitPrefix: Int) {
            if site == qubitCount {
                out[bitPrefix] = leftVec[0]
                return
            }
            let tensor = sites[site]
            for phys in 0..<2 {
                var next = [MPSComplex](repeating: .zero, count: tensor.rightDim)
                for beta in 0..<tensor.rightDim {
                    var sum = MPSComplex.zero
                    for alpha in 0..<min(tensor.leftDim, leftVec.count) {
                        sum = sum + leftVec[alpha] * tensor[alpha, phys, beta]
                    }
                    next[beta] = sum
                }
                walk(site: site + 1, leftVec: next, bitPrefix: bitPrefix | (phys << site))
            }
        }
        walk(site: 0, leftVec: [.one], bitPrefix: 0)
        return out
    }

    // MARK: - Local tensor 2Q + compress

    private mutating func applyAdjacent2QLocal(_ u: [MPSComplex], left: Int) throws {
        let right = left + 1
        let a = sites[left]
        let b = sites[right]
        let chiL = a.leftDim
        let chiM = a.rightDim
        precondition(b.leftDim == chiM)
        let chiR = b.rightDim

        var theta = [MPSComplex](repeating: .zero, count: chiL * 4 * chiR)
        for alpha in 0..<chiL {
            for s in 0..<2 {
                for t in 0..<2 {
                    for gamma in 0..<chiR {
                        var sum = MPSComplex.zero
                        for beta in 0..<chiM {
                            sum = sum + a[alpha, s, beta] * b[beta, t, gamma]
                        }
                        theta[((alpha * 2 + s) * 2 + t) * chiR + gamma] = sum
                    }
                }
            }
        }

        var thetaU = [MPSComplex](repeating: .zero, count: theta.count)
        for alpha in 0..<chiL {
            for sOut in 0..<2 {
                for tOut in 0..<2 {
                    let outPacked = sOut + 2 * tOut
                    for gamma in 0..<chiR {
                        var sum = MPSComplex.zero
                        for sIn in 0..<2 {
                            for tIn in 0..<2 {
                                let inPacked = sIn + 2 * tIn
                                let th = theta[((alpha * 2 + sIn) * 2 + tIn) * chiR + gamma]
                                sum = sum + u[outPacked * 4 + inPacked] * th
                            }
                        }
                        thetaU[((alpha * 2 + sOut) * 2 + tOut) * chiR + gamma] = sum
                    }
                }
            }
        }

        let rows = chiL * 2
        let cols = 2 * chiR
        var matrix = [MPSComplex](repeating: .zero, count: rows * cols)
        for alpha in 0..<chiL {
            for s in 0..<2 {
                for t in 0..<2 {
                    for gamma in 0..<chiR {
                        matrix[(alpha * 2 + s) * cols + (t * chiR + gamma)] =
                            thetaU[((alpha * 2 + s) * 2 + t) * chiR + gamma]
                    }
                }
            }
        }

        let (uFull, sAll, vtFull) = try MPSLinearAlgebra.svd(rows: rows, cols: cols, matrix: matrix)
        let keep = truncatedRank(singularValues: sAll)
        lastTruncationError = truncationWeight(singularValues: sAll, keep: keep)
        // Preserve the two-site Frobenius norm. Forcing ||θ||→1 is wrong under mixed gauge
        // (S absorbed into a neighboring site): that was dropping the global norm by ~½.
        let renorm = truncationScale(singularValues: sAll, keep: keep, forceUnitFrobenius: false)

        var leftData = [MPSComplex](repeating: .zero, count: rows * keep)
        for row in 0..<rows {
            for j in 0..<keep {
                let uv = uFull[row * sAll.count + j]
                let scale = sAll[j] * renorm
                leftData[row * keep + j] = MPSComplex(re: uv.re * scale, im: uv.im * scale)
            }
        }
        var rightData = [MPSComplex](repeating: .zero, count: keep * 2 * chiR)
        for j in 0..<keep {
            for t in 0..<2 {
                for gamma in 0..<chiR {
                    rightData[((j * 2) + t) * chiR + gamma] = vtFull[j * cols + (t * chiR + gamma)]
                }
            }
        }
        sites[left] = MPSSite(leftDim: chiL, rightDim: keep, data: leftData)
        sites[right] = MPSSite(leftDim: keep, rightDim: chiR, data: rightData)
    }

    mutating func replaceFromAmplitudes(_ amps: [MPSComplex]) throws {
        let dim = 1 << qubitCount
        precondition(amps.count == dim)
        // Successive SVD compression (left-to-right) with engineLSB bit order.
        var psi = amps
        var leftBond = 1
        var built: [MPSSite] = []
        for site in 0..<qubitCount {
            let rightQubits = qubitCount - site - 1
            let phys = 2
            let rightDimFull = 1 << rightQubits
            // Reshape psi to (leftBond * phys) × rightDimFull with LSB = current site.
            let rows = leftBond * phys
            var matrix = [MPSComplex](repeating: .zero, count: rows * rightDimFull)
            for left in 0..<leftBond {
                for s in 0..<phys {
                    for right in 0..<rightDimFull {
                        // Index in remaining register: bits [site..] as
                        // s + 2*right, and left is the virtual row group.
                        // psi is currently indexed as leftBond-major over remaining physical bits
                        // starting at `site`, with engineLSB for those bits.
                        let remIndex = s + 2 * right
                        let psiIndex = left * (phys * rightDimFull) + remIndex
                        matrix[(left * phys + s) * rightDimFull + right] = psi[psiIndex]
                    }
                }
            }

            if site == qubitCount - 1 {
                built.append(MPSSite(leftDim: leftBond, rightDim: 1, data: matrix))
                break
            }

            let (uFull, sAll, vtFull) = try MPSLinearAlgebra.svd(
                rows: rows,
                cols: rightDimFull,
                matrix: matrix
            )
            let keep = truncatedRank(singularValues: sAll)
            lastTruncationError = truncationWeight(singularValues: sAll, keep: keep)
            // Full-state compression: remaining ψ should stay unit-normalized.
            let renorm = truncationScale(singularValues: sAll, keep: keep, forceUnitFrobenius: true)

            var leftData = [MPSComplex](repeating: .zero, count: rows * keep)
            for row in 0..<rows {
                for j in 0..<keep {
                    let u = uFull[row * sAll.count + j]
                    let scale = sAll[j] * renorm
                    leftData[row * keep + j] = MPSComplex(re: u.re * scale, im: u.im * scale)
                }
            }
            built.append(MPSSite(leftDim: leftBond, rightDim: keep, data: leftData))

            // New psi: (keep) × rightDimFull from Vt, then reinterpret for next site.
            var next = [MPSComplex](repeating: .zero, count: keep * rightDimFull)
            for j in 0..<keep {
                for col in 0..<rightDimFull {
                    next[j * rightDimFull + col] = vtFull[j * rightDimFull + col]
                }
            }
            psi = next
            leftBond = keep
        }
        sites = built
    }

    /// Scale applied to kept singular values after truncation.
    /// - `forceUnitFrobenius`: set ||A||_F → 1 (full-state `replaceFromAmplitudes`).
    /// - otherwise: preserve ||A||_F, only undo weight lost to dropped σ (local 2Q updates).
    private func truncationScale(
        singularValues: [Double],
        keep: Int,
        forceUnitFrobenius: Bool
    ) -> Double {
        var keptWeight = 0.0
        var totalWeight = 0.0
        for (index, sigma) in singularValues.enumerated() {
            let w = sigma * sigma
            totalWeight += w
            if index < keep { keptWeight += w }
        }
        guard keptWeight > 0 else { return 1 }
        if forceUnitFrobenius {
            return 1.0 / sqrt(keptWeight)
        }
        return sqrt(totalWeight / keptWeight)
    }

    private func truncatedRank(singularValues: [Double]) -> Int {
        guard let sMax = singularValues.first, sMax > 0 else { return 1 }
        let threshold = configuration.svdTruncationThreshold * sMax
        var keep = 0
        for (index, sigma) in singularValues.enumerated() {
            if index >= configuration.maxBondDimension { break }
            if sigma < threshold { break }
            keep += 1
        }
        return max(keep, 1)
    }

    private func truncationWeight(singularValues: [Double], keep: Int) -> Double {
        guard !singularValues.isEmpty else { return 0 }
        var discarded = 0.0
        var total = 0.0
        for (index, sigma) in singularValues.enumerated() {
            let w = sigma * sigma
            total += w
            if index >= keep { discarded += w }
        }
        return total > 0 ? discarded / total : 0
    }
}
