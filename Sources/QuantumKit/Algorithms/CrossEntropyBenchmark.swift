import Foundation

// MARK: - Configuration

/// Options for small highly-entangling random circuits used by linear XEB.
///
/// Each depth layer applies a random single-qubit `u(θ,φ,λ)` on every qubit, then
/// nearest-neighbor CX on alternating edges (even layers `(0,1),(2,3),…`;
/// odd layers `(1,2),(3,4),…`).
public struct CrossEntropyBenchmarkOptions: Sendable, Equatable {
    /// Register width. Supported range for this session: `1...4`.
    public var qubitCount: Int
    /// Number of (1Q + CX) layers.
    public var depth: Int

    public init(qubitCount: Int, depth: Int) {
        self.qubitCount = qubitCount
        self.depth = depth
    }
}

public enum CrossEntropyBenchmarkError: Error, Equatable, Sendable {
    case invalidQubitCount(Int)
    case invalidDepth(Int)
    case emptyProbabilities
    case probabilityCountMismatch(expected: Int, actual: Int)
    case emptySamples
    case shotCountMismatch(expected: Int, actual: Int)
    case outcomeOutOfRange(outcome: Int, dimension: Int)
}

// MARK: - Result

/// Linear cross-entropy benchmarking (XEB) score and bookkeeping.
public struct LinearXEBResult: Sendable, Equatable {
    /// Estimated `F_XEB = ⟨2ⁿ p_U(x) − 1⟩` over the sample ensemble.
    public let fxeb: Double
    /// Exact expectation of the same estimator when shots are drawn from `p_U`
    /// itself: `2ⁿ Σ_x p_U(x)² − 1`.
    public let expectedUnderIdeal: Double
    public let qubitCount: Int
    public let shots: Int
    public let hilbertDimension: Int
}

// MARK: - Public API

/// Linear cross-entropy benchmarking of ``Sampler`` shots against ideal CPU SV Born probs.
///
/// ## Exact formula implemented
/// Let `n` = qubit count, `D = 2ⁿ`, and `p_U(x)` the ideal computational-basis
/// probabilities of circuit `U` (engine-LSB index `x`). For a sample of outcomes
/// `{x_i}_{i=1}^{S}` (or an equivalent histogram),
///
/// ```
/// F_XEB = (1/S) Σ_{i=1}^{S} ( D · p_U(x_i) − 1 )
/// ```
///
/// Equivalently with histogram counts `c(x)` (`Σ c = S`):
///
/// ```
/// F_XEB = (1/S) Σ_x c(x) · ( D · p_U(x) − 1 )
/// ```
///
/// When samples are drawn exactly from `p_U`, the population value is
///
/// ```
/// E[F_XEB] = D · Σ_x p_U(x)² − 1
/// ```
///
/// For chaotic / Porter–Thomas–like circuits this approaches `1` as `n` grows
/// (`≈ (D−1)/(D+1)`). MVP honesty: this module does **not** claim Google Sycamore
/// experimental parity, full Porter–Thomas certification, or hardware XEB protocols.
///
/// ## Ideal probabilities
/// From host **CPU statevector** Born probabilities
/// (``CPUStateVector/probabilitiesDouble()``), same indexing as ``ShotCounts/counts``.
///
/// ## Sampling backends
/// Primary path: ``Sampler`` + ``CPUStatevectorBackend``. Metal ``StatevectorBackend``
/// is optional when available (same formula; RNG schedule may differ).
public enum CrossEntropyBenchmark {

    /// Builds a seeded random circuit of random 1Q + adjacent CX layers.
    public static func makeRandomCircuit(
        options: CrossEntropyBenchmarkOptions,
        seed: UInt64
    ) throws -> QuantumCircuit {
        guard (1...4).contains(options.qubitCount) else {
            throw CrossEntropyBenchmarkError.invalidQubitCount(options.qubitCount)
        }
        guard options.depth >= 1 else {
            throw CrossEntropyBenchmarkError.invalidDepth(options.depth)
        }

        var rng = QuantumRNG.seeded(seed)
        var circuit = try QuantumCircuit(qubitCount: options.qubitCount)
        let n = options.qubitCount

        for layer in 0..<options.depth {
            for q in 0..<n {
                let theta = QFloat(rng.nextUnitDouble() * Double.pi)
                let phi = QFloat(rng.nextUnitDouble() * 2.0 * Double.pi)
                let lambda = QFloat(rng.nextUnitDouble() * 2.0 * Double.pi)
                try circuit.u(theta: theta, phi: phi, lambda: lambda, q)
            }
            for (c, t) in adjacentCXPairs(layer: layer, qubitCount: n) {
                try circuit.cx(c, t)
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

    /// Population linear XEB when samples are drawn from `p_U`: `D · Σ p² − 1`.
    public static func expectedLinearXEB(
        idealProbabilities: [Double]
    ) throws -> Double {
        guard !idealProbabilities.isEmpty else {
            throw CrossEntropyBenchmarkError.emptyProbabilities
        }
        let dimension = idealProbabilities.count
        guard dimension >= 2, dimension.nonzeroBitCount == 1 else {
            throw CrossEntropyBenchmarkError.probabilityCountMismatch(
                expected: dimension > 0 ? 1 << dimension.trailingZeroBitCount : 2,
                actual: dimension
            )
        }
        let sumSquares = idealProbabilities.reduce(0.0) { $0 + $1 * $1 }
        return Double(dimension) * sumSquares - 1.0
    }

    /// Linear XEB from an explicit list of engine-LSB outcome indices.
    public static func linearXEB(
        idealProbabilities: [Double],
        outcomes: [Int]
    ) throws -> Double {
        guard !outcomes.isEmpty else {
            throw CrossEntropyBenchmarkError.emptySamples
        }
        let dimension = idealProbabilities.count
        guard dimension > 0 else {
            throw CrossEntropyBenchmarkError.emptyProbabilities
        }

        var total = 0.0
        for x in outcomes {
            guard x >= 0, x < dimension else {
                throw CrossEntropyBenchmarkError.outcomeOutOfRange(outcome: x, dimension: dimension)
            }
            total += Double(dimension) * idealProbabilities[x] - 1.0
        }
        return total / Double(outcomes.count)
    }

    /// Linear XEB from a ``ShotCounts`` histogram (engine-LSB keys).
    public static func linearXEB(
        idealProbabilities: [Double],
        shotCounts: ShotCounts
    ) throws -> Double {
        guard shotCounts.shots > 0 else {
            throw CrossEntropyBenchmarkError.emptySamples
        }
        let dimension = idealProbabilities.count
        guard dimension > 0 else {
            throw CrossEntropyBenchmarkError.emptyProbabilities
        }

        var total = 0.0
        var accounted = 0
        for (x, count) in shotCounts.counts {
            guard x >= 0, x < dimension else {
                throw CrossEntropyBenchmarkError.outcomeOutOfRange(outcome: x, dimension: dimension)
            }
            guard count > 0 else { continue }
            total += Double(count) * (Double(dimension) * idealProbabilities[x] - 1.0)
            accounted += count
        }
        guard accounted == shotCounts.shots else {
            throw CrossEntropyBenchmarkError.shotCountMismatch(
                expected: shotCounts.shots,
                actual: accounted
            )
        }
        return total / Double(shotCounts.shots)
    }

    /// Draw Sampler shots and score them against ideal CPU SV probabilities of `circuit`.
    public static func evaluateLinearXEB(
        circuit: QuantumCircuit,
        backend: any QuantumBackend,
        shots: Int,
        seed: UInt64
    ) throws -> LinearXEBResult {
        let probs = try idealProbabilities(of: circuit)
        let expected = try expectedLinearXEB(idealProbabilities: probs)

        let sampler = Sampler()
        let options = QuantumRunOptions(seed: seed, shots: shots)
        let result = try sampler.run(circuit: circuit, backend: backend, options: options)
        guard let counts = result.shotCounts else {
            throw CrossEntropyBenchmarkError.emptySamples
        }
        let fxeb = try linearXEB(idealProbabilities: probs, shotCounts: counts)
        return LinearXEBResult(
            fxeb: fxeb,
            expectedUnderIdeal: expected,
            qubitCount: circuit.qubitCount,
            shots: shots,
            hilbertDimension: probs.count
        )
    }

    /// Seeded model circuit + noiseless linear XEB via the given backend.
    public static func evaluateLinearXEB(
        options: CrossEntropyBenchmarkOptions,
        circuitSeed: UInt64,
        sampleSeed: UInt64,
        shots: Int,
        backend: any QuantumBackend
    ) throws -> (circuit: QuantumCircuit, result: LinearXEBResult) {
        let circuit = try makeRandomCircuit(options: options, seed: circuitSeed)
        let result = try evaluateLinearXEB(
            circuit: circuit,
            backend: backend,
            shots: shots,
            seed: sampleSeed
        )
        return (circuit, result)
    }

    // MARK: - Porter–Thomas smoke helpers

    /// Mean ideal probability `⟨p⟩ = 1/D` for a normalized distribution.
    public static func meanProbability(_ probabilities: [Double]) -> Double {
        guard !probabilities.isEmpty else { return 0 }
        return probabilities.reduce(0, +) / Double(probabilities.count)
    }

    /// Collision probability `Σ p²` (Porter–Thomas expectation ≈ `2/(D+1)` for Haar).
    public static func collisionProbability(_ probabilities: [Double]) -> Double {
        probabilities.reduce(0.0) { $0 + $1 * $1 }
    }

    /// Alternating nearest-neighbor CX pairs for a depth layer.
    public static func adjacentCXPairs(layer: Int, qubitCount: Int) -> [(Int, Int)] {
        if qubitCount < 2 { return [] }
        if qubitCount == 2 { return [(0, 1)] }
        var pairs: [(Int, Int)] = []
        let start = layer % 2
        var q = start
        while q + 1 < qubitCount {
            pairs.append((q, q + 1))
            q += 2
        }
        return pairs
    }
}
