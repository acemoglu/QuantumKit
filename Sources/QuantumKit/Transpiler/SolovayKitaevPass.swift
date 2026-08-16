import Foundation

/// Opt-in 1Q Solovay–Kitaev rewrite over ``Discrete1QNet`` + iterative ``SolovayKitaev``.
///
/// **Opt-in only** — not part of the default transpiler pipeline. Enable via
/// ``TranspileOptions/enableSolovayKitaev`` (and related options) or run explicitly
/// through ``PassManager``.
///
/// ## Distance
/// Approximations are accepted when
/// ``Discrete1QNet/phaseAlignedFrobenius(target:candidate:)``
/// `d(U,V) = min_φ ‖U − e^{iφ} V‖_F` is `≤ epsilon`.
///
/// ## Gate cases rewritten
/// - ``Gate/unitary1`` (2×2 matrix)
/// - ``Gate/customUnitary`` on **exactly one** qubit (4 amplitudes)
/// - ``Gate/u`` **only when** ``Options/rewriteU`` is `true` and angles are literal
///
/// All other gates (including multi-qubit `customUnitary`, unbound `u`, and native
/// Clifford+T letters) are left unchanged.
///
/// Emitted sequences use the discrete alphabet `{H, T, T†}` (and `S` / `S†` only if they
/// appear as net words). Throws ``SolovayKitaevError`` if `epsilon` cannot be met within
/// ``Options/maxRefinementIterations`` refinement iterations.
public struct SolovayKitaevPass: CompilerPass, Sendable {
    /// Optional discovery id for ``CompilerPassRegistry`` (not auto-registered).
    public static let passID = "quantumkit.solovay_kitaev"

    /// Configuration for ``SolovayKitaevPass``.
    public struct Options: Sendable {
        /// Target phase-aligned Frobenius distance.
        public var epsilon: Double
        /// Max residual-GC refinement iterations (`0` = basic approx only: library NN / ZYZ polish).
        public var maxRefinementIterations: Int
        /// When `true`, also rewrite literal ``Gate/u``. Default `false`.
        public var rewriteU: Bool
        /// BFS depth used when building the default net (ignored if ``net`` is set).
        public var netMaxWordLength: Int
        /// Optional prebuilt net; when `nil`, built once from ``netMaxWordLength``.
        public var net: Discrete1QNet?

        public init(
            epsilon: Double = SolovayKitaev.defaultEpsilon,
            maxRefinementIterations: Int = SolovayKitaev.defaultMaxRefinementIterations,
            rewriteU: Bool = false,
            netMaxWordLength: Int = Discrete1QNet.defaultMaxWordLength,
            net: Discrete1QNet? = nil
        ) {
            self.epsilon = epsilon
            self.maxRefinementIterations = maxRefinementIterations
            self.rewriteU = rewriteU
            self.netMaxWordLength = netMaxWordLength
            self.net = net
        }

        public static let `default` = Options()
    }

    public let options: Options
    private let synthesizer: SolovayKitaevSynthesizer

    public init(options: Options = .default) {
        self.options = options
        let net: Discrete1QNet
        if let provided = options.net {
            net = provided
        } else {
            net = Discrete1QNet.build(
                maxWordLength: options.netMaxWordLength,
                calibrationSampleCount: 0
            )
        }
        self.synthesizer = SolovayKitaevSynthesizer(net: net)
    }

    public init(
        epsilon: Double,
        maxRefinementIterations: Int = SolovayKitaev.defaultMaxRefinementIterations,
        rewriteU: Bool = false,
        net: Discrete1QNet? = nil
    ) {
        self.init(
            options: Options(
                epsilon: epsilon,
                maxRefinementIterations: maxRefinementIterations,
                rewriteU: rewriteU,
                net: net
            )
        )
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        var output = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for gate in circuit.gates {
            let pieces = try expand(gate)
            for piece in pieces {
                try output.apply(piece)
            }
        }
        return output
    }

    // MARK: - Expand

    private func expand(_ gate: Gate) throws -> [Gate] {
        switch gate {
        case .unitary1(let matrix, let target):
            return try approximate(matrix: matrix, qubit: target)

        case .customUnitary(let matrix, let qubits) where qubits.count == 1 && matrix.count == 4:
            return try approximate(matrix: matrix, qubit: qubits[0])

        case .u(let theta, let phi, let lambda, let target) where options.rewriteU:
            guard
                theta.literalValue != nil,
                phi.literalValue != nil,
                lambda.literalValue != nil
            else {
                throw SolovayKitaevError.unboundParameters
            }
            let matrix = try GateFusionPass.singleQubitMatrix(gate)
            let amplitudes = matrix.elements.map {
                ComplexAmplitude(real: QFloat($0.re), imaginary: QFloat($0.im))
            }
            return try approximate(matrix: amplitudes, qubit: target)

        default:
            return [gate]
        }
    }

    private func approximate(matrix: [ComplexAmplitude], qubit: Int) throws -> [Gate] {
        let approx = try synthesizer.approximate(
            matrix,
            epsilon: options.epsilon,
            maxRefinementIterations: options.maxRefinementIterations
        )
        return approx.retargeted(to: qubit).gates
    }
}
