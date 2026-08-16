import Foundation

/// Rewrites two-qubit ``Gate/customUnitary`` ops into local 1Q gates + ≤3 ``Gate/cx``
/// via ``CartanKAK``.
///
/// **Opt-in only** — not part of the default transpiler pipeline. Enable via
/// ``TranspileOptions/enableKAKSynthesis`` or run explicitly through ``PassManager``.
///
/// ## Rewrite rules
/// - **2Q** ``customUnitary`` (16 amplitudes, distinct qubit pair): replace with
///   ``CartanKAK/decompose(_:qubits:absoluteTolerance:verifyRoundTrip:)`` factors
///   (``Gate/u`` / ``Gate/rx`` / ``Gate/ry`` / ``Gate/rz`` + ≤3 ``Gate/cx``).
/// - **1Q** ``customUnitary`` / ``Gate/unitary1`` and all non-custom gates: unchanged.
/// - **3Q+** ``customUnitary``: left unchanged (KAK is 2Q-only; no silent reject).
///
/// Does **not** implement Solovay–Kitaev, change Metal kernels, or alter default
/// optimization levels.
public struct KAKSynthesisPass: CompilerPass, Sendable {
    /// Optional discovery id for ``CompilerPassRegistry`` (not auto-registered).
    public static let passID = "quantumkit.kak_synthesis"

    private let absoluteTolerance: Double
    private let verifyRoundTrip: Bool

    public init(
        absoluteTolerance: Double = CartanKAK.absoluteTolerance,
        verifyRoundTrip: Bool = true
    ) {
        self.absoluteTolerance = absoluteTolerance
        self.verifyRoundTrip = verifyRoundTrip
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
        case .customUnitary(let matrix, let qubits) where qubits.count == 2:
            let decomp = try CartanKAK.decompose(
                matrix,
                qubits: (qubits[0], qubits[1]),
                absoluteTolerance: absoluteTolerance,
                verifyRoundTrip: verifyRoundTrip
            )
            return decomp.gates

        default:
            return [gate]
        }
    }
}
