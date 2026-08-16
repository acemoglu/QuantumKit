import Foundation

/// Fuses adjacent same-qubit single-qubit unitaries into one equivalent 1Q gate.
///
/// **Opt-in only** — not part of the default transpiler pipeline. Enable via
/// ``TranspileOptions/enableGateFusion`` or run explicitly through ``PassManager`` /
/// ``PassManager/dag(_:)``.
///
/// ## Fuse rules
/// - Consecutive **bound** 1Q unitaries on the same wire (nothing else on that qubit
///   between them) are multiplied into one equivalent 1Q (``Gate/u`` when Euler angles
///   recover cleanly, otherwise ``Gate/unitary1``).
/// - Wire adjacency spans intervening ops on *other* qubits; multi-qubit ops and
///   hard cuts on the wire split runs.
/// - Products that are approximately identity are dropped (so e.g. `H·H` collapses).
/// - Single-gate runs are left unchanged (named form preserved).
///
/// ## Hard cuts (never fuse across)
/// Same spirit as ``AlgebraicPreCompiler/isBarrier``:
/// `barrier`, `delay`, `measure`, `reset`, `c_if`, `initialize`.
///
/// ## Intentionally not fused
/// - Multi-qubit / controlled gates (`cx`, `cz`, `crx`, …)
/// - 1Q gates with **unbound** parameters
/// - Nested bodies inside `c_if` (the `c_if` itself is a fence)
/// - Cross-wire commutation sliding (compose with ``AlgebraicOptimizationPass`` first)
/// - Inverse-pair cancel tables already owned by ``AlgebraicOptimizationPass`` /
///   ``LocalUnitarySynthesisPass`` — this pass only collapses via matrix product
///
/// Does **not** change Metal kernels, ``ShotExecutionPolicy``, or flush/batching.
public struct GateFusionPass: DAGCompilerPass, CompilerPass, Sendable {
    private let identityTolerance: Double

    public init(identityTolerance: Double = 1e-10) {
        self.identityTolerance = identityTolerance
    }

    public func run(on dag: DAGCircuit) throws -> DAGCircuit {
        let circuit = try dag.toQuantumCircuit()
        let fused = try fuse(circuit)
        return try DAGCircuit(circuit: fused)
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        try fuse(circuit)
    }

    // MARK: - Core

    private func fuse(_ circuit: QuantumCircuit) throws -> QuantumCircuit {
        var pending: [[Gate]] = Array(repeating: [], count: circuit.qubitCount)
        var output: [Gate] = []
        output.reserveCapacity(circuit.gates.count)

        for gate in circuit.gates {
            if Self.isHardCut(gate) {
                let qubits = DAGCircuit.dependencyQubits(for: gate, qubitCount: circuit.qubitCount)
                try flush(qubits: qubits, pending: &pending, into: &output)
                output.append(gate)
                continue
            }

            if let target = Self.fusibleSingleQubitTarget(gate) {
                pending[target].append(gate)
                continue
            }

            // Multi-qubit or non-fusible 1Q: close runs on every touched wire, then emit.
            let qubits = DAGCircuit.dependencyQubits(for: gate, qubitCount: circuit.qubitCount)
            try flush(qubits: qubits, pending: &pending, into: &output)
            output.append(gate)
        }

        try flush(qubits: Array(0..<circuit.qubitCount), pending: &pending, into: &output)

        var result = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for gate in output {
            try result.apply(gate)
        }
        return result
    }

    private func flush(
        qubits: [Int],
        pending: inout [[Gate]],
        into output: inout [Gate]
    ) throws {
        var seen = Set<Int>()
        for q in qubits where seen.insert(q).inserted {
            let run = pending[q]
            pending[q] = []
            guard !run.isEmpty else { continue }
            if run.count == 1 {
                output.append(run[0])
                continue
            }
            if let fused = try fuseRun(run, target: q) {
                output.append(fused)
            }
        }
    }

    /// Returns fused ``Gate/u`` (preferred, basis-friendly) or ``Gate/unitary1``,
    /// or `nil` when the product is ≈ identity (dropped).
    private func fuseRun(_ run: [Gate], target: Int) throws -> Gate? {
        var product = UnitaryMatrix.identity(2)
        for gate in run {
            // ``CircuitUnitary.matrix(for:qubitCount:)`` embeds into a full register.
            // With `qubitCount: 1`, only `target == 0` is valid — remap first.
            let local = try Self.singleQubitMatrix(gate)
            product = local.multiplied(by: product)
        }

        if product.isApproximatelyEqual(to: .identity(2), tolerance: identityTolerance) {
            return nil
        }

        if let angles = Self.eulerUAngles(from: product, tolerance: identityTolerance) {
            return .u(
                theta: QFloatExpr(QFloat(angles.theta)),
                phi: QFloatExpr(QFloat(angles.phi)),
                lambda: QFloatExpr(QFloat(angles.lambda)),
                target: target
            )
        }

        let matrix: [ComplexAmplitude] = product.elements.map {
            ComplexAmplitude(real: QFloat($0.re), imaginary: QFloat($0.im))
        }
        return .unitary1(matrix: matrix, target: target)
    }

    /// ZYZ / Qiskit-``U`` Euler angles for a 2×2 unitary (global phase discarded).
    ///
    /// Matches ``CircuitUnitary``'s ``Gate/u`` convention:
    /// `[[c, -e^{iλ}s], [e^{iφ}s, e^{i(φ+λ)}c]]` with `c=cos(θ/2)`, `s=sin(θ/2)`.
    static func eulerUAngles(
        from matrix: UnitaryMatrix,
        tolerance: Double
    ) -> (theta: Double, phi: Double, lambda: Double)? {
        guard matrix.dimension == 2 else { return nil }

        let u00 = matrix[0, 0]
        let u01 = matrix[0, 1]
        let u10 = matrix[1, 0]
        let u11 = matrix[1, 1]

        // Peel global phase so u00 is real and non-negative when |u00| is usable.
        let phase00 = atan2(u00.im, u00.re)
        let mag00 = hypot(u00.re, u00.im)
        let peel: UnitaryComplex
        if mag00 > tolerance {
            peel = UnitaryComplex(re: cos(-phase00), im: sin(-phase00))
        } else {
            let phase10 = atan2(u10.im, u10.re)
            peel = UnitaryComplex(re: cos(-phase10), im: sin(-phase10))
        }

        let a00 = peel * u00
        let a01 = peel * u01
        let a10 = peel * u10
        let a11 = peel * u11

        let cosHalf = max(-1.0, min(1.0, a00.re))
        let theta = 2.0 * acos(cosHalf)
        let sinHalf = sin(theta / 2.0)

        let phi: Double
        let lambda: Double
        if abs(sinHalf) < tolerance {
            if cosHalf >= 0 {
                // θ ≈ 0: only φ+λ is identifiable — put phase on λ.
                phi = 0
                lambda = atan2(a11.im, a11.re)
            } else {
                // θ ≈ π: u01 ≈ -e^{iλ}, u10 ≈ e^{iφ}
                phi = atan2(a10.im, a10.re)
                lambda = atan2(-a01.im, -a01.re)
            }
        } else {
            phi = atan2(a10.im, a10.re)
            lambda = atan2(-a01.im, -a01.re)
        }

        return (theta, phi, lambda)
    }

    // MARK: - Classification

    /// Hard ordering fences — never fuse across (mirrors algebraic barriers + delay).
    public static func isHardCut(_ gate: Gate) -> Bool {
        switch gate {
        case .measure, .reset, .c_if, .while_c, .initialize, .barrier, .delay:
            return true
        default:
            return false
        }
    }

    /// Target qubit when `gate` is a bound single-qubit unitary eligible for fusion.
    public static func fusibleSingleQubitTarget(_ gate: Gate) -> Int? {
        if gate.containsUnboundParameters { return nil }
        switch gate {
        case .h(let t), .x(let t), .y(let t), .z(let t),
             .s(let t), .t(let t), .sdg(let t), .tdg(let t),
             .sx(let t), .sxdg(let t), .id(let t),
             .p(_, let t), .u(_, _, _, let t),
             .rx(_, let t), .ry(_, let t), .rz(_, let t):
            return t
        case .unitary1(let matrix, let t) where matrix.count == 4:
            return t
        case .customUnitary(let matrix, let qubits) where qubits.count == 1 && matrix.count == 4:
            return qubits[0]
        default:
            return nil
        }
    }

    /// 2×2 matrix of a fusible 1Q gate, independent of its wire index.
    static func singleQubitMatrix(_ gate: Gate) throws -> UnitaryMatrix {
        let onZero = gate.remappingQubits { _ in 0 }
        return try CircuitUnitary.matrix(for: onZero, qubitCount: 1)
    }
}
