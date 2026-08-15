import Foundation

/// Host CPU MPS engine (1D open-boundary). No Metal kernels.
///
/// **Gate / topology limits (B18 MVP):**
/// - Single-qubit unitaries (including `unitary1`) after ``QuantumEngine/expandForExecution``.
/// - Two-qubit unitaries: applied on adjacent sites when possible; **non-adjacent pairs use a
///   SWAP bubble chain** (then reverse) so GHZ-style `CX(0,n-1)` works on a 1D chain.
/// - Rejects native 3+-qubit unitaries that do not expand to 1Q/2Q, `initialize`, mid-circuit
///   `measure` / `reset` / `c_if` (terminal Z sampling only), and any noise model.
///
/// **χ:** ``MPSConfiguration/maxBondDimension``. Evolution always uses local adjacent SVD
/// updates with on-the-fly χ truncation. Amplitude export is capped by
/// ``MPSConfiguration/maxAmplitudeExportQubits``.
public final class MPSEngine: @unchecked Sendable {
    public let configuration: MPSConfiguration

    public init(configuration: MPSConfiguration = .default) {
        self.configuration = configuration
    }

    public func execute(
        _ circuit: QuantumCircuit,
        on state: inout MPSState
    ) throws -> CircuitExecutionResult {
        guard circuit.qubitCount == state.qubitCount else {
            throw MPSError.qubitCountMismatch(circuit: circuit.qubitCount, state: state.qubitCount)
        }
        try circuit.requireFullyBound()

        var applied = 0
        for gate in circuit.gates {
            try applyRuntimeGate(gate, on: &state)
            applied += 1
        }
        return CircuitExecutionResult(
            measurementOutcomes: [],
            classicalMemory: ClassicalMemory(
                registerWidths: circuit.classicalRegisters.map(\.bitCount)
            ),
            appliedGateCount: applied,
            unitaryRenormCount: nil
        )
    }

    func sampleTerminalOutcome(
        circuit: QuantumCircuit,
        state: inout MPSState,
        rng: inout QuantumRNG
    ) throws -> Int {
        _ = try execute(circuit, on: &state)
        return try state.sampleOutcome(rng: &rng)
    }

    // MARK: - Gate dispatch

    private func applyRuntimeGate(_ gate: Gate, on state: inout MPSState) throws {
        switch gate {
        case .barrier, .delay, .id:
            return

        case .measure, .reset, .c_if, .initialize:
            throw MPSError.unsupportedGate(gate)

        case .customUnitary(_, let qubits):
            if qubits.count > 2 {
                throw MPSError.unsupportedMultiQubitGate(qubitCount: qubits.count)
            }
            try applyUnitaryGate(gate, on: &state)

        default:
            try applyUnitaryGate(gate, on: &state)
        }
    }

    private func applyUnitaryGate(_ gate: Gate, on state: inout MPSState) throws {
        let pieces = try QuantumEngine.expandForExecution(gate)
        for piece in pieces {
            try applyExpandedPiece(piece, on: &state)
        }
    }

    private func applyExpandedPiece(_ gate: Gate, on state: inout MPSState) throws {
        switch gate {
        case .barrier, .delay, .id:
            return

        case .unitary1(let matrix, let target):
            try apply1QMatrix(matrix, target: target, on: &state)

        case .customUnitary(let matrix, let qubits):
            if qubits.count == 1, let q = qubits.first {
                try apply1QMatrix(matrix, target: q, on: &state)
            } else if qubits.count == 2 {
                try applyTwoQubitMatrix(
                    matrix.map { MPSComplex($0) },
                    q0: qubits[0],
                    q1: qubits[1],
                    on: &state
                )
            } else {
                throw MPSError.unsupportedMultiQubitGate(qubitCount: qubits.count)
            }

        case .h, .x, .y, .z, .s, .sdg, .t, .tdg, .sx, .sxdg, .p, .u, .rx, .ry, .rz:
            let q = gate.affectedQubits[0]
            let u = try CircuitUnitary.matrix(for: gate.remappingQubits { _ in 0 }, qubitCount: 1)
            let m = [
                MPSComplex(u[0, 0]), MPSComplex(u[0, 1]),
                MPSComplex(u[1, 0]), MPSComplex(u[1, 1]),
            ]
            state.apply1Q(m, target: q)

        case .cx, .cz, .swap, .cp, .crx, .cry, .crz, .rzz, .rxx, .ryy, .dcx, .iswap, .ecr:
            let qs = gate.affectedQubits
            guard qs.count == 2 else {
                throw MPSError.unsupportedGate(gate)
            }
            try applyTwoQubitGate(gate, q0: qs[0], q1: qs[1], on: &state)

        case .ccx, .mcx, .mcz, .cswap:
            // Expand further if possible; otherwise reject.
            if GateDecomposition.needsExecutionExpansion(gate) {
                try applyUnitaryGate(gate, on: &state)
            } else {
                let expanded = try GateDecomposition.expand(gate)
                for piece in expanded {
                    try applyExpandedPiece(piece, on: &state)
                }
            }

        default:
            throw MPSError.unsupportedGate(gate)
        }
    }

    private func apply1QMatrix(
        _ matrix: [ComplexAmplitude],
        target: Int,
        on state: inout MPSState
    ) throws {
        guard matrix.count == 4 else {
            throw MPSError.unsupportedGate(.unitary1(matrix: matrix, target: target))
        }
        let m = matrix.map { MPSComplex($0) }
        state.apply1Q(m, target: target)
    }

    private func applyTwoQubitGate(
        _ gate: Gate,
        q0: Int,
        q1: Int,
        on state: inout MPSState
    ) throws {
        // Fast path: adjacent CX with explicit bit packing (avoids embed layout ambiguity).
        if case .cx(let control, let target) = gate, abs(control - target) == 1 {
            let left = min(control, target)
            if control == left && target == left + 1 {
                try state.applyAdjacentCX(controlOnLeft: true, left: left)
                return
            }
            if target == left && control == left + 1 {
                try state.applyAdjacentCX(controlOnLeft: false, left: left)
                return
            }
        }

        let mapped = try gate.remappingQubits { q in
            if q == q0 { return 0 }
            if q == q1 { return 1 }
            throw MPSError.unsupportedGate(gate)
        }
        let u = try CircuitUnitary.matrix(for: mapped, qubitCount: 2)
        var matrix = [MPSComplex](repeating: .zero, count: 16)
        for row in 0..<4 {
            for col in 0..<4 {
                matrix[row * 4 + col] = MPSComplex(u[row, col])
            }
        }
        try applyTwoQubitMatrix(matrix, q0: q0, q1: q1, on: &state)
    }

    /// Apply a 4×4 on qubits `q0`,`q1` (any separation) via adjacent SVD + SWAP chain.
    private func applyTwoQubitMatrix(
        _ matrix: [MPSComplex],
        q0: Int,
        q1: Int,
        on state: inout MPSState
    ) throws {
        if abs(q0 - q1) == 1 {
            let left = min(q0, q1)
            let ordered: [MPSComplex]
            if q0 < q1 {
                ordered = matrix
            } else {
                // Physical packing is always (leftSite, rightSite) = (lo, hi).
                // If the gate was defined with q0=hi, q1=lo, permute basis: swap bit0↔bit1.
                ordered = Self.permuteQubitOrder(matrix)
            }
            try state.applyAdjacent2Q(ordered, left: left)
            return
        }

        // Non-adjacent: bubble the higher qubit down next to the lower, apply, bubble back.
        let lo = min(q0, q1)
        let hi = max(q0, q1)
        for i in stride(from: hi - 1, through: lo + 1, by: -1) {
            try state.applyAdjacentSwap(left: i)
        }
        // hi now at lo+1. Remap original indices to (lo, lo+1).
        let leftIsQ0 = q0 == lo
        let local: [MPSComplex]
        if leftIsQ0 {
            // q0=lo, q1=hi → now q1 at lo+1; packing matches matrix as built for (q0,q1)=(0,1)
            local = matrix
        } else {
            // q0=hi, q1=lo → after move, q0 at lo+1, q1 at lo → need bit swap of matrix
            local = Self.permuteQubitOrder(matrix)
        }
        try state.applyAdjacent2Q(local, left: lo)
        for i in (lo + 1)..<hi {
            try state.applyAdjacentSwap(left: i)
        }
    }

    /// Swap the two qubit wire order of a 4×4 (engineLSB bit0 ↔ bit1).
    private static func permuteQubitOrder(_ u: [MPSComplex]) -> [MPSComplex] {
        func swapBits(_ x: Int) -> Int {
            let b0 = x & 1
            let b1 = (x >> 1) & 1
            return b1 | (b0 << 1)
        }
        var out = [MPSComplex](repeating: .zero, count: 16)
        for row in 0..<4 {
            for col in 0..<4 {
                out[swapBits(row) * 4 + swapBits(col)] = u[row * 4 + col]
            }
        }
        return out
    }
}
