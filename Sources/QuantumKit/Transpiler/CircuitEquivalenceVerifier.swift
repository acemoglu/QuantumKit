import Foundation

/// Public circuit / unitary equivalence verification (up to global phase).
///
/// Wraps the internal ``CircuitUnitary`` matrix builder used by transpiler tests and
/// decomposition checks. Prefer this over engine-based Born-rule comparisons when both
/// circuits are unitary-only and small enough for exact matrix construction.
public enum CircuitEquivalenceVerifier: Sendable {

    /// Returns `true` when `lhs` and `rhs` implement the same unitary up to a global phase.
    public static func areEquivalent(
        _ lhs: QuantumCircuit,
        _ rhs: QuantumCircuit,
        tolerance: Double = 1e-4
    ) throws -> Bool {
        try CircuitUnitary.areEquivalent(lhs, rhs, tolerance: tolerance)
    }

    /// Returns `true` when `circuit` implements `unitary` (row-major 2ⁿ×2ⁿ complex entries)
    /// up to a global phase. `unitary.count` must equal `(1 << circuit.qubitCount)^2`.
    public static func circuitMatchesUnitary(
        _ circuit: QuantumCircuit,
        unitary: [ComplexAmplitude],
        tolerance: Double = 1e-4
    ) throws -> Bool {
        let dimension = 1 << circuit.qubitCount
        guard unitary.count == dimension * dimension else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "unitary has \(unitary.count) entries; expected \(dimension * dimension)"
            )
        }
        let left = try CircuitUnitary.build(circuit: circuit)
        let right = UnitaryMatrix(
            dimension: dimension,
            elements: unitary.map {
                UnitaryComplex(re: Double($0.real), im: Double($0.imaginary))
            }
        )
        return CircuitUnitary.areUnitarilyEquivalent(left, right, tolerance: tolerance)
    }

    /// Engine-based Born-rule action check (weaker than full unitary; allows input-dependent phases).
    public static func haveIdenticalBornAction(
        _ lhs: QuantumCircuit,
        _ rhs: QuantumCircuit,
        engine: QuantumEngine,
        tolerance: QFloat = 1e-4
    ) throws -> Bool {
        try CircuitEquivalence.haveIdenticalAction(lhs, rhs, engine: engine, tolerance: tolerance)
    }

    /// Compares `system` (width `systemQubitCount`) to `expanded` (wider) on the subspace where
    /// extra ancilla qubits (indices `systemQubitCount..<expanded.qubitCount`) are |0⟩.
    ///
    /// Used for clean-ancilla decompositions whose full unitary differs on dirty-ancilla inputs.
    public static func areEquivalentWithZeroAncillas(
        system: QuantumCircuit,
        expanded: QuantumCircuit,
        systemQubitCount: Int,
        tolerance: Double = 1e-4
    ) throws -> Bool {
        guard system.qubitCount == systemQubitCount else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "system circuit width \(system.qubitCount) != systemQubitCount \(systemQubitCount)"
            )
        }
        guard expanded.qubitCount >= systemQubitCount else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "expanded circuit is narrower than the system register"
            )
        }

        let systemU = try CircuitUnitary.build(circuit: system)
        let expandedU = try CircuitUnitary.build(circuit: expanded)
        let ancillaCount = expanded.qubitCount - systemQubitCount
        let systemDim = 1 << systemQubitCount

        // Embed system unitary as U ⊗ I_{ancilla} on the zero-ancilla block:
        // basis ordering is little-endian qubit 0 = LSB, so ancillas are the high bits.
        var embedded = UnitaryMatrix.identity(1 << expanded.qubitCount)
        for row in 0..<systemDim {
            for col in 0..<systemDim {
                // ancilla bits = 0 → full index equals system index
                embedded[row, col] = systemU[row, col]
            }
        }

        // Restrict expanded unitary to rows/cols with ancilla bits clear and compare.
        var leftBlock = UnitaryMatrix.identity(systemDim)
        var rightBlock = UnitaryMatrix.identity(systemDim)
        let ancillaMask = ((1 << ancillaCount) - 1) << systemQubitCount
        for row in 0..<expandedU.dimension {
            guard (row & ancillaMask) == 0 else { continue }
            for col in 0..<expandedU.dimension {
                guard (col & ancillaMask) == 0 else { continue }
                let r = row & (systemDim - 1)
                let c = col & (systemDim - 1)
                leftBlock[r, c] = embedded[row, col]
                rightBlock[r, c] = expandedU[row, col]
            }
        }
        return CircuitUnitary.areUnitarilyEquivalent(leftBlock, rightBlock, tolerance: tolerance)
    }
}
