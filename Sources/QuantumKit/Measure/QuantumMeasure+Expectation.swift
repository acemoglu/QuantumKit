import Foundation
import Metal

extension QuantumMeasurement {

    /// ⟨Z⟩ for a single qubit in the computational basis.
    public static func expectationZ(
        state: StateVector,
        engine: QuantumEngine,
        qubit: Int
    ) throws -> QFloat {
        try expectationPauliZ(state: state, engine: engine, qubits: [qubit])
    }

    /// ⟨Z_a Z_b⟩ for two qubits.
    public static func expectationZZ(
        state: StateVector,
        engine: QuantumEngine,
        qubitA: Int,
        qubitB: Int
    ) throws -> QFloat {
        try expectationPauliZ(state: state, engine: engine, qubits: [qubitA, qubitB])
    }

    /// ⟨Z_{i_0} Z_{i_1} …⟩ as the product of Pauli-Z on the listed qubits.
    public static func expectationPauliZ(
        state: StateVector,
        engine: QuantumEngine,
        qubits: [Int]
    ) throws -> QFloat {
        try validateQubits(qubits, qubitCount: state.qubitCount)

        // ⟨Π_q Z_q⟩ = Σⱼ (−1)^popcount(j & zMask)·|aⱼ|². XOR-folding the per-qubit bits preserves the
        // original per-occurrence eigenvalue product: a qubit listed an even number of times cancels
        // back to identity, an odd number leaves a single sign — matched by the parity over zMask.
        var zMask = 0
        for qubit in qubits {
            zMask ^= (1 << qubit)
        }

        // No X/Y factors ⇒ flipMask = 0 and the global phase is +1; the sign is carried entirely by
        // the parity of the measured Z bits, so this is the Z-only specialization of the general path.
        return try engine.pauliExpectation(
            on: state,
            flipMask: 0,
            signMask: zMask,
            phaseBaseReal: 1,
            phaseBaseImag: 0
        )
    }

    /// ⟨X⟩ for a single qubit in the computational basis.
    public static func expectationX(
        state: StateVector,
        engine: QuantumEngine,
        qubit: Int
    ) throws -> QFloat {
        try validateQubits([qubit], qubitCount: state.qubitCount)

        // ⟨X⟩ is the single-X specialization of the general Pauli path: flip the target qubit, no
        // sign mask, and a unit global phase (Σⱼ Re[ conj(a_{j⊕mask})·aⱼ ] = Σⱼ rⱼ·r_flip + iⱼ·i_flip).
        let mask = 1 << qubit
        return try engine.pauliExpectation(
            on: state,
            flipMask: mask,
            signMask: 0,
            phaseBaseReal: 1,
            phaseBaseImag: 0
        )
    }

    /// ⟨ψ|P|ψ⟩ for an arbitrary Pauli tensor product `P`.
    ///
    /// `paulis` maps a qubit index to its Pauli factor; any qubit absent from the map (or mapped
    /// to `.i`) acts as identity. Computed analytically on the CPU in O(2ⁿ) by reading amplitudes
    /// directly — no extra circuit or state clone is required.
    ///
    /// `P|j⟩ = phase(j)·|j ⊕ flipMask⟩`, where `flipMask` collects the X/Y qubits and
    /// `phase(j) = Π_{Y}(bit=0 → +i, bit=1 → −i) · Π_{Z}((−1)^bit)`. The `Π_Y` magnitude is the
    /// global factor `i^{#Y}`, and the per-state sign comes from the bits set under the Y and Z
    /// qubits. The expectation `Σⱼ conj(a_{j⊕flipMask})·phase(j)·aⱼ` is real, so its real part is returned.
    public static func expectation(
        state: StateVector,
        engine: QuantumEngine,
        paulis: [Int: Pauli]
    ) throws -> QFloat {
        var flipMask = 0
        var yMask = 0
        var zMask = 0
        var yCount = 0

        for (qubit, pauli) in paulis {
            guard qubit >= 0, qubit < state.qubitCount else {
                throw QuantumMeasurementError.qubitIndexOutOfBounds(index: qubit, qubitCount: state.qubitCount)
            }
            let bit = 1 << qubit
            switch pauli {
            case .i:
                continue
            case .x:
                flipMask |= bit
            case .y:
                flipMask |= bit
                yMask |= bit
                yCount += 1
            case .z:
                zMask |= bit
            }
        }

        // i^{#Y}: a global complex factor shared by every term.
        let phaseBaseReal: QFloat
        let phaseBaseImag: QFloat
        switch yCount & 3 {
        case 0: (phaseBaseReal, phaseBaseImag) = (1, 0)
        case 1: (phaseBaseReal, phaseBaseImag) = (0, 1)
        case 2: (phaseBaseReal, phaseBaseImag) = (-1, 0)
        default: (phaseBaseReal, phaseBaseImag) = (0, -1)
        }

        let signMask = yMask | zMask

        // Per-state fold `Σⱼ Re[ conj(a_{j⊕flipMask})·phase(j)·aⱼ ]` runs on the GPU with a compensated
        // (double-single) reduction, so the host reads a single scalar instead of touching every
        // amplitude. Bit-for-bit the same arithmetic as the previous O(2ⁿ) CPU loop.
        return try engine.pauliExpectation(
            on: state,
            flipMask: flipMask,
            signMask: signMask,
            phaseBaseReal: phaseBaseReal,
            phaseBaseImag: phaseBaseImag
        )
    }

    /// ⟨ψ|P|ψ⟩ for a Pauli label such as `"XYZ"` or `"IXZ"`.
    ///
    /// The label is MSB-first to match ``measure`` bit-array ordering: the leftmost character is
    /// the highest-index qubit. Its length must equal `state.qubitCount`.
    public static func expectation(
        state: StateVector,
        engine: QuantumEngine,
        pauliString: String
    ) throws -> QFloat {
        let paulis = try parsePauliString(pauliString, qubitCount: state.qubitCount)
        return try expectation(state: state, engine: engine, paulis: paulis)
    }

    static func parsePauliString(_ string: String, qubitCount: Int) throws -> [Int: Pauli] {
        let characters = Array(string.uppercased())
        guard characters.count == qubitCount else {
            throw QuantumMeasurementError.invalidPauliString(string)
        }

        var paulis: [Int: Pauli] = [:]
        for (position, character) in characters.enumerated() {
            let qubit = qubitCount - 1 - position
            let pauli: Pauli
            switch character {
            case "I": pauli = .i
            case "X": pauli = .x
            case "Y": pauli = .y
            case "Z": pauli = .z
            default:
                throw QuantumMeasurementError.invalidPauliString(string)
            }
            if pauli != .i {
                paulis[qubit] = pauli
            }
        }
        return paulis
    }
}
