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

        let distribution = try probabilities(state: state, engine: engine)
        var expectation = 0.0

        for (stateIndex, probability) in distribution.enumerated() {
            var eigenvalue: QFloat = 1
            for qubit in qubits {
                eigenvalue *= pauliZEigenvalue(stateIndex: stateIndex, qubit: qubit)
            }
            expectation += Double(probability) * Double(eigenvalue)
        }

        return QFloat(expectation)
    }

    /// ⟨X⟩ for a single qubit in the computational basis.
    public static func expectationX(
        state: StateVector,
        engine: QuantumEngine,
        qubit: Int
    ) throws -> QFloat {
        try validateQubits([qubit], qubitCount: state.qubitCount)

        let mask = 1 << qubit
        let realPointer = state.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = state.imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        var expectation = 0.0
        for index in 0..<state.stateCount {
            let flipped = index ^ mask
            let realProduct = realPointer[index] * realPointer[flipped]
            let imagProduct = imagPointer[index] * imagPointer[flipped]
            expectation += Double(realProduct + imagProduct)
        }

        return QFloat(expectation)
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

        let realPointer = state.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = state.imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        let signMask = yMask | zMask
        var expectation = 0.0

        for j in 0..<state.stateCount {
            let k = j ^ flipMask
            let negative = ((j & signMask).nonzeroBitCount & 1) == 1
            let pr = negative ? -phaseBaseReal : phaseBaseReal
            let pi = negative ? -phaseBaseImag : phaseBaseImag

            let rj = realPointer[j]
            let ij = imagPointer[j]
            // phase(j) · a_j
            let xr = pr * rj - pi * ij
            let xi = pr * ij + pi * rj
            // Re[ conj(a_k) · phase(j) · a_j ], accumulated in Double to avoid Float32 cancellation.
            expectation += Double(realPointer[k] * xr + imagPointer[k] * xi)
        }

        return QFloat(expectation)
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
