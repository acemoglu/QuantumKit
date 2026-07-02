import Foundation

public enum OperatorError: Error, Equatable {
    case invalidPauliLabel(String)
    case duplicateQubitInTerm(qubit: Int, label: String)
}

/// A single Pauli-string term with a real coefficient, e.g. `0.5 * Z0 Z1`.
public struct PauliTerm: Sendable, Equatable {
    public let coefficient: QFloat
    public let paulis: [Int: Pauli]

    public init(coefficient: QFloat, paulis: [Int: Pauli]) {
        self.coefficient = coefficient
        self.paulis = paulis
    }

    /// Sparse Qiskit-style label such as `"Z0 Z1"` or `"X0"`.
    public init(coefficient: QFloat, label: String) throws {
        self.coefficient = coefficient
        self.paulis = try Self.parseSparseLabel(label)
    }

    /// Full MSB-first Pauli string whose length equals `qubitCount` (e.g. `"IX"` on two qubits).
    public init(coefficient: QFloat, pauliString: String, qubitCount: Int) throws {
        self.coefficient = coefficient
        self.paulis = try QuantumMeasurement.parsePauliString(pauliString, qubitCount: qubitCount)
    }

    /// Exact ⟨ψ|P|ψ⟩ for this term on a state vector.
    public func expectation(state: StateVector, engine: QuantumEngine) throws -> QFloat {
        try QuantumMeasurement.expectation(state: state, engine: engine, paulis: paulis)
    }

    /// Exact Tr(ρP) for this term on a density matrix.
    public func expectation(density: DensityMatrix, engine: DensityMatrixEngine) throws -> QFloat {
        try QuantumMeasurement.expectationPauli(density: density, paulis: paulis, engine: engine)
    }

    private static func parseSparseLabel(_ label: String) throws -> [Int: Pauli] {
        let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw OperatorError.invalidPauliLabel(label)
        }

        var paulis: [Int: Pauli] = [:]
        for token in trimmed.split(whereSeparator: \.isWhitespace) {
            guard token.count >= 2 else {
                throw OperatorError.invalidPauliLabel(label)
            }

            let letter = token[token.startIndex]
            let qubitString = String(token[token.index(after: token.startIndex)...])
            guard let qubit = Int(qubitString), qubit >= 0 else {
                throw OperatorError.invalidPauliLabel(label)
            }

            let pauli: Pauli
            switch letter {
            case "I", "i": pauli = .i
            case "X", "x": pauli = .x
            case "Y", "y": pauli = .y
            case "Z", "z": pauli = .z
            default:
                throw OperatorError.invalidPauliLabel(label)
            }

            if pauli == .i { continue }
            if paulis[qubit] != nil {
                throw OperatorError.duplicateQubitInTerm(qubit: qubit, label: label)
            }
            paulis[qubit] = pauli
        }

        return paulis
    }
}
