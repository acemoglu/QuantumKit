import Foundation

/// A sum of weighted Pauli terms, ⟨H⟩ = Σᵢ cᵢ ⟨Pᵢ⟩.
///
/// Also known as a sparse Pauli operator (`SparsePauliOp` in Qiskit terminology).
public struct Hamiltonian: Sendable, Equatable {
    public let terms: [PauliTerm]

    public init(terms: [PauliTerm]) {
        self.terms = terms
    }

    public init(_ terms: PauliTerm...) {
        self.terms = terms
    }

    public func adding(_ term: PauliTerm) -> Hamiltonian {
        Hamiltonian(terms: terms + [term])
    }

    public func adding(_ other: Hamiltonian) -> Hamiltonian {
        Hamiltonian(terms: terms + other.terms)
    }

    /// Exact ⟨ψ|H|ψ⟩ on a state vector.
    public func expectation(state: StateVector, engine: QuantumEngine) throws -> QFloat {
        var total: QFloat = 0
        for term in terms {
            let termValue = try term.expectation(state: state, engine: engine)
            total += term.coefficient * termValue
        }
        return total
    }

    /// Exact Tr(ρH) on a density matrix.
    public func expectation(density: DensityMatrix, engine: DensityMatrixEngine) throws -> QFloat {
        var total: QFloat = 0
        for term in terms {
            let termValue = try term.expectation(density: density, engine: engine)
            total += term.coefficient * termValue
        }
        return total
    }
}

/// Qiskit-compatible alias for ``Hamiltonian``.
public typealias SparsePauliOp = Hamiltonian
