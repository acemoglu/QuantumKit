import Foundation

/// Partitions Pauli terms into **qubit-wise commuting (QWC)** measurement groups for
/// shot-based ``Estimator`` runs.
///
/// ## Grouping rule (QWC)
/// Two Pauli strings qubit-wise commute when, on every qubit, they agree or at least one
/// factor is `I`. Equivalent: their non-identity supports never place different Paulis
/// (`X`/`Y`/`Z`) on the same qubit.
///
/// QWC is stricter than full tensor-product commutation, but every QWC group admits a
/// shared product of *single-qubit* Clifford basis changes (`H` for `X`, `S†H` for `Y`,
/// identity for `Z`). One evolved measure circuit + one shot ensemble then yields
/// every term expectation in the group via per-term parity on that term's support.
///
/// Partitioning is greedy first-fit in input order: each non-identity term joins the
/// first existing group with which it QWC-commutes, otherwise it opens a new group.
///
/// ## When grouping / sampling is skipped
/// - **Empty term list**: no groups, identity contribution `0`.
/// - **Identity terms** (empty `paulis`, or only `I`): not placed in a sampling group;
///   their coefficients sum into ``Partition/identityContribution`` and contribute
///   exactly without a basis-change circuit.
/// - **Non-QWC leftovers**: a term that conflicts with every open group becomes its
///   own group (never merged incorrectly with `X`/`Z` on the same qubit, etc.).
///
/// Exact ``Estimator`` paths (`Tr(ρH)` / ⟨ψ|H|ψ⟩) do not use this partitioner.
/// Shot ``Estimator`` partitions by default (``EstimatorOptions/groupCommutingPaulis``);
/// set that flag to `false` for legacy per-term ensembles.
public enum PauliCommutingGroups {

    /// One QWC set sharing a single measurement basis (and thus one shot ensemble).
    public struct Group: Sendable, Equatable {
        public let terms: [PauliTerm]
        /// Union of non-identity Paulis per qubit; consistent by QWC construction.
        public let measurementAxes: [Int: Pauli]

        public init(terms: [PauliTerm], measurementAxes: [Int: Pauli]) {
            self.terms = terms
            self.measurementAxes = measurementAxes
        }
    }

    /// Result of partitioning a Hamiltonian or term list.
    public struct Partition: Sendable, Equatable {
        public let groups: [Group]
        /// Exact Σ cᵢ for identity / empty-support terms (no sampling).
        public let identityContribution: QFloat

        public init(groups: [Group], identityContribution: QFloat) {
            self.groups = groups
            self.identityContribution = identityContribution
        }
    }

    /// Qubit-wise commute check used by the greedy partitioner.
    public static func qubitWiseCommute(_ lhs: PauliTerm, _ rhs: PauliTerm) -> Bool {
        let qubits = Set(lhs.paulis.keys).union(rhs.paulis.keys)
        for qubit in qubits {
            let a = effectivePauli(lhs.paulis[qubit])
            let b = effectivePauli(rhs.paulis[qubit])
            if a == .i || b == .i { continue }
            if a != b { return false }
        }
        return true
    }

    public static func partition(_ hamiltonian: Hamiltonian) -> Partition {
        partition(hamiltonian.terms)
    }

    /// One sampling group per non-identity term (legacy Estimator shot schedule).
    public static func partitionSingletons(_ hamiltonian: Hamiltonian) -> Partition {
        partitionSingletons(hamiltonian.terms)
    }

    /// One sampling group per non-identity term (legacy Estimator shot schedule).
    public static func partitionSingletons(_ terms: [PauliTerm]) -> Partition {
        guard !terms.isEmpty else {
            return Partition(groups: [], identityContribution: 0)
        }

        var identityContribution: QFloat = 0
        var groups: [Group] = []
        for term in terms {
            let support = nonIdentitySupport(term)
            if support.isEmpty {
                identityContribution += term.coefficient
                continue
            }
            groups.append(Group(terms: [term], measurementAxes: support))
        }
        return Partition(groups: groups, identityContribution: identityContribution)
    }

    public static func partition(_ terms: [PauliTerm]) -> Partition {
        guard !terms.isEmpty else {
            return Partition(groups: [], identityContribution: 0)
        }

        var identityContribution: QFloat = 0
        var openGroups: [(terms: [PauliTerm], axes: [Int: Pauli])] = []

        for term in terms {
            let support = nonIdentitySupport(term)
            if support.isEmpty {
                identityContribution += term.coefficient
                continue
            }

            var placed = false
            for index in openGroups.indices {
                let candidate = PauliTerm(coefficient: 1, paulis: openGroups[index].axes)
                // QWC against the group's axis map (equivalent to all members under QWC).
                if qubitWiseCommute(term, candidate) {
                    openGroups[index].terms.append(term)
                    for (qubit, pauli) in support {
                        openGroups[index].axes[qubit] = pauli
                    }
                    placed = true
                    break
                }
            }

            if !placed {
                openGroups.append((terms: [term], axes: support))
            }
        }

        let groups = openGroups.map { Group(terms: $0.terms, measurementAxes: $0.axes) }
        return Partition(groups: groups, identityContribution: identityContribution)
    }

    /// Non-identity Paulis on this term (keys sorted only by callers that need order).
    static func nonIdentitySupport(_ term: PauliTerm) -> [Int: Pauli] {
        var support: [Int: Pauli] = [:]
        for (qubit, pauli) in term.paulis where pauli != .i {
            support[qubit] = pauli
        }
        return support
    }

    private static func effectivePauli(_ pauli: Pauli?) -> Pauli {
        guard let pauli, pauli != .i else { return .i }
        return pauli
    }
}
