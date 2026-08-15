import Foundation

public enum VariationalAlgorithmError: Error, Equatable, Sendable {
    case invalidLayerCount(Int)
    case emptyProblem
    case qubitOutOfRange(qubit: Int, qubitCount: Int)
    case duplicateEdge(Int, Int)
    case missingParameterBinding(String)
    case parameterCountMismatch(expected: Int, actual: Int)
}

/// Undirected Ising / MaxCut-style problem: `H = Σᵢ hᵢ Zᵢ + Σᵢⱼ Jᵢⱼ Zᵢ Zⱼ`.
///
/// Used by ``QAOA`` to build cost layers and by callers as a small Hamiltonian factory.
public struct IsingGraph: Sendable, Equatable {
    public struct Edge: Sendable, Equatable {
        public let qubitA: Int
        public let qubitB: Int
        public let weight: QFloat

        public init(qubitA: Int, qubitB: Int, weight: QFloat = 1) {
            self.qubitA = min(qubitA, qubitB)
            self.qubitB = max(qubitA, qubitB)
            self.weight = weight
        }
    }

    public let qubitCount: Int
    public let edges: [Edge]
    /// Linear Z fields `hᵢ` (missing keys → 0).
    public let fields: [Int: QFloat]

    public init(qubitCount: Int, edges: [Edge], fields: [Int: QFloat] = [:]) throws {
        guard qubitCount > 0 else {
            throw VariationalAlgorithmError.emptyProblem
        }
        for edge in edges {
            guard edge.qubitA >= 0, edge.qubitB < qubitCount, edge.qubitA != edge.qubitB else {
                throw VariationalAlgorithmError.qubitOutOfRange(
                    qubit: max(edge.qubitA, edge.qubitB),
                    qubitCount: qubitCount
                )
            }
        }
        var seen: Set<String> = []
        for edge in edges {
            let key = "\(edge.qubitA)-\(edge.qubitB)"
            if !seen.insert(key).inserted {
                throw VariationalAlgorithmError.duplicateEdge(edge.qubitA, edge.qubitB)
            }
        }
        for (qubit, _) in fields {
            guard qubit >= 0, qubit < qubitCount else {
                throw VariationalAlgorithmError.qubitOutOfRange(qubit: qubit, qubitCount: qubitCount)
            }
        }
        self.qubitCount = qubitCount
        self.edges = edges
        self.fields = fields
    }

    /// Unweighted MaxCut on an undirected edge list (`Jᵢⱼ = 1` unless `weight` given).
    public static func maxCut(
        qubitCount: Int,
        edges: [(Int, Int, QFloat)]
    ) throws -> IsingGraph {
        try IsingGraph(
            qubitCount: qubitCount,
            edges: edges.map { Edge(qubitA: $0.0, qubitB: $0.1, weight: $0.2) }
        )
    }

    public static func maxCut(
        qubitCount: Int,
        edges: [(Int, Int)]
    ) throws -> IsingGraph {
        try maxCut(qubitCount: qubitCount, edges: edges.map { ($0.0, $0.1, 1) })
    }

    /// Sparse Pauli form of the classical Ising cost Hamiltonian.
    public func costHamiltonian() throws -> Hamiltonian {
        var terms: [PauliTerm] = []
        for (qubit, weight) in fields where weight != 0 {
            terms.append(try PauliTerm(coefficient: weight, label: "Z\(qubit)"))
        }
        for edge in edges where edge.weight != 0 {
            terms.append(
                try PauliTerm(coefficient: edge.weight, label: "Z\(edge.qubitA) Z\(edge.qubitB)")
            )
        }
        return Hamiltonian(terms: terms)
    }
}

/// Built QAOA ansatz: `H⊗n` then `p` layers of cost(`γₖ`) + mixer(`βₖ`).
public struct QAOACircuit: Sendable, Equatable {
    public let circuit: QuantumCircuit
    /// Number of QAOA layers `p`.
    public let layers: Int
    /// `gamma0 … gamma{p-1}` (cost angles).
    public let gammaNames: [String]
    /// `beta0 … beta{p-1}` (mixer angles).
    public let betaNames: [String]

    public var parameterCount: Int { gammaNames.count + betaNames.count }

    /// Name-sorted union of gamma/beta names (matches ``QuantumCircuit/referencedParameters`` sort used by gradients).
    public var parameterNames: [String] {
        (gammaNames + betaNames).sorted()
    }
}

/// Thin QAOA circuit builder (no optimizer loop).
///
/// Cost/mixer use gate angles `2γJ`, `2γh`, `2β` (QuantumKit `R(θ)=exp(-iθP/2)`).
/// Parameter-shift / Hessian require a **homogeneous** linear scale per named parameter;
/// equal-weight MaxCut (`J=1` on every edge, uniform fields) is supported. Heterogeneous
/// `Jᵢⱼ` / `hᵢ` on the same `γ` need adjoint (RZZ unsupported) or custom differentiation.
public enum QAOA {

    /// Build a `p`-layer QAOA ansatz for an Ising / MaxCut cost.
    ///
    /// Cost layer `k`: `RZZ(2 γₖ Jᵢⱼ)` on each edge and `RZ(2 γₖ hᵢ)` on fields.
    /// Mixer layer `k`: `RX(2 βₖ)` on every qubit.
    /// Parameter names: `"\(gammaPrefix)\(k)"`, `"\(betaPrefix)\(k)"` for `k = 0..<p`.
    public static func build(
        problem: IsingGraph,
        layers: Int,
        gammaPrefix: String = "gamma",
        betaPrefix: String = "beta"
    ) throws -> QAOACircuit {
        guard layers >= 1 else {
            throw VariationalAlgorithmError.invalidLayerCount(layers)
        }
        guard !problem.edges.isEmpty || !problem.fields.isEmpty else {
            throw VariationalAlgorithmError.emptyProblem
        }

        var gammaNames: [String] = []
        var betaNames: [String] = []
        gammaNames.reserveCapacity(layers)
        betaNames.reserveCapacity(layers)

        var circuit = try QuantumCircuit(qubitCount: problem.qubitCount)
        for q in 0..<problem.qubitCount {
            try circuit.h(q)
        }

        for layer in 0..<layers {
            let gammaName = "\(gammaPrefix)\(layer)"
            let betaName = "\(betaPrefix)\(layer)"
            gammaNames.append(gammaName)
            betaNames.append(betaName)

            let gamma = Parameter(gammaName)
            let beta = Parameter(betaName)

            for (qubit, weight) in problem.fields where weight != 0 {
                // RZ(θ) = exp(-i θ Z / 2) ⇒ θ = 2 γ h
                try circuit.rz(theta: gamma.scaled(by: 2 * weight), qubit)
            }
            for edge in problem.edges where edge.weight != 0 {
                try circuit.rzz(theta: gamma.scaled(by: 2 * edge.weight), edge.qubitA, edge.qubitB)
            }
            for q in 0..<problem.qubitCount {
                try circuit.rx(theta: beta.scaled(by: 2), q)
            }
        }

        return QAOACircuit(
            circuit: circuit,
            layers: layers,
            gammaNames: gammaNames,
            betaNames: betaNames
        )
    }

    /// Bind QAOA angles from parallel `gammas` / `betas` arrays (`count == p`).
    public static func bindings(
        for qaoa: QAOACircuit,
        gammas: [QFloat],
        betas: [QFloat]
    ) throws -> [String: QFloat] {
        guard gammas.count == qaoa.layers, betas.count == qaoa.layers else {
            throw VariationalAlgorithmError.parameterCountMismatch(
                expected: qaoa.layers,
                actual: min(gammas.count, betas.count)
            )
        }
        var map: [String: QFloat] = [:]
        for i in 0..<qaoa.layers {
            map[qaoa.gammaNames[i]] = gammas[i]
            map[qaoa.betaNames[i]] = betas[i]
        }
        return map
    }
}
