import Foundation

/// Product-formula order for ``TrotterEvolution``.
///
/// Approximates `exp(-i H t)` for `H = Σⱼ Hⱼ` (Pauli terms). This is **not** exact
/// evolution when terms do not all commute.
public enum TrotterOrder: Int, Sendable, Equatable, Codable, CaseIterable {
    /// Lie–Trotter: `U₁(Δt) = ∏ⱼ exp(-i Hⱼ Δt)`, `Δt = t/r`.
    ///
    /// Local error per step `O(Δt²)`; global error over time `t` scales as **`O(t² / r)`**
    /// (roughly `O((t²/r) Σ_{j<k} ‖[Hⱼ, Hₖ]‖)`).
    case first = 1

    /// Symmetric Suzuki / Strang splitting (order 2):
    /// `U₂(Δt) = ∏ⱼ exp(-i Hⱼ Δt/2) · ∏ⱼ^{rev} exp(-i Hⱼ Δt/2)`.
    ///
    /// Local error `O(Δt³)`; global error scales as **`O(t³ / r²)`**.
    case second = 2
}

public enum TrotterError: Error, Equatable, CustomStringConvertible {
    case invalidStepCount(Int)
    case invalidQubitCount(Int)
    case qubitIndexOutOfRange(qubit: Int, qubitCount: Int)

    public var description: String {
        switch self {
        case .invalidStepCount(let steps):
            return "Trotter step count must be >= 1 (got \(steps))."
        case .invalidQubitCount(let n):
            return "Trotter qubitCount must be >= 1 (got \(n))."
        case .qubitIndexOutOfRange(let qubit, let qubitCount):
            return "Pauli qubit \(qubit) is outside 0..<\(qubitCount)."
        }
    }
}

/// Host-side Trotterized Hamiltonian evolution.
///
/// Builds a ``QuantumCircuit`` approximating `exp(-i H t)` via a Pauli-term product
/// formula. Gate angles follow QuantumKit’s convention `R_P(θ) = exp(-i θ P / 2)`, so a
/// term `c·P` over duration `Δt` uses **`θ = 2 c Δt`** (same as QAOA).
///
/// **Error (do not treat as exact):**
/// - ``TrotterOrder/first``: global error **`O(t² / r)`**
/// - ``TrotterOrder/second``: global error **`O(t³ / r²)`**
/// Single-term (or fully commuting) Hamiltonians are exact for any `r ≥ 1`.
///
/// **Supported Paulis:** weight-1 via `RX`/`RY`/`RZ`; weight-2 `XX`/`YY`/`ZZ` via
/// `RXX`/`RYY`/`RZZ`; all other strings (mixed 2-body and **3+ body**) via a basis-change
/// + CNOT parity ladder + `RZ` (documented synthesis, not a dense matrix exponential).
/// Identity terms (empty Pauli map) contribute only a global phase and emit no gates.
public enum TrotterEvolution {

    /// Approximate `exp(-i H t)` as a circuit on `qubitCount` qubits.
    ///
    /// - Parameters:
    ///   - hamiltonian: Sparse Pauli sum `H = Σ cⱼ Pⱼ`.
    ///   - time: Evolution time `t` (may be zero → identity).
    ///   - steps: Trotter steps `r` (`r >= 1`).
    ///   - qubitCount: Register width; must cover every Pauli index in `H`.
    ///   - order: ``TrotterOrder/first`` (default) or ``TrotterOrder/second``.
    public static func circuit(
        hamiltonian: Hamiltonian,
        time: QFloat,
        steps: Int,
        qubitCount: Int,
        order: TrotterOrder = .first
    ) throws -> QuantumCircuit {
        try validate(steps: steps, qubitCount: qubitCount, hamiltonian: hamiltonian)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)
        try append(
            to: &circuit,
            hamiltonian: hamiltonian,
            time: time,
            steps: steps,
            order: order
        )
        return circuit
    }

    /// Same as ``circuit(hamiltonian:time:steps:qubitCount:order:)`` wrapped as a
    /// ``GateSequence`` named `"trotter"`.
    public static func gateSequence(
        hamiltonian: Hamiltonian,
        time: QFloat,
        steps: Int,
        qubitCount: Int,
        order: TrotterOrder = .first
    ) throws -> GateSequence {
        let circuit = try self.circuit(
            hamiltonian: hamiltonian,
            time: time,
            steps: steps,
            qubitCount: qubitCount,
            order: order
        )
        return GateSequence(name: "trotter", circuit: circuit)
    }

    /// Appends Trotter layers onto an existing circuit (width must match Pauli support).
    public static func append(
        to circuit: inout QuantumCircuit,
        hamiltonian: Hamiltonian,
        time: QFloat,
        steps: Int,
        order: TrotterOrder = .first
    ) throws {
        try validate(steps: steps, qubitCount: circuit.qubitCount, hamiltonian: hamiltonian)
        guard time != 0, hamiltonian.terms.contains(where: { $0.coefficient != 0 }) else {
            return
        }

        let dt = time / QFloat(steps)
        for _ in 0..<steps {
            switch order {
            case .first:
                try appendProduct(to: &circuit, hamiltonian: hamiltonian, duration: dt)
            case .second:
                let half = dt / 2
                try appendProduct(to: &circuit, hamiltonian: hamiltonian, duration: half)
                try appendProductReversed(to: &circuit, hamiltonian: hamiltonian, duration: half)
            }
        }
    }

    /// Build Trotter evolution from `|0…0⟩` (or `initial`) and estimate `⟨observable⟩`
    /// with the existing ``Estimator`` (SV/DM/CPU backends).
    public static func expectation(
        hamiltonian: Hamiltonian,
        time: QFloat,
        steps: Int,
        observable: Hamiltonian,
        backend: any QuantumBackend,
        initial: QuantumCircuit? = nil,
        order: TrotterOrder = .first,
        options: QuantumRunOptions = QuantumRunOptions(),
        estimatorOptions: EstimatorOptions = .exact
    ) throws -> EstimatorResult {
        let qubitCount: Int
        if let initial {
            qubitCount = initial.qubitCount
        } else {
            qubitCount = try inferredQubitCount(
                hamiltonian: hamiltonian,
                observable: observable
            )
        }

        var circuit: QuantumCircuit
        if let initial {
            circuit = initial
        } else {
            circuit = try QuantumCircuit(qubitCount: qubitCount)
        }
        try append(
            to: &circuit,
            hamiltonian: hamiltonian,
            time: time,
            steps: steps,
            order: order
        )
        return try Estimator().run(
            circuit: circuit,
            hamiltonian: observable,
            backend: backend,
            options: options,
            estimatorOptions: estimatorOptions
        )
    }

    // MARK: - Product layers

    private static func appendProduct(
        to circuit: inout QuantumCircuit,
        hamiltonian: Hamiltonian,
        duration: QFloat
    ) throws {
        for term in hamiltonian.terms {
            try appendPauliRotation(to: &circuit, term: term, duration: duration)
        }
    }

    private static func appendProductReversed(
        to circuit: inout QuantumCircuit,
        hamiltonian: Hamiltonian,
        duration: QFloat
    ) throws {
        for term in hamiltonian.terms.reversed() {
            try appendPauliRotation(to: &circuit, term: term, duration: duration)
        }
    }

    /// Emit gates for `exp(-i c·duration · P)` using `θ = 2 c duration`.
    private static func appendPauliRotation(
        to circuit: inout QuantumCircuit,
        term: PauliTerm,
        duration: QFloat
    ) throws {
        let phi = term.coefficient * duration
        guard phi != 0 else { return }

        let support = term.paulis
            .filter { $0.value != .i }
            .map { ($0.key, $0.value) }
            .sorted { $0.0 < $1.0 }

        if support.isEmpty {
            // Identity: global phase only.
            return
        }

        let theta = 2 * phi

        if support.count == 1 {
            let (q, p) = support[0]
            switch p {
            case .x: try circuit.rx(theta: theta, q)
            case .y: try circuit.ry(theta: theta, q)
            case .z: try circuit.rz(theta: theta, q)
            case .i: return
            }
            return
        }

        if support.count == 2 {
            let (q0, p0) = support[0]
            let (q1, p1) = support[1]
            switch (p0, p1) {
            case (.x, .x):
                try circuit.rxx(theta: theta, q0, q1)
                return
            case (.y, .y):
                try circuit.ryy(theta: theta, q0, q1)
                return
            case (.z, .z):
                try circuit.rzz(theta: theta, q0, q1)
                return
            default:
                break
            }
        }

        // Mixed 2-body and weight ≥ 3: basis change + CNOT ladder + RZ.
        try appendLadderPauliRotation(to: &circuit, support: support, theta: theta)
    }

    /// Synthesize `exp(-i θ P / 2)` for arbitrary non-identity support.
    ///
    /// Maps each local Pauli to `Z` (`H` for `X`, `S†H` for `Y`), propagates parity onto
    /// the highest qubit with CNOTs, applies `RZ(θ)`, then uncomputes. Used for 3+ body
    /// terms and mixed two-qubit Paulis (e.g. `XZ`).
    private static func appendLadderPauliRotation(
        to circuit: inout QuantumCircuit,
        support: [(Int, Pauli)],
        theta: QFloat
    ) throws {
        let qubits = support.map(\.0)

        for (qubit, pauli) in support {
            try applyBasisChange(to: &circuit, qubit: qubit, pauli: pauli)
        }
        for index in 0..<(qubits.count - 1) {
            try circuit.cx(qubits[index], qubits[index + 1])
        }
        try circuit.rz(theta: theta, qubits[qubits.count - 1])
        for index in stride(from: qubits.count - 2, through: 0, by: -1) {
            try circuit.cx(qubits[index], qubits[index + 1])
        }
        for (qubit, pauli) in support.reversed() {
            try applyBasisChangeInverse(to: &circuit, qubit: qubit, pauli: pauli)
        }
    }

    private static func applyBasisChange(
        to circuit: inout QuantumCircuit,
        qubit: Int,
        pauli: Pauli
    ) throws {
        switch pauli {
        case .x:
            try circuit.h(qubit)
        case .y:
            try circuit.sdg(qubit)
            try circuit.h(qubit)
        case .z, .i:
            break
        }
    }

    private static func applyBasisChangeInverse(
        to circuit: inout QuantumCircuit,
        qubit: Int,
        pauli: Pauli
    ) throws {
        switch pauli {
        case .x:
            try circuit.h(qubit)
        case .y:
            try circuit.h(qubit)
            try circuit.s(qubit)
        case .z, .i:
            break
        }
    }

    // MARK: - Validation

    private static func validate(
        steps: Int,
        qubitCount: Int,
        hamiltonian: Hamiltonian
    ) throws {
        guard steps >= 1 else { throw TrotterError.invalidStepCount(steps) }
        guard qubitCount >= 1 else { throw TrotterError.invalidQubitCount(qubitCount) }
        for term in hamiltonian.terms {
            for qubit in term.paulis.keys {
                guard qubit >= 0, qubit < qubitCount else {
                    throw TrotterError.qubitIndexOutOfRange(qubit: qubit, qubitCount: qubitCount)
                }
            }
        }
    }

    private static func inferredQubitCount(
        hamiltonian: Hamiltonian,
        observable: Hamiltonian
    ) throws -> Int {
        var maxIndex = -1
        for term in hamiltonian.terms + observable.terms {
            for qubit in term.paulis.keys {
                maxIndex = max(maxIndex, qubit)
            }
        }
        let count = max(maxIndex + 1, 1)
        guard count >= 1 else { throw TrotterError.invalidQubitCount(count) }
        return count
    }
}
