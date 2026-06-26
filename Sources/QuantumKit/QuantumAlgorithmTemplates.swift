//
//  QuantumAlgorithmTemplates.swift
//  QuantumKit
//
//  Ready-made circuit templates for textbook quantum algorithms
//  (Deutsch-Jozsa, Grover, Quantum Phase Estimation).
//

import Foundation

extension QuantumCircuit {

    // MARK: - Deutsch-Jozsa

    /// Builds the Deutsch-Jozsa circuit for an oracle `Uf` acting on `inputQubits` (and `ancilla`).
    ///
    /// The ancilla is prepared in |−⟩ (X then H) so a phase-kickback oracle marks balanced inputs.
    /// After the closing Hadamards, measuring `inputQubits` yields all-zeros iff `f` is constant;
    /// any non-zero outcome means `f` is balanced. Measurement is left to the caller.
    ///
    /// The oracle is supplied as a closure that appends its gates to the circuit, e.g. a
    /// phase-kickback `Uf` that toggles `ancilla` based on `inputQubits`.
    public mutating func applyDeutschJozsa(
        inputQubits: [Int],
        ancilla: Int,
        oracle: (inout QuantumCircuit) throws -> Void
    ) throws {
        guard !inputQubits.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Deutsch-Jozsa requires at least one input qubit"
            )
        }
        guard Set(inputQubits).count == inputQubits.count else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Deutsch-Jozsa input qubits must be distinct"
            )
        }
        guard !inputQubits.contains(ancilla) else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Deutsch-Jozsa ancilla must differ from every input qubit"
            )
        }

        // Ancilla → |−⟩ so the oracle's phase kickback lands on the input register.
        try x(ancilla)
        try h(ancilla)

        for qubit in inputQubits {
            try h(qubit)
        }

        try oracle(&self)

        for qubit in inputQubits {
            try h(qubit)
        }
    }

    // MARK: - Grover

    /// Grover diffusion operator (inversion about the mean) over `qubits`:
    /// `H^⊗n · X^⊗n · (multi-controlled Z) · X^⊗n · H^⊗n`.
    public mutating func applyGroverDiffusion(qubits: [Int]) throws {
        guard !qubits.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Grover diffusion requires at least one qubit"
            )
        }
        guard Set(qubits).count == qubits.count else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Grover diffusion qubits must be distinct"
            )
        }

        for qubit in qubits {
            try h(qubit)
        }
        for qubit in qubits {
            try x(qubit)
        }

        if qubits.count == 1 {
            // A multi-controlled Z with no controls is just a phase flip on |1⟩, i.e. Z.
            try z(qubits[0])
        } else {
            try mcz(controls: Array(qubits.dropLast()), target: qubits[qubits.count - 1])
        }

        for qubit in qubits {
            try x(qubit)
        }
        for qubit in qubits {
            try h(qubit)
        }
    }

    /// Full Grover search template: prepares a uniform superposition over `qubits`, then alternates
    /// the supplied `oracle` (which must phase-flip the marked basis state(s)) with the diffusion
    /// operator for `iterations` rounds. Measurement is left to the caller.
    public mutating func applyGrover(
        qubits: [Int],
        iterations: Int,
        oracle: (inout QuantumCircuit) throws -> Void
    ) throws {
        guard !qubits.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Grover requires at least one qubit"
            )
        }
        guard Set(qubits).count == qubits.count else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Grover qubits must be distinct"
            )
        }
        guard iterations >= 0 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Grover iteration count must be non-negative"
            )
        }

        for qubit in qubits {
            try h(qubit)
        }

        for _ in 0..<iterations {
            try oracle(&self)
            try applyGroverDiffusion(qubits: qubits)
        }
    }

    /// Optimal number of Grover iterations ≈ ⌊(π/4)·√(N/M)⌋ for a search space of size `N`
    /// with `M` marked items.
    public static func groverOptimalIterations(searchSpaceSize: Int, markedCount: Int = 1) -> Int {
        guard searchSpaceSize > 0, markedCount > 0, markedCount <= searchSpaceSize else {
            return 0
        }
        let ratio = Double(searchSpaceSize) / Double(markedCount)
        return Int((Double.pi / 4.0) * ratio.squareRoot())
    }

    // MARK: - Quantum Phase Estimation

    /// Quantum Phase Estimation onto `countingRegister` (qubit `countingRegister[0]` is the least
    /// significant output bit). The eigenstate register is assumed to be prepared by the caller.
    ///
    /// `controlledUnitaryPower(&circuit, control, power)` must append a controlled-U^`power`
    /// conditioned on `control`. For counting qubit `j`, `power` is `2^j`. After the controlled
    /// applications an inverse QFT is run over the counting register; measuring it yields the
    /// `t`-bit estimate `m` with phase φ ≈ m / 2^t. Measurement is left to the caller.
    public mutating func applyPhaseEstimation(
        countingRegister: [Int],
        controlledUnitaryPower: (inout QuantumCircuit, _ control: Int, _ power: Int) throws -> Void
    ) throws {
        guard !countingRegister.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Phase estimation requires at least one counting qubit"
            )
        }
        guard Set(countingRegister).count == countingRegister.count else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Phase estimation counting qubits must be distinct"
            )
        }

        for qubit in countingRegister {
            try h(qubit)
        }

        for (index, control) in countingRegister.enumerated() {
            try controlledUnitaryPower(&self, control, 1 << index)
        }

        try applyInverseQFT(qubits: countingRegister)

        // The library's inverse QFT omits the final bit-reversal swaps, so its output lands on the
        // register in reverse. Append the swaps so counting qubit `j` carries output bit `j`, i.e.
        // the estimate `m` reads naturally with `countingRegister[0]` as the least significant bit.
        let count = countingRegister.count
        for i in 0..<(count / 2) {
            try swap(countingRegister[i], countingRegister[count - 1 - i])
        }
    }
}
