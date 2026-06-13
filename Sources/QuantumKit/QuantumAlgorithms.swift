//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation

extension QuantumCircuit {

    public mutating func applyCPHASE(theta: Double, control: Int, target: Int) throws {
        let halfTheta = QFloat(theta / 2.0)
        try cx(control, target)
        try rz(theta: -halfTheta, target)
        try cx(control, target)
        try rz(theta: halfTheta, control)
        try rz(theta: halfTheta, target)
    }

    public mutating func applyQFT() throws {
        let n = qubitCount
        for j in 0..<n {
            try h(j)
            for k in (j + 1)..<n {
                let theta = Double.pi / Double(1 << (k - j))
                try applyCPHASE(theta: theta, control: k, target: j)
            }
        }
    }

    public mutating func applySwap(q1: Int, q2: Int) throws {
        try cx(q1, q2)
        try cx(q2, q1)
        try cx(q1, q2)
    }

    public mutating func applyBellState(control: Int = 0, target: Int = 1) throws {
        try h(control)
        try cx(control, target)
    }

    // MARK: - Quantum Full Adder (Cuccaro gate sequence)

    /// Reversible single-bit full adder.
    /// Post-condition: b = a ⊕ b ⊕ cin (sum), cout = majority(a, b, cin) (carry out).
    /// Qubits `a` and `cin` are left unchanged. `cout` must be initialised to |0⟩.
    public mutating func applyQuantumFullAdder(cin: Int, a: Int, b: Int, cout: Int) throws {
        try validateRegisterIndex(cin)
        try validateRegisterIndex(a)
        try validateRegisterIndex(b)
        try validateRegisterIndex(cout)
        try ccx(a, b, cout)    // cout ^= a ∧ b
        try cx(a, b)           // b   ^= a
        try ccx(cin, b, cout)  // cout ^= cin ∧ (a ⊕ b_orig)
        try cx(cin, b)         // b    = a ⊕ b_orig ⊕ cin  (sum)
    }

    // MARK: - MAJ / UMA primitives (Cuccaro 2004)

    /// Majority gate on (c, b, a).
    /// Post: a = majority(c_in, b_in, a_in) = carry out, b = b_in ⊕ a_in, c = c_in ⊕ a_in.
    private mutating func applyMAJ(_ c: Int, _ b: Int, _ a: Int) throws {
        try cx(a, b)
        try cx(a, c)
        try ccx(c, b, a)
    }

    /// UnMajority-and-Add gate on (c, b, a) — inverse carry propagation of MAJ.
    /// Post: a = a_orig (restored), b = a_orig ⊕ b_orig ⊕ c_orig (sum), c = c_orig (restored).
    private mutating func applyUMA(_ c: Int, _ b: Int, _ a: Int) throws {
        try ccx(c, b, a)
        try cx(a, c)
        try cx(c, b)
    }

    // MARK: - n-qubit Ripple-Carry Adder

    /// In-place n-bit adder using Cuccaro MAJ-chain / CX / UMA-chain topology.
    /// Computes |a⟩|b⟩ → |a⟩|a + b⟩, overflow bit captured in `carryOut`.
    /// `registerA` and `registerB` must be non-empty and of equal length.
    /// `carryIn` must be |0⟩; `carryOut` receives the final overflow bit.
    public mutating func applyQuantumAdd(
        registerA: [Int],
        registerB: [Int],
        carryIn: Int,
        carryOut: Int
    ) throws {
        guard registerA.count == registerB.count, !registerA.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "registerA and registerB must be non-empty and of equal length"
            )
        }
        for idx in registerA { try validateRegisterIndex(idx) }
        for idx in registerB { try validateRegisterIndex(idx) }
        try validateRegisterIndex(carryIn)
        try validateRegisterIndex(carryOut)

        let n = registerA.count

        // Forward MAJ chain: carry[i+1] propagates into registerA[i]
        try applyMAJ(carryIn, registerB[0], registerA[0])
        for i in 1..<n {
            try applyMAJ(registerA[i - 1], registerB[i], registerA[i])
        }

        // Copy final carry to carryOut
        try cx(registerA[n - 1], carryOut)

        // Backward UMA chain: restore registerA, write sum into registerB
        for i in stride(from: n - 1, through: 1, by: -1) {
            try applyUMA(registerA[i - 1], registerB[i], registerA[i])
        }
        try applyUMA(carryIn, registerB[0], registerA[0])
    }

    // MARK: - Controlled Ripple-Carry Adder helpers

    /// Controlled MAJ: CX(x,y) → CCX(ctrl,x,y); CCX(c,b,a) → C3X(ctrl,c,b,a).
    /// C3X is decomposed as CCX(ctrl,c,anc)·CCX(b,anc,a)·CCX(ctrl,c,anc)
    /// using `ancilla` as a clean |0⟩ scratch qubit (restored on exit).
    private mutating func applyControlledMAJ(
        ctrl: Int, c: Int, b: Int, a: Int, ancilla: Int
    ) throws {
        try ccx(ctrl, a, b)          // ctrl-CX(a, b)
        try ccx(ctrl, a, c)          // ctrl-CX(a, c)
        // C3X(ctrl, c_new, b_new, a): a ^= ctrl ∧ c ∧ b
        try ccx(ctrl, c, ancilla)    // ancilla  = ctrl ∧ c
        try ccx(b, ancilla, a)       // a       ^= b ∧ ancilla
        try ccx(ctrl, c, ancilla)    // restore ancilla = 0
    }

    /// Controlled UMA: symmetric controlled decomposition of the UMA gate.
    private mutating func applyControlledUMA(
        ctrl: Int, c: Int, b: Int, a: Int, ancilla: Int
    ) throws {
        // C3X(ctrl, c, b, a): a ^= ctrl ∧ c ∧ b
        try ccx(ctrl, c, ancilla)
        try ccx(b, ancilla, a)
        try ccx(ctrl, c, ancilla)    // restore ancilla = 0
        try ccx(ctrl, a, c)          // ctrl-CX(a, c)
        try ccx(ctrl, c, b)          // ctrl-CX(c, b)
    }

    // MARK: - n-qubit Controlled Ripple-Carry Adder

    /// Applies n-bit addition conditioned on `control` being |1⟩.
    /// Semantics: |ctrl⟩|a⟩|b⟩ → |ctrl⟩|a⟩|b + ctrl·a⟩.
    /// `ancilla` must be a distinct clean |0⟩ qubit used to decompose C3X gates;
    /// it is guaranteed to return to |0⟩ after the operation.
    public mutating func applyControlledQuantumAdd(
        control: Int,
        registerA: [Int],
        registerB: [Int],
        carryIn: Int,
        carryOut: Int,
        ancilla: Int
    ) throws {
        guard registerA.count == registerB.count, !registerA.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "registerA and registerB must be non-empty and of equal length"
            )
        }
        try validateRegisterIndex(control)
        for idx in registerA { try validateRegisterIndex(idx) }
        for idx in registerB { try validateRegisterIndex(idx) }
        try validateRegisterIndex(carryIn)
        try validateRegisterIndex(carryOut)
        try validateRegisterIndex(ancilla)

        let n = registerA.count

        // Controlled MAJ forward chain
        try applyControlledMAJ(ctrl: control, c: carryIn,
                               b: registerB[0], a: registerA[0], ancilla: ancilla)
        for i in 1..<n {
            try applyControlledMAJ(ctrl: control, c: registerA[i - 1],
                                   b: registerB[i], a: registerA[i], ancilla: ancilla)
        }

        // Controlled carry propagation
        try ccx(control, registerA[n - 1], carryOut)

        // Controlled UMA backward chain
        for i in stride(from: n - 1, through: 1, by: -1) {
            try applyControlledUMA(ctrl: control, c: registerA[i - 1],
                                   b: registerB[i], a: registerA[i], ancilla: ancilla)
        }
        try applyControlledUMA(ctrl: control, c: carryIn,
                               b: registerB[0], a: registerA[0], ancilla: ancilla)
    }

    // MARK: - Modular Exponentiation (Shor's Algorithm scaffold)

    public mutating func applyModularExponentiation(
        a: Int,
        modulus N: Int,
        controlRegister: ClosedRange<Int>,
        targetRegister: ClosedRange<Int>
    ) throws {
        guard N > 1 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Modulus must be greater than 1")
        }

        guard a >= 0 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Base must be non-negative")
        }

        guard !controlRegister.isEmpty, !targetRegister.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Control and target registers must not be empty")
        }

        for index in controlRegister {
            try validateRegisterIndex(index)
        }

        for index in targetRegister {
            try validateRegisterIndex(index)
        }

        if controlRegister.overlaps(targetRegister) {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Control and target registers must not overlap")
        }

        let runningBase = a % N
        var currentMultiplier = runningBase

        for controlQubit in controlRegister {
            // TODO: Implement controlled modular multiplication:
            //       |c>|y> -> |c>|(y * currentMultiplier) mod N> when c = |1>
            
            try applyControlledModularMultiply(
                multiplier: currentMultiplier,
                modulus: N,
                control: controlQubit,
                targetRegister: targetRegister
            )

            currentMultiplier = (currentMultiplier * currentMultiplier) % N
        }
    }

    private mutating func applyControlledModularMultiply(
        multiplier: Int,
        modulus: Int,
        control: Int,
        targetRegister: ClosedRange<Int>
    ) throws {
        // TODO: Decompose into quantum adders/multipliers using ccx and swap primitives.
        _ = multiplier
        _ = modulus
        _ = control
        _ = targetRegister
    }

    private func validateRegisterIndex(_ index: Int) throws {
        guard index >= 0, index < qubitCount else {
            throw QuantumCircuitError.qubitIndexOutOfBounds(index: index, qubitCount: qubitCount)
        }
    }
}
