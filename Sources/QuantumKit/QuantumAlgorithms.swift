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

    /// Inverse QFT on all qubits (adjoint of ``applyQFT()``).
    public mutating func applyInverseQFT() throws {
        try applyInverseQFT(qubits: 0..<qubitCount)
    }

    /// Inverse QFT on a contiguous subset of qubits (e.g. Shor counting register).
    public mutating func applyInverseQFT(qubits: Range<Int>) throws {
        try applyInverseQFT(qubits: Array(qubits))
    }

    /// Inverse QFT over an explicit, ordered list of qubits (qubit `indices[0]` is the least
    /// significant). Useful when the target register is non-contiguous, e.g. a QPE counting register.
    public mutating func applyInverseQFT(qubits: [Int]) throws {
        guard !qubits.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "Inverse QFT requires at least one qubit"
            )
        }

        let indices = qubits
        for index in indices {
            try validateRegisterIndex(index)
        }

        let n = indices.count
        for offset in stride(from: n - 1, through: 0, by: -1) {
            let j = indices[offset]
            for kOffset in stride(from: n - 1, through: offset + 1, by: -1) {
                let k = indices[kOffset]
                let theta = -Double.pi / Double(1 << (kOffset - offset))
                try applyCPHASE(theta: theta, control: k, target: j)
            }
            try h(j)
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

    /// Computes |a⟩|b⟩ → |a⟩|b − a⟩ via two's-complement addition (b + ~a + 1).
    /// `registerA` and `registerB` must be non-empty and of equal length.
    /// `carryIn` must be |0⟩; `carryOut` receives the final borrow bit.
    public mutating func applyQuantumSubtract(
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

        for i in 0..<n {
            try x(registerA[i])
        }
        try x(carryIn)

        try applyQuantumAdd(
            registerA: registerA,
            registerB: registerB,
            carryIn: carryIn,
            carryOut: carryOut
        )

        try x(carryIn)
        for i in 0..<n {
            try x(registerA[i])
        }
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

    // MARK: - Wrapping (mod 2^n) Ripple-Carry Adders

    /// In-place ripple-carry adder that discards the final carry: |a⟩|b⟩ → |a⟩|(a + b) mod 2^n⟩.
    /// Identical to ``applyQuantumAdd(registerA:registerB:carryIn:carryOut:)`` but without an
    /// overflow qubit, so two's-complement arithmetic wraps cleanly inside the register.
    /// `carryIn` must be |0⟩ and is restored.
    private mutating func applyQuantumAddWrapping(
        registerA: [Int], registerB: [Int], carryIn: Int
    ) throws {
        let n = registerA.count
        try applyMAJ(carryIn, registerB[0], registerA[0])
        for i in 1..<n {
            try applyMAJ(registerA[i - 1], registerB[i], registerA[i])
        }
        for i in stride(from: n - 1, through: 1, by: -1) {
            try applyUMA(registerA[i - 1], registerB[i], registerA[i])
        }
        try applyUMA(carryIn, registerB[0], registerA[0])
    }

    /// Wrapping two's-complement subtraction: |a⟩|b⟩ → |a⟩|(b − a) mod 2^n⟩ (carry discarded).
    private mutating func applyQuantumSubtractWrapping(
        registerA: [Int], registerB: [Int], carryIn: Int
    ) throws {
        let n = registerA.count
        for i in 0..<n { try x(registerA[i]) }
        try x(carryIn)
        try applyQuantumAddWrapping(registerA: registerA, registerB: registerB, carryIn: carryIn)
        try x(carryIn)
        for i in 0..<n { try x(registerA[i]) }
    }

    /// Controlled wrapping adder: |ctrl⟩|a⟩|b⟩ → |ctrl⟩|a⟩|(b + ctrl·a) mod 2^n⟩ (carry discarded).
    /// `ancilla` is a clean |0⟩ scratch qubit for the C3X decomposition, restored on exit.
    private mutating func applyControlledQuantumAddWrapping(
        control: Int, registerA: [Int], registerB: [Int], carryIn: Int, ancilla: Int
    ) throws {
        let n = registerA.count
        try applyControlledMAJ(ctrl: control, c: carryIn,
                               b: registerB[0], a: registerA[0], ancilla: ancilla)
        for i in 1..<n {
            try applyControlledMAJ(ctrl: control, c: registerA[i - 1],
                                   b: registerB[i], a: registerA[i], ancilla: ancilla)
        }
        for i in stride(from: n - 1, through: 1, by: -1) {
            try applyControlledUMA(ctrl: control, c: registerA[i - 1],
                                   b: registerB[i], a: registerA[i], ancilla: ancilla)
        }
        try applyControlledUMA(ctrl: control, c: carryIn,
                               b: registerB[0], a: registerA[0], ancilla: ancilla)
    }

    /// Controlled wrapping subtraction: |ctrl⟩|a⟩|b⟩ → |ctrl⟩|a⟩|(b − ctrl·a) mod 2^n⟩.
    private mutating func applyControlledQuantumSubtractWrapping(
        control: Int, registerA: [Int], registerB: [Int], carryIn: Int, ancilla: Int
    ) throws {
        let n = registerA.count
        for i in 0..<n { try cx(control, registerA[i]) }
        try cx(control, carryIn)
        try applyControlledQuantumAddWrapping(
            control: control, registerA: registerA, registerB: registerB,
            carryIn: carryIn, ancilla: ancilla
        )
        try cx(control, carryIn)
        for i in 0..<n { try cx(control, registerA[i]) }
    }

    // MARK: - Modular Arithmetic (VBE)

    /// Vedral-Barenco-Ekert modular addition: |x⟩ → |(x + a) mod N⟩, valid for 0 ≤ x, a < N ≤ 2^n.
    ///
    /// `ancillaRegister` layout (n + 5 qubits, all restored to |0⟩ on exit):
    /// `[constReg (n) | constHigh | workHigh | flag | carryIn | c3xAncilla]`.
    /// `constReg`+`constHigh` form the (n+1)-bit constant register; `workHigh` is the (n+1)-th
    /// (sign/overflow) bit of the work value `registerX`; `flag` records the comparison `x+a < N`.
    /// Classical `a`/`N` are synthesised into `constReg` via X encoding (assumes |0⟩ initial state).
    public mutating func applyModularAdd(
        a: Int,
        modulus N: Int,
        registerX: [Int],
        ancillaRegister: [Int]
    ) throws {
        guard N > 0 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Modulus must be positive")
        }
        guard a >= 0 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Addend must be non-negative")
        }
        guard !registerX.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "registerX must not be empty")
        }
        let n = registerX.count
        guard ancillaRegister.count >= n + 5 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "ancillaRegister requires at least registerX.count + 5 qubits (constReg, constHigh, workHigh, flag, carryIn, c3xAncilla)"
            )
        }
        guard n < Int.bitWidth - 1, N <= (1 << n) - 1 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "modulus N must satisfy N < 2^registerX.count so it fits in the register"
            )
        }

        for idx in registerX { try validateRegisterIndex(idx) }
        for idx in ancillaRegister { try validateRegisterIndex(idx) }

        let constReg = Array(ancillaRegister[0..<n])
        let constHigh = ancillaRegister[n]
        let workHigh = ancillaRegister[n + 1]
        let flag = ancillaRegister[n + 2]
        let carryIn = ancillaRegister[n + 3]
        let c3xAncilla = ancillaRegister[n + 4]

        // (n+1)-bit two's-complement registers: workReg holds x (sign/overflow in workHigh),
        // constFull holds the classical operand (high bit constHigh stays |0⟩ since a, N < 2^n).
        let workReg = registerX + [workHigh]
        let constFull = constReg + [constHigh]

        let addend = a % N

        // Step 1: x → x + a   (wraps in n+1 bits; no overflow since x + a < 2N ≤ 2^(n+1))
        try encodeClassicalValue(addend, bitCount: n, register: constReg)
        try applyQuantumAddWrapping(registerA: constFull, registerB: workReg, carryIn: carryIn)
        try encodeClassicalValue(addend, bitCount: n, register: constReg)

        // Step 2: x → x + a − N   (workHigh becomes the sign bit: 1 iff x + a < N)
        try encodeClassicalValue(N, bitCount: n, register: constReg)
        try applyQuantumSubtractWrapping(registerA: constFull, registerB: workReg, carryIn: carryIn)
        try encodeClassicalValue(N, bitCount: n, register: constReg)

        // Step 3: copy the sign bit into flag.
        try cx(workHigh, flag)

        // Step 4: if flag (underflow), add N back → workReg = (x + a) mod N, workHigh = 0.
        try encodeClassicalValue(N, bitCount: n, register: constReg)
        try applyControlledQuantumAddWrapping(
            control: flag, registerA: constFull, registerB: workReg,
            carryIn: carryIn, ancilla: c3xAncilla
        )
        try encodeClassicalValue(N, bitCount: n, register: constReg)

        // Step 5: uncompute flag. Subtracting a makes the sign bit equal to NOT(flag),
        // so X·CNOT·X drives flag back to |0⟩ without disturbing the data.
        try encodeClassicalValue(addend, bitCount: n, register: constReg)
        try applyQuantumSubtractWrapping(registerA: constFull, registerB: workReg, carryIn: carryIn)
        try encodeClassicalValue(addend, bitCount: n, register: constReg)
        try x(workHigh)
        try cx(workHigh, flag)
        try x(workHigh)

        // Step 6: restore the addition → workReg = (x + a) mod N, all ancilla back to |0⟩.
        try encodeClassicalValue(addend, bitCount: n, register: constReg)
        try applyQuantumAddWrapping(registerA: constFull, registerB: workReg, carryIn: carryIn)
        try encodeClassicalValue(addend, bitCount: n, register: constReg)
    }

    /// Controlled modular addition: |ctrl⟩|x⟩ → |ctrl⟩|(x + ctrl·a) mod N⟩.
    private mutating func applyControlledModularAdd(
        control: Int,
        a: Int,
        modulus N: Int,
        registerX: [Int],
        ancillaRegister: [Int]
    ) throws {
        guard N > 0 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Modulus must be positive")
        }
        guard a >= 0 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Addend must be non-negative")
        }
        guard !registerX.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "registerX must not be empty")
        }
        let n = registerX.count
        guard ancillaRegister.count >= n + 5 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "ancillaRegister requires at least registerX.count + 5 qubits (constReg, constHigh, workHigh, flag, carryIn, c3xAncilla)"
            )
        }
        guard n < Int.bitWidth - 1, N <= (1 << n) - 1 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "modulus N must satisfy N < 2^registerX.count so it fits in the register"
            )
        }

        try validateRegisterIndex(control)
        for idx in registerX { try validateRegisterIndex(idx) }
        for idx in ancillaRegister { try validateRegisterIndex(idx) }

        let constReg = Array(ancillaRegister[0..<n])
        let constHigh = ancillaRegister[n]
        let workHigh = ancillaRegister[n + 1]
        let flag = ancillaRegister[n + 2]
        let carryIn = ancillaRegister[n + 3]
        let c3xAncilla = ancillaRegister[n + 4]

        let workReg = registerX + [workHigh]
        let constFull = constReg + [constHigh]

        let addend = a % N

        // Same VBE schedule as the uncontrolled adder, but every arithmetic step (and the
        // flag bookkeeping) is gated on `control`. When control = |0⟩ each operation is a
        // no-op, so the whole adder — including all ancilla — is a strict identity.

        // Step 1: x → x + ctrl·a
        try encodeClassicalValue(addend, bitCount: n, register: constReg)
        try applyControlledQuantumAddWrapping(
            control: control, registerA: constFull, registerB: workReg,
            carryIn: carryIn, ancilla: c3xAncilla
        )
        try encodeClassicalValue(addend, bitCount: n, register: constReg)

        // Step 2: x → x + ctrl·a − ctrl·N  (workHigh = sign bit when control = 1)
        try encodeClassicalValue(N, bitCount: n, register: constReg)
        try applyControlledQuantumSubtractWrapping(
            control: control, registerA: constFull, registerB: workReg,
            carryIn: carryIn, ancilla: c3xAncilla
        )
        try encodeClassicalValue(N, bitCount: n, register: constReg)

        // Step 3: copy the sign bit into flag (only when control = 1).
        try ccx(control, workHigh, flag)

        // Step 4: if flag, add N back → workReg = (x + a) mod N, workHigh = 0.
        try encodeClassicalValue(N, bitCount: n, register: constReg)
        try applyControlledQuantumAddWrapping(
            control: flag, registerA: constFull, registerB: workReg,
            carryIn: carryIn, ancilla: c3xAncilla
        )
        try encodeClassicalValue(N, bitCount: n, register: constReg)

        // Step 5: uncompute flag (sign bit becomes NOT(flag) after subtracting a).
        try encodeClassicalValue(addend, bitCount: n, register: constReg)
        try applyControlledQuantumSubtractWrapping(
            control: control, registerA: constFull, registerB: workReg,
            carryIn: carryIn, ancilla: c3xAncilla
        )
        try encodeClassicalValue(addend, bitCount: n, register: constReg)
        try cx(control, workHigh)
        try ccx(control, workHigh, flag)
        try cx(control, workHigh)

        // Step 6: restore the addition → workReg = (x + ctrl·a) mod N, all ancilla |0⟩.
        try encodeClassicalValue(addend, bitCount: n, register: constReg)
        try applyControlledQuantumAddWrapping(
            control: control, registerA: constFull, registerB: workReg,
            carryIn: carryIn, ancilla: c3xAncilla
        )
        try encodeClassicalValue(addend, bitCount: n, register: constReg)
    }

    /// Doubly-controlled modular addition conditioned on both `control1` and `control2` being |1⟩.
    private mutating func applyDoublyControlledModularAdd(
        control1: Int,
        control2: Int,
        a: Int,
        modulus N: Int,
        registerX: [Int],
        ancillaRegister: [Int],
        scratchFlag: Int
    ) throws {
        guard ancillaRegister.count >= registerX.count + 5 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "ancillaRegister requires at least registerX.count + 5 qubits (constReg, constHigh, workHigh, flag, carryIn, c3xAncilla)"
            )
        }
        try validateRegisterIndex(control1)
        try validateRegisterIndex(control2)
        try validateRegisterIndex(scratchFlag)

        try ccx(control1, control2, scratchFlag)
        try applyControlledModularAdd(
            control: scratchFlag,
            a: a,
            modulus: N,
            registerX: registerX,
            ancillaRegister: ancillaRegister
        )
        try ccx(control1, control2, scratchFlag)
    }

    private mutating func encodeClassicalValue(
        _ value: Int,
        bitCount: Int,
        register: [Int]
    ) throws {
        for i in 0..<bitCount {
            if (value >> i) & 1 != 0 {
                try x(register[i])
            }
        }
    }

    private func modularInverse(_ a: Int, modulus N: Int) throws -> Int {
        guard N > 1 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Modulus must be greater than 1 for modular inverse")
        }
        let reduced = ((a % N) + N) % N
        guard reduced != 0 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Addend must be coprime to modulus for modular inverse")
        }

        var t = 0, newT = 1
        var r = N, newR = reduced
        while newR != 0 {
            let quotient = r / newR
            (t, newT) = (newT, t - quotient * newT)
            (r, newR) = (newR, r - quotient * newR)
        }
        guard r == 1 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Addend must be coprime to modulus for modular inverse")
        }
        return ((t % N) + N) % N
    }

    // MARK: - Modular Exponentiation (Shor's Algorithm scaffold)

    public mutating func applyModularExponentiation(
        a: Int,
        modulus N: Int,
        controlRegister: ClosedRange<Int>,
        targetRegister: ClosedRange<Int>,
        ancillaRegister: ClosedRange<Int>
    ) throws {
        guard N > 1 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Modulus must be greater than 1")
        }

        guard a >= 0 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Base must be non-negative")
        }

        guard !controlRegister.isEmpty, !targetRegister.isEmpty, !ancillaRegister.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Control, target, and ancilla registers must not be empty")
        }

        for index in controlRegister {
            try validateRegisterIndex(index)
        }

        for index in targetRegister {
            try validateRegisterIndex(index)
        }

        for index in ancillaRegister {
            try validateRegisterIndex(index)
        }

        if controlRegister.overlaps(targetRegister) {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Control and target registers must not overlap")
        }

        if controlRegister.overlaps(ancillaRegister) {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Control and ancilla registers must not overlap")
        }

        if targetRegister.overlaps(ancillaRegister) {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Target and ancilla registers must not overlap")
        }

        let targetIndices = Array(targetRegister)
        let ancillaIndices = Array(ancillaRegister)
        let requiredAncillaCount = (targetIndices.count * 2) + 6
        guard ancillaIndices.count >= requiredAncillaCount else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "ancillaRegister requires at least (2 * targetRegister.count) + 6 qubits"
            )
        }

        let runningBase = a % N
        var currentMultiplier = runningBase

        for controlQubit in controlRegister {
            try applyControlledModularMultiply(
                control: controlQubit,
                a: currentMultiplier,
                modulus: N,
                targetRegister: targetIndices,
                ancillaRegister: Array(ancillaIndices.prefix(requiredAncillaCount))
            )

            currentMultiplier = (currentMultiplier * currentMultiplier) % N
        }
    }

    /// Controlled SWAP (Fredkin) of `q1` and `q2`, conditioned on `control`.
    /// Decomposed as CX(q2,q1)·CCX(control,q1,q2)·CX(q2,q1); leaves both qubits
    /// unchanged when `control = |0⟩` and exchanges them when `control = |1⟩`.
    private mutating func applyControlledSwap(control: Int, _ q1: Int, _ q2: Int) throws {
        try cx(q2, q1)
        try ccx(control, q1, q2)
        try cx(q2, q1)
    }

    /// VBE controlled modular multiplication: |c⟩|y⟩ → |c⟩|(c · a · y) mod N⟩.
    ///
    /// Implemented out-of-place: the product is accumulated into a clean `|0⟩`
    /// scratch register, a `control`-conditioned SWAP moves it into the target, and
    /// the scratch register is uncomputed with `a⁻¹`. Crucially every step is gated
    /// on `control`, so when `control = |0⟩` the whole operation is the identity and
    /// the ancilla register is guaranteed to return to `|0⟩` (no entanglement leak).
    private mutating func applyControlledModularMultiply(
        control: Int,
        a: Int,
        modulus N: Int,
        targetRegister: [Int],
        ancillaRegister: [Int]
    ) throws {
        guard N > 1 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Modulus must be greater than 1")
        }
        guard a >= 0 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Multiplier must be non-negative")
        }
        guard !targetRegister.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "targetRegister must not be empty")
        }

        let n = targetRegister.count
        guard ancillaRegister.count >= (n * 2) + 6 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "ancillaRegister requires at least (2 * targetRegister.count) + 6 qubits (accumulator, constReg, constHigh, workHigh, flag, carryIn, c3xAncilla, scratchFlag)"
            )
        }

        try validateRegisterIndex(control)
        for idx in targetRegister { try validateRegisterIndex(idx) }
        for idx in ancillaRegister { try validateRegisterIndex(idx) }

        let accumulator = Array(ancillaRegister[0..<n])
        let adderAncilla = Array(ancillaRegister[n..<(2 * n + 5)])
        let scratchFlag = ancillaRegister[(2 * n) + 5]

        let multiplier = a % N

        for i in 0..<n {
            let addend = (multiplier * (1 << i)) % N
            try applyDoublyControlledModularAdd(
                control1: control,
                control2: targetRegister[i],
                a: addend,
                modulus: N,
                registerX: accumulator,
                ancillaRegister: adderAncilla,
                scratchFlag: scratchFlag
            )
        }

        // Step 2: control-conditioned SWAP of target ↔ accumulator (Fredkin).
        // control = 1 → target now holds a·y mod N, accumulator holds the original y.
        // control = 0 → identity, accumulator stays |0⟩.
        for i in 0..<n {
            try applyControlledSwap(control: control, targetRegister[i], accumulator[i])
        }

        // Step 3: uncompute the accumulator back to |0⟩ using a⁻¹.
        // Modular subtraction of v is realised as modular addition of (N − v) mod N,
        // doubly controlled on `control` and the (post-swap) target bits. This drives
        // the accumulator to |0⟩ in the control = 1 branch and is a no-op otherwise,
        // so no which-path information is left behind in the ancilla.
        let aInverse = try modularInverse(multiplier, modulus: N)
        for i in stride(from: n - 1, through: 0, by: -1) {
            let inverseAddend = (aInverse * (1 << i)) % N
            let subtrahend = (N - inverseAddend) % N
            try applyDoublyControlledModularAdd(
                control1: control,
                control2: targetRegister[i],
                a: subtrahend,
                modulus: N,
                registerX: accumulator,
                ancillaRegister: adderAncilla,
                scratchFlag: scratchFlag
            )
        }
    }

    private func validateRegisterIndex(_ index: Int) throws {
        guard index >= 0, index < qubitCount else {
            throw QuantumCircuitError.qubitIndexOutOfBounds(index: index, qubitCount: qubitCount)
        }
    }
}
