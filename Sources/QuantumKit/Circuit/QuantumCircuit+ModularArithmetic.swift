import Foundation

extension QuantumCircuit {

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
    mutating func applyControlledModularAdd(
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
    mutating func applyDoublyControlledModularAdd(
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

    mutating func encodeClassicalValue(
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

    func modularInverse(_ a: Int, modulus N: Int) throws -> Int {
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

}
