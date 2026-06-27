import Foundation

extension QuantumCircuit {

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
    mutating func applyControlledSwap(control: Int, _ q1: Int, _ q2: Int) throws {
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
    mutating func applyControlledModularMultiply(
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

    func validateRegisterIndex(_ index: Int) throws {
        guard index >= 0, index < qubitCount else {
            throw QuantumCircuitError.qubitIndexOutOfBounds(index: index, qubitCount: qubitCount)
        }
    }
}
