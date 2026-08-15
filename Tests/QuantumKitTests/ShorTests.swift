import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Shor accuracy (N = 15 = 3 × 5)

    func testShorFactors15Accuracy() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let modulus = 15
        let base = 7
        let expectedPeriod = try ShorClassical.multiplicativeOrder(base: base, modulus: modulus)
        XCTAssertEqual(expectedPeriod, 4, "Precondition: multiplicative order of 7 mod 15 is 4")

        // 4 counting qubits suffice for r = 4 (phase peaks at k/4); more controls explode gate depth.
        let controlQubitCount = 4
        let targetQubitCount = 4
        let ancillaQubitCount = (targetQubitCount * 2) + 6
        let qubitCount = controlQubitCount + targetQubitCount + ancillaQubitCount

        let controlRange = 0..<controlQubitCount
        let targetRange = controlQubitCount..<(controlQubitCount + targetQubitCount)
        let ancillaRange = (controlQubitCount + targetQubitCount)..<qubitCount

        var circuit = try QuantumCircuit(qubitCount: qubitCount)

        // |y⟩ = |1⟩
        try circuit.x(targetRange.lowerBound)

        for control in controlRange {
            try circuit.h(control)
        }

        try circuit.applyModularExponentiation(
            a: base,
            modulus: modulus,
            controlRegister: controlRange.lowerBound...controlRange.upperBound - 1,
            targetRegister: targetRange.lowerBound...targetRange.upperBound - 1,
            ancillaRegister: ancillaRange.lowerBound...ancillaRange.upperBound - 1
        )

        try circuit.applyInverseQFT(qubits: controlRange)

        let state = try StateVector(qubitCount: qubitCount)
        try engine.execute(circuit, on: state)

        let shots = 512
        var rng: QuantumRNG = .seeded(0x5100_0015)
        // The inverse QFT emits its phase estimate in bit-reversed order, so the counting
        // register must be read most-significant qubit first to recover the true numerator.
        let counts = try QuantumMeasurement.sampleCountsRNG(
            state: state,
            engine: engine,
            qubits: Array(controlRange).reversed(),
            shots: shots,
            rng: &rng
        )

        let analysis = ShorClassical.analyze(
            counts: counts,
            controlQubitCount: controlQubitCount,
            base: base,
            modulus: modulus
        )

        print("🔐 SHOR N=15: recovered periods \(analysis.recoveredPeriods.sorted()), factors \(analysis.foundFactors.sorted())")

        XCTAssertTrue(
            analysis.recoveredPeriods.contains(expectedPeriod),
            "Expected to recover classical period r=\(expectedPeriod) from QFT peaks; got \(analysis.recoveredPeriods)"
        )
        XCTAssertEqual(
            analysis.foundFactors,
            [3, 5],
            "Shor post-processing should factor 15 into 3 and 5"
        )
    }

    // MARK: - Ancilla uncompute / no-entanglement-leak (Shor controlled modular multiply)

    /// Direct regression test for the controlled-modular-multiply ancilla cleanup
    /// bug ("Madde 4"): when the control qubit is |0⟩, the operation must be a strict
    /// identity and every ancilla qubit must be returned to a clean |0⟩.
    ///
    /// The previous implementation used an unconditional `cx` "swap" plus a
    /// wrong-direction uncompute, so even with control = |0⟩ it copied the target
    /// into the accumulator (`accumulator ^= target`) and never cleared it — leaving
    /// the ancilla entangled with the main register. Here we assert the full state
    /// vector is untouched (only |control=0, y, ancilla=0⟩ has amplitude), which fails
    /// on the buggy code and passes once the swap/uncompute are control-gated.
    func testControlledModularMultiplyControlZeroLeavesNoTrace() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // Layout (13 qubits): control = qubit 0, target = qubits 1,2 (n = 2),
        // ancilla = qubits 3...12 (2 * n + 6 = 10).
        let qubitCount = 13
        let controlRange = 0...0
        let targetRange = 1...2
        let ancillaRange = 3...12

        let state = try StateVector(qubitCount: qubitCount)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)

        // control = |0⟩ (left untouched), |y⟩ = |1⟩ on the target register.
        try circuit.x(targetRange.lowerBound)

        try circuit.applyModularExponentiation(
            a: 2,
            modulus: 3,
            controlRegister: controlRange,
            targetRegister: targetRange,
            ancillaRegister: ancillaRange
        )

        try engine.execute(circuit, on: state)

        let amplitudes = QuantumMeasurement.amplitudes(state: state)

        // basis index = Σ qubit_i · 2^i (qubit 0 = control = LSB).
        // Expected: identity → |control=0, y=1, ancilla=0⟩ → only index 2 (qubit 1 set).
        let expectedIndex = 2
        for (index, amplitude) in amplitudes.enumerated() {
            let magnitude = (Double(amplitude.real) * Double(amplitude.real)
                             + Double(amplitude.imaginary) * Double(amplitude.imaginary)).squareRoot()
            let expected: Double = (index == expectedIndex) ? 1.0 : 0.0
            XCTAssertEqual(magnitude, expected, accuracy: 1e-4,
                           "control = |0⟩ must be the identity with all ancilla restored to |0⟩; "
                           + "unexpected amplitude at basis state \(index)")
        }
    }

    /// Superposition / interference test ("Madde 4"). Puts the control qubit into a
    /// Hadamard superposition, applies the controlled modular multiply, then verifies
    /// that the resulting state is EXACTLY the clean two-branch superposition
    ///     (|c=0, y=1⟩ + |c=1, y=(a·y) mod N⟩) / √2  ⊗  |ancilla = 0⟩,
    /// i.e. every ancilla qubit is restored to |0⟩ in BOTH branches (no which-path
    /// information leaks anywhere) and the interference pattern is fully intact.
    ///
    /// The buggy code populated extra basis states (ancilla ≠ 0) in both branches; this
    /// strict full-state-vector check fails there and passes only with a leak-free
    /// controlled multiply and a fully reversible modular adder.
    func testControlledModularMultiplyAncillaUncomputeNoLeak() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let qubitCount = 13
        let controlRange = 0...0      // qubit 0
        let targetRange = 1...2       // qubits 1,2  (n = 2, LSB = qubit 1)
        let ancillaRange = 3...12     // qubits 3...12

        let base = 2
        let modulus = 3

        let state = try StateVector(qubitCount: qubitCount)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)

        // |y⟩ = |1⟩
        try circuit.x(targetRange.lowerBound)

        // Control into superposition (|0⟩ + |1⟩)/√2.
        try circuit.h(controlRange.lowerBound)

        try circuit.applyModularExponentiation(
            a: base,
            modulus: modulus,
            controlRegister: controlRange,
            targetRegister: targetRange,
            ancillaRegister: ancillaRange
        )

        try engine.execute(circuit, on: state)

        let amplitudes = QuantumMeasurement.amplitudes(state: state)
        let invSqrt2 = Double(1.0 / 2.0.squareRoot())

        func magnitude(_ a: ComplexAmplitude) -> Double {
            (Double(a.real) * Double(a.real) + Double(a.imaginary) * Double(a.imaginary)).squareRoot()
        }

        // basis index = Σ qubit_i · 2^i (qubit 0 = control = LSB), ancilla = qubits ≥ 3.
        //   c = 0 branch: control=0, y=1                 → index 2
        //   c = 1 branch: control=1, y=(2·1) mod 3 = 2   → index 5  (1 + 4)
        let cleanZeroBranchIndex = 2
        let cleanOneBranchIndex = 5

        for (index, amplitude) in amplitudes.enumerated() {
            let mag = magnitude(amplitude)
            if index == cleanZeroBranchIndex || index == cleanOneBranchIndex {
                XCTAssertEqual(mag, invSqrt2, accuracy: 1e-4,
                               "Branch \(index) must carry amplitude 1/√2 with all ancilla clean")
            } else {
                XCTAssertEqual(mag, 0, accuracy: 1e-4,
                               "No amplitude may leak to basis state \(index) — ancilla must stay |0⟩")
            }
        }
    }

    /// Interference (Hadamard-refocus) test. With an identity multiply (a = 1) a clean,
    /// leak-free controlled multiply leaves the control qubit completely disentangled,
    /// so H · multiply · H must refocus it deterministically back to |0⟩. Any residual
    /// ancilla entanglement (which-path information) would wash this interference out.
    func testControlledModularMultiplyPreservesInterference() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let qubitCount = 13
        let controlRange = 0...0
        let targetRange = 1...2
        let ancillaRange = 3...12

        let state = try StateVector(qubitCount: qubitCount)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)

        // |y⟩ = |1⟩
        try circuit.x(targetRange.lowerBound)

        // H · (controlled multiply by 1) · H — identity on the control if no leak.
        try circuit.h(controlRange.lowerBound)
        try circuit.applyModularExponentiation(
            a: 1,
            modulus: 3,
            controlRegister: controlRange,
            targetRegister: targetRange,
            ancillaRegister: ancillaRange
        )
        try circuit.h(controlRange.lowerBound)

        try engine.execute(circuit, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)

        // Control refocused to |0⟩, target still |y=1⟩, all ancilla |0⟩ → only index 2.
        let expectedIndex = 2
        for (index, probability) in probabilities.enumerated() {
            let expected: Double = (index == expectedIndex) ? 1.0 : 0.0
            XCTAssertEqual(Double(probability), expected, accuracy: 1e-4,
                           "Interference must refocus the control to |0⟩; unexpected probability at \(index)")
        }
    }

    /// End-to-end correctness of the modular-exponentiation oracle: with the counting
    /// register driven to a definite value `c`, the target must hold exactly `a^c mod N`
    /// and every ancilla qubit must be restored to |0⟩ (unitary, leak-free oracle).
    func testModularExponentiationOracleIsExactAndCleanForAllExponents() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let controlN = 4, targetN = 4
        let ancN = targetN * 2 + 6
        let total = controlN + targetN + ancN
        let controlRange = 0..<controlN
        let targetRange = controlN..<(controlN + targetN)
        let ancRange = (controlN + targetN)..<total
        let base = 7, modulus = 15

        for c in 0..<(1 << controlN) {
            let state = try StateVector(qubitCount: total)
            var circuit = try QuantumCircuit(qubitCount: total)

            try circuit.x(targetRange.lowerBound)   // |y⟩ = |1⟩
            for j in 0..<controlN where (c >> j) & 1 == 1 {
                try circuit.x(controlRange.lowerBound + j)
            }

            try circuit.applyModularExponentiation(
                a: base, modulus: modulus,
                controlRegister: controlRange.lowerBound...(controlRange.upperBound - 1),
                targetRegister: targetRange.lowerBound...(targetRange.upperBound - 1),
                ancillaRegister: ancRange.lowerBound...(ancRange.upperBound - 1)
            )

            try engine.execute(circuit, on: state)

            let amplitudes = QuantumMeasurement.amplitudes(state: state)
            var nonZero: [Int] = []
            for (index, a) in amplitudes.enumerated() {
                let mag = (Double(a.real) * Double(a.real)
                           + Double(a.imaginary) * Double(a.imaginary)).squareRoot()
                if mag > 1e-3 { nonZero.append(index) }
            }

            XCTAssertEqual(nonZero.count, 1,
                           "Oracle must be a permutation (single basis state) for exponent \(c)")
            let index = nonZero[0]
            let targetValue = (index >> controlN) & ((1 << targetN) - 1)
            let ancillaBits = index >> (controlN + targetN)
            XCTAssertEqual(targetValue, ShorClassical.modularPower(base, exponent: c, modulus: modulus),
                           "Target must equal \(base)^\(c) mod \(modulus)")
            XCTAssertEqual(ancillaBits, 0, "All ancilla must be restored to |0⟩ for exponent \(c)")
        }
    }

    /// General-N stress test for the modular-exponentiation oracle.
    ///
    /// For several coprime `(base, modulus)` pairs — moduli other than 15, including non-power and
    /// non-prime ones — and every counting-register value `c`, the target must hold exactly
    /// `base^c mod N`, the counting register must be preserved, and every ancilla qubit must return
    /// to |0⟩. This exercises the squaring schedule and the VBE modular adder/multiplier across a
    /// range of register widths, not just the N = 15 special case.
    func testModularExponentiationOracleGeneralModuli() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // (modulus, target bit width, coprime bases). Width must satisfy N <= 2^width - 1.
        let cases: [(modulus: Int, targetN: Int, bases: [Int])] = [
            (5, 3, [2, 3, 4]),
            (7, 3, [2, 3, 4, 5, 6]),
            (9, 4, [2, 4, 5, 7, 8]),
        ]
        let controlN = 2

        for testCase in cases {
            let targetN = testCase.targetN
            let modulus = testCase.modulus
            let ancN = targetN * 2 + 6
            let total = controlN + targetN + ancN

            let controlRange = 0..<controlN
            let targetRange = controlN..<(controlN + targetN)
            let ancRange = (controlN + targetN)..<total

            for base in testCase.bases {
                for c in 0..<(1 << controlN) {
                    let state = try StateVector(qubitCount: total)
                    var circuit = try QuantumCircuit(qubitCount: total)

                    // |y⟩ = |1⟩, then drive the counting register to the definite value c.
                    try circuit.x(targetRange.lowerBound)
                    for j in 0..<controlN where (c >> j) & 1 == 1 {
                        try circuit.x(controlRange.lowerBound + j)
                    }

                    try circuit.applyModularExponentiation(
                        a: base, modulus: modulus,
                        controlRegister: controlRange.lowerBound...(controlRange.upperBound - 1),
                        targetRegister: targetRange.lowerBound...(targetRange.upperBound - 1),
                        ancillaRegister: ancRange.lowerBound...(ancRange.upperBound - 1)
                    )

                    try engine.execute(circuit, on: state)

                    let amplitudes = QuantumMeasurement.amplitudes(state: state)
                    var nonZero: [Int] = []
                    for (index, a) in amplitudes.enumerated() {
                        let probability = Double(a.real) * Double(a.real)
                            + Double(a.imaginary) * Double(a.imaginary)
                        if probability > 1e-3 { nonZero.append(index) }
                    }

                    XCTAssertEqual(nonZero.count, 1,
                        "Oracle must be a permutation (single basis state) for base=\(base), N=\(modulus), c=\(c)")
                    guard let index = nonZero.first else { continue }

                    let controlValue = index & ((1 << controlN) - 1)
                    let targetValue = (index >> controlN) & ((1 << targetN) - 1)
                    let ancillaBits = index >> (controlN + targetN)

                    XCTAssertEqual(controlValue, c,
                        "Counting register must be preserved (base=\(base), N=\(modulus), c=\(c))")
                    XCTAssertEqual(targetValue,
                        ShorClassical.modularPower(base, exponent: c, modulus: modulus),
                        "Target must equal \(base)^\(c) mod \(modulus) (base=\(base), N=\(modulus), c=\(c))")
                    XCTAssertEqual(ancillaBits, 0,
                        "All ancilla must be restored to |0⟩ (base=\(base), N=\(modulus), c=\(c))")
                }
            }
        }
    }
}
