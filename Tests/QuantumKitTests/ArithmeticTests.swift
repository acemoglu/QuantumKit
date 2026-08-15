import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    func testModularExponentiationScaffold() throws {
        var circuit = try QuantumCircuit(qubitCount: 24)

        XCTAssertNoThrow(try circuit.applyModularExponentiation(
            a: 3,
            modulus: 7,
            controlRegister: 0...2,
            targetRegister: 3...7,
            ancillaRegister: 8...23
        ))
    }

    func testModularExponentiationRejectsInvalidParameters() throws {
        var circuit = try QuantumCircuit(qubitCount: 24)

        XCTAssertThrowsError(try circuit.applyModularExponentiation(
            a: 2,
            modulus: 1,
            controlRegister: 0...1,
            targetRegister: 2...5,
            ancillaRegister: 6...23
        )) { error in
            XCTAssertTrue(error is QuantumCircuitError)
        }

        XCTAssertThrowsError(try circuit.applyModularExponentiation(
            a: 2,
            modulus: 5,
            controlRegister: 0...3,
            targetRegister: 2...5,
            ancillaRegister: 6...23
        )) { error in
            XCTAssertTrue(error is QuantumCircuitError)
        }

        XCTAssertThrowsError(try circuit.applyModularExponentiation(
            a: -1,
            modulus: 5,
            controlRegister: 0...1,
            targetRegister: 2...5,
            ancillaRegister: 6...23
        )) { error in
            XCTAssertTrue(error is QuantumCircuitError)
        }
    }

    func testQuantumAdder() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 6)
        var circuit = try QuantumCircuit(qubitCount: 6)

        // Encode a = 1 (binary 01): set qubit 0 (LSB of registerA)
        try circuit.x(0)
        // Encode b = 2 (binary 10): set qubit 3 (MSB of registerB)
        try circuit.x(3)

        // registerA = [0, 1], registerB = [2, 3], carryIn = 4, carryOut = 5
        // Expected: 1 + 2 = 3 (binary 11) stored in registerB [2, 3]
        try circuit.applyQuantumAdd(
            registerA: [0, 1],
            registerB: [2, 3],
            carryIn: 4,
            carryOut: 5
        )

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        // Big-endian result layout for 6 qubits: result[k] = state of qubit(5 - k)
        //   result[0] = qubit 5 (carryOut)
        //   result[1] = qubit 4 (carryIn)
        //   result[2] = qubit 3 (registerB MSB)
        //   result[3] = qubit 2 (registerB LSB)
        //   result[4] = qubit 1 (registerA MSB)
        //   result[5] = qubit 0 (registerA LSB)
        print("➕ QUANTUM ADDER RESULT (1 + 2 = 3): \(result)")

        // Sum 3 = binary 11 in registerB (qubits 2 and 3)
        XCTAssertEqual(result[2], 1, "registerB MSB (qubit 3) should be 1 — sum = 3")
        XCTAssertEqual(result[3], 1, "registerB LSB (qubit 2) should be 1 — sum = 3")

        // Ancilla qubits restored
        XCTAssertEqual(result[0], 0, "carryOut (qubit 5) should be 0 — no overflow for 1+2")
        XCTAssertEqual(result[1], 0, "carryIn (qubit 4) should remain 0")

        // registerA preserved (a = 1 = binary 01: qubit 0 = 1, qubit 1 = 0)
        XCTAssertEqual(result[4], 0, "registerA MSB (qubit 1) should be restored to 0")
        XCTAssertEqual(result[5], 1, "registerA LSB (qubit 0) should be restored to 1")
    }

    func testQuantumSubtract() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 8)
        var circuit = try QuantumCircuit(qubitCount: 8)

        // Encode A = 2 (binary 010): set qubit 1 (middle bit of registerA)
        try circuit.x(1)
        // Encode B = 5 (binary 101): set qubits 3 (LSB) and 5 (MSB of registerB)
        try circuit.x(3)
        try circuit.x(5)

        // registerA = [0, 1, 2], registerB = [3, 4, 5], carryIn = 6, carryOut = 7
        // Expected: 5 - 2 = 3 (binary 011) stored in registerB [3, 4, 5]
        try circuit.applyQuantumSubtract(
            registerA: [0, 1, 2],
            registerB: [3, 4, 5],
            carryIn: 6,
            carryOut: 7
        )

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        // Big-endian result layout for 8 qubits: result[k] = state of qubit(7 - k)
        //   result[0] = qubit 7 (carryOut)
        //   result[1] = qubit 6 (carryIn)
        //   result[2] = qubit 5 (registerB MSB)
        //   result[3] = qubit 4 (registerB middle)
        //   result[4] = qubit 3 (registerB LSB)
        //   result[5] = qubit 2 (registerA MSB)
        //   result[6] = qubit 1 (registerA middle)
        //   result[7] = qubit 0 (registerA LSB)
        print("➖ QUANTUM SUBTRACT RESULT (5 - 2 = 3): \(result)")

        // Difference 3 = binary 011 in registerB (qubits 3, 4, 5)
        XCTAssertEqual(result[2], 0, "registerB MSB (qubit 5) should be 0 — difference = 3")
        XCTAssertEqual(result[3], 1, "registerB middle (qubit 4) should be 1 — difference = 3")
        XCTAssertEqual(result[4], 1, "registerB LSB (qubit 3) should be 1 — difference = 3")

        // carryIn restored
        XCTAssertEqual(result[1], 0, "carryIn (qubit 6) should remain 0")

        // registerA preserved (A = 2 = binary 010)
        XCTAssertEqual(result[5], 0, "registerA MSB (qubit 2) should be restored to 0")
        XCTAssertEqual(result[6], 1, "registerA middle (qubit 1) should be restored to 1")
        XCTAssertEqual(result[7], 0, "registerA LSB (qubit 0) should be restored to 0")
    }

    func testModularAdd() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 13)
        var circuit = try QuantumCircuit(qubitCount: 13)

        // Encode x = 4 (binary 0100): set qubit 2 (bit 2 of 4-bit registerX)
        try circuit.x(2)

        // registerX = [0, 1, 2, 3] (4-bit — holds intermediate x + a before reduction)
        // ancillaRegister = [4...12] → constantReg | carryInAdd | carryOutAdd | carryInSub | carryOutSub | c3xAncilla
        // Expected: (4 + 5) % 7 = 2 (binary 0010) in registerX
        try circuit.applyModularAdd(
            a: 5,
            modulus: 7,
            registerX: [0, 1, 2, 3],
            ancillaRegister: [4, 5, 6, 7, 8, 9, 10, 11, 12]
        )

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        // Big-endian result layout for 13 qubits: result[k] = state of qubit(12 - k)
        //   result[0]  = qubit 12 (c3xAncilla)
        //   result[1]  = qubit 11 (carryOutSub)
        //   result[2]  = qubit 10 (carryInSub)
        //   result[3]  = qubit 9  (carryOutAdd)
        //   result[4]  = qubit 8  (carryInAdd)
        //   result[5]  = qubit 7  (constantReg MSB)
        //   result[6]  = qubit 6  (constantReg bit 2)
        //   result[7]  = qubit 5  (constantReg bit 1)
        //   result[8]  = qubit 4  (constantReg LSB)
        //   result[9]  = qubit 3  (registerX MSB)
        //   result[10] = qubit 2  (registerX bit 2)
        //   result[11] = qubit 1  (registerX bit 1)
        //   result[12] = qubit 0  (registerX LSB)
        print("🔢 MODULAR ADD RESULT ((4 + 5) % 7 = 2): \(result)")

        // Result 2 = binary 0010 in registerX (qubits 0, 1, 2, 3)
        XCTAssertEqual(result[9], 0, "registerX MSB (qubit 3) should be 0 — (4 + 5) % 7 = 2")
        XCTAssertEqual(result[10], 0, "registerX bit 2 (qubit 2) should be 0 — (4 + 5) % 7 = 2")
        XCTAssertEqual(result[11], 1, "registerX bit 1 (qubit 1) should be 1 — (4 + 5) % 7 = 2")
        XCTAssertEqual(result[12], 0, "registerX LSB (qubit 0) should be 0 — (4 + 5) % 7 = 2")

        // Ancilla qubits restored (carryOutSub may hold subtract carry flag)
        XCTAssertEqual(result[0], 0, "c3xAncilla (qubit 12) should be restored to 0")
        XCTAssertEqual(result[2], 0, "carryInSub (qubit 10) should remain 0")
        XCTAssertEqual(result[4], 0, "carryInAdd (qubit 8) should remain 0")
        XCTAssertEqual(result[5], 0, "constantReg MSB (qubit 7) should be restored to 0")
        XCTAssertEqual(result[6], 0, "constantReg bit 2 (qubit 6) should be restored to 0")
        XCTAssertEqual(result[7], 0, "constantReg bit 1 (qubit 5) should be restored to 0")
        XCTAssertEqual(result[8], 0, "constantReg LSB (qubit 4) should be restored to 0")
    }

    /// Exhaustive reversibility sweep of the rewritten VBE modular adder: for several
    /// moduli (prime and composite) and every 0 ≤ x, a < N, the data register must hold
    /// exactly (x + a) mod N and ALL ancilla qubits must return to |0⟩.
    func testModularAddExhaustiveReversibility() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // (modulus, bit-width) pairs with N < 2^n.
        let cases: [(N: Int, n: Int)] = [(3, 2), (5, 3), (6, 3), (7, 3)]

        for (N, n) in cases {
            let qubitCount = 2 * n + 5
            let registerX = Array(0..<n)
            let ancilla = Array(n..<(2 * n + 5))

            for x in 0..<N {
                for a in 0..<N {
                    let state = try StateVector(qubitCount: qubitCount)
                    var circuit = try QuantumCircuit(qubitCount: qubitCount)

                    for i in 0..<n where (x >> i) & 1 == 1 { try circuit.x(registerX[i]) }

                    try circuit.applyModularAdd(
                        a: a, modulus: N, registerX: registerX, ancillaRegister: ancilla
                    )

                    try engine.execute(circuit, on: state)

                    let amplitudes = QuantumMeasurement.amplitudes(state: state)
                    var nonZero: [Int] = []
                    for (index, amp) in amplitudes.enumerated() {
                        let mag = (Double(amp.real) * Double(amp.real)
                                   + Double(amp.imaginary) * Double(amp.imaginary)).squareRoot()
                        if mag > 1e-3 { nonZero.append(index) }
                    }

                    XCTAssertEqual(nonZero.count, 1,
                                   "N=\(N) x=\(x) a=\(a): result must be a single basis state")
                    let index = nonZero[0]
                    let dataValue = index & ((1 << n) - 1)
                    let ancillaBits = index >> n
                    XCTAssertEqual(dataValue, (x + a) % N,
                                   "N=\(N) x=\(x) a=\(a): registerX must equal (x+a) mod N")
                    XCTAssertEqual(ancillaBits, 0,
                                   "N=\(N) x=\(x) a=\(a): all ancilla must be restored to |0⟩")
                }
            }
        }
    }

    func testCCXRejectsOutOfBoundsIndex() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)

        XCTAssertThrowsError(try circuit.ccx(0, 1, 5)) { error in
            guard case QuantumCircuitError.qubitIndexOutOfBounds = error else {
                XCTFail("Expected qubitIndexOutOfBounds")
                return
            }
        }
    }

    func testSwapRejectsOutOfBoundsIndex() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)

        XCTAssertThrowsError(try circuit.applySwap(q1: 0, q2: 4)) { error in
            guard case QuantumCircuitError.qubitIndexOutOfBounds = error else {
                XCTFail("Expected qubitIndexOutOfBounds")
                return
            }
        }
    }
}
