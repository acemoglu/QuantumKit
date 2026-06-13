import XCTest
import Metal
@testable import QuantumKit

final class QuantumKitTests: XCTestCase {

    private func makeDevice() -> MTLDevice? {
        MTLCreateSystemDefaultDevice()
    }

    func testBellStateEntanglement() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }
        let state = try StateVector(qubitCount: 2, device: device)

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🚀 QUANTUM COLLAPSE RESULT: \(result)")

        let isZeroZero = (result[0] == 0 && result[1] == 0)
        let isOneOne = (result[0] == 1 && result[1] == 1)

        XCTAssertTrue(isZeroZero || isOneOne, "Entanglement broken! Collapsed into an impossible state: \(result)")
    }

    func testQuantumFourierTransform() throws {
        let qubitCount = 3
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }
        let state = try StateVector(qubitCount: qubitCount, device: device)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)

        try circuit.applyQFT()

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🌊 QFT COLLAPSE RESULT (3 Qubit): \(result)")

        XCTAssertEqual(result.count, qubitCount)
    }

    func testCCXGate() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 3, device: device)
        var circuit = try QuantumCircuit(qubitCount: 3)

        try circuit.x(0)
        try circuit.x(1)
        try circuit.ccx(0, 1, 2)

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🔺 CCX COLLAPSE RESULT: \(result)")

        XCTAssertEqual(result, [1, 1, 1], "CCX should flip target when both controls are |1>")
    }

    func testSwapGate() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)

        try circuit.x(0)
        try circuit.applySwap(q1: 0, q2: 1)

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🔄 SWAP COLLAPSE RESULT: \(result)")

        XCTAssertEqual(result, [1, 0], "SWAP should exchange qubit amplitudes")
    }

    func testModularExponentiationScaffold() throws {
        var circuit = try QuantumCircuit(qubitCount: 8)

        XCTAssertNoThrow(try circuit.applyModularExponentiation(
            a: 3,
            modulus: 7,
            controlRegister: 0...2,
            targetRegister: 3...7
        ))
    }

    func testModularExponentiationRejectsInvalidParameters() throws {
        var circuit = try QuantumCircuit(qubitCount: 6)

        XCTAssertThrowsError(try circuit.applyModularExponentiation(
            a: 2,
            modulus: 1,
            controlRegister: 0...1,
            targetRegister: 2...5
        )) { error in
            XCTAssertTrue(error is QuantumCircuitError)
        }

        XCTAssertThrowsError(try circuit.applyModularExponentiation(
            a: 2,
            modulus: 5,
            controlRegister: 0...3,
            targetRegister: 2...5
        )) { error in
            XCTAssertTrue(error is QuantumCircuitError)
        }

        XCTAssertThrowsError(try circuit.applyModularExponentiation(
            a: -1,
            modulus: 5,
            controlRegister: 0...1,
            targetRegister: 2...5
        )) { error in
            XCTAssertTrue(error is QuantumCircuitError)
        }
    }

    func testQuantumAdder() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 6, device: device)
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
