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
