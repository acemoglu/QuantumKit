import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Task 1: amplitudes()

    func testAmplitudesOnPlusState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let amplitudes = QuantumMeasurement.amplitudes(state: state)
        let invSqrt2 = QFloat(1.0 / 2.0.squareRoot())

        XCTAssertEqual(amplitudes.count, 2)
        XCTAssertEqual(amplitudes[0].real, invSqrt2, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[0].imaginary, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[1].real, invSqrt2, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[1].imaginary, 0, accuracy: 1e-5)
    }

    func testAmplitudesOnBellState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try engine.execute(circuit, on: state)

        let amplitudes = QuantumMeasurement.amplitudes(state: state)
        let invSqrt2 = QFloat(1.0 / 2.0.squareRoot())

        XCTAssertEqual(amplitudes.count, 4)
        XCTAssertEqual(amplitudes[0].real, invSqrt2, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[0].imaginary, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[1].real, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[1].imaginary, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[2].real, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[2].imaginary, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[3].real, invSqrt2, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[3].imaginary, 0, accuracy: 1e-5)
    }

    // MARK: - Task 1: sxdg gate

    func testSXDaggerInvertsSX() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.sx(0)
        try circuit.sxdg(0)
        try engine.execute(circuit, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probabilities[0], 1, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 0, accuracy: 1e-5)
    }

    func testSXDaggerSquaredIsPauliX() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // SX†·SX† = X (up to global phase): |0> → |1>.
        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.sxdg(0)
        try circuit.sxdg(0)
        try engine.execute(circuit, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probabilities[0], 0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 1, accuracy: 1e-5)
    }

    // MARK: - Task 1: inverse()

    func testInverseReturnsToZeroState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // A mixed circuit exercising self-inverse, dagger-paired, and parametrized gates.
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.t(1)
        try circuit.sx(2)
        try circuit.rx(theta: 0.7, 0)
        try circuit.ry(theta: -1.3, 1)
        try circuit.rz(theta: 2.1, 2)
        try circuit.cx(0, 1)
        try circuit.cz(1, 2)
        try circuit.crx(theta: 0.9, control: 0, target: 2)
        try circuit.cp(theta: 1.1, control: 1, target: 0)
        try circuit.s(2)
        try circuit.u(theta: 0.5, phi: 1.2, lambda: -0.8, 1)
        try circuit.ccx(0, 1, 2)

        let inverse = try circuit.inverse()

        let state = try StateVector(qubitCount: 3, device: device)
        try engine.execute(circuit, on: state)
        try engine.execute(inverse, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probabilities[0], 1, accuracy: 1e-4)
        for index in 1..<probabilities.count {
            XCTAssertEqual(probabilities[index], 0, accuracy: 1e-4)
        }
    }

    func testInverseGateOrderIsReversedAndAdjointed() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.s(0)
        try circuit.t(0)
        try circuit.sx(0)

        let inverse = try circuit.inverse()

        XCTAssertEqual(inverse.gates, [.sxdg(target: 0), .tdg(target: 0), .sdg(target: 0)])
    }

    func testInverseThrowsForNonUnitaryCircuit() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)

        XCTAssertThrowsError(try circuit.inverse()) { error in
            guard case QuantumCircuitError.circuitNotUnitary = error else {
                XCTFail("Expected circuitNotUnitary")
                return
            }
        }
    }
}
