import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Task 2: general Pauli expectation

    func testPauliExpectationXOnPlusState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .x])
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)

        // Cross-validate against the dedicated single-qubit ⟨X⟩.
        let reference = try QuantumMeasurement.expectationX(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, reference, accuracy: 1e-5)
    }

    func testPauliExpectationZOnZeroState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        // |0⟩: no gates applied.
        let expectation = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .z])
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)

        let reference = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, reference, accuracy: 1e-5)
    }

    func testPauliExpectationOnBellState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try engine.execute(circuit, on: state)

        let xx = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .x, 1: .x])
        let yy = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .y, 1: .y])
        let zz = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .z, 1: .z])

        XCTAssertEqual(xx, 1, accuracy: 1e-5)
        XCTAssertEqual(yy, -1, accuracy: 1e-5)
        XCTAssertEqual(zz, 1, accuracy: 1e-5)

        // ⟨Z₀Z₁⟩ must agree with the existing ZZ helper.
        let referenceZZ = try QuantumMeasurement.expectationZZ(state: state, engine: engine, qubitA: 0, qubitB: 1)
        XCTAssertEqual(zz, referenceZZ, accuracy: 1e-5)
    }

    func testPauliExpectationIdentityFactorsAreIgnored() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try engine.execute(circuit, on: state)

        // ⟨Z₀⟩ on the Bell state is 0; an explicit identity on qubit 1 changes nothing.
        let withIdentity = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .z, 1: .i])
        let withoutIdentity = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .z])

        XCTAssertEqual(withIdentity, 0, accuracy: 1e-5)
        XCTAssertEqual(withIdentity, withoutIdentity, accuracy: 1e-5)
    }

    func testPauliExpectationStringMatchesDictionary() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // |+⟩ on qubit 0, |1⟩ on qubit 1, |0⟩ on qubit 2.
        let state = try StateVector(qubitCount: 3, device: device)
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.x(1)
        try engine.execute(circuit, on: state)

        // MSB-first label "ZZX" → qubit2=Z, qubit1=Z, qubit0=X.
        let fromString = try QuantumMeasurement.expectation(state: state, engine: engine, pauliString: "ZZX")
        let fromDict = try QuantumMeasurement.expectation(
            state: state,
            engine: engine,
            paulis: [2: .z, 1: .z, 0: .x]
        )

        XCTAssertEqual(fromString, fromDict, accuracy: 1e-5)
        // ⟨Z₂⟩=+1 (|0⟩), ⟨Z₁⟩=−1 (|1⟩), ⟨X₀⟩=+1 (|+⟩) ⇒ product −1.
        XCTAssertEqual(fromString, -1, accuracy: 1e-5)
    }

    func testPauliExpectationRejectsWrongLengthString() throws {
        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let engine = try QuantumEngine()
        let state = try StateVector(qubitCount: 2, device: device)

        XCTAssertThrowsError(try QuantumMeasurement.expectation(state: state, engine: engine, pauliString: "XYZ")) { error in
            guard case QuantumMeasurementError.invalidPauliString = error else {
                XCTFail("Expected invalidPauliString")
                return
            }
        }
    }
}
