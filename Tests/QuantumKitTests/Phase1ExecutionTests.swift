import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    func testConditionalXAppliesOnlyWhenClassicalRegisterMatches() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        var circuitOn = try QuantumCircuit(
            qubitCount: 2,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 2)]
        )
        try circuitOn.x(0)
        try circuitOn.measure(qubits: [0], classicalRegister: 0, classicalBitOffset: 0)
        try circuitOn.c_if(classicalRegister: 0, equals: 1, x: 1)

        let stateOn = try StateVector(qubitCount: 2)
        var rngOn: QuantumRNG = .seeded(1)
        let onResult = try engine.executeRNG(circuitOn, on: stateOn, rng: &rngOn)
        XCTAssertEqual(onResult.classicalMemory.value(ofRegister: 0), 1)
        let z1On = try QuantumMeasurement.expectationZ(state: stateOn, engine: engine, qubit: 1)
        XCTAssertEqual(z1On, -1, accuracy: 1e-5)

        var circuitOff = try QuantumCircuit(
            qubitCount: 2,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 2)]
        )
        try circuitOff.measure(qubits: [0], classicalRegister: 0, classicalBitOffset: 0)
        try circuitOff.c_if(classicalRegister: 0, equals: 1, x: 1)

        let stateOff = try StateVector(qubitCount: 2)
        var rngOff: QuantumRNG = .seeded(1)
        let offResult = try engine.executeRNG(circuitOff, on: stateOff, rng: &rngOff)
        XCTAssertEqual(offResult.classicalMemory.value(ofRegister: 0), 0)
        let z1Off = try QuantumMeasurement.expectationZ(state: stateOff, engine: engine, qubit: 1)
        XCTAssertEqual(z1Off, 1, accuracy: 1e-5)
    }

    func testMeasurementWritesExplicitClassicalRegisterBits() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(
            qubitCount: 2,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 4)]
        )
        try circuit.x(1)
        try circuit.measure(qubits: [1], classicalRegister: 0, classicalBitOffset: 2)

        var rng: QuantumRNG = .seeded(0)
        let result = try engine.executeRNG(circuit, on: state, rng: &rng)
        XCTAssertEqual(result.classicalMemory.value(ofRegister: 0), 1 << 2)
    }

    func testStateVectorAmplitudeInitialization() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let state = try StateVector(qubitCount: 1)
        let invSqrt2 = QFloat(0.5).squareRoot()
        try state.initialize(
            amplitudes: [
                ComplexAmplitude(real: invSqrt2, imaginary: 0),
                ComplexAmplitude(real: invSqrt2, imaginary: 0),
            ]
        )

        let engine = try QuantumEngine()
        let z = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(z, 0, accuracy: 1e-5)
    }

    func testCustomUnitary1MatchesHadamard() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let invSqrt2 = QFloat(0.5).squareRoot()
        let hadamard = [
            ComplexAmplitude(real: invSqrt2, imaginary: 0),
            ComplexAmplitude(real: invSqrt2, imaginary: 0),
            ComplexAmplitude(real: invSqrt2, imaginary: 0),
            ComplexAmplitude(real: -invSqrt2, imaginary: 0),
        ]

        let engine = try QuantumEngine()
        let custom = try StateVector(qubitCount: 1)
        var customCircuit = try QuantumCircuit(qubitCount: 1)
        try customCircuit.unitary1(matrix: hadamard, target: 0)
        try engine.execute(customCircuit, on: custom)

        let reference = try StateVector(qubitCount: 1)
        var referenceCircuit = try QuantumCircuit(qubitCount: 1)
        try referenceCircuit.h(0)
        try engine.execute(referenceCircuit, on: reference)

        XCTAssertEqual(
            try QuantumMeasurement.amplitudes(state: custom),
            try QuantumMeasurement.amplitudes(state: reference)
        )
    }

    func testDensityMatrixPauliZExpectationOnBellState() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try DensityMatrixEngine()
        let density = try DensityMatrix(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try engine.execute(circuit, on: density)

        let zz = try QuantumMeasurement.expectationPauliZ(
            density: density,
            engine: engine,
            qubits: [0, 1]
        )
        XCTAssertEqual(zz, 1, accuracy: 1e-4)
    }
}
