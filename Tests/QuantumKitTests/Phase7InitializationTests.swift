import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    func testBellStateInitializationViaAmplitudeInjection() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let invSqrt2 = QFloat(0.5).squareRoot()
        let bellAmplitudes = [
            ComplexAmplitude(real: invSqrt2, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: invSqrt2, imaginary: 0),
        ]

        let engine = try QuantumEngine()
        let initialized = try StateVector(qubitCount: 2)
        var initCircuit = try QuantumCircuit(qubitCount: 2)
        try initCircuit.initialize(qubits: [0, 1], amplitudes: bellAmplitudes)
        try engine.execute(initCircuit, on: initialized)

        let reference = try StateVector(qubitCount: 2)
        var referenceCircuit = try QuantumCircuit(qubitCount: 2)
        try referenceCircuit.applyBellState()
        try engine.execute(referenceCircuit, on: reference)

        XCTAssertEqual(
            QuantumMeasurement.amplitudes(state: initialized),
            QuantumMeasurement.amplitudes(state: reference)
        )

        let zz = try QuantumMeasurement.expectationPauliZ(
            state: initialized,
            engine: engine,
            qubits: [0, 1]
        )
        XCTAssertEqual(zz, 1, accuracy: 1e-5)
    }

    func testCustomUnitaryHadamardMatchesNativeH() throws {
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
        try customCircuit.customUnitary(matrix: hadamard, qubits: [0])
        try engine.execute(customCircuit, on: custom)

        let reference = try StateVector(qubitCount: 1)
        var referenceCircuit = try QuantumCircuit(qubitCount: 1)
        try referenceCircuit.h(0)
        try engine.execute(referenceCircuit, on: reference)

        XCTAssertEqual(
            QuantumMeasurement.amplitudes(state: custom),
            QuantumMeasurement.amplitudes(state: reference)
        )
    }

    func testInitializeRejectsNonNormalizedAmplitudes() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        XCTAssertThrowsError(
            try circuit.initialize(
                qubits: [0],
                amplitudes: [ComplexAmplitude(real: 1, imaginary: 0)]
            )
        ) { error in
            guard case QuantumCircuitError.invalidAlgorithmParameter = error else {
                return XCTFail("Expected invalidAlgorithmParameter, got \(error)")
            }
        }
    }

    func testCustomUnitaryRejectsNonUnitaryMatrix() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        let nonUnitary = [
            ComplexAmplitude(real: 1, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
        ]
        XCTAssertThrowsError(try circuit.customUnitary(matrix: nonUnitary, qubits: [0])) { error in
            guard case QuantumCircuitError.invalidAlgorithmParameter(let reason) = error else {
                return XCTFail("Expected invalidAlgorithmParameter, got \(error)")
            }
            XCTAssertTrue(reason.contains("not unitary"))
        }
    }

    func testDensityMatrixBellStateInitialization() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let invSqrt2 = QFloat(0.5).squareRoot()
        let bellAmplitudes = [
            ComplexAmplitude(real: invSqrt2, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: invSqrt2, imaginary: 0),
        ]

        let engine = try DensityMatrixEngine()
        let density = try DensityMatrix(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.initialize(qubits: [0, 1], amplitudes: bellAmplitudes)
        try engine.execute(circuit, on: density)

        let zz = try QuantumMeasurement.expectationPauliZ(
            density: density,
            engine: engine,
            qubits: [0, 1]
        )
        XCTAssertEqual(zz, 1, accuracy: 1e-4)
    }
}
