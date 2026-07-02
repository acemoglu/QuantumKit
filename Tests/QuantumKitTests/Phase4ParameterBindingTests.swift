import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    func testParameterizedAnsatzMatchesHardcodedCircuit() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let pi = QFloat(Double.pi)
        let halfPi = pi / 2

        var parameterized = try QuantumCircuit(qubitCount: 1)
        try parameterized.rx(theta: Parameter("theta"), 0)
        try parameterized.ry(theta: Parameter("phi"), 0)

        let bound = try parameterized.bind(parameters: [
            "theta": halfPi,
            "phi": pi,
        ])

        var reference = try QuantumCircuit(qubitCount: 1)
        try reference.rx(theta: halfPi, 0)
        try reference.ry(theta: pi, 0)

        let engine = try QuantumEngine()
        let boundState = try StateVector(qubitCount: 1)
        let referenceState = try StateVector(qubitCount: 1)
        var rng: QuantumRNG = .seeded(7)
        _ = try engine.executeRNG(bound, on: boundState, rng: &rng)
        rng = .seeded(7)
        _ = try engine.executeRNG(reference, on: referenceState, rng: &rng)

        let boundAmplitudes = QuantumMeasurement.amplitudes(state: boundState)
        let referenceAmplitudes = QuantumMeasurement.amplitudes(state: referenceState)

        XCTAssertEqual(boundAmplitudes.count, referenceAmplitudes.count)
        for index in boundAmplitudes.indices {
            XCTAssertEqual(boundAmplitudes[index].real, referenceAmplitudes[index].real, accuracy: 1e-5)
            XCTAssertEqual(boundAmplitudes[index].imaginary, referenceAmplitudes[index].imaginary, accuracy: 1e-5)
        }
    }

    func testBindRejectsMissingParameter() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rz(theta: Parameter("alpha"), 0)

        XCTAssertThrowsError(try circuit.bind(parameters: [:])) { error in
            guard case ParameterBindingError.missingBinding(let name) = error else {
                return XCTFail("Expected missingBinding, got \(error)")
            }
            XCTAssertEqual(name, "alpha")
        }
    }

    func testExecutionRejectsUnboundParameterizedCircuit() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)

        let backend = try StatevectorBackend()

        XCTAssertThrowsError(try backend.run(circuit: circuit)) { error in
            guard case ParameterBindingError.circuitContainsUnboundParameters(let names) = error else {
                return XCTFail("Expected circuitContainsUnboundParameters, got \(error)")
            }
            XCTAssertEqual(names, ["theta"])
        }
    }

    func testTranspileOnceThenBindPreservesUnitary() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let pi = QFloat(Double.pi)
        let halfPi = pi / 2

        var ansatz = try QuantumCircuit(qubitCount: 1)
        try ansatz.rx(theta: Parameter("theta"), 0)
        try ansatz.ry(theta: Parameter("phi"), 0)

        let transpiled = try Transpiler.transpile(ansatz, targetBasis: .ibmEagle)
        XCTAssertTrue(transpiled.containsUnboundParameters)

        let boundTranspiled = try transpiled.bind(parameters: [
            "theta": halfPi,
            "phi": pi,
        ])
        XCTAssertFalse(boundTranspiled.containsUnboundParameters)

        var reference = try QuantumCircuit(qubitCount: 1)
        try reference.rx(theta: halfPi, 0)
        try reference.ry(theta: pi, 0)
        let boundReference = try Transpiler.transpile(reference, targetBasis: .ibmEagle)

        let engine = try QuantumEngine()
        XCTAssertTrue(
            try CircuitEquivalence.haveIdenticalAction(
                boundTranspiled,
                boundReference,
                engine: engine,
                tolerance: 1e-4
            )
        )
    }

    func testParameterBindingPassIntegratesWithPassManager() throws {
        let value = QFloat(Double.pi / 3)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rz(theta: Parameter("gamma"), 0)

        let bound = try PassManager(passes: [
            ParameterBindingPass(parameters: ["gamma": value]),
        ]).run(on: circuit)

        XCTAssertFalse(bound.containsUnboundParameters)
        guard case .rz(let theta, _) = bound.gates[0] else {
            return XCTFail("Expected rz gate")
        }
        XCTAssertEqual(try theta.requireLiteral(), value, accuracy: 1e-6)
    }
}
