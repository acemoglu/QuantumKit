import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    func testParameterShiftRXAnsatzMatchesAnalyticZGradient() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let theta = QFloat(Double.pi / 4)
        let expectedGradient = -sin(theta)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)

        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let backend = try StatevectorBackend()

        let result = try GradientCalculator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: backend,
            options: QuantumRunOptions(seed: 11)
        )

        XCTAssertEqual(result.expectationValue, cos(theta), accuracy: 1e-5)
        XCTAssertEqual(result.circuitEvaluations, 3)
        XCTAssertEqual(result.parameterGradients.count, 1)
        XCTAssertEqual(result.parameterGradients[0].name, "theta")
        XCTAssertEqual(result.parameterGradients[0].gradient, expectedGradient, accuracy: 1e-4)
        XCTAssertEqual(result.gradient(for: "theta") ?? 0, expectedGradient, accuracy: 1e-4)
        XCTAssertEqual(result.metadata.method, .statevector)
    }

    func testParameterShiftRuleHelper() {
        let plus: QFloat = 0.25
        let minus: QFloat = -0.25
        XCTAssertEqual(ParameterShift.gradient(plusExpectation: plus, minusExpectation: minus), 0.25, accuracy: 1e-6)
        XCTAssertEqual(ParameterShift.shift, QFloat(Double.pi / 2), accuracy: 1e-6)
    }

    func testGradientCalculatorRejectsMissingParameterBinding() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.ry(theta: Parameter("phi"), 0)

        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let backend = try StatevectorBackend()

        XCTAssertThrowsError(
            try GradientCalculator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                parameters: [:],
                backend: backend
            )
        ) { error in
            guard case GradientCalculatorError.missingParameterBinding(let name) = error else {
                return XCTFail("Expected missingParameterBinding, got \(error)")
            }
            XCTAssertEqual(name, "phi")
        }
    }

    func testGradientCalculatorRejectsFullyBoundCircuit() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: QFloat(Double.pi / 6), 0)

        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let backend = try StatevectorBackend()

        XCTAssertThrowsError(
            try GradientCalculator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                parameters: ["theta": QFloat(Double.pi / 6)],
                backend: backend
            )
        ) { error in
            guard case GradientCalculatorError.noDifferentiableParameters = error else {
                return XCTFail("Expected noDifferentiableParameters, got \(error)")
            }
        }
    }

    func testBatchedShiftEvaluationsMatchSequentialEstimator() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let theta = QFloat(Double.pi / 3)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)

        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let backend = try StatevectorBackend()
        let options = QuantumRunOptions(seed: 5)

        let batched = try GradientCalculator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: backend,
            options: options,
            gradientOptions: GradientOptions(batchSize: 2)
        )

        let sequential = try GradientCalculator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: backend,
            options: options,
            gradientOptions: GradientOptions(batchSize: 1)
        )

        XCTAssertEqual(batched.expectationValue, sequential.expectationValue, accuracy: 1e-5)
        XCTAssertEqual(
            batched.parameterGradients[0].gradient,
            sequential.parameterGradients[0].gradient,
            accuracy: 1e-5
        )
        XCTAssertEqual(batched.circuitEvaluations, 3)
    }
}
