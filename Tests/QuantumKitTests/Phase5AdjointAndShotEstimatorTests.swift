import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Shot-based Estimator (D2 / E5)

    func testShotEstimatorBellZZApproachesExact() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0 Z1"))
        let backend = try StatevectorBackend()

        let exact = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 1)
        )
        XCTAssertEqual(exact.value, 1.0, accuracy: 1e-5)
        XCTAssertNil(exact.shots)

        let sampled = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 42),
            estimatorOptions: EstimatorOptions(shots: 2048)
        )

        XCTAssertEqual(sampled.shots, 2048)
        XCTAssertNotNil(sampled.standardError)
        XCTAssertEqual(sampled.value, 1.0, accuracy: 0.05)
    }

    func testEstimatorPrecisionResolvesToShots() throws {
        let options = EstimatorOptions(precision: 0.1)
        XCTAssertEqual(try options.resolvedShots(), 100)
    }

    func testShotEstimatorRejectsDensityMatrixBackend() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let backend = try DensityMatrixBackend()

        XCTAssertThrowsError(
            try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: backend,
                estimatorOptions: EstimatorOptions(shots: 64)
            )
        ) { error in
            XCTAssertEqual(error as? EstimatorError, .shotsNotSupportedForDensityMatrix)
        }
    }

    // MARK: - Adjoint differentiation (E4)

    func testAdjointRXAnsatzMatchesAnalyticZGradient() throws {
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
            options: QuantumRunOptions(seed: 11),
            gradientOptions: GradientOptions(method: .adjoint)
        )

        XCTAssertEqual(result.expectationValue, cos(theta), accuracy: 1e-5)
        XCTAssertEqual(result.circuitEvaluations, 1)
        XCTAssertEqual(result.parameterGradients[0].gradient, expectedGradient, accuracy: 1e-4)
    }

    func testAdjointMatchesParameterShiftOnRX() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let theta = QFloat(Double.pi / 3)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)

        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let backend = try StatevectorBackend()
        let options = QuantumRunOptions(seed: 5)

        let shift = try GradientCalculator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: backend,
            options: options,
            gradientOptions: GradientOptions(method: .parameterShift)
        )
        let adjoint = try GradientCalculator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: backend,
            options: options,
            gradientOptions: GradientOptions(method: .adjoint)
        )

        XCTAssertEqual(shift.expectationValue, adjoint.expectationValue, accuracy: 1e-5)
        XCTAssertEqual(
            shift.parameterGradients[0].gradient,
            adjoint.parameterGradients[0].gradient,
            accuracy: 1e-4
        )
    }

    func testAdjointRejectsDensityMatrixBackend() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.ry(theta: Parameter("phi"), 0)

        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let backend = try DensityMatrixBackend()

        XCTAssertThrowsError(
            try GradientCalculator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                parameters: ["phi": 0.1],
                backend: backend,
                gradientOptions: GradientOptions(method: .adjoint)
            )
        ) { error in
            XCTAssertEqual(error as? GradientCalculatorError, .adjointRequiresStatevectorBackend)
        }
    }
}
