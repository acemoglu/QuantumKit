import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Observable Jacobian / Hessian

    func testObservableJacobianRXMatchesAnalyticDZ() throws {
        let theta = QFloat(Double.pi / 4)
        // RX(θ)|0⟩: ⟨Z⟩ = cos(θ), ∂⟨Z⟩/∂θ = −sin(θ)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))

        let result = try ObservableDifferentiator().jacobian(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 11)
        )

        XCTAssertEqual(result.method, .parameterShift)
        XCTAssertEqual(result.parameterNames, ["theta"])
        XCTAssertEqual(result.circuitEvaluations, 3) // base + 2 shifts
        XCTAssertEqual(result.expectationValue, cos(theta), accuracy: 1e-10)
        XCTAssertEqual(result.values[0], -sin(theta), accuracy: 1e-8)
        XCTAssertEqual(result.value(for: "theta") ?? 0, -sin(theta), accuracy: 1e-8)
    }

    func testObservableHessianTwoParamSymmetricAndDiagonal() throws {
        let theta = QFloat(Double.pi / 5)
        let phi = QFloat(Double.pi / 7)
        // RX(θ) then RZ(φ): ⟨Z⟩ = cos(θ) independent of φ.
        // Hessian = [[−cos(θ), 0], [0, 0]] in (theta, phi) name order.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)
        try circuit.rz(theta: Parameter("phi"), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))

        let hessian = try ObservableDifferentiator().hessian(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta, "phi": phi],
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 3)
        )

        XCTAssertEqual(hessian.parameterNames, ["phi", "theta"]) // sorted
        XCTAssertEqual(hessian.circuitEvaluations, 1 + 2 * 4) // 1 + 2P² = 9
        XCTAssertEqual(hessian.expectationValue, cos(theta), accuracy: 1e-10)

        let h = hessian.matrix
        XCTAssertEqual(h.count, 2)
        // Symmetry
        XCTAssertEqual(h[0][1], h[1][0], accuracy: 1e-10)

        let iPhi = hessian.parameterNames.firstIndex(of: "phi")!
        let iTheta = hessian.parameterNames.firstIndex(of: "theta")!
        XCTAssertEqual(h[iTheta][iTheta], -cos(theta), accuracy: 1e-7)
        XCTAssertEqual(h[iPhi][iPhi], 0, accuracy: 1e-7)
        XCTAssertEqual(h[iTheta][iPhi], 0, accuracy: 1e-7)
        XCTAssertEqual(hessian.value(row: "theta", column: "theta") ?? 0, -cos(theta), accuracy: 1e-7)
    }

    func testObservableJacobianScaledRXMatchesAnalytic() throws {
        let theta = QFloat(Double.pi / 6)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta").scaled(by: 2), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))

        let jac = try ObservableDifferentiator().jacobian(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: CPUStatevectorBackend()
        )
        XCTAssertEqual(jac.values[0], -2 * sin(2 * theta), accuracy: 1e-6)

        let hess = try ObservableDifferentiator().hessian(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: CPUStatevectorBackend()
        )
        XCTAssertEqual(hess.matrix[0][0], -4 * cos(2 * theta), accuracy: 1e-5)
    }

    func testParameterShiftSecondDerivativeHelpers() {
        // f(θ)=cos(θ) at θ=0: f(±π)=−1, f(0)=1 → second deriv = ¼(−1 − 2 + −1) = −1
        XCTAssertEqual(
            ParameterShift.secondDerivative(
                plusPiExpectation: -1,
                centerExpectation: 1,
                minusPiExpectation: -1
            ),
            -1,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            ParameterShift.mixedPartial(
                plusPlus: 1,
                plusMinus: -1,
                minusPlus: -1,
                minusMinus: 1
            ),
            1,
            accuracy: 1e-12
        )
    }

    func testObservableJacobianMatchesGradientCalculatorOnCPU() throws {
        let theta = QFloat(Double.pi / 5)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta").scaled(by: 2), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let backend = CPUStatevectorBackend()
        let parameters = ["theta": theta]
        let options = QuantumRunOptions(seed: 19)

        let jac = try ObservableDifferentiator().jacobian(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: parameters,
            backend: backend,
            options: options
        )
        let grad = try GradientCalculator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: parameters,
            backend: backend,
            options: options
        )

        XCTAssertEqual(jac.expectationValue, grad.expectationValue, accuracy: 1e-10)
        XCTAssertEqual(jac.values[0], grad.gradient(for: "theta") ?? 0, accuracy: 1e-10)
        XCTAssertEqual(jac.circuitEvaluations, grad.circuitEvaluations)
    }

    func testObservableJacobianNegativeScaleMatchesAnalytic() throws {
        // Audit counterexample: RX((-2)·θ)|0⟩, ⟨Z⟩=cos(-2θ), ∂/∂θ = -s sin(sθ) with s=-2.
        // At θ=π/6 → −√3 (not +√3 from abs-based δ).
        let theta = QFloat(Double.pi / 6)
        let scale: QFloat = -2
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta").scaled(by: scale), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))

        XCTAssertEqual(
            ParameterShift.parameterShiftAmount(scale: scale),
            ParameterShift.shift / scale,
            accuracy: 1e-15
        )
        XCTAssertLessThan(ParameterShift.parameterShiftAmount(scale: scale), 0)

        let jac = try ObservableDifferentiator().jacobian(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: CPUStatevectorBackend()
        )
        let analytic = -scale * sin(scale * theta)
        XCTAssertEqual(jac.expectationValue, cos(scale * theta), accuracy: 1e-6)
        XCTAssertEqual(jac.values[0], analytic, accuracy: 1e-5)
        XCTAssertEqual(jac.values[0], QFloat(-sqrt(3.0)), accuracy: 1e-5)
        XCTAssertLessThan(jac.values[0], 0)

        let grad = try GradientCalculator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta],
            backend: CPUStatevectorBackend()
        )
        XCTAssertEqual(grad.gradient(for: "theta") ?? 0, analytic, accuracy: 1e-5)
    }

    func testObservableHessianNegativeScaleMatchesAnalytic() throws {
        // RX((-2)·θ)|0⟩: ⟨Z⟩=cos(-2θ), ∂²/∂θ² = (-2)² (−cos(-2θ)) = -4 cos(2θ)
        let theta = QFloat(Double.pi / 5)
        let scale: QFloat = -2
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta").scaled(by: scale), 0)
        try circuit.rz(theta: Parameter("phi"), 0) // spectator; Hessian block should stay 0
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let phi = QFloat(0.3)

        XCTAssertEqual(
            ParameterShift.parameterPiShiftAmount(scale: scale),
            (ParameterShift.shift * 2) / scale,
            accuracy: 1e-15
        )

        let hess = try ObservableDifferentiator().hessian(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: ["theta": theta, "phi": phi],
            backend: CPUStatevectorBackend()
        )
        let iTheta = hess.parameterNames.firstIndex(of: "theta")!
        let iPhi = hess.parameterNames.firstIndex(of: "phi")!
        let expectedDiag = scale * scale * (-cos(scale * theta))

        XCTAssertEqual(hess.matrix[iTheta][iTheta], expectedDiag, accuracy: 1e-6)
        XCTAssertEqual(hess.matrix[iPhi][iPhi], 0, accuracy: 1e-7)
        XCTAssertEqual(hess.matrix[iTheta][iPhi], hess.matrix[iPhi][iTheta], accuracy: 1e-10)
        XCTAssertEqual(hess.matrix[iTheta][iPhi], 0, accuracy: 1e-7)
        XCTAssertEqual(hess.expectationValue, cos(scale * theta), accuracy: 1e-10)
    }

    func testObservableJacobianAdjointMatchesParameterShiftWhenMetalAvailable() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        let theta = QFloat(Double.pi / 5)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1.0, label: "Z0"))
        let backend = try StatevectorBackend()
        let parameters = ["theta": theta]
        let options = QuantumRunOptions(seed: 8)

        let shift = try ObservableDifferentiator().jacobian(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: parameters,
            backend: backend,
            options: options,
            differentiatorOptions: ObservableDifferentiatorOptions(jacobianMethod: .parameterShift)
        )
        let adjoint = try ObservableDifferentiator().jacobian(
            circuit: circuit,
            hamiltonian: hamiltonian,
            parameters: parameters,
            backend: backend,
            options: options,
            differentiatorOptions: ObservableDifferentiatorOptions(jacobianMethod: .adjoint)
        )

        XCTAssertEqual(adjoint.method, .adjoint)
        XCTAssertEqual(adjoint.circuitEvaluations, 1)
        XCTAssertEqual(shift.values[0], adjoint.values[0], accuracy: 1e-4)
        XCTAssertEqual(shift.expectationValue, adjoint.expectationValue, accuracy: 1e-5)
        XCTAssertEqual(adjoint.values[0], -sin(theta), accuracy: 1e-4)
    }
}
