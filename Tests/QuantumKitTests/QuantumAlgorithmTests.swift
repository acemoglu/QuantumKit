import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Task 3: Deutsch-Jozsa

    func testDeutschJozsaConstantOracleYieldsAllZeros() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let inputs = [0, 1, 2]
        let ancilla = 3
        let state = try StateVector(qubitCount: 4)
        var circuit = try QuantumCircuit(qubitCount: 4)

        // Constant f(x) = 0: the oracle does nothing at all.
        try circuit.applyDeutschJozsa(inputQubits: inputs, ancilla: ancilla) { _ in }
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: inputs)
        XCTAssertEqual(marginal[0], 1, accuracy: 1e-5, "Constant function must collapse inputs to all-zeros")
    }

    func testDeutschJozsaConstantOneOracleYieldsAllZeros() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let inputs = [0, 1, 2]
        let ancilla = 3
        let state = try StateVector(qubitCount: 4)
        var circuit = try QuantumCircuit(qubitCount: 4)

        // Constant f(x) = 1: flip the ancilla unconditionally (global phase only).
        try circuit.applyDeutschJozsa(inputQubits: inputs, ancilla: ancilla) { c in
            try c.x(ancilla)
        }
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: inputs)
        XCTAssertEqual(marginal[0], 1, accuracy: 1e-5, "Constant function must collapse inputs to all-zeros")
    }

    func testDeutschJozsaBalancedOracleYieldsNonZero() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let inputs = [0, 1, 2]
        let ancilla = 3
        let state = try StateVector(qubitCount: 4)
        var circuit = try QuantumCircuit(qubitCount: 4)

        // Balanced f(x) = parity(x): CNOT each input into the ancilla.
        try circuit.applyDeutschJozsa(inputQubits: inputs, ancilla: ancilla) { c in
            for input in inputs {
                try c.cx(input, ancilla)
            }
        }
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: inputs)
        // Parity oracle collapses every input qubit to |1⟩ → outcome 0b111 = 7.
        XCTAssertEqual(marginal[7], 1, accuracy: 1e-5, "Balanced parity function must yield all-ones")
        XCTAssertEqual(marginal[0], 0, accuracy: 1e-5, "Balanced function must not yield all-zeros")
    }

    // MARK: - Task 3: Grover

    func testGroverOptimalIterations() {
        XCTAssertEqual(QuantumCircuit.groverOptimalIterations(searchSpaceSize: 8, markedCount: 1), 2)
        XCTAssertEqual(QuantumCircuit.groverOptimalIterations(searchSpaceSize: 16, markedCount: 1), 3)
        XCTAssertEqual(QuantumCircuit.groverOptimalIterations(searchSpaceSize: 4, markedCount: 1), 1)
    }

    func testGroverFindsMarkedState() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let qubits = [0, 1, 2]
        let target = 5 // 0b101: qubit0 = 1, qubit1 = 0, qubit2 = 1
        let state = try StateVector(qubitCount: 3)
        var circuit = try QuantumCircuit(qubitCount: 3)

        // Phase oracle marking |101⟩: open-control with X on the zero bits, MCZ, then undo the X's.
        let markOracle: (inout QuantumCircuit) throws -> Void = { c in
            try c.x(1)
            try c.mcz(controls: [0, 1], target: 2)
            try c.x(1)
        }

        let iterations = QuantumCircuit.groverOptimalIterations(searchSpaceSize: 8, markedCount: 1)
        try circuit.applyGrover(qubits: qubits, iterations: iterations, oracle: markOracle)
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: qubits)
        let peak = marginal.enumerated().max(by: { $0.element < $1.element })!
        XCTAssertEqual(peak.offset, target, "Grover should amplify the marked state |101⟩ = 5")
        XCTAssertGreaterThan(marginal[target], 0.9, "Marked state should dominate the distribution")
    }

    // MARK: - Task 3: Quantum Phase Estimation

    func testPhaseEstimationRecoversTGatePhase() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // 3 counting qubits [0,1,2]; eigenstate qubit 3 prepared in |1⟩, eigenvalue of T = e^{2πi·(1/8)}.
        let counting = [0, 1, 2]
        let eigenstate = 3
        let state = try StateVector(qubitCount: 4)
        var circuit = try QuantumCircuit(qubitCount: 4)

        try circuit.x(eigenstate) // |1⟩ eigenstate of T

        // Controlled-T^power on the eigenstate qubit ≡ controlled phase of (π/4)·power on |1⟩.
        let controlledPower: (inout QuantumCircuit, Int, Int) throws -> Void = { c, control, power in
            let theta = QFloat(Double.pi / 4.0 * Double(power))
            try c.cp(theta: theta, control: control, target: eigenstate)
        }

        try circuit.applyPhaseEstimation(countingRegister: counting, controlledUnitaryPower: controlledPower)
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: counting)
        // φ = 1/8 with t = 3 → exact estimate m = 1.
        XCTAssertEqual(marginal[1], 1, accuracy: 1e-4, "QPE should read m = 1 (phase 1/8) deterministically")
    }
}
