import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - G10 lite: bounded classical while (`while_c`)

    /// while creg==0: X then measure → writes 1 and exits after one iteration.
    func testWhileCFlipsClassicalBitThenExitsCPU() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.while_c(
            classicalRegister: 0,
            equals: 0,
            maxIterations: 8,
            body: [
                .x(target: 0),
                .measure(MeasureSpec(qubits: [0], classicalRegister: 0, classicalBitOffset: 0)),
            ]
        )

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 1)
        var rng: QuantumRNG = .seeded(42)
        let result = try engine.executeRNG(circuit, on: state, rng: &rng)

        XCTAssertEqual(result.classicalMemory.value(ofRegister: 0), 1)
        XCTAssertEqual(result.measurementOutcomes.count, 1)
        XCTAssertEqual(result.measurementOutcomes[0], [1])
        // |1⟩ after X + measure collapse
        let probs = state.probabilities()
        XCTAssertEqual(probs[1], 1, accuracy: 1e-6)
        XCTAssertTrue(ShotExecutionPolicy.mustSerial(circuit: circuit))
        XCTAssertFalse(circuit.isUnitaryOnly)
    }

    /// Body never updates the classical register → hard cap throws a typed error.
    func testWhileCMaxIterationsExceededCPU() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.while_c(
            classicalRegister: 0,
            equals: 0,
            maxIterations: 3,
            body: [.x(target: 0)]
        )

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 1)
        var rng: QuantumRNG = .seeded(7)

        XCTAssertThrowsError(try engine.executeRNG(circuit, on: state, rng: &rng)) { error in
            guard case QuantumCircuitError.maxLoopIterationsExceeded(let maxIterations) = error else {
                return XCTFail("expected maxLoopIterationsExceeded, got \(error)")
            }
            XCTAssertEqual(maxIterations, 3)
        }
    }

    /// Construction rejects unbounded / non-positive caps.
    func testWhileCRejectsNonPositiveMaxIterations() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        XCTAssertThrowsError(
            try circuit.while_c(
                classicalRegister: 0,
                equals: 0,
                maxIterations: 0,
                body: [.x(target: 0)]
            )
        ) { error in
            guard case QuantumCircuitError.invalidAlgorithmParameter(let reason) = error else {
                return XCTFail("expected invalidAlgorithmParameter, got \(error)")
            }
            XCTAssertTrue(reason.contains("maxIterations"))
        }
    }

    /// GateSequence body helper inlines gates into the nested `while_c` body (not flat IR).
    func testWhileCAcceptsGateSequenceBody() throws {
        var sequence = try GateSequence(
            name: "flip_measure",
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try sequence.apply(.x(target: 0))
        try sequence.apply(
            .measure(MeasureSpec(qubits: [0], classicalRegister: 0, classicalBitOffset: 0))
        )

        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.while_c(
            classicalRegister: 0,
            equals: 0,
            maxIterations: 4,
            sequence: sequence
        )

        XCTAssertEqual(circuit.gates.count, 1)
        guard case .while_c(_, _, let body, let maxIterations) = circuit.gates[0] else {
            return XCTFail("expected while_c")
        }
        XCTAssertEqual(body.count, 2)
        XCTAssertEqual(maxIterations, 4)

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 1)
        var rng: QuantumRNG = .seeded(1)
        let result = try engine.executeRNG(circuit, on: state, rng: &rng)
        XCTAssertEqual(result.classicalMemory.value(ofRegister: 0), 1)
    }

    /// Schema v1: `while_c` is in-memory only (not serialized yet).
    func testWhileCNotSerializedUnderSchemaV1() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.while_c(
            classicalRegister: 0,
            equals: 0,
            maxIterations: 2,
            body: [.x(target: 0)]
        )

        XCTAssertEqual(CircuitIRSchema.current, 1)
        XCTAssertThrowsError(try JSONEncoder().encode(circuit)) { error in
            guard case CircuitIRError.controlFlowNotSerialized(let op) = error else {
                return XCTFail("expected controlFlowNotSerialized, got \(error)")
            }
            XCTAssertEqual(op, "while_c")
        }
    }

    /// Static circuits without `while_c` still round-trip under schema v1.
    func testStaticCircuitSerializationUnchanged() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.x(0)
        try circuit.measure(qubits: [0], classicalRegister: 0, classicalBitOffset: 0)
        try circuit.c_if(classicalRegister: 0, equals: 1, x: 0)

        let data = try JSONEncoder().encode(circuit)
        let decoded = try JSONDecoder().decode(QuantumCircuit.self, from: data)
        XCTAssertEqual(decoded, circuit)
    }

    /// Metal host path: same flip-then-exit semantics as CPU (seeded).
    func testWhileCFlipsClassicalBitThenExitsMetalMatchesCPU() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.while_c(
            classicalRegister: 0,
            equals: 0,
            maxIterations: 8,
            body: [
                .x(target: 0),
                .measure(MeasureSpec(qubits: [0], classicalRegister: 0, classicalBitOffset: 0)),
            ]
        )

        let cpuEngine = CPUStatevectorEngine()
        let cpuState = try CPUStateVector(qubitCount: 1)
        var cpuRNG: QuantumRNG = .seeded(42)
        let cpuResult = try cpuEngine.executeRNG(circuit, on: cpuState, rng: &cpuRNG)

        let metalEngine = try QuantumEngine()
        let metalState = try StateVector(qubitCount: 1)
        var metalRNG: QuantumRNG = .seeded(42)
        let metalResult = try metalEngine.executeRNG(circuit, on: metalState, rng: &metalRNG)

        XCTAssertEqual(cpuResult.classicalMemory.value(ofRegister: 0), 1)
        XCTAssertEqual(metalResult.classicalMemory.value(ofRegister: 0), 1)
        XCTAssertEqual(cpuResult.measurementOutcomes, metalResult.measurementOutcomes)
        let metalProbs = try QuantumMeasurement.probabilities(state: metalState, engine: metalEngine)
        XCTAssertEqual(metalProbs[1], 1, accuracy: 1e-5)
    }

    /// Metal host path: max-iteration cap throws the same typed error as CPU.
    func testWhileCMaxIterationsExceededMetal() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try circuit.while_c(
            classicalRegister: 0,
            equals: 0,
            maxIterations: 3,
            body: [.x(target: 0)]
        )

        let engine = try QuantumEngine()
        let state = try StateVector(qubitCount: 1)
        var rng: QuantumRNG = .seeded(7)

        XCTAssertThrowsError(try engine.executeRNG(circuit, on: state, rng: &rng)) { error in
            guard case QuantumCircuitError.maxLoopIterationsExceeded(let maxIterations) = error else {
                return XCTFail("expected maxLoopIterationsExceeded, got \(error)")
            }
            XCTAssertEqual(maxIterations, 3)
        }
    }

    /// `c_if` inside `while_c` body shares classical memory (seeded CPU).
    func testWhileCBodyMayContainCIf() throws {
        var circuit = try QuantumCircuit(
            qubitCount: 2,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        // while creg==0: X q0, measure→creg; if creg==1 then X q1 (runs once then exits).
        try circuit.while_c(
            classicalRegister: 0,
            equals: 0,
            maxIterations: 4,
            body: [
                .x(target: 0),
                .measure(MeasureSpec(qubits: [0], classicalRegister: 0, classicalBitOffset: 0)),
                .c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 1)),
            ]
        )

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 2)
        var rng: QuantumRNG = .seeded(11)
        let result = try engine.executeRNG(circuit, on: state, rng: &rng)

        XCTAssertEqual(result.classicalMemory.value(ofRegister: 0), 1)
        let probs = state.probabilities()
        // |q1 q0⟩ = |1 1⟩ → basis 3
        XCTAssertEqual(probs[3], 1, accuracy: 1e-6)
    }
}
