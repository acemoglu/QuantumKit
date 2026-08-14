import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Circuit composition (append / tensor / control)

    func testComposeAppendMatchesHandBuiltGateListCPUAndMetal() throws {
        var partA = try QuantumCircuit(qubitCount: 2)
        try partA.h(0)

        var partB = try QuantumCircuit(qubitCount: 2)
        try partB.cx(0, 1)

        let composed = try partA.compose(partB)

        var handBuilt = try QuantumCircuit(qubitCount: 2)
        try handBuilt.h(0)
        try handBuilt.cx(0, 1)

        XCTAssertEqual(composed.gates, handBuilt.gates)
        XCTAssertEqual(composed.qubitCount, 2)

        let cpuEngine = CPUStatevectorEngine()
        let composedState = try CPUStateVector(qubitCount: 2)
        let handState = try CPUStateVector(qubitCount: 2)
        _ = try cpuEngine.execute(composed, on: composedState)
        _ = try cpuEngine.execute(handBuilt, on: handState)
        let composedProbs = composedState.probabilities()
        let handProbs = handState.probabilities()
        for index in 0..<4 {
            XCTAssertEqual(composedProbs[index], handProbs[index], accuracy: 1e-6)
        }

        if MetalRuntime.isAvailable {
            let metalEngine = try QuantumEngine()
            let metalComposed = try StateVector(qubitCount: 2, device: metalEngine.device)
            let metalHand = try StateVector(qubitCount: 2, device: metalEngine.device)
            _ = try metalEngine.execute(composed, on: metalComposed)
            _ = try metalEngine.execute(handBuilt, on: metalHand)
            let mComposed = try QuantumMeasurement.probabilities(state: metalComposed, engine: metalEngine)
            let mHand = try QuantumMeasurement.probabilities(state: metalHand, engine: metalEngine)
            for index in 0..<4 {
                XCTAssertEqual(mComposed[index], mHand[index], accuracy: 1e-4)
            }
        }
    }

    func testAppendPreservesInstructionMetadata() throws {
        var a = try QuantumCircuit(qubitCount: 1)
        try a.apply(.h(target: 0), metadata: InstructionMetadata(label: "hadamard"))
        var b = try QuantumCircuit(qubitCount: 1)
        try b.apply(.x(target: 0), metadata: InstructionMetadata(label: "pauli-x"))

        try a.append(b)
        XCTAssertEqual(a.gates.count, 2)
        XCTAssertEqual(a.metadata(at: 0)?.label, "hadamard")
        XCTAssertEqual(a.metadata(at: 1)?.label, "pauli-x")
    }

    func testTensorTwoOneQubitCircuitsProductState() throws {
        var left = try QuantumCircuit(qubitCount: 1)
        try left.h(0)
        var right = try QuantumCircuit(qubitCount: 1)
        try right.x(0)

        let product = try left.tensor(right)
        XCTAssertEqual(product.qubitCount, 2)
        XCTAssertEqual(product.gates, [.h(target: 0), .x(target: 1)])

        // |+⟩ ⊗ |1⟩ → amplitudes on |01⟩ and |11⟩ (little-endian: bit0 = q0).
        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 2)
        _ = try engine.execute(product, on: state)
        let probs = state.probabilities()
        // Qubit 0 = LSB: H(q0)⊗X(q1) → (|10⟩+|11⟩)/√2 → indices 2 and 3.
        XCTAssertEqual(probs[0], 0, accuracy: 1e-6)
        XCTAssertEqual(probs[1], 0, accuracy: 1e-6)
        XCTAssertEqual(probs[2], 0.5, accuracy: 1e-6)
        XCTAssertEqual(probs[3], 0.5, accuracy: 1e-6)
    }

    func testControlledXCircuitEqualsCX() throws {
        var xCircuit = try QuantumCircuit(qubitCount: 1)
        try xCircuit.x(0)

        let controlled = try xCircuit.controlled(controlCount: 1)
        XCTAssertEqual(controlled.qubitCount, 2)
        XCTAssertEqual(controlled.gates, [.cx(control: 0, target: 1)])

        var handCX = try QuantumCircuit(qubitCount: 2)
        try handCX.cx(0, 1)

        let engine = CPUStatevectorEngine()
        // Prepare |11⟩ input via X⊗X then apply controlled-X vs CX — compare final probs from |10⟩.
        var prep = try QuantumCircuit(qubitCount: 2)
        try prep.x(0) // control |1⟩
        // target starts |0⟩; CX should flip target → |11⟩

        let fromControlled = try prep.compose(controlled)
        let fromHand = try prep.compose(handCX)

        let s1 = try CPUStateVector(qubitCount: 2)
        let s2 = try CPUStateVector(qubitCount: 2)
        _ = try engine.execute(fromControlled, on: s1)
        _ = try engine.execute(fromHand, on: s2)
        let p1 = s1.probabilities()
        let p2 = s2.probabilities()
        for index in 0..<4 {
            XCTAssertEqual(p1[index], p2[index], accuracy: 1e-6)
        }
        XCTAssertEqual(p1[3], 1.0, accuracy: 1e-6) // |11⟩
    }

    func testControlledUnsupportedOpsThrow() throws {
        var measured = try QuantumCircuit(qubitCount: 1, classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)])
        try measured.measure(qubits: [0], classicalRegister: 0, classicalBitOffset: 0)
        XCTAssertThrowsError(try measured.controlled()) { error in
            guard case QuantumCircuitError.unsupportedControlledGate = error else {
                return XCTFail("expected unsupportedControlledGate, got \(error)")
            }
        }

        var resetCircuit = try QuantumCircuit(qubitCount: 1)
        try resetCircuit.apply(.reset(qubit: 0))
        XCTAssertThrowsError(try resetCircuit.controlled()) { error in
            guard case QuantumCircuitError.unsupportedControlledGate = error else {
                return XCTFail("expected unsupportedControlledGate, got \(error)")
            }
        }

        var initCircuit = try QuantumCircuit(qubitCount: 1)
        try initCircuit.apply(
            .initialize(
                qubits: [0],
                amplitudes: [
                    ComplexAmplitude(real: 1, imaginary: 0),
                    ComplexAmplitude(real: 0, imaginary: 0),
                ]
            )
        )
        XCTAssertThrowsError(try initCircuit.controlled()) { error in
            guard case QuantumCircuitError.unsupportedControlledGate = error else {
                return XCTFail("expected unsupportedControlledGate, got \(error)")
            }
        }

        var cif = try QuantumCircuit(qubitCount: 1, classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)])
        try cif.apply(.c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 0)))
        XCTAssertThrowsError(try cif.controlled()) { error in
            guard case QuantumCircuitError.unsupportedControlledGate = error else {
                return XCTFail("expected unsupportedControlledGate, got \(error)")
            }
        }
    }

    func testControlledControlZeroLeavesTarget() throws {
        var xCircuit = try QuantumCircuit(qubitCount: 1)
        try xCircuit.x(0)
        let controlled = try xCircuit.controlled(controlCount: 1)

        var prep = try QuantumCircuit(qubitCount: 2)
        try prep.x(1) // target |1⟩, control |0⟩
        let circuit = try prep.compose(controlled)

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 2)
        _ = try engine.execute(circuit, on: state)
        let probs = state.probabilities()
        // |10⟩ (q1=1, q0=0) must stay; CX must not fire.
        XCTAssertEqual(probs[2], 1.0, accuracy: 1e-6)
        XCTAssertEqual(probs[0], 0, accuracy: 1e-6)
        XCTAssertEqual(probs[1], 0, accuracy: 1e-6)
        XCTAssertEqual(probs[3], 0, accuracy: 1e-6)
    }

    func testControlledHAndCustomUnitaryThrow() throws {
        var hadamard = try QuantumCircuit(qubitCount: 1)
        try hadamard.h(0)
        XCTAssertThrowsError(try hadamard.controlled()) { error in
            guard case QuantumCircuitError.unsupportedControlledGate = error else {
                return XCTFail("expected unsupportedControlledGate, got \(error)")
            }
        }

        var custom = try QuantumCircuit(qubitCount: 1)
        try custom.customUnitary(
            matrix: [
                ComplexAmplitude(real: 1, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: 1, imaginary: 0),
            ],
            qubits: [0]
        )
        XCTAssertThrowsError(try custom.controlled()) { error in
            guard case QuantumCircuitError.unsupportedControlledGate = error else {
                return XCTFail("expected unsupportedControlledGate, got \(error)")
            }
        }
    }

    func testAppendClassicalRegisterMapMeasureCIfExecuteParity() throws {
        let destRegs = [try ClassicalRegisterSpec(bitCount: 1), try ClassicalRegisterSpec(bitCount: 1)]
        var dest = try QuantumCircuit(qubitCount: 2, classicalRegisters: destRegs)
        try dest.x(1)

        var block = try QuantumCircuit(qubitCount: 1, classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)])
        try block.measure(qubits: [0], classicalRegister: 0, classicalBitOffset: 0)
        try block.apply(.c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 0)))

        try dest.append(block, qubitMap: [1], classicalRegisterMap: [1])

        var hand = try QuantumCircuit(qubitCount: 2, classicalRegisters: destRegs)
        try hand.x(1)
        try hand.measure(qubits: [1], classicalRegister: 1, classicalBitOffset: 0)
        try hand.apply(.c_if(classicalRegister: 1, expectedValue: 1, gate: .x(target: 1)))

        XCTAssertEqual(dest.gates, hand.gates)
        XCTAssertEqual(dest.classicalRegisters.count, 2)

        let engine = CPUStatevectorEngine()
        let composedState = try CPUStateVector(qubitCount: 2)
        let handState = try CPUStateVector(qubitCount: 2)
        let composedRun = try engine.execute(dest, on: composedState)
        let handRun = try engine.execute(hand, on: handState)

        let p1 = composedState.probabilities()
        let p2 = handState.probabilities()
        for index in 0..<4 {
            XCTAssertEqual(p1[index], p2[index], accuracy: 1e-6)
        }
        // x(1) → measure 1 → c_if flips q1 back to |00⟩.
        XCTAssertEqual(p1[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(composedRun.classicalMemory, handRun.classicalMemory)
        XCTAssertEqual(composedRun.classicalMemory.value(ofRegister: 1), 1)
    }
}
