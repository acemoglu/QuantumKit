import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    func testBellStateEntanglement() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }
        let state = try StateVector(qubitCount: 2)

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🚀 QUANTUM COLLAPSE RESULT: \(result)")

        let isZeroZero = (result[0] == 0 && result[1] == 0)
        let isOneOne = (result[0] == 1 && result[1] == 1)

        XCTAssertTrue(isZeroZero || isOneOne, "Entanglement broken! Collapsed into an impossible state: \(result)")
    }

    func testQuantumFourierTransform() throws {
        let qubitCount = 3
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }
        let state = try StateVector(qubitCount: qubitCount)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)

        try circuit.applyQFT()

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🌊 QFT COLLAPSE RESULT (3 Qubit): \(result)")

        XCTAssertEqual(result.count, qubitCount)
    }

    func testCCXGate() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 3)
        var circuit = try QuantumCircuit(qubitCount: 3)

        try circuit.x(0)
        try circuit.x(1)
        try circuit.ccx(0, 1, 2)

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🔺 CCX COLLAPSE RESULT: \(result)")

        XCTAssertEqual(result, [1, 1, 1], "CCX should flip target when both controls are |1>")
    }

    func testSwapGate() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)

        try circuit.x(0)
        try circuit.applySwap(q1: 0, q2: 1)

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🔄 SWAP COLLAPSE RESULT: \(result)")

        XCTAssertEqual(result, [1, 0], "SWAP should exchange qubit amplitudes")
    }

    func testRyPiRotatesZeroToOne() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.ry(theta: QFloat(Double.pi), 0)

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)
        XCTAssertEqual(result, [1], "RY(pi) should rotate |0> to |1>")
    }

    func testSGateMatchesRzPiOverTwo() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let sState = try StateVector(qubitCount: 1)
        var sCircuit = try QuantumCircuit(qubitCount: 1)
        try sCircuit.h(0)
        try sCircuit.s(0)
        try engine.execute(sCircuit, on: sState)

        let rzState = try StateVector(qubitCount: 1)
        var rzCircuit = try QuantumCircuit(qubitCount: 1)
        try rzCircuit.h(0)
        try rzCircuit.rz(theta: QFloat(Double.pi / 2.0), 0)
        try engine.execute(rzCircuit, on: rzState)

        let sProbabilities = try QuantumMeasurement.probabilities(state: sState, engine: engine)
        let rzProbabilities = try QuantumMeasurement.probabilities(state: rzState, engine: engine)

        XCTAssertEqual(sProbabilities.count, rzProbabilities.count)
        for index in 0..<sProbabilities.count {
            XCTAssertEqual(sProbabilities[index], rzProbabilities[index], accuracy: 1e-5)
        }
    }

    func testTGateMatchesRzPiOverFour() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let tState = try StateVector(qubitCount: 1)
        var tCircuit = try QuantumCircuit(qubitCount: 1)
        try tCircuit.h(0)
        try tCircuit.t(0)
        try engine.execute(tCircuit, on: tState)

        let rzState = try StateVector(qubitCount: 1)
        var rzCircuit = try QuantumCircuit(qubitCount: 1)
        try rzCircuit.h(0)
        try rzCircuit.rz(theta: QFloat(Double.pi / 4.0), 0)
        try engine.execute(rzCircuit, on: rzState)

        let tProbabilities = try QuantumMeasurement.probabilities(state: tState, engine: engine)
        let rzProbabilities = try QuantumMeasurement.probabilities(state: rzState, engine: engine)

        XCTAssertEqual(tProbabilities.count, rzProbabilities.count)
        for index in 0..<tProbabilities.count {
            XCTAssertEqual(tProbabilities[index], rzProbabilities[index], accuracy: 1e-5)
        }
    }

    func testTSquaredEqualsS() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let ttState = try StateVector(qubitCount: 1)
        var ttCircuit = try QuantumCircuit(qubitCount: 1)
        try ttCircuit.h(0)
        try ttCircuit.t(0)
        try ttCircuit.t(0)
        try engine.execute(ttCircuit, on: ttState)

        let sState = try StateVector(qubitCount: 1)
        var sCircuit = try QuantumCircuit(qubitCount: 1)
        try sCircuit.h(0)
        try sCircuit.s(0)
        try engine.execute(sCircuit, on: sState)

        let ttProbabilities = try QuantumMeasurement.probabilities(state: ttState, engine: engine)
        let sProbabilities = try QuantumMeasurement.probabilities(state: sState, engine: engine)

        XCTAssertEqual(ttProbabilities.count, sProbabilities.count)
        for index in 0..<ttProbabilities.count {
            XCTAssertEqual(ttProbabilities[index], sProbabilities[index], accuracy: 1e-5)
        }
    }

    // MARK: - Extended gate set (Wave A)

    func testSDaggerInvertsS() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.s(0)
        try circuit.sdg(0)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5, "S·S† = I should return |0⟩")
    }

    func testTDaggerInvertsT() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.t(0)
        try circuit.tdg(0)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5, "T·T† = I should return |0⟩")
    }

    func testSXSquaredEqualsX() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.sx(0)
        try circuit.sx(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5, "SX·SX = X should map |0⟩ → |1⟩")
    }

    func testSingleSXProducesEqualSuperposition() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.sx(0)
        try engine.execute(circuit, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probabilities[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 0.5, accuracy: 1e-5)
    }

    func testPhaseGatePiActsAsZ() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.p(theta: QFloat(Double.pi), 0)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5, "H·P(π)·H = X should map |0⟩ → |1⟩")
    }

    func testControlledZActsAsZWhenControlIsOne() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)            // control = |1⟩
        try circuit.h(1)            // target = |+⟩
        try circuit.cz(0, 1)        // acts as Z on target
        try circuit.h(1)            // H·Z·H = X → target = |1⟩
        try engine.execute(circuit, on: state)

        let z0 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z0, -1, accuracy: 1e-5)
        XCTAssertEqual(z1, -1, accuracy: 1e-5)
    }

    func testControlledZLeavesTargetWhenControlIsZero() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(1)            // target = |+⟩, control stays |0⟩
        try circuit.cz(0, 1)        // no-op since control = |0⟩
        try circuit.h(1)            // H·I·H = I → target back to |0⟩
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, 1, accuracy: 1e-5)
    }

    func testNativeSwapExchangesQubits() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)            // qubit0 = |1⟩, qubit1 = |0⟩
        try circuit.swap(0, 1)      // → qubit0 = |0⟩, qubit1 = |1⟩
        try engine.execute(circuit, on: state)

        let z0 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z0, 1, accuracy: 1e-5)
        XCTAssertEqual(z1, -1, accuracy: 1e-5)
    }

    func testUniversalGateReproducesPauliX() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.u(theta: QFloat(Double.pi), phi: 0, lambda: QFloat(Double.pi), 0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5, "U(π,0,π) = X should map |0⟩ → |1⟩")
    }

    func testUniversalGateReproducesHadamard() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.u(theta: QFloat(Double.pi / 2), phi: 0, lambda: QFloat(Double.pi), 0)
        try engine.execute(circuit, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probabilities[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 0.5, accuracy: 1e-5)

        let expectationX = try QuantumMeasurement.expectationX(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectationX, 1, accuracy: 1e-5, "U(π/2,0,π) = H should produce |+⟩")
    }

    func testUniversalGateMatchesPhaseGate() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        // U(0,0,λ) = P(λ): H · U(0,0,π) · H should behave like H · Z · H = X.
        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.u(theta: 0, phi: 0, lambda: QFloat(Double.pi), 0)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5, "U(0,0,π) = P(π) = Z")
    }

    // MARK: - Extended gate set (Wave B: controlled & multi-controlled)

    func testControlledRYRotatesTargetWhenControlIsOne() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.cry(theta: QFloat(Double.pi), control: 0, target: 1)
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, -1, accuracy: 1e-5, "CRY(π) with control |1⟩ should map target |0⟩ → |1⟩")
    }

    func testControlledRYLeavesTargetWhenControlIsZero() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.cry(theta: QFloat(Double.pi), control: 0, target: 1)
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, 1, accuracy: 1e-5, "CRY with control |0⟩ should leave the target untouched")
    }

    func testControlledRXExcitesTargetWhenControlIsOne() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.crx(theta: QFloat(Double.pi), control: 0, target: 1)
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, -1, accuracy: 1e-5, "CRX(π) with control |1⟩ should fully excite the target")
    }

    func testControlledRZActsAsZWhenControlIsOne() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.h(1)
        try circuit.crz(theta: QFloat(Double.pi), control: 0, target: 1)
        try circuit.h(1)
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, -1, accuracy: 1e-5, "H·CRZ(π)·H with control |1⟩ should flip the target")
    }

    func testControlledPhasePiMatchesControlledZ() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.h(1)
        try circuit.cp(theta: QFloat(Double.pi), control: 0, target: 1)
        try circuit.h(1)
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, -1, accuracy: 1e-5, "CP(π) should behave like CZ")
    }

    func testMultiControlledXFlipsTargetWhenAllControlsAreOne() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 4)
        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.x(0)
        try circuit.x(1)
        try circuit.x(2)
        try circuit.mcx(controls: [0, 1, 2], target: 3)
        try engine.execute(circuit, on: state)

        let z3 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 3)
        XCTAssertEqual(z3, -1, accuracy: 1e-5, "MCX should flip the target when all controls are |1⟩")
    }

    func testMultiControlledXLeavesTargetWhenAControlIsZero() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 4)
        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.x(0)
        try circuit.x(1)
        // qubit 2 stays |0⟩
        try circuit.mcx(controls: [0, 1, 2], target: 3)
        try engine.execute(circuit, on: state)

        let z3 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 3)
        XCTAssertEqual(z3, 1, accuracy: 1e-5, "MCX must not flip the target when a control is |0⟩")
    }

    func testMultiControlledZActsAsZWhenAllControlsAreOne() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 3)
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.x(0)
        try circuit.x(1)
        try circuit.h(2)
        try circuit.mcz(controls: [0, 1], target: 2)
        try circuit.h(2)
        try engine.execute(circuit, on: state)

        let z2 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 2)
        XCTAssertEqual(z2, -1, accuracy: 1e-5, "H·MCZ·H with all controls |1⟩ should flip the target")
    }
}
