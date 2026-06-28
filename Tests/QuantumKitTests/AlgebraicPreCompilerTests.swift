import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Algebraic pre-compiler

    func testAlgebraicPreCompilerCancelsDoubleHadamard() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.h(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.originalGateCount, 2)
        XCTAssertEqual(result.optimizedGateCount, 0)
        XCTAssertTrue(result.gates.isEmpty)
    }

    func testAlgebraicPreCompilerSSBecomesZ() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.s(0)
        try circuit.s(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.z(target: 0)])
    }

    func testAlgebraicPreCompilerTTBecomesS() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.t(0)
        try circuit.t(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.s(target: 0)])
    }

    func testAlgebraicPreCompilerMergesAdjacentRotations() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        let quarter = QFloat(Double.pi / 4.0)
        try circuit.rx(theta: quarter, 0)
        try circuit.rx(theta: quarter, 0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.rx(theta: QFloat(Double.pi / 2.0), target: 0)])
    }

    func testAlgebraicPreCompilerCancelsSWithSDagger() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.s(0)
        try circuit.sdg(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 0, "S·S† should cancel completely")
    }

    func testAlgebraicPreCompilerCancelsTWithTDagger() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.t(0)
        try circuit.tdg(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 0, "T·T† should cancel completely")
    }

    func testAlgebraicPreCompilerMergesPhaseGatesIntoS() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        let quarter = QFloat(Double.pi / 4.0)
        try circuit.p(theta: quarter, 0)
        try circuit.p(theta: quarter, 0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.s(target: 0)], "P(π/4)·P(π/4) = P(π/2) = S")
    }

    func testAlgebraicPreCompilerMergesTWithPhaseIntoSDagger() throws {
        // T (π/4) followed by P(-3π/4) sums to -π/2 → S†.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.t(0)
        try circuit.p(theta: QFloat(-3.0 * Double.pi / 4.0), 0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.sdg(target: 0)])
    }

    func testAlgebraicPreCompilerCancelsDoubleCZ() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.cz(0, 1)
        try circuit.cz(0, 1)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 0, "CZ·CZ should cancel completely")
    }

    func testAlgebraicPreCompilerCancelsDoubleSwap() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.swap(0, 1)
        try circuit.swap(0, 1)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 0, "SWAP·SWAP should cancel completely")
    }

    func testAlgebraicPreCompilerSlidesZAxisGatesThroughCZ() throws {
        // S(0), CZ(0,1), S(0): the two S gates commute through CZ and fold into Z.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.s(0)
        try circuit.cz(0, 1)
        try circuit.s(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 2)
        // Z and CZ commute, so either ordering is a valid optimization.
        XCTAssertTrue(result.gates.contains(.z(target: 0)))
        XCTAssertTrue(result.gates.contains(.cz(control: 0, target: 1)))
    }

    func testCommutationSlidesHadamardThroughDisjointPauli() throws {
        let gates: [Gate] = [.h(target: 0), .x(target: 1), .h(target: 0)]

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.x(1)
        try circuit.h(0)

        let slid = AlgebraicPreCompiler.slideGates(gates)
        XCTAssertEqual(slid, [.x(target: 1), .h(target: 0), .h(target: 0)])

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.x(target: 1)])
    }

    func testCommutationDoesNotCancelSameQubitHadamardSandwich() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.x(0)
        try circuit.h(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 3)
    }

    func testCommutationDoesNotSlidePastMeasurement() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.measure(1)
        try circuit.h(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 3)
        XCTAssertEqual(result.gates, circuit.gates)
    }

    func testCommutationDoesNotSlideZThroughCXTarget() throws {
        // Z on the *target* of a CX does NOT commute: Z(1)·CX(0,1)·Z(1) = Z(0)·CX(0,1) ≠ CX(0,1).
        // The two Z(1) gates must not slide together and cancel, so all three gates survive.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.z(1)
        try circuit.cx(0, 1)
        try circuit.z(1)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 3)
        XCTAssertEqual(result.gates, [.z(target: 1), .cx(control: 0, target: 1), .z(target: 1)])
    }

    func testCommutationSlidesZThroughCXControl() throws {
        // Z on the *control* of a CX commutes (both diagonal in the control's basis), so the two
        // Z(0) gates slide together through CX and fold into identity.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.z(0)
        try circuit.cx(0, 1)
        try circuit.z(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.cx(control: 0, target: 1)])
    }

    func testCommutationSlidesXThroughCXTarget() throws {
        // X on the *target* of a CX commutes, so the two X(1) gates fold into identity.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(1)
        try circuit.cx(0, 1)
        try circuit.x(1)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.cx(control: 0, target: 1)])
    }

    func testCommutationDoesNotSlideXThroughCXControl() throws {
        // X on the *control* of a CX does NOT commute, so no cancellation occurs.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.cx(0, 1)
        try circuit.x(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 3)
        XCTAssertEqual(result.gates, [.x(target: 0), .cx(control: 0, target: 1), .x(target: 0)])
    }

    func testCommutationDoesNotSlideZThroughCCXTarget() throws {
        // Z on the *target* of a CCX does NOT commute: Z anticommutes with the doubly-controlled X
        // on the active subspace. The two Z(2) gates must not slide together and cancel, so all
        // three gates survive.
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.z(2)
        try circuit.ccx(0, 1, 2)
        try circuit.z(2)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 3)
        XCTAssertEqual(result.gates, [.z(target: 2), .ccx(control1: 0, control2: 1, target: 2), .z(target: 2)])
    }

    func testCommutationSlidesZThroughCCXControl() throws {
        // Z on a *control* of a CCX commutes (diagonal in the control's basis), so the two Z(0)
        // gates slide together through CCX and fold into identity.
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.z(0)
        try circuit.ccx(0, 1, 2)
        try circuit.z(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.ccx(control1: 0, control2: 1, target: 2)])
    }

    func testCommutationSlidesZThroughCCXSecondControl() throws {
        // The same holds for the *second* control: Z(1) is diagonal in control 1's basis, so the
        // two Z(1) gates slide together through CCX and fold into identity.
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.z(1)
        try circuit.ccx(0, 1, 2)
        try circuit.z(1)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.ccx(control1: 0, control2: 1, target: 2)])
    }

    func testCommutationSlidesXThroughCCXTarget() throws {
        // X on the *target* of a CCX commutes (X·X under the doubly-controlled flip), so the two
        // X(2) gates fold into identity.
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.x(2)
        try circuit.ccx(0, 1, 2)
        try circuit.x(2)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.ccx(control1: 0, control2: 1, target: 2)])
    }

    func testCommutationDoesNotSlideXThroughCCXControl() throws {
        // X on a *control* of a CCX does NOT commute (it flips which branch fires), so no
        // cancellation occurs and all three gates survive.
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.x(0)
        try circuit.ccx(0, 1, 2)
        try circuit.x(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 3)
        XCTAssertEqual(result.gates, [.x(target: 0), .ccx(control1: 0, control2: 1, target: 2), .x(target: 0)])
    }

    func testCommutationDoesNotSlideSThroughCCXTarget() throws {
        // A Z-axis phase gate (S) on the target must also be blocked, not just Z. The two S(2)
        // gates must not slide together across the CCX (where they would fold into Z), so all
        // three gates survive in order.
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.s(2)
        try circuit.ccx(0, 1, 2)
        try circuit.s(2)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 3)
        XCTAssertEqual(result.gates, [.s(target: 2), .ccx(control1: 0, control2: 1, target: 2), .s(target: 2)])
    }

    func testAlgebraicPreCompilerPreservesBellStateProbabilities() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        var original = try QuantumCircuit(qubitCount: 2)
        try original.applyBellState()

        var redundant = try QuantumCircuit(qubitCount: 2)
        try redundant.applyBellState()
        try redundant.h(0)
        try redundant.h(0)
        try redundant.cx(0, 1)
        try redundant.cx(0, 1)

        let optimized = try redundant.algebraicallyOptimized()

        let originalState = try StateVector(qubitCount: 2, device: device)
        try engine.execute(original, on: originalState)

        let optimizedState = try StateVector(qubitCount: 2, device: device)
        try engine.execute(optimized, on: optimizedState)

        let originalProbabilities = try QuantumMeasurement.probabilities(state: originalState, engine: engine)
        let optimizedProbabilities = try QuantumMeasurement.probabilities(state: optimizedState, engine: engine)

        XCTAssertLessThan(optimized.gates.count, redundant.gates.count)
        XCTAssertEqual(originalProbabilities.count, optimizedProbabilities.count)
        for index in 0..<originalProbabilities.count {
            XCTAssertEqual(originalProbabilities[index], optimizedProbabilities[index], accuracy: 1e-5)
        }
    }
}
