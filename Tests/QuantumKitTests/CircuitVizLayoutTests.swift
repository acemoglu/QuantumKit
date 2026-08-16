import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - CircuitVizLayout / ASCII (Epic H12)

    func testCircuitVizMomentPackingIndependentGatesShareMoment() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.x(2)

        let layout = CircuitVizLayout(circuit: circuit)
        XCTAssertEqual(layout.momentCount, 1)
        XCTAssertEqual(layout.moments[0].gateIndices, [0, 1])
        XCTAssertEqual(layout.cell(row: 0, moment: 0), .gate("H"))
        XCTAssertEqual(layout.cell(row: 1, moment: 0), .idle)
        XCTAssertEqual(layout.cell(row: 2, moment: 0), .gate("X"))
    }

    func testCircuitVizMomentPackingIntersectingGatesSplitMoments() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.x(0)

        let layout = CircuitVizLayout(circuit: circuit)
        XCTAssertEqual(layout.momentCount, 3)
        XCTAssertEqual(layout.moments.map(\.gateIndices), [[0], [1], [2]])
        XCTAssertEqual(layout.cell(row: 0, moment: 0), .gate("H"))
        XCTAssertEqual(layout.cell(row: 0, moment: 1), .control)
        XCTAssertEqual(layout.cell(row: 1, moment: 1), .target("X"))
        XCTAssertEqual(layout.cell(row: 0, moment: 2), .gate("X"))
    }

    func testCircuitVizVerticalWireSpanAcrossIntermediateQubit() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.cx(0, 2)

        let layout = CircuitVizLayout(circuit: circuit)
        XCTAssertEqual(layout.momentCount, 1)
        XCTAssertEqual(layout.cell(row: 0, moment: 0), .control)
        XCTAssertEqual(layout.cell(row: 1, moment: 0), .wire)
        XCTAssertEqual(layout.cell(row: 2, moment: 0), .target("X"))

        // Intermediate wire occupancy blocks packing another op on q1 into the same moment.
        var blocked = try QuantumCircuit(qubitCount: 3)
        try blocked.cx(0, 2)
        try blocked.h(1)
        let blockedLayout = CircuitVizLayout(circuit: blocked)
        XCTAssertEqual(blockedLayout.momentCount, 2)
        XCTAssertEqual(blockedLayout.cell(row: 1, moment: 1), .gate("H"))
    }

    func testCircuitVizClassicalRowsAndMeasureMapping() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.apply(
            .measure(MeasureSpec(qubits: [0], classicalRegister: 0, classicalBitOffset: 1))
        )

        let layout = CircuitVizLayout(circuit: circuit)
        XCTAssertEqual(layout.qubitCount, 2)
        XCTAssertEqual(layout.classicalBitCount, 2)
        XCTAssertEqual(layout.rowCount, 4)
        XCTAssertEqual(layout.rowLabel(2), "c0")
        XCTAssertEqual(layout.rowLabel(3), "c1")

        XCTAssertEqual(layout.momentCount, 2)
        XCTAssertEqual(layout.cell(row: 0, moment: 1), .measure)
        XCTAssertEqual(layout.cell(row: 2, moment: 1), .idle) // classical bit 0 unused
        XCTAssertEqual(layout.cell(row: 3, moment: 1), .measureClassical)
    }

    func testCircuitVizPlaceholdersForBarrierDelayAndControlFlow() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try circuit.apply(.barrier(qubits: [0, 1]))
        try circuit.apply(.delay(duration: 1.0, qubit: 0))
        try circuit.apply(.c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 1)))
        try circuit.apply(
            .while_c(
                classicalRegister: 0,
                expectedValue: 0,
                body: [.h(target: 0)],
                maxIterations: 2
            )
        )

        let layout = CircuitVizLayout(circuit: circuit)
        // delay(q0) and c_if→X(q1) are independent, so they share a moment.
        XCTAssertEqual(layout.momentCount, 3)
        XCTAssertEqual(layout.cell(row: 0, moment: 0), .placeholder("║"))
        XCTAssertEqual(layout.cell(row: 1, moment: 0), .placeholder("║"))
        XCTAssertEqual(layout.cell(row: 0, moment: 1), .placeholder("τ"))
        XCTAssertEqual(layout.cell(row: 1, moment: 1), .placeholder("IF"))
        XCTAssertEqual(layout.cell(row: 0, moment: 2), .placeholder("W"))
    }

    func testCircuitVizAsciiDiagramBellState() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let diagram = circuit.asciiDiagram()
        let expected = """
        q0: ─H──■─
        q1: ────X─
        """
        XCTAssertEqual(diagram, expected)

        var sequence = try GateSequence(name: "bell", qubitCount: 2)
        try sequence.apply(.h(target: 0))
        try sequence.apply(.cx(control: 0, target: 1))
        XCTAssertEqual(sequence.asciiDiagram(), expected)
        XCTAssertEqual(CircuitVizLayout(sequence: sequence).momentCount, 2)
    }

    func testCircuitVizAsciiDiagramGHZWithSpanAndMeasure() throws {
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 3, classicalRegisters: [creg])
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.cx(0, 2)
        try circuit.apply(
            .measure(MeasureSpec(qubits: [1], classicalRegister: 0, classicalBitOffset: 0))
        )

        let layout = CircuitVizLayout(circuit: circuit)
        XCTAssertEqual(layout.momentCount, 4)
        // CX(0,2) must wire through q1.
        XCTAssertEqual(layout.cell(row: 0, moment: 2), .control)
        XCTAssertEqual(layout.cell(row: 1, moment: 2), .wire)
        XCTAssertEqual(layout.cell(row: 2, moment: 2), .target("X"))

        let diagram = circuit.asciiDiagram()
        let expected = """
        q0: ─H──■──■────
        q1: ────X──┼─[M]
        q2: ───────X────
        c0: ══════════╩═
        """
        XCTAssertEqual(diagram, expected)
    }

    func testCircuitVizLSBQubitAtTopRow() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(1)

        let layout = CircuitVizLayout(circuit: circuit)
        XCTAssertEqual(layout.rowLabel(0), "q0")
        XCTAssertEqual(layout.cell(row: 0, moment: 0), .idle)
        XCTAssertEqual(layout.cell(row: 1, moment: 0), .gate("X"))
    }
}
