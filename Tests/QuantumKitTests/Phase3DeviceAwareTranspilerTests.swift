import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - CouplingMap / Layout

    func testCouplingMapLinearAdjacencyAndShortestPath() throws {
        let map = try CouplingMap.linear(4)

        XCTAssertTrue(map.areAdjacent(0, 1))
        XCTAssertTrue(map.areAdjacent(2, 3))
        XCTAssertFalse(map.areAdjacent(0, 2))
        XCTAssertEqual(map.shortestPath(from: 0, to: 3), [0, 1, 2, 3])
        XCTAssertEqual(map.shortestPath(from: 2, to: 0), [2, 1, 0])
    }

    func testCouplingMapRejectsSelfLoopAndOutOfRange() {
        XCTAssertThrowsError(try CouplingMap(qubitCount: 2, edges: [(0, 0)])) { error in
            guard case TranspilerError.invalidCouplingMap = error else {
                return XCTFail("Expected invalidCouplingMap, got \(error)")
            }
        }
        XCTAssertThrowsError(try CouplingMap(qubitCount: 2, edges: [(0, 2)])) { error in
            guard case TranspilerError.invalidCouplingMap = error else {
                return XCTFail("Expected invalidCouplingMap, got \(error)")
            }
        }
    }

    func testLayoutIdentityAndInjectivity() throws {
        let layout = try Layout.identity(qubitCount: 3)
        XCTAssertEqual(layout.logicalToPhysical, [0, 1, 2])
        XCTAssertEqual(try layout.physical(forLogical: 1), 1)

        XCTAssertThrowsError(try Layout(logicalToPhysical: [0, 0, 1])) { error in
            guard case TranspilerError.invalidLayout = error else {
                return XCTFail("Expected invalidLayout, got \(error)")
            }
        }
    }

    // MARK: - Initial layout

    func testInitialLayoutPassRemapsQubitsOntoWiderDevice() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let layout = try Layout(logicalToPhysical: [2, 0])
        let remapped = try InitialLayoutPass(layout: layout, physicalQubitCount: 4).run(on: circuit)

        XCTAssertEqual(remapped.qubitCount, 4)
        XCTAssertEqual(remapped.gates[0], .h(target: 2))
        XCTAssertEqual(remapped.gates[1], .cx(control: 2, target: 0))
    }

    // MARK: - Unroll + routing

    func testUnrollMultiQubitPassExpandsCCX() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.ccx(0, 1, 2)

        let unrolled = try UnrollMultiQubitPass().run(on: circuit)
        XCTAssertGreaterThan(unrolled.gates.count, 1)
        for gate in unrolled.gates {
            XCTAssertLessThanOrEqual(Set(gate.affectedQubits).count, 2)
        }
    }

    func testBasicSwapRoutingInsertsSwapForNonAdjacentCX() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.cx(0, 2)

        let map = try CouplingMap.linear(3)
        let routed = try BasicSwapRoutingPass(couplingMap: map).run(on: circuit)

        XCTAssertEqual(routed.qubitCount, 3)
        XCTAssertTrue(routed.gates.contains { if case .swap = $0 { return true }; return false })

        for gate in routed.gates {
            let qubits = gate.affectedQubits
            guard qubits.count == 2 else { continue }
            XCTAssertTrue(
                map.areAdjacent(qubits[0], qubits[1]),
                "Two-qubit gate \(gate) must lie on a coupling edge"
            )
        }
    }

    func testBasicSwapRoutingLeavesAdjacentCXUntouched() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let map = try CouplingMap.linear(3)
        let routed = try BasicSwapRoutingPass(couplingMap: map).run(on: circuit)

        XCTAssertFalse(routed.gates.contains { if case .swap = $0 { return true }; return false })
        XCTAssertEqual(routed.gates, [.h(target: 0), .cx(control: 0, target: 1)])
    }

    func testRoutingRejectsMultiQubitGateWithoutUnroll() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.ccx(0, 1, 2)

        let map = try CouplingMap.linear(3)
        XCTAssertThrowsError(try BasicSwapRoutingPass(couplingMap: map).run(on: circuit)) { error in
            guard case TranspilerError.routingRequiresTwoQubitGates = error else {
                return XCTFail("Expected routingRequiresTwoQubitGates, got \(error)")
            }
        }
    }

    // MARK: - Device-aware pipeline (A1 + A5)

    func testDeviceAwarePresetUnrollsRoutesAndBasisTranslates() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 2)

        let map = try CouplingMap.linear(3)
        let transpiled = try Transpiler.transpile(
            circuit,
            preset: .deviceAware(couplingMap: map, basis: .ibmEagle, layout: nil)
        )

        XCTAssertEqual(transpiled.qubitCount, 3)
        XCTAssertFalse(transpiled.gates.isEmpty)

        for gate in transpiled.gates {
            guard let kind = BasisGateKind(gate: gate) else {
                XCTFail("Gate outside target basis after device-aware transpile: \(gate)")
                return
            }
            XCTAssertTrue(BasisGateSet.ibmEagle.contains(kind))
        }

        // After basis translation every remaining two-qubit op is CX; all must be adjacent.
        for gate in transpiled.gates {
            if case .cx(let control, let target) = gate {
                XCTAssertTrue(map.areAdjacent(control, target))
            }
        }
    }

    func testTranspileOptionsWithCouplingMapMatchesPreset() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.cx(0, 2)

        let map = try CouplingMap.linear(3)
        let viaOptions = try Transpiler.transpile(
            circuit,
            options: TranspileOptions(targetBasis: .ibmEagle, couplingMap: map, optimizationLevel: 1)
        )
        let viaPreset = try Transpiler.transpile(
            circuit,
            preset: .deviceAware(couplingMap: map, basis: .ibmEagle, layout: nil)
        )

        XCTAssertEqual(viaOptions.gates, viaPreset.gates)
    }

    func testLegacyBasisOnlyAPIUnchanged() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let transpiled = try Transpiler.transpile(circuit, targetBasis: .ibmEagle)
        for gate in transpiled.gates {
            XCTAssertNotNil(BasisGateKind(gate: gate))
        }
        XCTAssertFalse(transpiled.gates.contains { if case .h = $0 { return true }; return false })
    }

    func testDeviceAwareAdjacentBellPreservesUnitary() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let map = try CouplingMap.linear(2)
        let transpiled = try Transpiler.transpile(
            circuit,
            preset: .deviceAware(couplingMap: map, basis: .ibmEagle, layout: nil)
        )

        if try CircuitUnitary.areEquivalent(circuit, transpiled, tolerance: 1e-4) {
            return
        }

        let engine = try QuantumEngine()
        XCTAssertTrue(
            try CircuitEquivalence.haveIdenticalAction(
                circuit,
                transpiled,
                engine: engine,
                tolerance: 1e-4
            )
        )
    }
}
