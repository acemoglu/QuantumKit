import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - GateFusionPass

    func testGateFusionCollapsesParameterized1QChain() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: QFloatExpr(0.3), 0)
        try circuit.ry(theta: QFloatExpr(-0.7), 0)
        try circuit.rz(theta: QFloatExpr(1.1), 0)
        try circuit.h(0)

        let fused = try GateFusionPass().run(on: circuit)
        XCTAssertEqual(fused.gates.count, 1)
        XCTAssertTrue(
            Self.isFused1Q(fused.gates[0]),
            "expected u or unitary1 fusion product, got \(fused.gates)"
        )

        assertCPUProbabilitiesMatch(circuit, fused)

        if makeDevice() != nil {
            let engine = try QuantumEngine()
            XCTAssertTrue(
                try CircuitEquivalenceVerifier.haveIdenticalBornAction(
                    circuit,
                    fused,
                    engine: engine,
                    tolerance: 1e-4
                )
            )
        }
    }

    func testGateFusionDoesNotCrossBarrierMeasureOrCIf() throws {
        var withBarrier = try QuantumCircuit(qubitCount: 1)
        try withBarrier.h(0)
        try withBarrier.barrier([0])
        try withBarrier.x(0)
        let fusedBarrier = try GateFusionPass().run(on: withBarrier)
        XCTAssertEqual(fusedBarrier.gates.count, 3)
        XCTAssertEqual(fusedBarrier.gates[1], .barrier(qubits: [0]))

        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var withMeasure = try QuantumCircuit(qubitCount: 1, classicalRegisters: [creg])
        try withMeasure.h(0)
        try withMeasure.measure(qubits: [0], classicalRegister: 0)
        try withMeasure.x(0)
        let fusedMeasure = try GateFusionPass().run(on: withMeasure)
        XCTAssertEqual(fusedMeasure.gates.count, 3)
        XCTAssertTrue(Self.isMeasure(fusedMeasure.gates[1]))

        var withCIf = try QuantumCircuit(qubitCount: 1, classicalRegisters: [creg])
        try withCIf.h(0)
        try withCIf.apply(.c_if(classicalRegister: 0, expectedValue: 1, gate: .x(target: 0)))
        try withCIf.y(0)
        let fusedCIf = try GateFusionPass().run(on: withCIf)
        XCTAssertEqual(fusedCIf.gates.count, 3)
        XCTAssertTrue(Self.isCIf(fusedCIf.gates[1]))
    }

    func testGateFusionDoesNotCrossTwoQubitGate() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.rx(theta: QFloatExpr(0.4), 0)
        try circuit.cx(0, 1)
        try circuit.ry(theta: QFloatExpr(0.5), 0)

        let fused = try GateFusionPass().run(on: circuit)
        XCTAssertEqual(fused.gates.count, 3)
        XCTAssertEqual(fused.gates[1], .cx(control: 0, target: 1))
        assertCPUProbabilitiesMatch(circuit, fused)
    }

    func testGateFusionPassManagerSmoke() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: QFloatExpr(0.2), 0)
        try circuit.h(0)
        try circuit.rz(theta: QFloatExpr(-0.5), 0)

        let manager = PassManager.dag([GateFusionPass()])
        let out = try manager.run(on: circuit)
        XCTAssertEqual(out.gates.count, 1)
        XCTAssertTrue(Self.isFused1Q(out.gates[0]))
        assertCPUProbabilitiesMatch(circuit, out)
    }

    func testGateFusionDefaultOffInTranspileOptions() throws {
        let off = try TranspileOptions(targetBasis: .ibmEagle, optimizationLevel: 0).makePasses()
        XCTAssertFalse(off.contains { $0 is GateFusionPass })

        let on = try TranspileOptions(
            targetBasis: .ibmEagle,
            optimizationLevel: 0,
            enableGateFusion: true
        ).makePasses()
        XCTAssertTrue(on.contains { $0 is GateFusionPass })

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: QFloatExpr(0.3), 0)
        try circuit.ry(theta: QFloatExpr(0.4), 0)

        let fused = try Transpiler.transpile(
            circuit,
            options: TranspileOptions(
                targetBasis: .ibmEagle,
                optimizationLevel: 0,
                enableGateFusion: true
            )
        )
        // Fusion collapses to one U, then basis expands — still fewer than two full RX/RY expansions.
        XCTAssertLessThan(fused.gates.count, 10)
        assertCPUProbabilitiesMatch(circuit, fused)
    }

    func testGateFusionSkipsUnboundParameters() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: Parameter("theta"), 0)
        try circuit.h(0)

        let fused = try GateFusionPass().run(on: circuit)
        XCTAssertEqual(fused.gates.count, 2)
        XCTAssertEqual(fused.gates[0], .rx(theta: Parameter("theta"), target: 0))
    }

    func testGateFusionComposesAfterAlgebraic() throws {
        // Algebraic cancels H·H; fusion collapses the remaining cross-axis chain.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.h(0)
        try circuit.rx(theta: QFloatExpr(0.6), 0)
        try circuit.ry(theta: QFloatExpr(-0.2), 0)

        let manager = PassManager(passes: [
            AlgebraicOptimizationPass(),
            GateFusionPass(),
        ])
        let out = try manager.run(on: circuit)
        XCTAssertEqual(out.gates.count, 1)
        assertCPUProbabilitiesMatch(circuit, out)
    }

    func testGateFusionOnQubit1MatchesCPUProbs() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.rx(theta: QFloatExpr(0.4), 1)
        try circuit.ry(theta: QFloatExpr(0.5), 1)

        let fused = try GateFusionPass().run(on: circuit)
        XCTAssertEqual(fused.gates.count, 1)
        guard case .u(_, _, _, let target) = fused.gates[0] else {
            return XCTFail("expected fused u on qubit 1, got \(fused.gates)")
        }
        XCTAssertEqual(target, 1)
        assertCPUProbabilitiesMatch(circuit, fused)
    }

    func testGateFusionOnQubit1AfterEntanglingMatchesCPUProbs() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.rx(theta: QFloatExpr(0.3), 1)
        try circuit.ry(theta: QFloatExpr(-0.7), 1)
        try circuit.rz(theta: QFloatExpr(1.1), 1)

        let fused = try GateFusionPass().run(on: circuit)
        XCTAssertEqual(fused.gates.count, 3)
        XCTAssertEqual(fused.gates[0], .h(target: 0))
        XCTAssertEqual(fused.gates[1], .cx(control: 0, target: 1))
        XCTAssertTrue(Self.isFused1Q(fused.gates[2]))
        assertCPUProbabilitiesMatch(circuit, fused)
    }

    func testGateFusionWireAdjacentAcrossDisjointQubitOps() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.rx(theta: QFloatExpr(0.3), 0)
        try circuit.x(1)
        try circuit.ry(theta: QFloatExpr(-0.5), 0)

        let fused = try GateFusionPass().run(on: circuit)
        XCTAssertEqual(fused.gates.count, 2)
        assertCPUProbabilitiesMatch(circuit, fused)
    }

    func testGateFusionDoesNotCrossDelayResetOrInitialize() throws {
        var withDelay = try QuantumCircuit(qubitCount: 1)
        try withDelay.h(0)
        try withDelay.apply(.delay(duration: 1.0, qubit: 0))
        try withDelay.x(0)
        let fusedDelay = try GateFusionPass().run(on: withDelay)
        XCTAssertEqual(fusedDelay.gates.count, 3)
        XCTAssertEqual(fusedDelay.gates[1], .delay(duration: 1.0, qubit: 0))

        var withReset = try QuantumCircuit(qubitCount: 1)
        try withReset.h(0)
        try withReset.apply(.reset(qubit: 0))
        try withReset.x(0)
        XCTAssertEqual(try GateFusionPass().run(on: withReset).gates.count, 3)

        var withInit = try QuantumCircuit(qubitCount: 1)
        try withInit.h(0)
        try withInit.apply(
            .initialize(
                qubits: [0],
                amplitudes: [
                    ComplexAmplitude(real: 1, imaginary: 0),
                    ComplexAmplitude(real: 0, imaginary: 0),
                ]
            )
        )
        try withInit.x(0)
        XCTAssertEqual(try GateFusionPass().run(on: withInit).gates.count, 3)
    }

    func testGateFusionDoesNotCrossCZOnWire() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.rx(theta: QFloatExpr(0.4), 0)
        try circuit.cz(0, 1)
        try circuit.ry(theta: QFloatExpr(0.5), 0)
        let fused = try GateFusionPass().run(on: circuit)
        XCTAssertEqual(fused.gates.count, 3)
        XCTAssertEqual(fused.gates[1], .cz(control: 0, target: 1))
        assertCPUProbabilitiesMatch(circuit, fused)
    }

    func testGateFusionCustomUnitaryOnQubit1() throws {
        let inv = QFloat(1 / sqrt(2))
        let hadamard: [ComplexAmplitude] = [
            ComplexAmplitude(real: inv, imaginary: 0),
            ComplexAmplitude(real: inv, imaginary: 0),
            ComplexAmplitude(real: inv, imaginary: 0),
            ComplexAmplitude(real: -inv, imaginary: 0),
        ]
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.apply(.customUnitary(matrix: hadamard, qubits: [1]))
        try circuit.x(1)
        let fused = try GateFusionPass().run(on: circuit)
        XCTAssertEqual(fused.gates.count, 1)
        assertCPUProbabilitiesMatch(circuit, fused)
    }

    func testGateFusionOnQubit2ChainMatchesCPUProbs() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.rx(theta: QFloatExpr(1.7), 2)
        try circuit.ry(theta: QFloatExpr(0.9), 2)
        try circuit.rz(theta: QFloatExpr(-1.1), 2)
        try circuit.h(2)
        let fused = try GateFusionPass().run(on: circuit)
        XCTAssertEqual(fused.gates.count, 1)
        assertCPUProbabilitiesMatch(circuit, fused)
    }

    // MARK: - Helpers

    private func assertCPUProbabilitiesMatch(
        _ original: QuantumCircuit,
        _ fused: QuantumCircuit,
        accuracy: QFloat = 1e-6,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            let engine = CPUStatevectorEngine()
            let a = try CPUStateVector(qubitCount: original.qubitCount)
            let b = try CPUStateVector(qubitCount: fused.qubitCount)
            _ = try engine.execute(original, on: a)
            _ = try engine.execute(fused, on: b)
            let pa = a.probabilities()
            let pb = b.probabilities()
            XCTAssertEqual(pa.count, pb.count, file: file, line: line)
            for index in pa.indices {
                XCTAssertEqual(pa[index], pb[index], accuracy: accuracy, file: file, line: line)
            }
        } catch {
            XCTFail("CPU SV compare failed: \(error)", file: file, line: line)
        }
    }

    private static func isFused1Q(_ gate: Gate) -> Bool {
        switch gate {
        case .u, .unitary1:
            return true
        default:
            return false
        }
    }

    private static func isMeasure(_ gate: Gate) -> Bool {
        if case .measure = gate { return true }
        return false
    }

    private static func isCIf(_ gate: Gate) -> Bool {
        if case .c_if = gate { return true }
        return false
    }
}
