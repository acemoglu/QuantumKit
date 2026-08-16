import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - TemplateMatchingPass (A6 lite)

    func testTemplateMatchingCatalogHXHAndHZH() throws {
        var hxh = try QuantumCircuit(qubitCount: 1)
        try hxh.h(0)
        try hxh.x(0)
        try hxh.h(0)

        let outHXH = try TemplateMatchingPass().run(on: hxh)
        XCTAssertEqual(outHXH.gates, [.z(target: 0)])
        XCTAssertLessThan(outHXH.gates.count, hxh.gates.count)
        assertCPUBornActionMatch(hxh, outHXH)

        var hzh = try QuantumCircuit(qubitCount: 1)
        try hzh.h(0)
        try hzh.z(0)
        try hzh.h(0)

        let outHZH = try TemplateMatchingPass().run(on: hzh)
        XCTAssertEqual(outHZH.gates, [.x(target: 0)])
        XCTAssertLessThan(outHZH.gates.count, hzh.gates.count)
        assertCPUBornActionMatch(hzh, outHZH)
    }

    func testTemplateMatchingCatalogSXSdgToY() throws {
        var sXSdg = try QuantumCircuit(qubitCount: 1)
        try sXSdg.s(0)
        try sXSdg.x(0)
        try sXSdg.sdg(0)

        let outS = try TemplateMatchingPass().run(on: sXSdg)
        XCTAssertEqual(outS.gates, [.y(target: 0)])
        XCTAssertLessThan(outS.gates.count, sXSdg.gates.count)
        assertCPUBornActionMatch(sXSdg, outS)

        // Catalog `sdgxsd`: Sdg·X·S → Y
        var sdgXS = try QuantumCircuit(qubitCount: 1)
        try sdgXS.sdg(0)
        try sdgXS.x(0)
        try sdgXS.s(0)

        let outSdg = try TemplateMatchingPass().run(on: sdgXS)
        XCTAssertEqual(outSdg.gates, [.y(target: 0)])
        XCTAssertLessThan(outSdg.gates.count, sdgXS.gates.count)
        assertCPUBornActionMatch(sdgXS, outSdg)
    }

    func testTemplateMatchingCatalogHCXHToCZ() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(1)
        try circuit.cx(0, 1)
        try circuit.h(1)

        let out = try TemplateMatchingPass().run(on: circuit)
        XCTAssertEqual(out.gates.count, 1)
        XCTAssertEqual(out.gates[0], .cz(control: 0, target: 1))
        XCTAssertLessThan(out.gates.count, circuit.gates.count)
        assertCPUBornActionMatch(circuit, out)
    }

    func testTemplateMatchingCatalogHCZHToCX() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(1)
        try circuit.cz(0, 1)
        try circuit.h(1)

        let out = try TemplateMatchingPass().run(on: circuit)
        XCTAssertEqual(out.gates, [.cx(control: 0, target: 1)])
        XCTAssertLessThan(out.gates.count, circuit.gates.count)
        assertCPUBornActionMatch(circuit, out)
    }

    func testTemplateMatchingCatalogCXTripleToSWAP() throws {
        var forward = try QuantumCircuit(qubitCount: 2)
        try forward.cx(0, 1)
        try forward.cx(1, 0)
        try forward.cx(0, 1)

        let outFwd = try TemplateMatchingPass().run(on: forward)
        XCTAssertEqual(outFwd.gates.count, 1)
        XCTAssertEqual(outFwd.gates[0], .swap(q1: 0, q2: 1))
        XCTAssertLessThan(outFwd.gates.count, forward.gates.count)
        assertCPUBornActionMatch(forward, outFwd)

        var reverse = try QuantumCircuit(qubitCount: 2)
        try reverse.cx(1, 0)
        try reverse.cx(0, 1)
        try reverse.cx(1, 0)

        let outRev = try TemplateMatchingPass().run(on: reverse)
        XCTAssertEqual(outRev.gates.count, 1)
        XCTAssertEqual(outRev.gates[0], .swap(q1: 1, q2: 0))
        XCTAssertLessThan(outRev.gates.count, reverse.gates.count)
        assertCPUBornActionMatch(reverse, outRev)
    }

    func testTemplateMatchingCatalogCXConjugations() throws {
        var xt = try QuantumCircuit(qubitCount: 2)
        try xt.cx(0, 1)
        try xt.x(1)
        try xt.cx(0, 1)
        let outXT = try TemplateMatchingPass().run(on: xt)
        XCTAssertEqual(outXT.gates, [.x(target: 1)])
        XCTAssertLessThan(outXT.gates.count, xt.gates.count)
        assertCPUBornActionMatch(xt, outXT)

        var xc = try QuantumCircuit(qubitCount: 2)
        try xc.cx(0, 1)
        try xc.x(0)
        try xc.cx(0, 1)
        let outXC = try TemplateMatchingPass().run(on: xc)
        XCTAssertEqual(outXC.gates, [.x(target: 0), .x(target: 1)])
        XCTAssertLessThan(outXC.gates.count, xc.gates.count)
        assertCPUBornActionMatch(xc, outXC)

        var zc = try QuantumCircuit(qubitCount: 2)
        try zc.cx(0, 1)
        try zc.z(0)
        try zc.cx(0, 1)
        let outZC = try TemplateMatchingPass().run(on: zc)
        XCTAssertEqual(outZC.gates, [.z(target: 0)])
        XCTAssertLessThan(outZC.gates.count, zc.gates.count)
        assertCPUBornActionMatch(zc, outZC)

        var zt = try QuantumCircuit(qubitCount: 2)
        try zt.cx(0, 1)
        try zt.z(1)
        try zt.cx(0, 1)
        let outZT = try TemplateMatchingPass().run(on: zt)
        XCTAssertEqual(outZT.gates, [.z(target: 0), .z(target: 1)])
        XCTAssertLessThan(outZT.gates.count, zt.gates.count)
        assertCPUBornActionMatch(zt, outZT)
    }

    func testTemplateMatchingLeavesNonMatchingUnchanged() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.t(1)
        try circuit.rz(theta: QFloat(0.3), 0)

        let out = try TemplateMatchingPass().run(on: circuit)
        XCTAssertEqual(out.gates, circuit.gates)
        assertCPUBornActionMatch(circuit, out)
    }

    func testTemplateMatchingDoesNotMatchAcrossInterveningGate() throws {
        // H · X · barrier · H is not adjacent H·X·H
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.x(0)
        try circuit.barrier([0])
        try circuit.h(0)

        let out = try TemplateMatchingPass().run(on: circuit)
        XCTAssertEqual(out.gates.count, 4)
        XCTAssertEqual(out.gates, circuit.gates)
    }

    func testTemplateMatchingDefaultOffInTranspileOptions() throws {
        XCTAssertEqual(TemplateMatchingPass.catalogEntryCount, 11)

        let off = try TranspileOptions(targetBasis: .ibmEagle, optimizationLevel: 2).makePasses()
        XCTAssertFalse(off.contains { $0 is TemplateMatchingPass })

        let on = try TranspileOptions(
            targetBasis: .ibmEagle,
            optimizationLevel: 0,
            enableTemplateMatching: true
        ).makePasses()
        XCTAssertTrue(on.contains { $0 is TemplateMatchingPass })

        // Include `.x` so the rewrite product stays native (ibmEagle expands X via SX/RZ).
        let basis = BasisGateSet(.rz, .sx, .sxdg, .x, .cx)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.cx(0, 1)
        try circuit.x(1)
        try circuit.cx(0, 1)

        let defaultOut = try Transpiler.transpile(
            circuit,
            options: TranspileOptions(targetBasis: basis, optimizationLevel: 0)
        )
        let withFlag = try Transpiler.transpile(
            circuit,
            options: TranspileOptions(
                targetBasis: basis,
                optimizationLevel: 0,
                enableTemplateMatching: true
            )
        )
        XCTAssertEqual(defaultOut.gates.count, 3)
        XCTAssertEqual(withFlag.gates, [.x(target: 1)])
        XCTAssertLessThan(withFlag.gates.count, circuit.gates.count)
        assertCPUBornActionMatch(circuit, withFlag)
        assertCPUBornActionMatch(circuit, defaultOut)
    }

    // MARK: - TemplateMatching helpers

    /// CPU statevector Born-rule check on every computational-basis input.
    /// Prefer this over ``CircuitEquivalenceVerifier/areEquivalent`` (CircuitUnitary)
    /// so template tests never depend on the known CRX/CP matrix pitfalls.
    private func assertCPUBornActionMatch(
        _ original: QuantumCircuit,
        _ rewritten: QuantumCircuit,
        accuracy: QFloat = 1e-6,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            XCTAssertEqual(
                original.qubitCount,
                rewritten.qubitCount,
                "qubit counts differ",
                file: file,
                line: line
            )
            let engine = CPUStatevectorEngine()
            let dimension = 1 << original.qubitCount
            for basisIndex in 0..<dimension {
                let a = try CPUStateVector(qubitCount: original.qubitCount)
                let b = try CPUStateVector(qubitCount: rewritten.qubitCount)
                prepareCPUBasisState(basisIndex, on: a)
                prepareCPUBasisState(basisIndex, on: b)
                _ = try engine.execute(original, on: a)
                _ = try engine.execute(rewritten, on: b)
                let pa = a.probabilities()
                let pb = b.probabilities()
                XCTAssertEqual(pa.count, pb.count, file: file, line: line)
                for index in pa.indices {
                    XCTAssertEqual(
                        pa[index],
                        pb[index],
                        accuracy: accuracy,
                        "basis \(basisIndex) outcome \(index)",
                        file: file,
                        line: line
                    )
                }
            }
        } catch {
            XCTFail("CPU SV Born compare failed: \(error)", file: file, line: line)
        }
    }

    private func prepareCPUBasisState(_ basisIndex: Int, on state: CPUStateVector) {
        var real = Array(repeating: 0.0, count: state.stateCount)
        let imag = Array(repeating: 0.0, count: state.stateCount)
        real[basisIndex] = 1
        state.setAmplitudes(real: real, imag: imag)
    }
}
