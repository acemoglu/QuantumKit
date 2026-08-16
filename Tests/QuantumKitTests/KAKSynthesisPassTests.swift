import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - KAKSynthesisPass

    func testKAKSynthesisHaarCustomUnitaryFidelity() throws {
        var rng = QuantumRNG.seeded(20260816)
        let samples = 24
        // Independent oracle: pass skips CartanKAK internal verify so frob/fid asserts are not tautological.
        let pass = KAKSynthesisPass(verifyRoundTrip: false)
        for i in 0..<samples {
            let matrix = QuantumVolume.sampleHaarSU4(rng: &rng)
            var circuit = try QuantumCircuit(qubitCount: 2)
            try circuit.customUnitary(matrix: matrix, qubits: [0, 1])

            let out = try pass.run(on: circuit)
            for (j, gate) in out.gates.enumerated() {
                XCTAssertTrue(
                    Self.isKAKEmittedGate(gate),
                    "sample \(i) gate[\(j)]: expected u/rx/ry/rz/cx, got \(gate)"
                )
            }
            let cxCount = out.gates.reduce(0) { count, gate in
                if case .cx = gate { return count + 1 }
                return count
            }
            XCTAssertLessThanOrEqual(cxCount, 3, "sample \(i) CX count")

            let rebuilt = try CartanKAK.matrix(ofGates: out.gates)
            let frob = CartanKAK.phaseAlignedFrobenius(target: matrix, candidate: rebuilt)
            let fid = CartanKAK.averageGateFidelity(target: matrix, candidate: rebuilt)
            XCTAssertLessThanOrEqual(
                frob,
                CartanKAK.roundTripFrobeniusTolerance,
                "sample \(i): ‖U−V‖_F=\(frob)"
            )
            XCTAssertGreaterThanOrEqual(
                fid,
                CartanKAK.fidelityFloor,
                "sample \(i): fidelity=\(fid)"
            )
        }
    }

    /// Locals + CX only — no leftover ``customUnitary`` / ``unitary1`` (basis-hostile).
    private static func isKAKEmittedGate(_ gate: Gate) -> Bool {
        switch gate {
        case .u, .rx, .ry, .rz, .cx:
            return true
        default:
            return false
        }
    }

    func testKAKSynthesisLeavesCircuitsWithout2QCustomUnitaryUnchanged() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.rz(theta: QFloatExpr(0.3), 1)

        let out = try KAKSynthesisPass().run(on: circuit)
        XCTAssertEqual(out.gates, circuit.gates)

        // 1Q customUnitary is left alone (not a 2Q KAK target).
        let hadamard: [ComplexAmplitude] = [
            ComplexAmplitude(real: QFloat(1 / sqrt(2)), imaginary: 0),
            ComplexAmplitude(real: QFloat(1 / sqrt(2)), imaginary: 0),
            ComplexAmplitude(real: QFloat(1 / sqrt(2)), imaginary: 0),
            ComplexAmplitude(real: QFloat(-1 / sqrt(2)), imaginary: 0),
        ]
        var with1Q = try QuantumCircuit(qubitCount: 1)
        try with1Q.customUnitary(matrix: hadamard, qubits: [0])
        let out1Q = try KAKSynthesisPass().run(on: with1Q)
        XCTAssertEqual(out1Q.gates, with1Q.gates)

        // 3Q+ customUnitary is left alone (KAK is 2Q-only).
        var i8 = [ComplexAmplitude](
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: 64
        )
        for k in 0..<8 {
            i8[k * 8 + k] = ComplexAmplitude(real: 1, imaginary: 0)
        }
        var with3Q = try QuantumCircuit(qubitCount: 3)
        try with3Q.customUnitary(matrix: i8, qubits: [0, 1, 2])
        let out3Q = try KAKSynthesisPass().run(on: with3Q)
        XCTAssertEqual(out3Q.gates, with3Q.gates)
        guard case .customUnitary(_, let qubits) = out3Q.gates[0] else {
            return XCTFail("expected untouched 3Q customUnitary")
        }
        XCTAssertEqual(qubits, [0, 1, 2])
    }

    func testKAKSynthesisDefaultOffInTranspileOptions() throws {
        let off = try TranspileOptions(targetBasis: .ibmEagle, optimizationLevel: 0).makePasses()
        XCTAssertFalse(off.contains { $0 is KAKSynthesisPass })
        XCTAssertFalse(off.contains { $0 is SolovayKitaevPass })

        let on = try TranspileOptions(
            targetBasis: .ibmEagle,
            optimizationLevel: 0,
            enableKAKSynthesis: true
        ).makePasses()
        XCTAssertTrue(on.contains { $0 is KAKSynthesisPass })
        XCTAssertEqual(KAKSynthesisPass.passID, "quantumkit.kak_synthesis")

        let both = try TranspileOptions(
            targetBasis: .ibmEagle,
            optimizationLevel: 0,
            enableKAKSynthesis: true,
            enableSolovayKitaev: true
        ).makePasses()
        let kakIdx = both.firstIndex { $0 is KAKSynthesisPass }
        let skIdx = both.firstIndex { $0 is SolovayKitaevPass }
        let unrollIdx = both.firstIndex { $0 is UnrollMultiQubitPass }
        let basisIdx = both.firstIndex { $0 is BasisTranslatorPass }
        XCTAssertNotNil(kakIdx)
        XCTAssertNotNil(skIdx)
        XCTAssertNotNil(basisIdx)
        XCTAssertLessThan(kakIdx!, skIdx!)
        if let unrollIdx {
            XCTAssertLessThan(skIdx!, unrollIdx)
            XCTAssertLessThan(unrollIdx, basisIdx!)
        } else {
            XCTAssertLessThan(skIdx!, basisIdx!)
        }
    }

    func testKAKSynthesisDefaultTranspileBitIdentical() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.rz(theta: QFloatExpr(0.3), 1)

        let defaultOpts = TranspileOptions(targetBasis: .ibmEagle, optimizationLevel: 0)
        let explicitOff = TranspileOptions(
            targetBasis: .ibmEagle,
            optimizationLevel: 0,
            enableKAKSynthesis: false
        )
        let a = try Transpiler.transpile(circuit, options: defaultOpts)
        let b = try Transpiler.transpile(circuit, options: explicitOff)
        XCTAssertEqual(a.gates, b.gates, "default must match explicit enableKAKSynthesis: false")

        // No 2Q customUnitary → enabling KAK is a no-op on the gate list.
        let withKAK = try Transpiler.transpile(
            circuit,
            options: TranspileOptions(
                targetBasis: .ibmEagle,
                optimizationLevel: 0,
                enableKAKSynthesis: true
            )
        )
        XCTAssertEqual(a.gates, withKAK.gates)
    }

    func testKAKSynthesisFlagOnRewritesTinyCustomUnitaryFixture() throws {
        // CX as 2Q customUnitary — without KAK, ibmEagle basis rejects it.
        let cx: [ComplexAmplitude] = [
            ComplexAmplitude(real: 1, imaginary: 0), ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0), ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0), ComplexAmplitude(real: 1, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0), ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0), ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0), ComplexAmplitude(real: 1, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0), ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 1, imaginary: 0), ComplexAmplitude(real: 0, imaginary: 0),
        ]
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.customUnitary(matrix: cx, qubits: [0, 1])

        XCTAssertThrowsError(
            try Transpiler.transpile(
                circuit,
                options: TranspileOptions(targetBasis: .ibmEagle, optimizationLevel: 0)
            )
        )

        let withKAK = try Transpiler.transpile(
            circuit,
            options: TranspileOptions(
                targetBasis: .ibmEagle,
                optimizationLevel: 0,
                enableKAKSynthesis: true
            )
        )
        XCTAssertFalse(
            withKAK.gates.contains { if case .customUnitary = $0 { return true }; return false },
            "flags on: KAK must expand 2Q customUnitary before basis"
        )
        XCTAssertTrue(
            withKAK.gates.contains { if case .cx = $0 { return true }; return false },
            "expected at least one CX after KAK + basis"
        )
    }
}
