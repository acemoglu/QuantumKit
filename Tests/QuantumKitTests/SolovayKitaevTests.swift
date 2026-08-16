import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Solovay–Kitaev

    func testSolovayKitaevRandomTargetsMeetEpsilon() throws {
        let net = Discrete1QNet.build(maxWordLength: 8, calibrationSampleCount: 0)
        let synthesizer = SolovayKitaevSynthesizer(net: net)
        let epsilons: [Double] = [0.1, 0.05, 0.01]
        let sampleCount = 12

        for epsilon in epsilons {
            var rng = QuantumRNG.seeded(20260816 &+ UInt64(epsilon * 1_000_000))
            let pass = SolovayKitaevPass(
                epsilon: epsilon,
                maxRefinementIterations: 12,
                net: net
            )
            for i in 0..<sampleCount {
                let target = Discrete1QNet.sampleHaarSU2(rng: &rng)
                var circuit = try QuantumCircuit(qubitCount: 1)
                try circuit.apply(.unitary1(matrix: target, target: 0))

                let out = try pass.run(on: circuit)
                for gate in out.gates {
                    XCTAssertTrue(
                        Self.isCliffordTLetter(gate),
                        "ε=\(epsilon) sample \(i): unexpected gate \(gate)"
                    )
                }

                let rebuilt = try Self.skCircuitUnitaryAmplitudes(gates: out.gates)
                let distance = Discrete1QNet.phaseAlignedFrobenius(
                    target: target,
                    candidate: rebuilt
                )
                XCTAssertLessThanOrEqual(
                    distance,
                    epsilon,
                    "ε=\(epsilon) sample \(i): independent d=\(distance)"
                )

                let approx = try synthesizer.approximate(
                    target,
                    epsilon: epsilon,
                    maxRefinementIterations: 12
                )
                XCTAssertLessThanOrEqual(approx.distance, epsilon)
                // Tracked SK matrix vs CircuitUnitary product can drift slightly on long words.
                XCTAssertEqual(approx.distance, distance, accuracy: 1e-3)
                // Library word cap + Rz polish growth cap should prevent pathological blow-up.
                XCTAssertLessThan(
                    approx.gates.count,
                    4_000,
                    "ε=\(epsilon) sample \(i): unexpected gate count \(approx.gates.count)"
                )
            }
        }
    }

    func testSolovayKitaevTighterEpsilonTypicallyLonger() throws {
        let net = Discrete1QNet.build(maxWordLength: 8, calibrationSampleCount: 0)
        let synthesizer = SolovayKitaevSynthesizer(net: net)
        var rng = QuantumRNG.seeded(424242)
        var longerCount = 0
        let trials = 16

        for _ in 0..<trials {
            let target = Discrete1QNet.sampleHaarSU2(rng: &rng)
            let loose = try synthesizer.approximate(
                target,
                epsilon: 0.1,
                maxRefinementIterations: 12
            )
            let tight = try synthesizer.approximate(
                target,
                epsilon: 0.01,
                maxRefinementIterations: 12
            )
            XCTAssertLessThanOrEqual(loose.distance, 0.1)
            XCTAssertLessThanOrEqual(tight.distance, 0.01)
            if tight.gates.count >= loose.gates.count {
                longerCount += 1
            }
        }
        XCTAssertGreaterThanOrEqual(
            longerCount,
            trials * 3 / 4,
            "expected tighter ε to lengthen sequences on most trials (got \(longerCount)/\(trials))"
        )
    }

    func testSolovayKitaevPassGateCasesAndDefaultOff() throws {
        let net = Discrete1QNet.build(maxWordLength: 8, calibrationSampleCount: 0)
        var rng = QuantumRNG.seeded(7)
        let target = Discrete1QNet.sampleHaarSU2(rng: &rng)

        var u1 = try QuantumCircuit(qubitCount: 1)
        try u1.apply(.unitary1(matrix: target, target: 0))
        let pass = SolovayKitaevPass(epsilon: 0.1, maxRefinementIterations: 12, net: net)
        let out1 = try pass.run(on: u1)
        XCTAssertFalse(out1.gates.contains { if case .unitary1 = $0 { return true }; return false })

        var cu = try QuantumCircuit(qubitCount: 1)
        try cu.apply(.customUnitary(matrix: target, qubits: [0]))
        let outCU = try pass.run(on: cu)
        XCTAssertFalse(outCU.gates.contains { if case .customUnitary = $0 { return true }; return false })

        var withU = try QuantumCircuit(qubitCount: 1)
        try withU.apply(.u(theta: QFloatExpr(0.3), phi: QFloatExpr(0.4), lambda: QFloatExpr(0.5), target: 0))
        let outUOff = try pass.run(on: withU)
        XCTAssertEqual(outUOff.gates, withU.gates)

        let passU = SolovayKitaevPass(
            epsilon: 0.1,
            maxRefinementIterations: 12,
            rewriteU: true,
            net: net
        )
        let outUOn = try passU.run(on: withU)
        XCTAssertFalse(outUOn.gates.contains { if case .u = $0 { return true }; return false })
        let uMatrix = try GateFusionPass.singleQubitMatrix(withU.gates[0])
        let uAmps = uMatrix.elements.map {
            ComplexAmplitude(real: QFloat($0.re), imaginary: QFloat($0.im))
        }
        let dU = Discrete1QNet.phaseAlignedFrobenius(
            target: uAmps,
            candidate: try Self.skCircuitUnitaryAmplitudes(gates: outUOn.gates)
        )
        XCTAssertLessThanOrEqual(dU, 0.1)

        var other = try QuantumCircuit(qubitCount: 2)
        try other.h(0)
        try other.cx(0, 1)
        XCTAssertEqual(try pass.run(on: other).gates, other.gates)

        let off = try TranspileOptions(targetBasis: .ibmEagle, optimizationLevel: 0).makePasses()
        XCTAssertFalse(off.contains { $0 is SolovayKitaevPass })
        XCTAssertFalse(off.contains { $0 is KAKSynthesisPass })

        let on = try TranspileOptions(
            targetBasis: .ibmEagle,
            optimizationLevel: 0,
            enableSolovayKitaev: true,
            solovayKitaevEpsilon: 0.05
        ).makePasses()
        XCTAssertTrue(on.contains { $0 is SolovayKitaevPass })
        XCTAssertEqual(SolovayKitaevPass.passID, "quantumkit.solovay_kitaev")
        let skIdx = on.firstIndex { $0 is SolovayKitaevPass }!
        let basisIdx = on.firstIndex { $0 is BasisTranslatorPass }!
        XCTAssertLessThan(skIdx, basisIdx, "SK must run before basis")

        let defaultOpts = TranspileOptions(targetBasis: .ibmEagle, optimizationLevel: 0)
        let explicitOff = TranspileOptions(
            targetBasis: .ibmEagle,
            optimizationLevel: 0,
            enableSolovayKitaev: false
        )
        let a = try Transpiler.transpile(other, options: defaultOpts)
        let b = try Transpiler.transpile(other, options: explicitOff)
        XCTAssertEqual(a.gates, b.gates)
    }

    func testSolovayKitaevFlagOnRewritesTinyUnitary1Fixture() throws {
        // Exact H — discrete net hits it; without SK, ibmEagle basis rejects unitary1.
        let invSqrt2 = 1 / sqrt(2.0)
        let hadamard: [ComplexAmplitude] = [
            ComplexAmplitude(real: QFloat(invSqrt2), imaginary: 0),
            ComplexAmplitude(real: QFloat(invSqrt2), imaginary: 0),
            ComplexAmplitude(real: QFloat(invSqrt2), imaginary: 0),
            ComplexAmplitude(real: QFloat(-invSqrt2), imaginary: 0),
        ]
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.apply(.unitary1(matrix: hadamard, target: 0))

        XCTAssertThrowsError(
            try Transpiler.transpile(
                circuit,
                options: TranspileOptions(targetBasis: .ibmEagle, optimizationLevel: 0)
            )
        )

        let withSK = try Transpiler.transpile(
            circuit,
            options: TranspileOptions(
                targetBasis: .ibmEagle,
                optimizationLevel: 0,
                enableSolovayKitaev: true,
                solovayKitaevEpsilon: 0.1,
                solovayKitaevMaxRefinementIterations: 4
            )
        )
        XCTAssertFalse(
            withSK.gates.contains { if case .unitary1 = $0 { return true }; return false },
            "flags on: SK must rewrite unitary1 before basis"
        )
        XCTAssertFalse(withSK.gates.isEmpty)
        // After SK + ibmEagle basis, only native letters remain.
        for gate in withSK.gates {
            XCTAssertTrue(
                BasisGateSet.ibmEagle.contains(gate),
                "unexpected non-basis gate \(gate)"
            )
        }
    }

    func testSolovayKitaevFailsClearlyWhenDepthTooSmall() throws {
        let net = Discrete1QNet.build(maxWordLength: 3, calibrationSampleCount: 0)
        let synthesizer = SolovayKitaevSynthesizer(net: net, commutatorExpansionRounds: 0)
        var rng = QuantumRNG.seeded(99)
        var sawFailure = false
        for _ in 0..<32 {
            let target = Discrete1QNet.sampleHaarSU2(rng: &rng)
            do {
                _ = try synthesizer.approximate(
                    target,
                    epsilon: 1e-4,
                    maxRefinementIterations: 0
                )
            } catch let SolovayKitaevError.approximationFailed(achieved, epsilon, iterations) {
                XCTAssertEqual(epsilon, 1e-4, accuracy: 0)
                XCTAssertEqual(iterations, 0)
                XCTAssertGreaterThan(achieved, epsilon)
                sawFailure = true
                break
            }
        }
        XCTAssertTrue(sawFailure, "expected approximationFailed for tiny net at 0 refinement iterations")
    }

    func testSolovayKitaevDeterministicForFixedSeedTargets() throws {
        let net = Discrete1QNet.build(maxWordLength: 8, calibrationSampleCount: 0)
        let synthesizer = SolovayKitaevSynthesizer(net: net)
        var rngA = QuantumRNG.seeded(12345)
        var rngB = QuantumRNG.seeded(12345)
        let targetA = Discrete1QNet.sampleHaarSU2(rng: &rngA)
        let targetB = Discrete1QNet.sampleHaarSU2(rng: &rngB)
        XCTAssertEqual(targetA, targetB)

        let a = try synthesizer.approximate(targetA, epsilon: 0.05, maxRefinementIterations: 12)
        let b = try synthesizer.approximate(targetB, epsilon: 0.05, maxRefinementIterations: 12)
        XCTAssertEqual(a.gates, b.gates)
        XCTAssertEqual(a.refinementIterations, b.refinementIterations)
        XCTAssertEqual(a.distance, b.distance, accuracy: 0)
    }

    private static func isCliffordTLetter(_ gate: Gate) -> Bool {
        switch gate {
        case .h, .t, .tdg, .s, .sdg:
            return true
        default:
            return false
        }
    }

    private static func skCircuitUnitaryAmplitudes(gates: [Gate]) throws -> [ComplexAmplitude] {
        var circuit = try QuantumCircuit(qubitCount: 1)
        for gate in gates {
            try circuit.apply(gate)
        }
        let matrix = try CircuitUnitary.build(circuit: circuit)
        return matrix.elements.map {
            ComplexAmplitude(real: QFloat($0.re), imaginary: QFloat($0.im))
        }
    }
}
