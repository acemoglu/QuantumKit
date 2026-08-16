import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - E8 Pauli twirling lite

    func testPauliTwirlingConjugationPreservesHadamardAndCX() throws {
        // H: X↔Z
        XCTAssertEqual(PauliTwirling.conjugatePauliString(gate: .h(target: 0), paulis: [.x]), [.z])
        XCTAssertEqual(PauliTwirling.conjugatePauliString(gate: .h(target: 0), paulis: [.z]), [.x])

        // CX: Xc → Xc Xt, Zt → Zc Zt
        let cx = Gate.cx(control: 0, target: 1)
        XCTAssertEqual(
            PauliTwirling.conjugatePauliString(gate: cx, paulis: [.x, .i]),
            [.x, .x]
        )
        XCTAssertEqual(
            PauliTwirling.conjugatePauliString(gate: cx, paulis: [.i, .z]),
            [.z, .z]
        )
    }

    func testPauliTwirlingISWAPConjugationIsNotSWAP() throws {
        let iswap = Gate.iswap(q1: 0, q2: 1)
        let swap = Gate.swap(q1: 0, q2: 1)
        // SWAP swaps factors; iSWAP maps X⊗I → Z⊗Y (phase dropped).
        XCTAssertEqual(
            PauliTwirling.conjugatePauliString(gate: swap, paulis: [.x, .i]),
            [.i, .x]
        )
        XCTAssertEqual(
            PauliTwirling.conjugatePauliString(gate: iswap, paulis: [.x, .i]),
            [.z, .y]
        )
        XCTAssertEqual(
            PauliTwirling.conjugatePauliString(gate: iswap, paulis: [.i, .x]),
            [.y, .z]
        )
        XCTAssertEqual(
            PauliTwirling.conjugatePauliString(gate: iswap, paulis: [.z, .i]),
            [.i, .z]
        )
    }

    func testPauliTwirlingISWAPRewritePreservesNoiselessExpectation() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.iswap(0, 1)
        let sites = try PauliTwirling.twirlSites(in: circuit)
        let iswapIndex = try XCTUnwrap(sites.first { circuit.gates[$0] == .iswap(q1: 0, q2: 1) })
        // Twirl only the iSWAP site with P = X⊗I → P' = Z⊗Y.
        let leftBySite = sites.map { site -> [Pauli] in
            site == iswapIndex ? [.x, .i] : [.i]
        }
        let twirled = try PauliTwirling.circuitByTwirlingSites(
            circuit,
            siteGateIndices: sites,
            leftPaulis: leftBySite
        )
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let ideal = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(),
            estimatorOptions: .exact
        )
        let rewritten = try Estimator().run(
            circuit: twirled,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(),
            estimatorOptions: .exact
        )
        XCTAssertEqual(ideal.value, rewritten.value, accuracy: 1e-6)
    }

    func testPauliTwirlingRewritePreservesNoiselessUnitaryOnX() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let sites = try PauliTwirling.twirlSites(in: circuit)
        // Twirl with P=Y: Y → X → Y  (X Y X = −Y; phase dropped ⇒ right = Y)
        let twirled = try PauliTwirling.circuitByTwirlingSites(
            circuit,
            siteGateIndices: sites,
            leftPaulis: [[.y]]
        )
        // Exact ⟨Z⟩ on |0⟩ after X is −1; after Y X Y = X (up to phase) same.
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let ideal = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(),
            estimatorOptions: .exact
        )
        let rewritten = try Estimator().run(
            circuit: twirled,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(),
            estimatorOptions: .exact
        )
        XCTAssertEqual(ideal.value, rewritten.value, accuracy: 1e-6)
        XCTAssertEqual(ideal.value, -1, accuracy: 1e-6)
    }

    /// Exhaustive noiseless rewrite check for **every** Clifford twirl site.
    ///
    /// Uses non-|0⟩ RX prep (not a twirl site) and all Pauli left-strings on the gate
    /// support; compares single-qubit ⟨X/Y/Z⟩. Catches composition-order bugs (DCX) and
    /// hand-table errors that ⟨Z⟩ on |0⟩ can miss.
    func testPauliTwirlingAllSitesRewritePreserveNoiselessExpectations() throws {
        func expect(_ circuit: QuantumCircuit, label: String) throws -> QFloat {
            let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: label))
            return try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: CPUStatevectorBackend(),
                options: QuantumRunOptions(),
                estimatorOptions: .exact
            ).value
        }
        func allPauliStrings(width: Int) -> [[Pauli]] {
            let letters: [Pauli] = [.i, .x, .y, .z]
            var out: [[Pauli]] = [[]]
            for _ in 0..<width {
                out = out.flatMap { prefix in letters.map { prefix + [$0] } }
            }
            return out
        }
        func check(name: String, qubits: Int, apply: (inout QuantumCircuit) throws -> Void) throws {
            let labels = (0..<qubits).flatMap { q in ["X\(q)", "Y\(q)", "Z\(q)"] }
            for left in allPauliStrings(width: qubits) {
                var circuit = try QuantumCircuit(qubitCount: qubits)
                for q in 0..<qubits {
                    try circuit.rx(theta: QFloat(0.37 + Double(q) * 0.11), q)
                }
                try apply(&circuit)
                let sites = try PauliTwirling.twirlSites(in: circuit)
                let focus = try XCTUnwrap(sites.last, "\(name): missing twirl site")
                let leftBySite = sites.map { site -> [Pauli] in
                    site == focus ? left : Array(repeating: .i, count: circuit.gates[site].affectedQubits.count)
                }
                let twirled = try PauliTwirling.circuitByTwirlingSites(
                    circuit,
                    siteGateIndices: sites,
                    leftPaulis: leftBySite
                )
                for label in labels {
                    let ideal = try expect(circuit, label: label)
                    let rewritten = try expect(twirled, label: label)
                    XCTAssertEqual(
                        ideal,
                        rewritten,
                        accuracy: 1e-5,
                        "\(name) P=\(left) <\(label)>"
                    )
                }
            }
        }

        // 1Q Cliffords
        try check(name: "H", qubits: 1) { try $0.h(0) }
        try check(name: "X", qubits: 1) { try $0.x(0) }
        try check(name: "Y", qubits: 1) { try $0.y(0) }
        try check(name: "Z", qubits: 1) { try $0.z(0) }
        try check(name: "S", qubits: 1) { try $0.s(0) }
        try check(name: "Sdg", qubits: 1) { try $0.sdg(0) }
        try check(name: "SX", qubits: 1) { try $0.sx(0) }
        try check(name: "SXdg", qubits: 1) { try $0.sxdg(0) }

        // 2Q Cliffords (both argument orders where meaningful)
        try check(name: "CX01", qubits: 2) { try $0.cx(0, 1) }
        try check(name: "CX10", qubits: 2) { try $0.cx(1, 0) }
        try check(name: "CZ01", qubits: 2) { try $0.cz(0, 1) }
        try check(name: "CZ10", qubits: 2) { try $0.cz(1, 0) }
        try check(name: "SWAP", qubits: 2) { try $0.swap(0, 1) }
        try check(name: "iSWAP01", qubits: 2) { try $0.iswap(0, 1) }
        try check(name: "iSWAP10", qubits: 2) { try $0.iswap(1, 0) }
        try check(name: "DCX01", qubits: 2) { try $0.dcx(0, 1) }
        try check(name: "DCX10", qubits: 2) { try $0.dcx(1, 0) }
    }

    func testPauliTwirlingDisabledMatchesBaselineEstimator() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let backend = CPUStatevectorBackend()

        let baseline = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 21),
            estimatorOptions: EstimatorOptions(shots: 256)
        )
        let explicitOff = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 21, resilience: .disabled),
            estimatorOptions: EstimatorOptions(shots: 256, resilience: .disabled)
        )
        XCTAssertEqual(baseline.value, explicitOff.value)
        XCTAssertNil(baseline.pauliTwirling)
        XCTAssertEqual(baseline.metadata.pipelineHash, explicitOff.metadata.pipelineHash)
    }

    func testPipelineFingerprintIncludesPauliTwirling() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let base = QuantumRunOptions(seed: 3, shots: 64)
        let withTwirl = QuantumRunOptions(
            seed: 3,
            shots: 64,
            resilience: ResilienceOptions(pauliTwirling: .default)
        )
        let h0 = PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: base)
        let h1 = PipelineFingerprint.hash(circuit: circuit, method: .statevector, options: withTwirl)
        XCTAssertNotEqual(h0, h1)
    }

    func testPauliTwirlingSeededRepeatableAndNoiselessUnbiased() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.z(0)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let backend = CPUStatevectorBackend()
        let ideal: QFloat = -1

        let twirlOpts = EstimatorOptions(
            shots: 2048,
            resilience: ResilienceOptions(
                pauliTwirling: PauliTwirlingOptions(ensembleSize: 64)
            )
        )
        let a = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 99),
            estimatorOptions: twirlOpts
        )
        let b = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 99),
            estimatorOptions: twirlOpts
        )
        XCTAssertEqual(a.value, b.value)
        XCTAssertEqual(a.metadata.pipelineHash, b.metadata.pipelineHash)
        let meta = try XCTUnwrap(a.pauliTwirling)
        XCTAssertEqual(meta.siteCount, 3)
        XCTAssertEqual(meta.ensembleSize, 64)
        XCTAssertEqual(meta.ensembleOverhead, 64)
        XCTAssertEqual(meta.shotsPerMember, 32)

        // Noiseless: twirled ensemble must match untwirled shot estimate / ideal within tol.
        let untwirled = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 99),
            estimatorOptions: EstimatorOptions(shots: 2048)
        )
        XCTAssertEqual(a.value, ideal, accuracy: 0.08)
        XCTAssertEqual(untwirled.value, ideal, accuracy: 0.08)
        XCTAssertEqual(a.value, untwirled.value, accuracy: 0.12)
        XCTAssertNotEqual(a.metadata.pipelineHash, untwirled.metadata.pipelineHash)
    }

    func testPauliTwirlingIncompatibleWithZNEAndPEC() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let noise = NoiseModel(depolarizingProbability: 0.05)

        XCTAssertThrowsError(
            try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: CPUStatevectorBackend(),
                options: QuantumRunOptions(noise: noise, seed: 1),
                estimatorOptions: EstimatorOptions(
                    shots: 32,
                    resilience: ResilienceOptions(
                        zne: .default,
                        pauliTwirling: .default
                    )
                )
            )
        ) { error in
            XCTAssertEqual(error as? PauliTwirlingError, .incompatibleWithZNE)
        }

        XCTAssertThrowsError(
            try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: CPUStatevectorBackend(),
                options: QuantumRunOptions(noise: noise, seed: 1),
                estimatorOptions: EstimatorOptions(
                    shots: 32,
                    resilience: ResilienceOptions(
                        pec: .default,
                        pauliTwirling: .default
                    )
                )
            )
        ) { error in
            XCTAssertEqual(error as? PauliTwirlingError, .incompatibleWithPEC)
        }
    }

    func testPauliTwirlingRejectsEnsembleExceedingShots() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))

        XCTAssertThrowsError(
            try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: CPUStatevectorBackend(),
                options: QuantumRunOptions(seed: 3),
                estimatorOptions: EstimatorOptions(
                    shots: 32,
                    resilience: ResilienceOptions(
                        pauliTwirling: PauliTwirlingOptions(ensembleSize: 100)
                    )
                )
            )
        ) { error in
            guard case PauliTwirlingError.ensembleExceedsShots(let ensemble, let shots) = error else {
                return XCTFail("expected ensembleExceedsShots, got \(error)")
            }
            XCTAssertEqual(ensemble, 100)
            XCTAssertEqual(shots, 32)
        }
    }

    func testPauliTwirlingExactPathIgnoresTwirlOptions() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let backend = CPUStatevectorBackend()

        let baseline = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 12),
            estimatorOptions: .exact
        )
        let withTwirlKnob = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(seed: 12),
            estimatorOptions: EstimatorOptions(
                resilience: ResilienceOptions(pauliTwirling: .default)
            )
        )

        XCTAssertEqual(baseline.value, withTwirlKnob.value)
        XCTAssertEqual(baseline.metadata.pipelineHash, withTwirlKnob.metadata.pipelineHash)
        XCTAssertNil(baseline.pauliTwirling)
        XCTAssertNil(withTwirlKnob.pauliTwirling)
        XCTAssertNil(baseline.shots)
        XCTAssertNil(withTwirlKnob.shots)
    }

    func testPauliTwirlingInheritsRunOptionsReadoutMitigation() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let matrix = try ReadoutConfusionMatrix.singleQubit(p01: 0.0, p10: 0.0)

        let withoutInherit = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 4),
            estimatorOptions: EstimatorOptions(
                shots: 64,
                resilience: ResilienceOptions(
                    pauliTwirling: PauliTwirlingOptions(ensembleSize: 16)
                )
            )
        )
        let withInherit = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(
                seed: 4,
                resilience: ResilienceOptions(readoutMitigation: matrix)
            ),
            estimatorOptions: EstimatorOptions(
                shots: 64,
                resilience: ResilienceOptions(
                    pauliTwirling: PauliTwirlingOptions(ensembleSize: 16)
                )
            )
        )

        // Identity confusion matrix: values match, but fingerprint must include readout.
        XCTAssertEqual(withoutInherit.value, withInherit.value, accuracy: 1e-6)
        XCTAssertNotEqual(
            withoutInherit.metadata.pipelineHash,
            withInherit.metadata.pipelineHash
        )
        XCTAssertNotNil(withInherit.pauliTwirling)
    }

    /// Coherent over-rotation on a twirl site: untwirled ⟨Z⟩ is first-order biased;
    /// the full 1Q Pauli twirl average cancels that linear bias (RC motivation).
    func testPauliTwirlingReducesCoherentOverRotationBias() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let epsilon: QFloat = 0.40
        // Noise only on H — inserted frame Paulis are X/Y/Z, so they do not pick up this rule.
        let noise = NoiseModel().adding(
            .coherentOverRotation(axis: .y, angle: epsilon),
            for: .gate(.h)
        )
        let backend = CPUDensityMatrixBackend()

        let raw = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: backend,
            options: QuantumRunOptions(noise: noise),
            estimatorOptions: .exact
        )
        // Ideal noiseless ⟨Z⟩ after H is 0; coherent RY(ε) after H biases it.
        XCTAssertGreaterThan(abs(raw.value), 0.15, "coherent RY(\(epsilon)) should bias ⟨Z⟩ away from 0")

        let sites = try PauliTwirling.twirlSites(in: circuit)
        let letters: [Pauli] = [.i, .x, .y, .z]
        var sum: QFloat = 0
        for left in letters {
            let twirled = try PauliTwirling.circuitByTwirlingSites(
                circuit,
                siteGateIndices: sites,
                leftPaulis: [[left]]
            )
            let member = try Estimator().run(
                circuit: twirled,
                hamiltonian: hamiltonian,
                backend: backend,
                options: QuantumRunOptions(noise: noise),
                estimatorOptions: .exact
            )
            sum += member.value
        }
        let twirlMean = sum / QFloat(letters.count)

        XCTAssertLessThan(
            abs(twirlMean),
            abs(raw.value) * 0.35,
            "twirl average |⟨Z⟩|=\(twirlMean) should be much smaller than raw |⟨Z⟩|=\(raw.value)"
        )
        XCTAssertEqual(twirlMean, 0, accuracy: 0.05)
    }

    func testPauliTwirlingRejectsCircuitWithoutCliffordSites() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.rx(theta: 0.3, 0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))

        XCTAssertThrowsError(
            try Estimator().run(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: CPUStatevectorBackend(),
                options: QuantumRunOptions(seed: 2),
                estimatorOptions: EstimatorOptions(
                    shots: 32,
                    resilience: ResilienceOptions(pauliTwirling: .default)
                )
            )
        ) { error in
            XCTAssertEqual(error as? PauliTwirlingError, .emptyCircuitNoTwirlSites)
        }
    }

    func testPauliTwirlingDefaultEnsembleUsesOneShotPerMember() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))

        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 6),
            estimatorOptions: EstimatorOptions(
                shots: 48,
                resilience: ResilienceOptions(pauliTwirling: .default)
            )
        )
        let meta = try XCTUnwrap(result.pauliTwirling)
        XCTAssertEqual(meta.ensembleSize, 48)
        XCTAssertEqual(meta.shotsPerMember, 1)
        XCTAssertEqual(meta.ensembleOverhead, 48)
        XCTAssertEqual(result.shots, 48)
    }

    func testInactiveZNEDoesNotBlockPauliTwirling() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        let hamiltonian = try Hamiltonian(PauliTerm(coefficient: 1, label: "Z0"))
        let inactive = ZNEOptions(scaleFactors: [1])
        XCTAssertFalse(inactive.isActive)

        let result = try Estimator().run(
            circuit: circuit,
            hamiltonian: hamiltonian,
            backend: CPUStatevectorBackend(),
            options: QuantumRunOptions(seed: 8),
            estimatorOptions: EstimatorOptions(
                shots: 64,
                resilience: ResilienceOptions(
                    zne: inactive,
                    pauliTwirling: PauliTwirlingOptions(ensembleSize: 16)
                )
            )
        )
        XCTAssertNil(result.zne)
        XCTAssertNotNil(result.pauliTwirling)
    }
}
