import XCTest
@testable import QuantumKit

/// B16: optional cumulative global-phase metadata (SV), Born/shots remain phase-blind.
extension QuantumKitTests {

    /// Closed-form S/T contributions are exact Doubles.
    private static let phaseTol = 1e-12
    /// `unitary1` / `p` angles pass through ``QFloat`` (Float32).
    private static let phaseTolQFloat = 1e-6
    private static let bornTol = 1e-12

    func testGlobalPhase_eIphiIdentity_accumulatesPhi() throws {
        let phi = Double.pi / 3.0
        let matrix = globalPhaseIdentity(phi: phi)
        let expected = GlobalPhaseTracking.contribution(matrix: matrix, qubitWidth: 1)
        XCTAssertEqual(expected, phi, accuracy: Self.phaseTolQFloat)

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.unitary1(matrix: matrix, target: 0)

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 2)
        let result = try engine.execute(circuit, on: state)

        XCTAssertEqual(state.cumulativeGlobalPhaseRadians, expected, accuracy: Self.phaseTol)
        XCTAssertEqual(result.cumulativeGlobalPhaseRadians ?? .nan, expected, accuracy: Self.phaseTol)
        XCTAssertEqual(state.probabilitiesDouble()[0], 1.0, accuracy: Self.bornTol)
    }

    func testGlobalPhase_STStack_accumulatesKnownSum() throws {
        // S → π/4, T → π/8 ⇒ Φ = 3π/8 (det convention; amplitudes stay |0⟩).
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.s(0)
        try circuit.t(0)

        let expected = Double.pi / 4.0 + Double.pi / 8.0
        XCTAssertEqual(try GlobalPhaseTracking.contribution(of: .s(target: 0)), Double.pi / 4.0, accuracy: Self.phaseTol)
        XCTAssertEqual(try GlobalPhaseTracking.contribution(of: .t(target: 0)), Double.pi / 8.0, accuracy: Self.phaseTol)

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 1)
        let result = try engine.execute(circuit, on: state)

        XCTAssertEqual(state.cumulativeGlobalPhaseRadians, expected, accuracy: Self.phaseTol)
        XCTAssertEqual(result.cumulativeGlobalPhaseRadians ?? .nan, expected, accuracy: Self.phaseTol)
        assertComputationalZero(state.probabilitiesDouble(), qubitCount: 1)
    }

    func testGlobalPhase_bornUnchangedVsPhaseStrippedCompare() throws {
        var base = try QuantumCircuit(qubitCount: 2)
        try base.h(0)
        try base.cx(0, 1)
        try base.s(0)
        try base.t(1)

        let phi = Double.pi / 5.0
        let phaseGate = globalPhaseIdentity(phi: phi)
        let phiTracked = GlobalPhaseTracking.contribution(matrix: phaseGate, qubitWidth: 1)
        var phased = base
        try phased.unitary1(matrix: phaseGate, target: 0)

        let engine = CPUStatevectorEngine()
        let stateU = try CPUStateVector(qubitCount: 2)
        let stateUP = try CPUStateVector(qubitCount: 2)
        let resultU = try engine.execute(base, on: stateU)
        let resultUP = try engine.execute(phased, on: stateUP)

        let probsU = stateU.probabilitiesDouble()
        let probsUP = stateUP.probabilitiesDouble()
        XCTAssertEqual(probsU.count, probsUP.count)
        for index in probsU.indices {
            XCTAssertEqual(probsU[index], probsUP[index], accuracy: Self.bornTol)
        }

        let phaseU = try XCTUnwrap(resultU.cumulativeGlobalPhaseRadians)
        let phaseUP = try XCTUnwrap(resultUP.cumulativeGlobalPhaseRadians)
        XCTAssertEqual(phaseUP - phaseU, phiTracked, accuracy: Self.phaseTol)

        // Phase-stripped amplitude compare: |⟨ψ|φ⟩| ≈ 1.
        let overlap = complexOverlapSV(
            realA: stateU.real, imagA: stateU.imag,
            realB: stateUP.real, imagB: stateUP.imag
        )
        let mag = sqrt(overlap.re * overlap.re + overlap.im * overlap.im)
        XCTAssertEqual(mag, 1.0, accuracy: 1e-10)
    }

    func testGlobalPhase_backendMetadataExposesPhase_CPU() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.s(0)
        try circuit.t(0)

        let backend = CPUStatevectorBackend()
        let result = try backend.run(circuit: circuit)
        let expected = Double.pi / 4.0 + Double.pi / 8.0
        XCTAssertEqual(
            result.metadata.cumulativeGlobalPhaseRadians ?? .nan,
            expected,
            accuracy: Self.phaseTol
        )
        XCTAssertEqual(
            result.execution?.cumulativeGlobalPhaseRadians ?? .nan,
            expected,
            accuracy: Self.phaseTol
        )
        // Shot path leaves phase unset (evolve metadata not retained).
        let shots = try backend.run(
            circuit: circuit,
            options: QuantumRunOptions(seed: 7, shots: 32)
        )
        XCTAssertNil(shots.metadata.cumulativeGlobalPhaseRadians)
    }

    func testGlobalPhase_densityMatrixLeavesPhaseNil() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.s(0)
        let backend = CPUDensityMatrixBackend()
        let result = try backend.run(circuit: circuit)
        XCTAssertNil(result.metadata.cumulativeGlobalPhaseRadians)
        XCTAssertNil(result.execution?.cumulativeGlobalPhaseRadians)
    }

    func testGlobalPhase_metadataJSONRoundTripIncludesPhase() throws {
        let metadata = QuantumResultMetadata(
            method: .statevector,
            seed: 1,
            deviceName: "CPU",
            wallClockNanoseconds: 10,
            qubitCount: 1,
            gateCount: 2,
            noiseSnapshot: nil,
            cumulativeGlobalPhaseRadians: 0.375 * Double.pi
        )
        let data = try JSONEncoder().encode(metadata)
        let decoded = try JSONDecoder().decode(QuantumResultMetadata.self, from: data)
        XCTAssertEqual(decoded, metadata)
        XCTAssertEqual(
            decoded.cumulativeGlobalPhaseRadians ?? .nan,
            0.375 * Double.pi,
            accuracy: Self.phaseTol
        )
    }

    func testGlobalPhase_preB16SnapshotsDecodeMissingPhaseAsZero() throws {
        let metalJSON = Data(#"{"qubitCount":1,"real":[1],"imag":[0]}"#.utf8)
        let metal = try JSONDecoder().decode(StateVectorSnapshot.self, from: metalJSON)
        XCTAssertEqual(metal.cumulativeGlobalPhaseRadians, 0, accuracy: Self.phaseTol)
        XCTAssertEqual(metal.real, [1])
        XCTAssertEqual(metal.imag, [0])

        let cpuJSON = Data(#"{"qubitCount":1,"real":[1.0],"imag":[0.0]}"#.utf8)
        let cpu = try JSONDecoder().decode(CPUStateVectorSnapshot.self, from: cpuJSON)
        XCTAssertEqual(cpu.cumulativeGlobalPhaseRadians, 0, accuracy: Self.phaseTol)
        XCTAssertEqual(cpu.real, [1.0])
        XCTAssertEqual(cpu.imag, [0.0])

        let withPhase = StateVectorSnapshot(
            qubitCount: 1,
            real: [1],
            imag: [0],
            cumulativeGlobalPhaseRadians: Double.pi / 4
        )
        let roundTrip = try JSONDecoder().decode(
            StateVectorSnapshot.self,
            from: try JSONEncoder().encode(withPhase)
        )
        XCTAssertEqual(roundTrip, withPhase)
    }

    func testGlobalPhase_rzDoesNotContribute_pDoes() throws {
        XCTAssertEqual(try GlobalPhaseTracking.contribution(of: .rz(theta: .literal(1.2), target: 0)), 0, accuracy: Self.phaseTol)
        let theta = 0.8
        XCTAssertEqual(
            try GlobalPhaseTracking.contribution(of: .p(theta: .literal(QFloat(theta)), target: 0)),
            Double(QFloat(theta)) / 2.0,
            accuracy: Self.phaseTol
        )
    }

    func testGlobalPhase_noiseAndResetUnitariesDoNotUpdatePhi() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.s(0)
        try circuit.t(0)
        let expected = Double.pi / 4.0 + Double.pi / 8.0

        let engine = CPUStatevectorEngine()
        let state = try CPUStateVector(qubitCount: 1)
        _ = try engine.execute(circuit, on: state)
        XCTAssertEqual(state.cumulativeGlobalPhaseRadians, expected, accuracy: Self.phaseTol)

        // Noise / reset helpers apply Paulis via applyUnitaryGate — must not change Φ.
        try engine.applyUnitaryGate(.x(target: 0), on: state)
        try engine.applyUnitaryGate(.y(target: 0), on: state)
        try engine.applyUnitaryGate(.z(target: 0), on: state)
        XCTAssertEqual(state.cumulativeGlobalPhaseRadians, expected, accuracy: Self.phaseTol)

        // Stochastic reset-error path (p=1) after a circuit unitary must keep Φ = circuit-only.
        var withReset = try QuantumCircuit(
            qubitCount: 1,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: 1)]
        )
        try withReset.s(0)
        try withReset.t(0)
        try withReset.reset(0)
        let noisy = try CPUStateVector(qubitCount: 1)
        var rng: QuantumRNG = .seeded(99)
        let noise = NoiseModel(resetErrorProbability: 1.0)
        let result = try engine.executeRNG(withReset, on: noisy, rng: &rng, noise: noise)
        XCTAssertEqual(result.cumulativeGlobalPhaseRadians ?? .nan, expected, accuracy: Self.phaseTol)
        XCTAssertEqual(noisy.cumulativeGlobalPhaseRadians, expected, accuracy: Self.phaseTol)
    }

    // MARK: - Helpers

    private func globalPhaseIdentity(phi: Double) -> [ComplexAmplitude] {
        let c = QFloat(cos(phi))
        let s = QFloat(sin(phi))
        return [
            ComplexAmplitude(real: c, imaginary: s),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: c, imaginary: s),
        ]
    }

    private func assertComputationalZero(_ probs: [Double], qubitCount: Int) {
        XCTAssertEqual(probs.count, 1 << qubitCount)
        XCTAssertEqual(probs[0], 1.0, accuracy: Self.bornTol)
        for index in 1..<probs.count {
            XCTAssertEqual(probs[index], 0.0, accuracy: Self.bornTol)
        }
    }

    private func complexOverlapSV(
        realA: [Double], imagA: [Double],
        realB: [Double], imagB: [Double]
    ) -> (re: Double, im: Double) {
        var re = 0.0
        var im = 0.0
        for index in realA.indices {
            re += realA[index] * realB[index] + imagA[index] * imagB[index]
            im += realA[index] * imagB[index] - imagA[index] * realB[index]
        }
        return (re, im)
    }
}
