import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Cartan / KAK (2Q)

    func testCartanKAKIdentity() throws {
        var i4 = [ComplexAmplitude](
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: 16
        )
        for k in 0..<4 {
            i4[k * 4 + k] = ComplexAmplitude(real: 1, imaginary: 0)
        }

        let decomp = try CartanKAK.decompose(i4)
        XCTAssertLessThanOrEqual(decomp.cxCount, 3)
        XCTAssertEqual(decomp.cxCount, 0, "identity should need no CX")
        XCTAssertEqual(decomp.interaction.x, 0, accuracy: 1e-8)
        XCTAssertEqual(decomp.interaction.y, 0, accuracy: 1e-8)
        XCTAssertEqual(decomp.interaction.z, 0, accuracy: 1e-8)

        let rebuilt = try CartanKAK.matrix(ofGates: decomp.gates)
        let frob = CartanKAK.phaseAlignedFrobenius(target: i4, candidate: rebuilt)
        let fid = CartanKAK.averageGateFidelity(target: i4, candidate: rebuilt)
        XCTAssertLessThanOrEqual(frob, CartanKAK.roundTripFrobeniusTolerance)
        XCTAssertGreaterThanOrEqual(fid, CartanKAK.fidelityFloor)
    }

    func testCartanKAKKnownCX() throws {
        // CX(0→1) on LSB qubit 0 = control.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.cx(0, 1)
        let target = try CircuitUnitary.build(circuit: circuit)
        let unitary = target.elements.map {
            ComplexAmplitude(real: QFloat($0.re), imaginary: QFloat($0.im))
        }

        let decomp = try CartanKAK.decompose(unitary)
        // XX-only Cartan interaction uses at most 2 CX (never the full 3-CX template).
        XCTAssertLessThanOrEqual(decomp.cxCount, 2)
        XCTAssertEqual(decomp.interaction.x, .pi / 4, accuracy: 1e-6)
        XCTAssertEqual(decomp.interaction.y, 0, accuracy: 1e-6)
        XCTAssertEqual(decomp.interaction.z, 0, accuracy: 1e-6)

        let rebuilt = try CartanKAK.matrix(ofGates: decomp.gates)
        let frob = CartanKAK.phaseAlignedFrobenius(target: unitary, candidate: rebuilt)
        let fid = CartanKAK.averageGateFidelity(target: unitary, candidate: rebuilt)
        XCTAssertLessThanOrEqual(frob, CartanKAK.roundTripFrobeniusTolerance)
        XCTAssertGreaterThanOrEqual(fid, CartanKAK.fidelityFloor)
    }

    func testCartanKAKLocalTensorLocal() throws {
        // U = u(θ,φ,λ) ⊗ u(θ',φ',λ') — zero Cartan vector.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.u(theta: 0.3, phi: 0.4, lambda: 0.5, 0)
        try circuit.u(theta: 0.7, phi: -0.2, lambda: 1.1, 1)
        let target = try CircuitUnitary.build(circuit: circuit)
        let unitary = target.elements.map {
            ComplexAmplitude(real: QFloat($0.re), imaginary: QFloat($0.im))
        }

        let decomp = try CartanKAK.decompose(unitary)
        XCTAssertEqual(decomp.cxCount, 0)
        XCTAssertEqual(decomp.interaction.x, 0, accuracy: 1e-6)
        XCTAssertEqual(decomp.interaction.y, 0, accuracy: 1e-6)
        XCTAssertEqual(decomp.interaction.z, 0, accuracy: 1e-6)

        let rebuilt = try CartanKAK.matrix(ofGates: decomp.gates)
        let frob = CartanKAK.phaseAlignedFrobenius(target: unitary, candidate: rebuilt)
        let fid = CartanKAK.averageGateFidelity(target: unitary, candidate: rebuilt)
        XCTAssertLessThanOrEqual(frob, CartanKAK.roundTripFrobeniusTolerance)
        XCTAssertGreaterThanOrEqual(fid, CartanKAK.fidelityFloor)
    }

    func testCartanKAKHaarRandomSU4RoundTrip() throws {
        var rng = QuantumRNG.seeded(20260816)
        var worstFrob = 0.0
        var worstFid = 1.0
        let samples = 100
        for i in 0..<samples {
            let matrix = QuantumVolume.sampleHaarSU4(rng: &rng)
            // Independent oracle: skip internal verifyRoundTrip so frob/fid asserts below are not tautological.
            let decomp: CartanKAKDecomposition
            do {
                decomp = try CartanKAK.decompose(matrix, verifyRoundTrip: false)
            } catch {
                return XCTFail("sample \(i) decompose failed: \(error)")
            }
            XCTAssertLessThanOrEqual(decomp.cxCount, 3, "sample \(i) CX count")

            let rebuilt = try CartanKAK.matrix(ofGates: decomp.gates)
            let frob = CartanKAK.phaseAlignedFrobenius(target: matrix, candidate: rebuilt)
            let fid = CartanKAK.averageGateFidelity(target: matrix, candidate: rebuilt)
            worstFrob = max(worstFrob, frob)
            worstFid = min(worstFid, fid)
            XCTAssertLessThanOrEqual(
                frob,
                CartanKAK.roundTripFrobeniusTolerance,
                "sample \(i): ‖U−V‖_F=\(frob) (binding ε)"
            )
            XCTAssertGreaterThanOrEqual(
                fid,
                CartanKAK.fidelityFloor,
                "sample \(i): fidelity=\(fid)"
            )
        }
        XCTAssertLessThanOrEqual(worstFrob, CartanKAK.roundTripFrobeniusTolerance)
        XCTAssertGreaterThanOrEqual(worstFid, CartanKAK.fidelityFloor)
    }

    func testCartanKAKRejectsNonUnitary() {
        var bad = [ComplexAmplitude](
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: 16
        )
        bad[0] = ComplexAmplitude(real: 2, imaginary: 0)
        XCTAssertThrowsError(try CartanKAK.decompose(bad, verifyRoundTrip: false))
    }

    func testCartanKAKU4GlobalPhaseStripped() throws {
        // e^{iφ} · I should decompose like identity with global phase ≈ e^{iφ}.
        let phase = 0.37
        var u = [ComplexAmplitude](
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: 16
        )
        let g = ComplexAmplitude(real: QFloat(cos(phase)), imaginary: QFloat(sin(phase)))
        for k in 0..<4 {
            u[k * 4 + k] = g
        }
        let decomp = try CartanKAK.decompose(u)
        XCTAssertEqual(decomp.cxCount, 0)

        let gp = decomp.globalPhase
        let gpMag = hypot(Double(gp.real), Double(gp.imaginary))
        XCTAssertEqual(gpMag, 1.0, accuracy: 1e-5)
        // Re(conj(globalPhase) · e^{iφ}) ≈ 1 ⇒ phases match.
        let phaseAlign = Double(gp.real) * cos(phase) + Double(gp.imaginary) * sin(phase)
        XCTAssertEqual(phaseAlign, 1.0, accuracy: 1e-4)

        let rebuilt = try CartanKAK.matrix(ofGates: decomp.gates)
        let frob = CartanKAK.phaseAlignedFrobenius(target: u, candidate: rebuilt)
        let fid = CartanKAK.averageGateFidelity(target: u, candidate: rebuilt)
        XCTAssertLessThanOrEqual(frob, CartanKAK.roundTripFrobeniusTolerance)
        XCTAssertGreaterThanOrEqual(fid, CartanKAK.fidelityFloor)
    }

    func testCartanKAKUnitarityToleranceMatchesValidation() {
        XCTAssertEqual(CartanKAK.unitarityTolerance, UnitaryValidation.unitarityTolerance)
    }
}
