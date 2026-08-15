import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - C12 Choi / superoperator → kraus1Q

    func testFromChoi1QRejectsBadShape() {
        let bad = [ComplexAmplitude](
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: 9
        )
        XCTAssertThrowsError(try QuantumChannel.fromChoi1Q(bad)) { error in
            guard case QuantumChannelError.invalidProcessMatrixDimension(let count, let expected) = error else {
                return XCTFail("expected invalidProcessMatrixDimension, got \(error)")
            }
            XCTAssertEqual(count, 9)
            XCTAssertEqual(expected, 16)
        }
    }

    func testFromChoi1QRejectsTwoQubitChoi() {
        let twoQ = [ComplexAmplitude](
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: 256
        )
        XCTAssertThrowsError(try QuantumChannel.fromChoi1Q(twoQ)) { error in
            guard case QuantumChannelError.multiQubitProcessMatrixUnsupported(let n) = error else {
                return XCTFail("expected multiQubitProcessMatrixUnsupported, got \(error)")
            }
            XCTAssertEqual(n, 2)
        }
    }

    func testFromChoi1QRejectsNonHermitian() {
        var matrix = ProcessMatrixTestSupport.zero4x4()
        // |0⟩⟨0| ⊗ |0⟩⟨0| plus a non-Hermitian off-diagonal.
        matrix[0] = ComplexAmplitude(real: 1, imaginary: 0)
        matrix[1] = ComplexAmplitude(real: 0.5, imaginary: 0)
        // matrix[4] left at 0 ⇒ |H01 - conj(H10)| = 0.5
        XCTAssertThrowsError(try QuantumChannel.fromChoi1Q(matrix, hermiticityTolerance: 1e-6)) { error in
            guard case QuantumChannelError.choiNotHermitian = error else {
                return XCTFail("expected choiNotHermitian, got \(error)")
            }
        }
    }

    func testFromChoi1QRejectsNegativeEigenvalue() {
        // diag(1, 1, 1, -0.1) — Hermitian but not PSD beyond tol.
        var matrix = ProcessMatrixTestSupport.zero4x4()
        matrix[0] = ComplexAmplitude(real: 1, imaginary: 0)
        matrix[5] = ComplexAmplitude(real: 1, imaginary: 0)
        matrix[10] = ComplexAmplitude(real: 1, imaginary: 0)
        matrix[15] = ComplexAmplitude(real: -0.1, imaginary: 0)
        XCTAssertThrowsError(
            try QuantumChannel.fromChoi1Q(matrix, eigenvalueTolerance: 1e-6)
        ) { error in
            guard case QuantumChannelError.choiNotPositiveSemidefinite = error else {
                return XCTFail("expected choiNotPositiveSemidefinite, got \(error)")
            }
        }
    }

    func testFromChoi1QDepolarizingRoundTripMatchesNative() throws {
        let p: QFloat = 0.2
        let kraus = ProcessMatrixTestSupport.depolarizingKraus(probability: p)
        let choi = ProcessMatrixTestSupport.choiFromKraus1Q(kraus)
        let imported = try QuantumChannel.fromChoi1Q(choi)

        try ProcessMatrixTestSupport.assertLocalizedChannelsMatchOnGroundState(
            native: .depolarizing(probability: p),
            imported: imported,
            accuracy: 1e-5
        )
    }

    func testFromChoi1QAmplitudeDampingRoundTripMatchesNative() throws {
        let gamma: QFloat = 0.35
        let keep = sqrt(max(0, 1 - gamma))
        let relax = sqrt(max(0, gamma))
        let kraus: [[ComplexAmplitude]] = [
            [
                ComplexAmplitude(real: 1, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: keep, imaginary: 0),
            ],
            [
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: relax, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
            ],
        ]
        let choi = ProcessMatrixTestSupport.choiFromKraus1Q(kraus)
        let imported = try QuantumChannel.fromChoi1Q(choi)

        try ProcessMatrixTestSupport.assertLocalizedChannelsMatchOnGroundState(
            native: .amplitudeDamping(probability: gamma),
            imported: imported,
            accuracy: 1e-5
        )
        // Also check |1⟩ population after X then channel.
        try ProcessMatrixTestSupport.assertLocalizedChannelsMatchAfterX(
            native: .amplitudeDamping(probability: gamma),
            imported: imported,
            accuracy: 1e-5
        )
    }

    func testFromSuperoperator1QMatchesChoiImport() throws {
        let p: QFloat = 0.15
        let kraus = ProcessMatrixTestSupport.depolarizingKraus(probability: p)
        let choi = ProcessMatrixTestSupport.choiFromKraus1Q(kraus)
        let superop = ProcessMatrixTestSupport.superoperatorFromChoi1Q(choi)

        let fromChoi = try QuantumChannel.fromChoi1Q(choi)
        let fromSuper = try QuantumChannel.fromSuperoperator1Q(superop)

        try ProcessMatrixTestSupport.assertLocalizedChannelsMatchOnGroundState(
            native: fromChoi,
            imported: fromSuper,
            accuracy: 1e-5
        )
    }

    func testFromChoi1QWiresIntoNoiseModelLikeAnyChannel() throws {
        let p: QFloat = 0.1
        let choi = ProcessMatrixTestSupport.choiFromKraus1Q(
            ProcessMatrixTestSupport.depolarizingKraus(probability: p)
        )
        let channel = try QuantumChannel.fromChoi1Q(choi)
        let noise = NoiseModel().adding(channel, for: .gate(.z))
        XCTAssertEqual(noise.localizedRules.count, 1)
        XCTAssertTrue(noise.hasLocalizedGateNoise)
        guard case .kraus1Q(let ops) = noise.localizedRules[0].channel else {
            return XCTFail("expected kraus1Q attachment")
        }
        XCTAssertFalse(ops.isEmpty)
    }
}

/// Shared builders for C12 process-matrix tests (column-vec / Choi conventions).
private enum ProcessMatrixTestSupport {

    static func zero4x4() -> [ComplexAmplitude] {
        [ComplexAmplitude](
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: 16
        )
    }

    static func depolarizingKraus(probability p: QFloat) -> [[ComplexAmplitude]] {
        let keep = sqrt(max(0, 1 - p))
        let jump = sqrt(max(0, p / 3))
        return [
            [
                ComplexAmplitude(real: keep, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: keep, imaginary: 0),
            ],
            [
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: jump, imaginary: 0),
                ComplexAmplitude(real: jump, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
            ],
            [
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: -jump),
                ComplexAmplitude(real: 0, imaginary: jump),
                ComplexAmplitude(real: 0, imaginary: 0),
            ],
            [
                ComplexAmplitude(real: jump, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: 0, imaginary: 0),
                ComplexAmplitude(real: -jump, imaginary: 0),
            ],
        ]
    }

    /// `J = Σ_k vec(E_k) vec(E_k)†` with column-vec matching `fromChoi1Q`.
    static func choiFromKraus1Q(_ operators: [[ComplexAmplitude]]) -> [ComplexAmplitude] {
        var choi = zero4x4()
        for op in operators {
            // column-vec from row-major [E00, E01, E10, E11] → [E00, E10, E01, E11]
            let v: [ComplexAmplitude] = [op[0], op[2], op[1], op[3]]
            for i in 0..<4 {
                for j in 0..<4 {
                    let re = v[i].real * v[j].real + v[i].imaginary * v[j].imaginary
                    let im = v[i].imaginary * v[j].real - v[i].real * v[j].imaginary
                    let idx = i * 4 + j
                    choi[idx] = ComplexAmplitude(
                        real: choi[idx].real + re,
                        imaginary: choi[idx].imaginary + im
                    )
                }
            }
        }
        return choi
    }

    /// Inverse of `choiFromSuperoperator1Q`: `S[a+2*b, i+2*j] = J[i*2+a, j*2+b]`.
    static func superoperatorFromChoi1Q(_ choi: [ComplexAmplitude]) -> [ComplexAmplitude] {
        let d = 2
        var superop = zero4x4()
        for i in 0..<d {
            for a in 0..<d {
                for j in 0..<d {
                    for b in 0..<d {
                        let choiIndex = (i * d + a) * 4 + (j * d + b)
                        let sopIndex = (a + d * b) * 4 + (i + d * j)
                        superop[sopIndex] = choi[choiIndex]
                    }
                }
            }
        }
        return superop
    }

    static func assertLocalizedChannelsMatchOnGroundState(
        native: QuantumChannel,
        imported: QuantumChannel,
        accuracy: QFloat
    ) throws {
        let engine = CPUDensityMatrixEngine()

        let densityNative = try CPUDensityMatrix(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.z(0)
        _ = try engine.execute(
            circuit,
            on: densityNative,
            noise: NoiseModel().adding(native, for: .gate(.z))
        )

        let densityImported = try CPUDensityMatrix(qubitCount: 1)
        _ = try engine.execute(
            circuit,
            on: densityImported,
            noise: NoiseModel().adding(imported, for: .gate(.z))
        )

        let pN = densityNative.probabilities()
        let pI = densityImported.probabilities()
        XCTAssertEqual(pN[0], pI[0], accuracy: accuracy)
        XCTAssertEqual(pN[1], pI[1], accuracy: accuracy)
    }

    static func assertLocalizedChannelsMatchAfterX(
        native: QuantumChannel,
        imported: QuantumChannel,
        accuracy: QFloat
    ) throws {
        let engine = CPUDensityMatrixEngine()

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        let densityNative = try CPUDensityMatrix(qubitCount: 1)
        _ = try engine.execute(
            circuit,
            on: densityNative,
            noise: NoiseModel().adding(native, for: .gate(.x))
        )

        let densityImported = try CPUDensityMatrix(qubitCount: 1)
        _ = try engine.execute(
            circuit,
            on: densityImported,
            noise: NoiseModel().adding(imported, for: .gate(.x))
        )

        let pN = densityNative.probabilities()
        let pI = densityImported.probabilities()
        XCTAssertEqual(pN[0], pI[0], accuracy: accuracy)
        XCTAssertEqual(pN[1], pI[1], accuracy: accuracy)
    }
}
