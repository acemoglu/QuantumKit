import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Discrete 1Q ε₀-net (SK foundation)

    func testDiscrete1QNetNonEmptyAndContainsIdentity() {
        let net = Discrete1QNet.build(maxWordLength: 6)
        XCTAssertFalse(net.elements.isEmpty)
        XCTAssertGreaterThan(net.count, 1)
        XCTAssertEqual(net.generatingSet, .ht)
        XCTAssertTrue(net.includeInverses)
        XCTAssertEqual(net.elements[0].wordLength, 0)
        XCTAssertTrue(net.elements[0].gates.isEmpty)

        let identity = [
            ComplexAmplitude(real: 1, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 1, imaginary: 0)
        ]
        let nn = net.nearestNeighbor(to: identity)
        XCTAssertEqual(nn.index, 0)
        XCTAssertLessThan(nn.distance, 1e-12)
    }

    func testDiscrete1QNetLookupWithinClaimedCoveringRadius() throws {
        let net = Discrete1QNet.build(
            maxWordLength: 8,
            calibrationSampleCount: 256,
            calibrationSeed: Discrete1QNet.defaultCalibrationSeed
        )
        XCTAssertGreaterThan(net.claimedCoveringRadius, 0)
        XCTAssertLessThanOrEqual(net.empiricalCoveringRadius, net.claimedCoveringRadius)

        // Independent Haar draws (different seed from calibration).
        var rng = QuantumRNG.seeded(424242)
        var worst = 0.0
        for _ in 0..<64 {
            let target = Discrete1QNet.sampleHaarSU2(rng: &rng)
            let nn = net.nearestNeighbor(to: target)

            // Independent oracle: rebuild word via CircuitUnitary, recompute distance.
            let rebuilt = try Self.circuitUnitaryAmplitudes(gates: nn.element.gates)
            let independentDistance = Discrete1QNet.phaseAlignedFrobenius(
                target: target,
                candidate: rebuilt
            )
            XCTAssertEqual(nn.distance, independentDistance, accuracy: 1e-12)
            XCTAssertLessThan(
                Discrete1QNet.phaseAlignedFrobenius(target: nn.element.matrix, candidate: rebuilt),
                1e-12,
                "stored net matrix drifted from CircuitUnitary product of its word"
            )

            worst = max(worst, independentDistance)
            XCTAssertLessThanOrEqual(
                independentDistance,
                net.claimedCoveringRadius,
                "independent NN distance \(independentDistance) exceeded claimed covering radius \(net.claimedCoveringRadius)"
            )
        }
        XCTAssertLessThanOrEqual(worst, net.claimedCoveringRadius)
    }

    func testDiscrete1QNetBuildDeterministicForFixedSeed() {
        let a = Discrete1QNet.build(
            maxWordLength: 7,
            calibrationSampleCount: 64,
            calibrationSeed: 99
        )
        let b = Discrete1QNet.build(
            maxWordLength: 7,
            calibrationSampleCount: 64,
            calibrationSeed: 99
        )
        XCTAssertEqual(a.count, b.count)
        XCTAssertEqual(a.empiricalCoveringRadius, b.empiricalCoveringRadius)
        XCTAssertEqual(a.claimedCoveringRadius, b.claimedCoveringRadius)
        XCTAssertEqual(a.elements.map(\.wordLength), b.elements.map(\.wordLength))
        for (lhs, rhs) in zip(a.elements, b.elements) {
            XCTAssertEqual(lhs.gates, rhs.gates)
            XCTAssertEqual(lhs.matrix, rhs.matrix)
        }
    }

    func testDiscrete1QNetExactGeneratorLookup() throws {
        let net = Discrete1QNet.build(maxWordLength: 4, calibrationSampleCount: 0)
        let h = try Self.circuitUnitaryAmplitudes(gate: .h(target: 0))
        let nn = net.nearestNeighbor(to: h)
        XCTAssertLessThan(nn.distance, 1e-10)
        XCTAssertEqual(nn.element.gates, [.h(target: 0)])
    }

    func testDiscrete1QNetGeneratorsMatchCircuitUnitary() throws {
        let net = Discrete1QNet.build(maxWordLength: 3, calibrationSampleCount: 0)
        let letters: [Gate] = [.h(target: 0), .t(target: 0), .tdg(target: 0)]
        for gate in letters {
            let oracle = try Self.circuitUnitaryAmplitudes(gate: gate)
            let match = net.elements.first { $0.gates == [gate] }
            XCTAssertNotNil(match, "net missing length-1 word \(gate)")
            guard let match else { continue }
            XCTAssertLessThan(
                Discrete1QNet.phaseAlignedFrobenius(target: oracle, candidate: match.matrix),
                1e-12
            )
            for (lhs, rhs) in zip(oracle, match.matrix) {
                XCTAssertEqual(Double(lhs.real), Double(rhs.real), accuracy: 1e-12)
                XCTAssertEqual(Double(lhs.imaginary), Double(rhs.imaginary), accuracy: 1e-12)
            }
        }

        // Short composite word: stored product must match CircuitUnitary left-to-right apply.
        let word: [Gate] = [.h(target: 0), .t(target: 0), .h(target: 0)]
        let oracleWord = try Self.circuitUnitaryAmplitudes(gates: word)
        let netWord = net.elements.first { $0.gates == word }
        XCTAssertNotNil(netWord)
        guard let netWord else { return }
        XCTAssertLessThan(
            Discrete1QNet.phaseAlignedFrobenius(target: oracleWord, candidate: netWord.matrix),
            1e-12
        )
    }

    func testDiscrete1QNetDocumentsHTGeneratingSetSize() {
        // Documented defaults: {H,T} (+ T†), maxWordLength 8 → 505 distinct elements
        // (phase-collapsed); empirical covering radius ≈ 0.41 (claimed ≈ 0.52 with 1.25×).
        let net = Discrete1QNet.build()
        XCTAssertEqual(net.generatingSet, .ht)
        XCTAssertEqual(net.maxWordLength, Discrete1QNet.defaultMaxWordLength)
        XCTAssertEqual(net.count, 505)
        XCTAssertGreaterThan(net.empiricalCoveringRadius, 0.3)
        XCTAssertLessThan(net.empiricalCoveringRadius, 0.6)
        XCTAssertEqual(
            net.claimedCoveringRadius,
            net.empiricalCoveringRadius * Discrete1QNet.defaultCoveringRadiusSafetyFactor,
            accuracy: 1e-12
        )
    }

    func testDiscrete1QNetPhaseAlignedFrobeniusIdentity() {
        let i2 = [
            ComplexAmplitude(real: 1, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 1, imaginary: 0)
        ]
        let phased = [
            ComplexAmplitude(real: 0, imaginary: 1),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 0),
            ComplexAmplitude(real: 0, imaginary: 1)
        ]
        let d = Discrete1QNet.phaseAlignedFrobenius(target: i2, candidate: phased)
        XCTAssertLessThan(d, 1e-12)
        let fid = Discrete1QNet.averageGateFidelity(target: i2, candidate: phased)
        XCTAssertEqual(fid, 1.0, accuracy: 1e-12)
    }

    // MARK: - CircuitUnitary oracles (Gate semantics)

    private static func circuitUnitaryAmplitudes(gate: Gate) throws -> [ComplexAmplitude] {
        let matrix = try CircuitUnitary.matrix(for: gate, qubitCount: 1)
        return amplitudes(from: matrix)
    }

    private static func circuitUnitaryAmplitudes(gates: [Gate]) throws -> [ComplexAmplitude] {
        var circuit = try QuantumCircuit(qubitCount: 1)
        for gate in gates {
            try circuit.apply(gate)
        }
        return amplitudes(from: try CircuitUnitary.build(circuit: circuit))
    }

    private static func amplitudes(from matrix: UnitaryMatrix) -> [ComplexAmplitude] {
        matrix.elements.map {
            ComplexAmplitude(real: QFloat($0.re), imaginary: QFloat($0.im))
        }
    }
}

