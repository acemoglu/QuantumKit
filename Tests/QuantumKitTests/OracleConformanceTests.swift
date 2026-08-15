import XCTest
@testable import QuantumKit

/// Frozen reference-oracle conformance (item 40).
///
/// Source of truth: ``Resources/ReferenceOracles.json`` (analytic; no Aer/Stim).
/// Cross-checks CPU SV (and cheap DM) amplitudes / Born / Pauli expectations.
/// Clifford-tagged entries also spot-check ``StabilizerBackend`` shot histograms.
extension QuantumKitTests {

    func testFrozenOracleCatalogLoadsAndHasExpectedEntries() throws {
        let catalog = try ReferenceOracleCatalog.load()
        XCTAssertEqual(catalog.version, 1)
        let ids = Set(catalog.entries.map(\.id))
        XCTAssertTrue(ids.isSuperset(of: [
            "zero_state", "plus_state", "bell_phi_plus", "ghz3", "ry_pi_over_3"
        ]))
        XCTAssertTrue(catalog.entries.contains { $0.stabilizerClifford == true })
    }

    func testFrozenOraclesMatchCPUStatevectorAmplitudesProbsAndPaulis() throws {
        let catalog = try ReferenceOracleCatalog.load()
        let engine = CPUStatevectorEngine()

        for entry in catalog.entries {
            let circuit = try ReferenceOracleCatalog.makeCircuit(entry)
            let state = try CPUStateVector(qubitCount: entry.qubitCount)
            _ = try engine.execute(circuit, on: state)

            XCTAssertEqual(
                state.real.count, entry.amplitudesReal.count,
                "amplitude width mismatch for \(entry.id)"
            )
            for index in state.real.indices {
                XCTAssertEqual(
                    state.real[index], entry.amplitudesReal[index],
                    accuracy: catalog.amplitudeTolerance,
                    "SV real[\(index)] for \(entry.id)"
                )
                XCTAssertEqual(
                    state.imag[index], entry.amplitudesImag[index],
                    accuracy: catalog.amplitudeTolerance,
                    "SV imag[\(index)] for \(entry.id)"
                )
            }

            let probs = state.probabilitiesDouble()
            for index in probs.indices {
                XCTAssertEqual(
                    probs[index], entry.probabilities[index],
                    accuracy: catalog.probabilityTolerance,
                    "SV p[\(index)] for \(entry.id)"
                )
            }

            for pe in entry.pauliExpectations {
                let map = try ReferenceOracleCatalog.pauliMap(pe)
                let value = try QuantumMeasurement.expectation(state: state, paulis: map)
                XCTAssertEqual(
                    Double(value), pe.value,
                    accuracy: catalog.pauliTolerance,
                    "⟨\(pe.paulis)⟩ for \(entry.id)"
                )
            }
        }
    }

    func testFrozenOraclesMatchCPUDensityMatrixProbsAndPaulis() throws {
        let catalog = try ReferenceOracleCatalog.load()
        let engine = CPUDensityMatrixEngine()

        for entry in catalog.entries {
            let circuit = try ReferenceOracleCatalog.makeCircuit(entry)
            let density = try CPUDensityMatrix(qubitCount: entry.qubitCount)
            _ = try engine.execute(circuit, on: density)

            let probs = density.probabilitiesDouble()
            for index in probs.indices {
                XCTAssertEqual(
                    probs[index], entry.probabilities[index],
                    accuracy: catalog.probabilityTolerance,
                    "DM p[\(index)] for \(entry.id)"
                )
            }

            for pe in entry.pauliExpectations {
                let map = try ReferenceOracleCatalog.pauliMap(pe)
                let value = try QuantumMeasurement.expectation(density: density, paulis: map)
                XCTAssertEqual(
                    Double(value), pe.value,
                    accuracy: catalog.pauliTolerance,
                    "DM ⟨\(pe.paulis)⟩ for \(entry.id)"
                )
            }
        }
    }

    func testFrozenCliffordOraclesMatchStabilizerBackendHistogram() throws {
        let catalog = try ReferenceOracleCatalog.load()
        let clifford = catalog.entries.filter { $0.stabilizerClifford == true }
        XCTAssertFalse(clifford.isEmpty)

        let backend = StabilizerBackend()
        let shots = 6000
        // Seeded for reproducibility; TV threshold leaves room for multinomial noise.
        let maxTV = 0.05

        for entry in clifford {
            let circuit = try ReferenceOracleCatalog.makeCircuit(entry)
            let result = try backend.run(
                circuit: circuit,
                options: QuantumRunOptions(seed: 4040, shots: shots)
            )
            let counts = try XCTUnwrap(result.shotCounts?.counts)
            let tv = ShotStatistics.totalVariation(
                counts: counts,
                reference: entry.probabilities,
                shots: shots
            )
            XCTAssertLessThanOrEqual(
                tv, maxTV,
                "Stabilizer TV=\(tv) exceeds \(maxTV) for \(entry.id)"
            )
        }
    }
}
