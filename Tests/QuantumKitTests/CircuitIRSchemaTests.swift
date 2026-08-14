import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - Circuit IR schema versioning

    func testCircuitIRSchemaCurrentIsOne() {
        XCTAssertEqual(CircuitIRSchema.current, 1)
    }

    func testQuantumCircuitRoundTripIncludesSchemaVersion() throws {
        var original = try QuantumCircuit(qubitCount: 2)
        try original.h(0)
        try original.cx(0, 1)
        try original.apply(.x(target: 1), metadata: InstructionMetadata(label: "x1"))

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(original)

        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(object?["schemaVersion"] as? Int, CircuitIRSchema.current)

        let decoded = try JSONDecoder().decode(QuantumCircuit.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    func testQuantumCircuitLegacyUnversionedMigrates() throws {
        // Pre-contract Codable shape: no schemaVersion key.
        let legacy = Data(
            #"""
            {
              "qubitCount": 2,
              "gates": [
                { "type": "h", "target": 0 },
                { "type": "cx", "control": 0, "target": 1 }
              ]
            }
            """#.utf8
        )

        let decoded = try JSONDecoder().decode(QuantumCircuit.self, from: legacy)
        XCTAssertEqual(decoded.qubitCount, 2)
        XCTAssertEqual(decoded.gates, [.h(target: 0), .cx(control: 0, target: 1)])

        // Re-encode stamps the current schema version.
        let reencoded = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(decoded)
        ) as? [String: Any]
        XCTAssertEqual(reencoded?["schemaVersion"] as? Int, CircuitIRSchema.current)
    }

    func testQuantumCircuitFutureSchemaVersionThrows() throws {
        let future = Data(
            #"""
            {
              "schemaVersion": 99,
              "qubitCount": 1,
              "gates": [{ "type": "x", "target": 0 }]
            }
            """#.utf8
        )

        XCTAssertThrowsError(try JSONDecoder().decode(QuantumCircuit.self, from: future)) { error in
            guard case CircuitIRError.unsupportedSchemaVersion(let found, let supported) = error else {
                return XCTFail("expected unsupportedSchemaVersion, got \(error)")
            }
            XCTAssertEqual(found, 99)
            XCTAssertEqual(supported, CircuitIRSchema.current)
        }
    }

    func testGateSequenceRoundTripAndLegacyMigration() throws {
        var seq = try GateSequence(name: "bell", qubitCount: 2)
        try seq.apply(.h(target: 0))
        try seq.apply(.cx(control: 0, target: 1))

        let data = try JSONEncoder().encode(seq)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(object?["schemaVersion"] as? Int, CircuitIRSchema.current)

        let decoded = try JSONDecoder().decode(GateSequence.self, from: data)
        XCTAssertEqual(decoded, seq)

        let legacy = Data(
            #"""
            {
              "name": "bell",
              "body": {
                "qubitCount": 2,
                "gates": [
                  { "type": "h", "target": 0 },
                  { "type": "cx", "control": 0, "target": 1 }
                ]
              }
            }
            """#.utf8
        )
        let migrated = try JSONDecoder().decode(Subcircuit.self, from: legacy)
        XCTAssertEqual(migrated.name, "bell")
        XCTAssertEqual(migrated.gates, [.h(target: 0), .cx(control: 0, target: 1)])
    }

    func testGateSequenceFutureSchemaVersionThrows() throws {
        let future = Data(
            #"""
            {
              "schemaVersion": 7,
              "name": "x",
              "body": {
                "schemaVersion": 1,
                "qubitCount": 1,
                "gates": [{ "type": "x", "target": 0 }]
              }
            }
            """#.utf8
        )

        XCTAssertThrowsError(try JSONDecoder().decode(GateSequence.self, from: future)) { error in
            guard case CircuitIRError.unsupportedSchemaVersion(let found, let supported) = error else {
                return XCTFail("expected unsupportedSchemaVersion, got \(error)")
            }
            XCTAssertEqual(found, 7)
            XCTAssertEqual(supported, 1)
        }
    }

    func testQuantumCircuitRejectsOutOfRangeQubitOnDecode() throws {
        let payload = Data(
            #"""
            {
              "schemaVersion": 1,
              "qubitCount": 1,
              "gates": [{ "type": "x", "target": 3 }]
            }
            """#.utf8
        )

        XCTAssertThrowsError(try JSONDecoder().decode(QuantumCircuit.self, from: payload)) { error in
            guard case CircuitIRError.invalidCircuit = error else {
                return XCTFail("expected invalidCircuit, got \(error)")
            }
        }
    }

    func testQuantumCircuitRejectsMetadataLengthMismatch() throws {
        let tooLong = Data(
            #"""
            {
              "schemaVersion": 1,
              "qubitCount": 1,
              "gates": [{ "type": "x", "target": 0 }],
              "instructionMetadata": [
                { "label": "a" },
                { "label": "b" }
              ]
            }
            """#.utf8
        )
        XCTAssertThrowsError(try JSONDecoder().decode(QuantumCircuit.self, from: tooLong)) { error in
            guard case CircuitIRError.metadataLengthMismatch(let metadataCount, let gateCount) = error else {
                return XCTFail("expected metadataLengthMismatch, got \(error)")
            }
            XCTAssertEqual(metadataCount, 2)
            XCTAssertEqual(gateCount, 1)
        }

        let tooShort = Data(
            #"""
            {
              "schemaVersion": 1,
              "qubitCount": 1,
              "gates": [
                { "type": "x", "target": 0 },
                { "type": "h", "target": 0 }
              ],
              "instructionMetadata": [{ "label": "only-one" }]
            }
            """#.utf8
        )
        XCTAssertThrowsError(try JSONDecoder().decode(QuantumCircuit.self, from: tooShort)) { error in
            guard case CircuitIRError.metadataLengthMismatch(let metadataCount, let gateCount) = error else {
                return XCTFail("expected metadataLengthMismatch, got \(error)")
            }
            XCTAssertEqual(metadataCount, 1)
            XCTAssertEqual(gateCount, 2)
        }
    }

    func testQuantumCircuitRejectsUndeclaredClassicalRegister() throws {
        let payload = Data(
            #"""
            {
              "schemaVersion": 1,
              "qubitCount": 1,
              "gates": [{
                "type": "measure",
                "qubits": [0],
                "classicalRegister": 0,
                "classicalBitOffset": 0
              }]
            }
            """#.utf8
        )

        XCTAssertThrowsError(try JSONDecoder().decode(QuantumCircuit.self, from: payload)) { error in
            guard case CircuitIRError.invalidCircuit(let reason) = error else {
                return XCTFail("expected invalidCircuit, got \(error)")
            }
            XCTAssertTrue(reason.contains("classical register"))
        }
    }
}
