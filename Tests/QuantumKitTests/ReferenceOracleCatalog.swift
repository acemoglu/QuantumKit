import Foundation
@testable import QuantumKit

/// Loader for ``Resources/ReferenceOracles.json`` (item 40 frozen oracles).
enum ReferenceOracleCatalog {
    struct File: Decodable {
        let version: Int
        let description: String
        let amplitudeTolerance: Double
        let probabilityTolerance: Double
        let pauliTolerance: Double
        let entries: [Entry]
    }

    struct Entry: Decodable {
        let id: String
        let qubitCount: Int
        let ops: [Op]
        let amplitudesReal: [Double]
        let amplitudesImag: [Double]
        let probabilities: [Double]
        let pauliExpectations: [PauliExpectation]
        let stabilizerClifford: Bool?
        let comment: String?
    }

    struct PauliExpectation: Decodable {
        /// Qubit index (string key) → pauli letter `i|x|y|z`.
        let paulis: [String: String]
        let value: Double
    }

    /// JSON op: `["h", q]`, `["cx", c, t]`, `["ry", theta, q]`, …
    enum Op: Decodable {
        case h(Int)
        case x(Int)
        case y(Int)
        case z(Int)
        case s(Int)
        case t(Int)
        case cx(Int, Int)
        case cz(Int, Int)
        case rx(Double, Int)
        case ry(Double, Int)
        case rz(Double, Int)

        init(from decoder: Decoder) throws {
            var container = try decoder.unkeyedContainer()
            let name = try container.decode(String.self).lowercased()
            switch name {
            case "h":
                self = .h(try container.decode(Int.self))
            case "x":
                self = .x(try container.decode(Int.self))
            case "y":
                self = .y(try container.decode(Int.self))
            case "z":
                self = .z(try container.decode(Int.self))
            case "s":
                self = .s(try container.decode(Int.self))
            case "t":
                self = .t(try container.decode(Int.self))
            case "cx":
                self = .cx(try container.decode(Int.self), try container.decode(Int.self))
            case "cz":
                self = .cz(try container.decode(Int.self), try container.decode(Int.self))
            case "rx":
                self = .rx(try container.decode(Double.self), try container.decode(Int.self))
            case "ry":
                self = .ry(try container.decode(Double.self), try container.decode(Int.self))
            case "rz":
                self = .rz(try container.decode(Double.self), try container.decode(Int.self))
            default:
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Unknown oracle op '\(name)'"
                )
            }
        }
    }

    static func load() throws -> File {
        let url = try resourceURL()
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(File.self, from: data)
    }

    static func entry(id: String) throws -> Entry {
        let file = try load()
        guard let entry = file.entries.first(where: { $0.id == id }) else {
            throw OracleLoadError.missingEntry(id)
        }
        return entry
    }

    static func makeCircuit(_ entry: Entry) throws -> QuantumCircuit {
        var circuit = try QuantumCircuit(qubitCount: entry.qubitCount)
        for op in entry.ops {
            switch op {
            case .h(let q): try circuit.h(q)
            case .x(let q): try circuit.x(q)
            case .y(let q): try circuit.y(q)
            case .z(let q): try circuit.z(q)
            case .s(let q): try circuit.s(q)
            case .t(let q): try circuit.t(q)
            case .cx(let c, let t): try circuit.cx(c, t)
            case .cz(let c, let t): try circuit.cz(c, t)
            case .rx(let theta, let q): try circuit.rx(theta: QFloat(theta), q)
            case .ry(let theta, let q): try circuit.ry(theta: QFloat(theta), q)
            case .rz(let theta, let q): try circuit.rz(theta: QFloat(theta), q)
            }
        }
        return circuit
    }

    static func pauliMap(_ expectation: PauliExpectation) throws -> [Int: Pauli] {
        var map: [Int: Pauli] = [:]
        for (key, letter) in expectation.paulis {
            guard let qubit = Int(key) else {
                throw OracleLoadError.badPauliKey(key)
            }
            switch letter.lowercased() {
            case "i": map[qubit] = .i
            case "x": map[qubit] = .x
            case "y": map[qubit] = .y
            case "z": map[qubit] = .z
            default:
                throw OracleLoadError.badPauliLetter(letter)
            }
        }
        return map
    }

    private static func resourceURL() throws -> URL {
        guard let url = Bundle.module.url(
            forResource: "ReferenceOracles",
            withExtension: "json",
            subdirectory: "Resources"
        ) ?? Bundle.module.url(forResource: "ReferenceOracles", withExtension: "json") else {
            throw OracleLoadError.missingResource
        }
        return url
    }
}

enum OracleLoadError: Error, CustomStringConvertible {
    case missingResource
    case missingEntry(String)
    case badPauliKey(String)
    case badPauliLetter(String)

    var description: String {
        switch self {
        case .missingResource:
            return "ReferenceOracles.json not found in test bundle"
        case .missingEntry(let id):
            return "Oracle entry '\(id)' not found"
        case .badPauliKey(let key):
            return "Invalid Pauli qubit key '\(key)'"
        case .badPauliLetter(let letter):
            return "Invalid Pauli letter '\(letter)'"
        }
    }
}
