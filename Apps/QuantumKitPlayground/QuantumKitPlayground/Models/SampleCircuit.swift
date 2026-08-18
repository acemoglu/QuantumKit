import Foundation

struct SampleCircuit: Identifiable, Hashable {
    let id: String
    let name: String
    let summary: String
    let filename: String

    static let bundled: [SampleCircuit] = [
        SampleCircuit(
            id: "bell",
            name: "Bell State",
            summary: "Hadamard and CNOT, then measure.",
            filename: "bell.qasm"
        ),
        SampleCircuit(
            id: "toffoli",
            name: "Toffoli",
            summary: "Three-qubit Toffoli (CCX), then measure.",
            filename: "toffoli.qasm"
        ),
        SampleCircuit(
            id: "teleport",
            name: "Teleport",
            summary: "Bell pair, measure, then classically controlled X and Z.",
            filename: "teleport.qasm"
        ),
        SampleCircuit(
            id: "grover_2q",
            name: "Grover (2 qubits)",
            summary: "Two-qubit Grover-style amplitude amplification.",
            filename: "grover_2q.qasm"
        ),
        SampleCircuit(
            id: "parametric",
            name: "Rotations",
            summary: "RX, RY, and RZ with explicit angles.",
            filename: "parametric.qasm"
        ),
        SampleCircuit(
            id: "ghz4",
            name: "GHZ (4 qubits)",
            summary: "Four-qubit GHZ state with terminal measurement.",
            filename: "ghz4.qasm"
        ),
    ]

    func loadSource() -> String? {
        let name = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension
        guard let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "Samples") else {
            return nil
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }
}

struct SavedCircuit: Identifiable, Codable, Hashable {
    var id: UUID
    var name: String
    var source: String
    var updatedAt: Date
}

enum CircuitLibraryStore {
    static func load() -> [SavedCircuit] {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let data = try? Data(contentsOf: fileURL) else { return [] }
        return (try? decoder.decode([SavedCircuit].self, from: data)) ?? []
    }

    static func persist(_ items: [SavedCircuit]) {
        let directory = fileURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(items) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }

    private static var fileURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("QuantumKit", isDirectory: true)
            .appendingPathComponent("saved-circuits.json")
    }
}
