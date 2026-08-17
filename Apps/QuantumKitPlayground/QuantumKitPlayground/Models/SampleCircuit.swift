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
            summary: "H + CNOT entangler with terminal measurement.",
            filename: "bell.qasm"
        ),
        SampleCircuit(
            id: "toffoli",
            name: "Toffoli",
            summary: "Three-qubit CCX gate (OpenQASM 2 / qelib1).",
            filename: "toffoli.qasm"
        ),
        SampleCircuit(
            id: "teleport",
            name: "Teleport (sketch)",
            summary: "Bell pair, bilateral measure, classically conditioned X/Z.",
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
            name: "Parametric",
            summary: "π-scaled rotations via OpenQASM 2 angle expressions.",
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
