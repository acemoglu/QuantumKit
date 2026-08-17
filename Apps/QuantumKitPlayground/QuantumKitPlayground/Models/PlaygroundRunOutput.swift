import Foundation
import QuantumKit

struct PlaygroundRunOutput: Equatable, Sendable {
    let circuit: QuantumCircuit
    let result: QuantumResult
    let asciiDiagram: String
    let wallClockMilliseconds: Double
    let histogram: [String: Int]

    var metadata: QuantumResultMetadata { result.metadata }

    /// MSB bitstrings sorted lexicographically for the chart x-axis.
    var bitstringHistogram: [(label: String, count: Int)] {
        histogram
            .sorted { $0.key < $1.key }
            .map { (label: $0.key, count: $0.value) }
    }

    var summaryLines: [String] {
        var lines: [String] = []
        lines.append("QuantumKit \(metadata.quantumKitVersion)")
        lines.append("Method: \(metadata.method.rawValue)")
        lines.append("Qubits: \(metadata.qubitCount) · Gates: \(metadata.gateCount)")
        if let seed = metadata.seed {
            lines.append("Seed: \(seed)")
        }
        lines.append(String(format: "Wall time: %.2f ms", wallClockMilliseconds))
        if let device = metadata.deviceName {
            lines.append("Device: \(device)")
        }
        if let shots = result.shotCounts?.shots {
            lines.append("Shots: \(shots)")
        }
        if let phase = metadata.cumulativeGlobalPhaseRadians {
            lines.append(String(format: "Global phase: %.4f rad", phase))
        }
        return lines
    }
}
