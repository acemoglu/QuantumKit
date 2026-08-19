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

    /// Shot counts for spreadsheets. `#` lines are metadata; then `bitstring,count,probability`.
    var histogramCSV: String {
        let shots = result.shotCounts?.shots ?? histogram.values.reduce(0, +)
        var lines: [String] = []
        lines.append("# QuantumKit \(metadata.quantumKitVersion)")
        lines.append("# method,\(csvEscape(metadata.method.rawValue))")
        if let device = metadata.deviceName, !device.isEmpty {
            lines.append("# device,\(csvEscape(device))")
        }
        lines.append(String(format: "# wall_time_ms,%.4f", wallClockMilliseconds))
        lines.append("# qubits,\(metadata.qubitCount)")
        lines.append("# gates,\(metadata.gateCount)")
        if let seed = metadata.seed {
            lines.append("# seed,\(seed)")
        }
        lines.append("# shots,\(shots)")
        lines.append("bitstring,count,probability")
        for row in bitstringHistogram {
            let probability = shots > 0 ? Double(row.count) / Double(shots) : 0
            lines.append("\(row.label),\(row.count),\(Self.csvProbability(probability))")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private func csvEscape(_ value: String) -> String {
        if value.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" }) {
            return "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
        }
        return value
    }

    private static func csvProbability(_ value: Double) -> String {
        String(format: "%.12g", value)
    }
}
