import Foundation

/// Renders a ``CircuitVizLayout`` as a multi-line ASCII diagram for terminal debugging.
///
/// Quantum idle segments use `─`; classical idle segments use `═`. Column width is the
/// max glyph width in that moment (minimum 3), with content centered.
public struct CircuitASCIIRenderer: Sendable {
    public init() {}

    /// Renders `layout` to a multi-line string (no trailing newline after the last row).
    public func render(_ layout: CircuitVizLayout) -> String {
        let rowCount = layout.rowCount
        guard rowCount > 0 else { return "" }

        let columnWidths: [Int] = layout.moments.map { moment in
            let widest = moment.cells.map { glyph(for: $0).count }.max() ?? 3
            return max(3, widest)
        }

        var lines: [String] = []
        lines.reserveCapacity(rowCount)

        for row in 0..<rowCount {
            let isClassical = row >= layout.qubitCount
            var body = ""
            if layout.moments.isEmpty {
                body = isClassical ? "═══" : "───"
            } else {
                for (momentIndex, moment) in layout.moments.enumerated() {
                    let cell = moment.cells[row]
                    let raw = glyph(for: cell)
                    body += padCentered(raw, width: columnWidths[momentIndex], isClassical: isClassical)
                }
            }
            lines.append("\(layout.rowLabel(row)): \(body)")
        }

        return lines.joined(separator: "\n")
    }

    private func glyph(for cell: CircuitVizCell) -> String {
        switch cell {
        case .idle:
            return ""
        case .wire:
            return "┼"
        case .gate(let name):
            return name
        case .control:
            return "■"
        case .target(let label):
            return label
        case .swap:
            return "×"
        case .measure:
            return "[M]"
        case .measureClassical:
            return "╩"
        case .placeholder(let label):
            return label
        }
    }

    private func padCentered(_ text: String, width: Int, isClassical: Bool) -> String {
        let fill = isClassical ? Character("═") : Character("─")
        if text.isEmpty {
            return String(repeating: fill, count: width)
        }
        if text.count >= width {
            return text
        }
        let pad = width - text.count
        let left = pad / 2
        let right = pad - left
        return String(repeating: fill, count: left) + text + String(repeating: fill, count: right)
    }
}

extension QuantumCircuit {
    /// ASCII wire diagram for terminal debugging (see ``CircuitASCIIRenderer``).
    public func asciiDiagram() -> String {
        CircuitASCIIRenderer().render(CircuitVizLayout(circuit: self))
    }
}

extension GateSequence {
    /// ASCII wire diagram of ``body``.
    public func asciiDiagram() -> String {
        body.asciiDiagram()
    }
}
