import QuantumKit
import SwiftUI

struct CircuitCanvasView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel

    private let rowHeight: CGFloat = 52
    private let columnWidth: CGFloat = 52
    private let labelWidth: CGFloat = 44

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            ScrollView([.horizontal, .vertical]) {
                canvas
                    .padding(.vertical, 8)
                    .padding(.trailing, 12)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.12))
        )
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "point.3.connected.trianglepath.dotted")
                .foregroundStyle(.secondary)
            Text("Circuit")
                .font(.headline)
            if let circuit = viewModel.editableCircuit {
                Text("\(circuit.qubitCount)q · \(circuit.gates.count)g")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                viewModel.addQubit()
            } label: {
                Label("Add qubit", systemImage: "plus")
            }
            .controlSize(.small)
            .disabled(!viewModel.canAddQubit)

            Button {
                viewModel.removeLastQubit()
            } label: {
                Label("Remove qubit", systemImage: "minus")
            }
            .controlSize(.small)
            .disabled(!viewModel.canRemoveQubit)

            Button {
                viewModel.undoCanvas()
            } label: {
                Label("Undo", systemImage: "arrow.uturn.backward")
            }
            .controlSize(.small)
            .disabled(!viewModel.canUndoCanvas)

            Button(role: .destructive) {
                viewModel.deleteSelectedGate()
            } label: {
                Label("Delete gate", systemImage: "trash")
            }
            .controlSize(.small)
            .disabled(viewModel.selectedGateIndex == nil)

            Button("Clear") {
                viewModel.clearCanvas()
            }
            .controlSize(.small)
        }
    }

    @ViewBuilder
    private var canvas: some View {
        if let circuit = viewModel.editableCircuit {
            let layout = CircuitVizLayout(circuit: circuit)
            VStack(alignment: .leading, spacing: 0) {
                ForEach(0..<circuit.qubitCount, id: \.self) { qubit in
                    qubitRow(qubit: qubit, circuit: circuit, layout: layout)
                }
            }
        } else {
            VStack(spacing: 8) {
                Image(systemName: "waveform.path.ecg")
                    .font(.title2)
                    .foregroundStyle(.secondary)
                Text("No circuit yet")
                    .font(.headline)
                Text("Load a sample, or click a gate and then a qubit wire.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity, minHeight: 180)
        }
    }

    private func qubitRow(qubit: Int, circuit: QuantumCircuit, layout: CircuitVizLayout) -> some View {
        let isPending = viewModel.pendingPlacement?.pickedQubits.contains(qubit) == true
        return HStack(spacing: 0) {
            Text("q\(qubit)")
                .font(.system(.caption, design: .monospaced).weight(.semibold))
                .frame(width: labelWidth, alignment: .leading)
                .foregroundStyle(isPending ? Color.orange : Color.secondary)

            ZStack(alignment: .leading) {
                Rectangle()
                    .fill(Color.primary.opacity(0.28))
                    .frame(height: 2)

                HStack(spacing: 0) {
                    ForEach(Array(layout.moments.enumerated()), id: \.offset) { momentIndex, moment in
                        momentCell(qubit: qubit, momentIndex: momentIndex, moment: moment, circuit: circuit)
                    }
                    dropSlot(qubit: qubit)
                }
            }
            .frame(height: rowHeight)
            .contentShape(Rectangle())
            .onTapGesture {
                viewModel.handleQubitTap(qubit)
            }
            .dropDestination(for: PaletteTool.self) { tools, _ in
                guard let tool = tools.first else { return false }
                viewModel.placePaletteTool(tool, on: qubit)
                return true
            }
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(isPending ? Color.orange.opacity(0.12) : Color.clear)
            )
        }
    }

    private func momentCell(
        qubit: Int,
        momentIndex: Int,
        moment: CircuitVizMoment,
        circuit: QuantumCircuit
    ) -> some View {
        let cell = moment.cells.indices.contains(qubit) ? moment.cells[qubit] : .idle
        let gateIndex = gateIndex(atQubit: qubit, moment: moment, circuit: circuit)
        let selected = gateIndex != nil && gateIndex == viewModel.selectedGateIndex

        return ZStack {
            if drawsSpan(moment.cells, qubit: qubit) {
                Rectangle()
                    .fill(Color.accentColor.opacity(0.55))
                    .frame(width: 2)
                    .frame(maxHeight: .infinity)
            }
            GateGlyphView(cell: cell, isSelected: selected)
        }
        .frame(width: columnWidth, height: rowHeight)
        .contentShape(Rectangle())
        .onTapGesture {
            if let gateIndex {
                viewModel.selectGate(at: gateIndex)
            } else {
                viewModel.handleQubitTap(qubit)
            }
        }
        .accessibilityLabel(cellAccessibility(cell, qubit: qubit, moment: momentIndex))
    }

    private func dropSlot(qubit: Int) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                .foregroundStyle(Color.secondary.opacity(0.45))
            Image(systemName: "plus")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(width: columnWidth, height: 36)
        .padding(.horizontal, 4)
        .contentShape(Rectangle())
        .onTapGesture { viewModel.handleQubitTap(qubit) }
        .dropDestination(for: PaletteTool.self) { tools, _ in
            guard let tool = tools.first else { return false }
            viewModel.placePaletteTool(tool, on: qubit)
            return true
        }
        .help("Drop a gate on q\(qubit)")
    }

    private func drawsSpan(_ cells: [CircuitVizCell], qubit: Int) -> Bool {
        guard cells.indices.contains(qubit) else { return false }
        switch cells[qubit] {
        case .wire, .control, .target, .swap:
            return true
        default:
            return false
        }
    }

    private func gateIndex(atQubit qubit: Int, moment: CircuitVizMoment, circuit: QuantumCircuit) -> Int? {
        moment.gateIndices.first { index in
            guard circuit.gates.indices.contains(index) else { return false }
            return circuit.gates[index].affectedQubits.contains(qubit)
        }
    }

    private func cellAccessibility(_ cell: CircuitVizCell, qubit: Int, moment: Int) -> String {
        "q\(qubit) column \(moment + 1): \(String(describing: cell))"
    }
}

struct GateGlyphView: View {
    let cell: CircuitVizCell
    var isSelected: Bool

    var body: some View {
        switch cell {
        case .idle:
            Color.clear
        case .wire:
            Rectangle()
                .fill(Color.accentColor.opacity(0.55))
                .frame(width: 2, height: 20)
        case .control:
            Circle()
                .fill(isSelected ? Color.accentColor : Color.primary)
                .frame(width: 12, height: 12)
        case .swap:
            Text("×")
                .font(.system(.body, design: .rounded).weight(.bold))
                .frame(width: 28, height: 28)
                .background(chipBackground)
        case .gate(let name):
            Text(name)
                .font(.system(.caption, design: .rounded).weight(.bold))
                .frame(width: 36, height: 32)
                .background(chipBackground)
        case .target(let label):
            Text(label)
                .font(.system(.caption, design: .rounded).weight(.bold))
                .frame(width: 36, height: 32)
                .background(chipBackground)
        case .measure:
            Text("M")
                .font(.system(.caption, design: .rounded).weight(.bold))
                .frame(width: 36, height: 32)
                .background(chipBackground)
        case .measureClassical:
            Image(systemName: "arrow.down")
                .font(.caption2)
        case .placeholder(let label):
            Text(label)
                .font(.system(.caption2, design: .rounded).weight(.bold))
                .frame(width: 36, height: 32)
                .background(chipBackground)
        }
    }

    private var chipBackground: some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .fill(Color.playgroundEditorBackground)
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(isSelected ? Color.accentColor : Color.primary.opacity(0.25), lineWidth: isSelected ? 2 : 1)
            )
            .shadow(color: .black.opacity(0.06), radius: isSelected ? 3 : 0, y: 1)
    }
}

#Preview {
    CircuitCanvasView()
        .environmentObject(PlaygroundViewModel.previewFactory())
        .frame(height: 280)
}
