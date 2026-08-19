import QuantumKit
import SwiftUI

struct CircuitCanvasView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel
    @Environment(\.isPhoneLayout) private var isPhoneLayout

    private var rowHeight: CGFloat { isPhoneLayout ? 44 : 52 }
    private var columnWidth: CGFloat { isPhoneLayout ? 44 : 52 }
    private var labelWidth: CGFloat { isPhoneLayout ? 36 : 44 }

    var body: some View {
        VStack(alignment: .leading, spacing: isPhoneLayout ? 6 : 8) {
            if isPhoneLayout {
                phoneHeader
            } else {
                desktopHeader
            }
            ScrollView([.horizontal, .vertical]) {
                canvas
                    .padding(.vertical, 8)
                    .padding(.trailing, 12)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .playgroundPanel(enabled: !isPhoneLayout)
        #if os(macOS)
        .onDeleteCommand {
            viewModel.deleteSelectedGate()
        }
        #endif
    }

    private var phoneHeader: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                if let circuit = viewModel.editableCircuit {
                    Text(circuitSizeLabel(circuit))
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                if let hint = viewModel.insertHint {
                    Text(hint)
                        .font(.caption)
                        .foregroundStyle(Color.quantumPending)
                        .lineLimit(1)
                }
                Spacer(minLength: 4)
                Button {
                    viewModel.isPresentingHelp = true
                } label: {
                    Image(systemName: "questionmark.circle")
                }
                .accessibilityLabel("How to use Circuit")
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    iconButton("plus", "Add qubit", enabled: viewModel.canAddQubit, action: viewModel.addQubit)
                    iconButton("minus", "Remove last qubit", enabled: viewModel.canRemoveQubit, action: viewModel.removeLastQubit)
                    iconButton("arrow.uturn.backward", "Undo", enabled: viewModel.canUndoCanvas, action: viewModel.undoCanvas)
                    iconButton("chevron.left", "Move gate left", enabled: viewModel.canMoveSelectedGateLeft) {
                        viewModel.moveSelectedGate(by: -1)
                    }
                    iconButton("chevron.right", "Move gate right", enabled: viewModel.canMoveSelectedGateRight) {
                        viewModel.moveSelectedGate(by: 1)
                    }
                    iconButton("trash", "Delete gate", enabled: viewModel.selectedGateIndex != nil, role: .destructive, action: viewModel.deleteSelectedGate)
                    Button("Clear", action: viewModel.clearCanvas)
                        .font(.caption)
                        .controlSize(.small)
                }
            }
        }
    }

    private func iconButton(
        _ systemImage: String,
        _ label: String,
        enabled: Bool,
        role: ButtonRole? = nil,
        action: @escaping () -> Void
    ) -> some View {
        Button(role: role, action: action) {
            Image(systemName: systemImage)
                .frame(width: 32, height: 32)
        }
        .disabled(!enabled)
        .accessibilityLabel(label)
    }

    private func circuitSizeLabel(_ circuit: QuantumCircuit) -> String {
        let qubits = circuit.qubitCount == 1 ? "1 qubit" : "\(circuit.qubitCount) qubits"
        let gates = circuit.gates.count == 1 ? "1 gate" : "\(circuit.gates.count) gates"
        return "\(qubits) · \(gates)"
    }

    private var desktopHeader: some View {
        HStack(spacing: 8) {
            Image(systemName: "point.3.connected.trianglepath.dotted")
                .foregroundStyle(.secondary)
            Text("Circuit")
                .font(.headline)
            Button {
                viewModel.isPresentingHelp = true
            } label: {
                Image(systemName: "questionmark.circle")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("How to use Circuit")
            .accessibilityLabel("How to use Circuit")
            if let circuit = viewModel.editableCircuit {
                Text(circuitSizeLabel(circuit))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            if let hint = viewModel.insertHint {
                Text(hint)
                    .font(.caption)
                    .foregroundStyle(Color.quantumPending)
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

            Button {
                viewModel.moveSelectedGate(by: -1)
            } label: {
                Label("Move left", systemImage: "chevron.left")
            }
            .controlSize(.small)
            .disabled(!viewModel.canMoveSelectedGateLeft)

            Button {
                viewModel.moveSelectedGate(by: 1)
            } label: {
                Label("Move right", systemImage: "chevron.right")
            }
            .controlSize(.small)
            .disabled(!viewModel.canMoveSelectedGateRight)

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
                Text(isPhoneLayout
                     ? "Load a sample, or tap a gate and then a qubit wire."
                     : "Load a sample, or click a gate and then a qubit wire.")
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
            HStack(spacing: 2) {
                Text("q\(qubit)")
                    .font(.system(.caption, design: .monospaced).weight(.semibold))
                if isPhoneLayout {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(width: isPhoneLayout ? 52 : labelWidth, alignment: .leading)
            .frame(minHeight: rowHeight)
            .contentShape(Rectangle())
            .foregroundStyle(isPending ? Color.quantumPending : Color.secondary)
            .help(isPhoneLayout ? "Touch and hold for qubit options" : "Right-click to insert or remove this qubit")
            .contextMenu {
                Button("Insert qubit here") {
                    viewModel.insertQubit(at: qubit)
                }
                .disabled(!viewModel.canAddQubit)
                Button("Insert qubit below") {
                    viewModel.insertQubit(at: qubit + 1)
                }
                .disabled(!viewModel.canAddQubit)
                Button("Remove this qubit", role: .destructive) {
                    viewModel.removeQubit(at: qubit)
                }
                .disabled(circuit.qubitCount <= 1)
            }
            .accessibilityHint("Opens qubit insert and remove options")

            ZStack(alignment: .leading) {
                Rectangle()
                    .fill(Color.primary.opacity(0.28))
                    .frame(height: 2)

                HStack(spacing: 0) {
                    insertGutter(qubit: qubit, beforeGateIndex: 0, circuit: circuit)
                    ForEach(Array(layout.moments.enumerated()), id: \.offset) { momentIndex, moment in
                        momentAndGutter(
                            qubit: qubit,
                            momentIndex: momentIndex,
                            moment: moment,
                            circuit: circuit,
                            layout: layout
                        )
                    }
                    dropSlot(qubit: qubit, gateCount: circuit.gates.count)
                }
            }
            .frame(height: rowHeight)
            .contentShape(Rectangle())
            .dropDestination(for: PaletteTool.self) { tools, _ in
                guard let tool = tools.first else { return false }
                viewModel.placePaletteTool(tool, on: qubit)
                return true
            }
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(isPending ? Color.quantumPending.opacity(0.12) : Color.clear)
            )
        }
    }

    @ViewBuilder
    private func momentAndGutter(
        qubit: Int,
        momentIndex: Int,
        moment: CircuitVizMoment,
        circuit: QuantumCircuit,
        layout: CircuitVizLayout
    ) -> some View {
        momentCell(qubit: qubit, momentIndex: momentIndex, moment: moment, circuit: circuit)
        if momentIndex + 1 < layout.moments.count {
            let nextIndex = layout.moments[momentIndex + 1].gateIndices.min() ?? circuit.gates.count
            insertGutter(qubit: qubit, beforeGateIndex: nextIndex, circuit: circuit)
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
        let insertBefore = moment.gateIndices.min() ?? circuit.gates.count

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
            if let gateIndex,
               viewModel.selectedPaletteTool == nil,
               viewModel.selectedCircuitBlock == nil,
               viewModel.pendingPlacement == nil {
                viewModel.selectGate(at: gateIndex)
            } else {
                viewModel.handleCanvasTap(qubit: qubit, insertBefore: insertBefore)
            }
        }
        .dropDestination(for: PaletteTool.self) { tools, _ in
            guard let tool = tools.first else { return false }
            viewModel.insertionIndex = insertBefore
            viewModel.placePaletteTool(tool, on: qubit)
            return true
        }
        .accessibilityLabel(cellAccessibility(cell, qubit: qubit, moment: momentIndex))
    }

    private func insertGutter(qubit: Int, beforeGateIndex: Int, circuit: QuantumCircuit) -> some View {
        let active = viewModel.insertionIndex == beforeGateIndex
        let width: CGFloat = isPhoneLayout ? 24 : 12
        return ZStack {
            Rectangle()
                .fill(active ? Color.accentColor : Color.secondary.opacity(0.28))
                .frame(width: active ? 3 : 1)
                .frame(maxHeight: .infinity)
                .padding(.vertical, 10)
            if isPhoneLayout {
                Image(systemName: "plus")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(active ? Color.accentColor : Color.secondary.opacity(0.55))
            }
        }
        .frame(width: width)
        .contentShape(Rectangle())
        .onTapGesture {
            if viewModel.selectedPaletteTool != nil
                || viewModel.selectedCircuitBlock != nil
                || viewModel.pendingPlacement != nil {
                viewModel.handleCanvasTap(qubit: qubit, insertBefore: beforeGateIndex)
            } else {
                viewModel.setInsertionPoint(beforeGateIndex)
            }
        }
        .help("Insert before this column")
        .accessibilityLabel("Insert before column")
    }

    private func dropSlot(qubit: Int, gateCount: Int) -> some View {
        let appending = viewModel.insertionIndex == nil || viewModel.insertionIndex == gateCount
        return ZStack {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(style: StrokeStyle(lineWidth: appending ? 2 : 1, dash: [4, 3]))
                .foregroundStyle(appending ? Color.accentColor.opacity(0.7) : Color.secondary.opacity(0.45))
            Image(systemName: "plus")
                .font(.caption)
                .foregroundStyle(appending ? Color.accentColor : Color.secondary.opacity(0.45))
        }
        .frame(width: columnWidth, height: 36)
        .padding(.horizontal, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            viewModel.setInsertionPoint(nil)
            viewModel.handleQubitTap(qubit)
        }
        .dropDestination(for: PaletteTool.self) { tools, _ in
            guard let tool = tools.first else { return false }
            viewModel.setInsertionPoint(nil)
            viewModel.placePaletteTool(tool, on: qubit)
            return true
        }
        .help("Append a gate on q\(qubit)")
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
        RoundedRectangle(cornerRadius: PlaygroundChrome.chipRadius, style: .continuous)
            .fill(Color.quantumCard)
            .overlay(
                RoundedRectangle(cornerRadius: PlaygroundChrome.chipRadius, style: .continuous)
                    .strokeBorder(isSelected ? Color.accentColor : Color.quantumInk.opacity(0.22), lineWidth: isSelected ? 2 : 1)
            )
    }
}

#Preview {
    CircuitCanvasView()
        .environmentObject(PlaygroundViewModel.previewFactory())
        .frame(height: 280)
}
