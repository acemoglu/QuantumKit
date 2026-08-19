import SwiftUI

struct GatePaletteView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel
    @Environment(\.isPhoneLayout) private var isPhoneLayout

    var body: some View {
        Group {
            if isPhoneLayout {
                phoneBody
            } else {
                desktopBody
            }
        }
    }

    private var phoneBody: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if let pending = viewModel.pendingPlacement {
                    Text(pendingHint(pending))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Color.quantumPending)
                        .lineLimit(2)
                    Spacer(minLength: 0)
                    Button("Cancel") { viewModel.cancelPendingPlacement() }
                        .font(.caption.weight(.semibold))
                } else {
                    Text("Tap a gate or block, then a qubit")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                phoneChipRow(PaletteTool.allCases)
                phoneBlockRow()
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color.quantumCanvas)
    }

    private func phoneChipRow(_ tools: [PaletteTool]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(tools) { tool in
                    chip(tool, minWidth: 44, minHeight: 36)
                        .draggable(tool) {
                            PaletteChipView(tool: tool, isSelected: true)
                        }
                }
            }
            .padding(.vertical, 1)
        }
    }

    private var desktopBody: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Label("Gates", systemImage: "square.grid.2x2")
                    .font(.headline)
                if let pending = viewModel.pendingPlacement {
                    Text(pendingHint(pending))
                        .font(.caption)
                        .foregroundStyle(Color.quantumPending)
                        .lineLimit(1)
                    Button("Cancel") { viewModel.cancelPendingPlacement() }
                        .controlSize(.small)
                }
                Spacer(minLength: 0)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 14) {
                    ForEach(PaletteSection.allCases) { section in
                        HStack(spacing: 6) {
                            Text(section.title)
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.secondary)
                            ForEach(PaletteTool.allCases.filter { $0.section == section }) { tool in
                                chip(tool, minWidth: 34, minHeight: 28)
                                    .draggable(tool) {
                                        PaletteChipView(tool: tool, isSelected: true)
                                    }
                            }
                        }
                    }
                    HStack(spacing: 6) {
                        Text("Blocks")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                        ForEach(CircuitBlock.allCases) { block in
                            blockChip(block, minWidth: 44, minHeight: 28)
                        }
                    }
                }
                .padding(.vertical, 2)
            }
        }
        .padding(10)
        .playgroundPanel()
    }

    private func phoneBlockRow() -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(CircuitBlock.allCases) { block in
                    blockChip(block, minWidth: 52, minHeight: 36)
                }
            }
            .padding(.vertical, 1)
        }
    }

    private func chip(_ tool: PaletteTool, minWidth: CGFloat, minHeight: CGFloat) -> some View {
        PaletteChipView(tool: tool, isSelected: viewModel.selectedPaletteTool == tool, minWidth: minWidth, minHeight: minHeight)
            .onTapGesture { viewModel.selectPaletteTool(tool) }
            .help(tool.help)
            .accessibilityLabel(tool.help)
    }

    private func blockChip(_ block: CircuitBlock, minWidth: CGFloat, minHeight: CGFloat) -> some View {
        PaletteChipView(
            title: block.title,
            isSelected: viewModel.selectedCircuitBlock == block,
            minWidth: minWidth,
            minHeight: minHeight
        )
        .onTapGesture { viewModel.selectCircuitBlock(block) }
        .help(block.help)
        .accessibilityLabel(block.help)
    }

    private func pendingHint(_ pending: PendingGatePlacement) -> String {
        let remain = pending.tool.qubitCount - pending.pickedQubits.count
        if remain <= 0 { return pending.tool.help }
        if pending.tool == .cx || pending.tool == .cz {
            return pending.pickedQubits.isEmpty ? "Tap the control qubit" : "Now tap the target"
        }
        return "Tap \(remain) more \(remain == 1 ? "qubit" : "qubits") for \(pending.tool.title)"
    }
}

struct PaletteChipView: View {
    let title: String
    var isSelected: Bool
    var minWidth: CGFloat = 34
    var minHeight: CGFloat = 28

    init(tool: PaletteTool, isSelected: Bool, minWidth: CGFloat = 34, minHeight: CGFloat = 28) {
        self.title = tool.title
        self.isSelected = isSelected
        self.minWidth = minWidth
        self.minHeight = minHeight
    }

    init(title: String, isSelected: Bool, minWidth: CGFloat = 34, minHeight: CGFloat = 28) {
        self.title = title
        self.isSelected = isSelected
        self.minWidth = minWidth
        self.minHeight = minHeight
    }

    var body: some View {
        Text(title)
            .font(.system(.caption, design: .rounded).weight(.bold))
            .foregroundStyle(isSelected ? Color.quantumOnAccent : Color.quantumInk)
            .frame(minWidth: minWidth, minHeight: minHeight)
            .padding(.horizontal, 7)
            .background(
                RoundedRectangle(cornerRadius: PlaygroundChrome.chipRadius, style: .continuous)
                    .fill(isSelected ? Color.accentColor : Color.quantumCard)
            )
            .overlay(
                RoundedRectangle(cornerRadius: PlaygroundChrome.chipRadius, style: .continuous)
                    .strokeBorder(isSelected ? Color.accentColor : Color.quantumInk.opacity(0.16), lineWidth: isSelected ? 2 : 1)
            )
    }
}

#Preview {
    GatePaletteView()
        .environmentObject(PlaygroundViewModel.previewFactory())
        .padding()
}
