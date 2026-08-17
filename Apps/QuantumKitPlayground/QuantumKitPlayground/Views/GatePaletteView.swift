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
                    Text("Tap a gate, then a qubit wire")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                phoneChipRow(PaletteTool.allCases.filter { $0.section == .oneQubit || $0.section == .rotations })
                phoneChipRow(PaletteTool.allCases.filter { $0.section == .multiQubit || $0.section == .ops })
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
                }
                .padding(.vertical, 2)
            }
        }
        .padding(10)
        .playgroundPanel()
    }

    private func chip(_ tool: PaletteTool, minWidth: CGFloat, minHeight: CGFloat) -> some View {
        PaletteChipView(tool: tool, isSelected: viewModel.selectedPaletteTool == tool, minWidth: minWidth, minHeight: minHeight)
            .onTapGesture { viewModel.selectPaletteTool(tool) }
            .help(tool.help)
            .accessibilityLabel(tool.help)
    }

    private func pendingHint(_ pending: PendingGatePlacement) -> String {
        let remain = pending.tool.qubitCount - pending.pickedQubits.count
        if remain <= 0 { return pending.tool.help }
        if pending.tool == .cx || pending.tool == .cz {
            return pending.pickedQubits.isEmpty ? "Tap the control qubit" : "Now tap the target"
        }
        return "Tap \(remain) more qubit(s) for \(pending.tool.title)"
    }
}

struct PaletteChipView: View {
    let tool: PaletteTool
    var isSelected: Bool
    var minWidth: CGFloat = 34
    var minHeight: CGFloat = 28

    var body: some View {
        Text(tool.title)
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
