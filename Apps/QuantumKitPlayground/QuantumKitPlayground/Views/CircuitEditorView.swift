import SwiftUI
import UniformTypeIdentifiers

struct CircuitEditorView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel
    @Environment(\.horizontalSizeClass) private var sizeClass
    @State private var isDropTargeted = false

    var showsSettings: Bool = false
    var compactEditor: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !compactEditor {
                header
            }

            TextEditor(text: $viewModel.sourceText)
                .font(.system(.body, design: .monospaced))
                .scrollContentBackground(.hidden)
                .padding(8)
                .background(Color.playgroundEditorBackground)
                .clipShape(RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous)
                        .strokeBorder(isDropTargeted ? Color.accentColor : Color.primary.opacity(0.18), lineWidth: isDropTargeted ? 2 : 1)
                )
                .overlay(alignment: .topTrailing) {
                    if isDropTargeted {
                        Label("Drop OpenQASM", systemImage: "square.and.arrow.down")
                            .font(.caption.weight(.semibold))
                            .padding(8)
                            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                            .padding(10)
                    }
                }

            if showsSettings {
                SettingsPanelView()
            }
        }
        .padding(compactEditor ? 0 : 16)
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            viewModel.applyDroppedProviders(providers)
        }
        // v1: LLM generate-circuit UI is intentionally omitted.
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "chevron.left.forwardslash.chevron.right")
                .foregroundStyle(.secondary)
            Text("OpenQASM")
                .font(.headline)
            Spacer()
            if sizeClass == .compact {
                sampleMenu
            }
        }
    }

    private var sampleMenu: some View {
        Menu {
            ForEach(SampleCircuit.bundled) { sample in
                Button(sample.name) {
                    viewModel.loadSample(sample)
                }
            }
        } label: {
            Label(viewModel.selectedSampleName ?? "Samples", systemImage: "square.stack")
        }
    }
}

#Preview {
    CircuitEditorView()
        .environmentObject(PlaygroundViewModel.previewFactory())
}
