import SwiftUI
import UniformTypeIdentifiers

struct CircuitEditorView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel
    @Environment(\.isPhoneLayout) private var isPhoneLayout
    @State private var isDropTargeted = false

    var showsSettings: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: isPhoneLayout ? 0 : 12) {
            if !isPhoneLayout {
                header
            }

            TextEditor(text: $viewModel.sourceText)
                .font(.system(.body, design: .monospaced))
                .scrollContentBackground(.hidden)
                .padding(isPhoneLayout ? 12 : 8)
                .background(Color.playgroundEditorBackground)
                .clipShape(RoundedRectangle(cornerRadius: isPhoneLayout ? 0 : PlaygroundChrome.cornerRadius, style: .continuous))
                .overlay {
                    if !isPhoneLayout {
                        RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous)
                            .strokeBorder(isDropTargeted ? Color.accentColor : Color.quantumInk.opacity(0.14), lineWidth: isDropTargeted ? 2 : 1)
                    }
                }
                .overlay(alignment: .topTrailing) {
                    if isDropTargeted {
                        Label("Drop OpenQASM", systemImage: "square.and.arrow.down")
                            .font(.caption.weight(.semibold))
                            .padding(8)
                            .background(Color.quantumCard, in: RoundedRectangle(cornerRadius: PlaygroundChrome.chipRadius, style: .continuous))
                            .padding(10)
                    }
                }

            if showsSettings {
                SettingsPanelView()
            }
        }
        .padding(isPhoneLayout ? 0 : 16)
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            viewModel.applyDroppedProviders(providers)
        }
        .playgroundKeyboardDismissToolbar()
        #if os(iOS)
        .scrollDismissesKeyboard(.interactively)
        #endif
        // v1: LLM generate-circuit UI is intentionally omitted.
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "chevron.left.forwardslash.chevron.right")
                .foregroundStyle(.secondary)
            Text("OpenQASM")
                .font(.headline)
            Spacer()
        }
    }
}

#Preview {
    CircuitEditorView()
        .environmentObject(PlaygroundViewModel.previewFactory())
}
