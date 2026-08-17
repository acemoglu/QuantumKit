import SwiftUI

struct SamplePickerView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel

    var body: some View {
        List(selection: $viewModel.selectedSampleID) {
            Section("Samples") {
                ForEach(SampleCircuit.bundled) { sample in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(sample.name)
                            .font(.headline)
                        Text(sample.summary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .tag(sample.id)
                    .contentShape(Rectangle())
                }
            }

            Section("Actions") {
                Button {
                    viewModel.resetToSelectedSample()
                } label: {
                    Label("Reset Sample", systemImage: "arrow.counterclockwise")
                }
            }
        }
        #if os(iOS)
        .listStyle(.insetGrouped)
        #else
        .listStyle(.sidebar)
        #endif
        .modifier(SampleSelectionChangeModifier(viewModel: viewModel))
    }
}

private struct SampleSelectionChangeModifier: ViewModifier {
    @ObservedObject var viewModel: PlaygroundViewModel

    func body(content: Content) -> some View {
        #if os(macOS)
        content.onChange(of: viewModel.selectedSampleID) { _, newID in
            viewModel.selectSample(id: newID)
        }
        #else
        if #available(iOS 17.0, *) {
            content.onChange(of: viewModel.selectedSampleID) { _, newID in
                viewModel.selectSample(id: newID)
            }
        } else {
            content.onChange(of: viewModel.selectedSampleID) { newID in
                viewModel.selectSample(id: newID)
            }
        }
        #endif
    }
}

#Preview {
    SamplePickerView()
        .environmentObject(PlaygroundViewModel.previewFactory())
}
