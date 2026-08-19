import SwiftUI

struct SamplePickerView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel

    var body: some View {
        List(selection: $viewModel.selectedLibraryID) {
            Section("Samples") {
                ForEach(SampleCircuit.bundled) { sample in
                    libraryRow(title: sample.name, summary: sample.summary)
                        .tag(sample.id)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            viewModel.loadSample(sample)
                        }
                }
            }

            Section("My Circuits") {
                if viewModel.savedCircuits.isEmpty {
                    Text("Save a circuit to see it here.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                ForEach(viewModel.savedCircuits, id: \.id.uuidString) { circuit in
                    libraryRow(title: circuit.name, summary: Self.dateText(circuit.updatedAt))
                        .tag(circuit.id.uuidString)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            viewModel.loadSavedCircuit(circuit)
                        }
                        .contextMenu {
                            Button("Delete", role: .destructive) {
                                viewModel.deleteSavedCircuit(circuit)
                            }
                        }
                }
                .onDelete { offsets in
                    let items = offsets.map { viewModel.savedCircuits[$0] }
                    items.forEach(viewModel.deleteSavedCircuit)
                }
            }

            Section("Actions") {
                Button {
                    viewModel.newBlankCircuit()
                } label: {
                    Label("New Circuit", systemImage: "plus")
                }

                Button {
                    viewModel.presentSaveToLibrary()
                } label: {
                    Label("Save Circuit", systemImage: "square.and.arrow.down")
                }

                Button {
                    viewModel.resetToSelectedSample()
                } label: {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                }

                #if os(macOS)
                Button {
                    viewModel.isPresentingHelp = true
                } label: {
                    Label("How to Use", systemImage: "questionmark.circle")
                }
                #endif
            }
        }
        #if os(iOS)
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .background(Color.quantumCanvas)
        #else
        .listStyle(.sidebar)
        .scrollContentBackground(.hidden)
        .background(Color.quantumCanvas)
        #endif
        .modifier(LibrarySelectionChangeModifier(viewModel: viewModel))
    }

    private func libraryRow(title: String, summary: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.headline)
            Text(summary)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .contentShape(Rectangle())
    }

    private static func dateText(_ date: Date) -> String {
        date.formatted(date: .abbreviated, time: .shortened)
    }
}

private struct LibrarySelectionChangeModifier: ViewModifier {
    @ObservedObject var viewModel: PlaygroundViewModel

    func body(content: Content) -> some View {
        #if os(macOS)
        content.onChange(of: viewModel.selectedLibraryID) { _, newID in
            viewModel.selectLibraryItem(id: newID)
        }
        #else
        if #available(iOS 17.0, *) {
            content.onChange(of: viewModel.selectedLibraryID) { _, newID in
                viewModel.selectLibraryItem(id: newID)
            }
        } else {
            content.onChange(of: viewModel.selectedLibraryID) { newID in
                viewModel.selectLibraryItem(id: newID)
            }
        }
        #endif
    }
}

#Preview {
    SamplePickerView()
        .environmentObject(PlaygroundViewModel.previewFactory())
}
