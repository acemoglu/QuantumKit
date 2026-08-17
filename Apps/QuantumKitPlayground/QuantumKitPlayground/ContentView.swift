import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel
    @Environment(\.horizontalSizeClass) private var sizeClass

    var body: some View {
        Group {
            if usesCompactTabs {
                compactLayout
            } else {
                regularLayout
            }
        }
        .navigationTitle("QuantumKit Playground")
        .toolbar { playgroundToolbar }
        .safeAreaInset(edge: .top, spacing: 0) {
            errorBanners
        }
        .safeAreaInset(edge: .bottom) {
            StatusBarView(
                message: viewModel.statusMessage,
                isBusy: viewModel.isBusy,
                lastRunSummary: viewModel.lastRunSummary
            )
        }
        .fileImporter(
            isPresented: $viewModel.isPresentingOpen,
            allowedContentTypes: QASMFileIO.readableContentTypes,
            allowsMultipleSelection: false
        ) { result in
            viewModel.handleOpenResult(result)
        }
        .fileExporter(
            isPresented: $viewModel.isPresentingSave,
            document: viewModel.exportDocument,
            contentType: QASMFileIO.qasmContentType,
            defaultFilename: viewModel.suggestedFilename
        ) { result in
            viewModel.handleSaveResult(result)
        }
    }

    private var usesCompactTabs: Bool {
        #if os(iOS)
        sizeClass == .compact
        #else
        false
        #endif
    }

    private var regularLayout: some View {
        NavigationSplitView {
            SamplePickerView()
                .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 260)
        } detail: {
            splitWorkspace
        }
    }

    @ViewBuilder
    private var splitWorkspace: some View {
        #if os(macOS)
        HSplitView {
            centerStage
                .frame(minWidth: 420)
            ResultsPanelView(showsSettings: true)
                .frame(minWidth: 280, idealWidth: 340)
                .padding([.vertical, .trailing], 8)
        }
        .padding(.leading, 4)
        #else
        HStack(spacing: 8) {
            centerStage
            ResultsPanelView(showsSettings: true)
                .frame(minWidth: 280)
        }
        .padding(.trailing, 8)
        #endif
    }

    @ViewBuilder
    private var centerStage: some View {
        VStack(spacing: 0) {
            Picker("Workspace", selection: $viewModel.centerPane) {
                ForEach(CenterPane.allCases) { pane in
                    Text(pane.title).tag(pane)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 260)
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 6)

            switch viewModel.centerPane {
            case .circuit:
                CircuitComposerView()
                    .padding(.horizontal, 8)
                    .padding(.bottom, 8)
            case .code:
                CircuitEditorView(showsSettings: false)
            }
        }
    }

    private var compactLayout: some View {
        TabView(selection: $viewModel.selectedCompactTab) {
            NavigationStack {
                VStack(spacing: 8) {
                    CircuitComposerView()
                    SettingsPanelView()
                        .padding(.horizontal)
                }
                .padding(.bottom, 8)
                .navigationTitle("Circuit")
            }
            .tabItem {
                Label(PlaygroundTab.circuit.title, systemImage: PlaygroundTab.circuit.systemImage)
            }
            .tag(PlaygroundTab.circuit)

            NavigationStack {
                CircuitEditorView(showsSettings: false)
                    .navigationTitle("Code")
            }
            .tabItem {
                Label(PlaygroundTab.editor.title, systemImage: PlaygroundTab.editor.systemImage)
            }
            .tag(PlaygroundTab.editor)

            NavigationStack {
                ResultsPanelView(showsSettings: false)
                    .padding()
                    .navigationTitle("Results")
            }
            .tabItem {
                Label(PlaygroundTab.results.title, systemImage: PlaygroundTab.results.systemImage)
            }
            .tag(PlaygroundTab.results)
        }
    }

    @ToolbarContentBuilder
    private var playgroundToolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .automatic) {
            Button {
                viewModel.presentOpen()
            } label: {
                Label("Open…", systemImage: "folder")
            }
            .help("Open an OpenQASM file")

            #if os(macOS)
            Button {
                viewModel.presentSave()
            } label: {
                Label("Save…", systemImage: "square.and.arrow.down")
            }
            .help("Save OpenQASM")
            #endif
        }

        ToolbarItemGroup(placement: .primaryAction) {
            Button {
                viewModel.parse()
            } label: {
                Label("Parse", systemImage: "text.alignleft")
            }
            .disabled(viewModel.isBusy)
            .help("Parse OpenQASM")

            if viewModel.isBusy {
                ProgressView()
                    .controlSize(.small)
            }

            Button {
                viewModel.run()
            } label: {
                Label("Run", systemImage: "play.fill")
            }
            .keyboardShortcut("r", modifiers: .command)
            .disabled(viewModel.isBusy)
            .help("Run the circuit (⌘R)")
        }
    }

    @ViewBuilder
    private var errorBanners: some View {
        if viewModel.parseError != nil || viewModel.runError != nil {
            VStack(spacing: 8) {
                if let parseError = viewModel.parseError {
                    ErrorBannerView(title: "Parse Error", message: parseError)
                }
                if let runError = viewModel.runError {
                    ErrorBannerView(title: "Run Error", message: runError)
                }
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 4)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(PlaygroundViewModel.previewFactory())
}
