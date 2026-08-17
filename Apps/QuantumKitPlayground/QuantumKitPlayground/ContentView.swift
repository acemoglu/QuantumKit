import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel
    @Environment(\.horizontalSizeClass) private var sizeClass
    @Environment(\.verticalSizeClass) private var verticalSizeClass
    @State private var isPresentingSamples = false

    var body: some View {
        Group {
            #if os(iOS)
            iosTabLayout
            #else
            regularChrome
            #endif
        }
        .environment(\.isPhoneLayout, usesCompactChrome)
        .overlay(alignment: .top) {
            #if os(iOS)
            errorBanners
                .padding(.horizontal, 12)
                .padding(.top, 8)
            #endif
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            #if os(macOS)
            errorBanners
            #endif
        }
        .safeAreaInset(edge: .bottom) {
            #if os(macOS)
            StatusBarView(
                message: viewModel.statusMessage,
                isBusy: viewModel.isBusy,
                lastRunSummary: viewModel.lastRunSummary
            )
            #endif
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
        .sheet(isPresented: $isPresentingSamples) {
            NavigationStack {
                SamplePickerView()
                    .navigationTitle("Samples")
                    #if os(iOS)
                    .navigationBarTitleDisplayMode(.inline)
                    #endif
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { isPresentingSamples = false }
                        }
                    }
            }
            #if os(iOS)
            .presentationDetents(usesCompactChrome ? [.medium, .large] : [.large])
            #endif
            .modifier(DismissSamplesWhenSelectionChanges(
                isPresented: $isPresentingSamples,
                sampleID: viewModel.selectedSampleID
            ))
        }
    }

    /// Compact chrome: iPhone (including Plus/Max landscape) and iPad slide over.
    private var usesCompactChrome: Bool {
        #if os(iOS)
        sizeClass == .compact || verticalSizeClass == .compact
        #else
        false
        #endif
    }

    private var regularChrome: some View {
        regularLayout
            .navigationTitle("QuantumKit")
            .toolbar {
                ToolbarItem(placement: .navigation) {
                    QuantumKitMark(size: 22)
                        .help("QuantumKit")
                }
                desktopToolbar
            }
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

    #if os(iOS)
    private var iosTabLayout: some View {
        TabView(selection: $viewModel.selectedCompactTab) {
            iosTab(PlaygroundTab.circuit) {
                CircuitComposerView()
            }
            iosTab(PlaygroundTab.editor) {
                CircuitEditorView(showsSettings: false)
            }
            iosTab(PlaygroundTab.results) {
                ResultsPanelView(showsSettings: true)
            }
        }
        .modifier(IOSTabBarOnlyStyle())
    }

    private func iosTab<Content: View>(
        _ tab: PlaygroundTab,
        @ViewBuilder content: () -> Content
    ) -> some View {
        NavigationStack {
            content()
                .navigationTitle(tab.title)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { iosToolbar }
        }
        .tabItem {
            Label(tab.title, systemImage: tab.systemImage)
        }
        .tag(tab)
    }

    @ToolbarContentBuilder
    private var iosToolbar: some ToolbarContent {
        ToolbarItem(placement: .navigationBarLeading) {
            Button {
                isPresentingSamples = true
            } label: {
                Image(systemName: "square.stack")
            }
            .accessibilityLabel("Samples")
        }

        ToolbarItem(placement: .navigationBarTrailing) {
            Menu {
                Button("Open…", systemImage: "folder") {
                    viewModel.presentOpen()
                }
                Button("Save…", systemImage: "square.and.arrow.down") {
                    viewModel.presentSave()
                }
                Button("Parse", systemImage: "text.alignleft") {
                    viewModel.parse()
                }
                .disabled(viewModel.isBusy)
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .accessibilityLabel("More")
        }

        ToolbarItem(placement: .navigationBarTrailing) {
            if viewModel.isBusy {
                ProgressView()
            } else {
                Button {
                    viewModel.run()
                    viewModel.selectedCompactTab = .results
                } label: {
                    Image(systemName: "play.fill")
                }
                .accessibilityLabel("Run")
            }
        }
    }
    #endif

    @ToolbarContentBuilder
    private var desktopToolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .automatic) {
            Button {
                viewModel.presentOpen()
            } label: {
                Label("Open…", systemImage: "folder")
            }
            .help("Open an OpenQASM file")

            Button {
                viewModel.presentSave()
            } label: {
                Label("Save…", systemImage: "square.and.arrow.down")
            }
            .help("Save OpenQASM")
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
            .padding(.horizontal, usesCompactChrome ? 0 : 12)
            .padding(.top, usesCompactChrome ? 0 : 8)
            .padding(.bottom, usesCompactChrome ? 0 : 4)
        }
    }
}

#if os(iOS)
/// iOS 18+ TabView becomes a sidebar on iPad; keep a real bottom tab bar.
private struct IOSTabBarOnlyStyle: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 18.0, *) {
            content.tabViewStyle(.tabBarOnly)
        } else {
            content
        }
    }
}
#endif

#Preview {
    ContentView()
        .environmentObject(PlaygroundViewModel.previewFactory())
}

private struct DismissSamplesWhenSelectionChanges: ViewModifier {
    @Binding var isPresented: Bool
    let sampleID: SampleCircuit.ID?

    func body(content: Content) -> some View {
        #if os(iOS)
        if #available(iOS 17.0, *) {
            content.onChange(of: sampleID) { _, _ in
                if isPresented { isPresented = false }
            }
        } else {
            content.onChange(of: sampleID) { _ in
                if isPresented { isPresented = false }
            }
        }
        #else
        content.onChange(of: sampleID) { _, _ in
            if isPresented { isPresented = false }
        }
        #endif
    }
}
