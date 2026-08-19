import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel
    @Environment(\.horizontalSizeClass) private var sizeClass
    @Environment(\.verticalSizeClass) private var verticalSizeClass
    @State private var isPresentingSamples = false
    @State private var toolbarHint: String?

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
            errorBanners
                .padding(.horizontal, 12)
                .padding(.top, 8)
        }
        .overlay(alignment: .topTrailing) {
            if let toolbarHint {
                Text(toolbarHint)
                    .font(.caption)
                    .foregroundStyle(Color.quantumInk)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.quantumCard, in: Capsule())
                    .overlay(
                        Capsule()
                            .strokeBorder(Color.quantumInk.opacity(0.12))
                    )
                    .padding(.top, 6)
                    .padding(.trailing, 16)
                    .allowsHitTesting(false)
            }
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
        .fileExporter(
            isPresented: $viewModel.isPresentingHistogramExport,
            document: viewModel.histogramExportDocument,
            contentType: .commaSeparatedText,
            defaultFilename: viewModel.suggestedHistogramFilename
        ) { result in
            viewModel.handleHistogramExportResult(result)
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
                sampleID: viewModel.selectedLibraryID
            ))
        }
        .alert("Save Circuit", isPresented: $viewModel.isPresentingSaveToLibrary) {
            TextField("Name", text: $viewModel.libraryNameDraft)
            Button("Save") {
                viewModel.confirmSaveToLibrary()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            #if os(iOS)
            Text("Stored under My Circuits in Samples.")
            #else
            Text("Stored in the sidebar under My Circuits.")
            #endif
        }
        .sheet(isPresented: $viewModel.isPresentingHelp) {
            MacHowToView {
                MacHowTo.markSeen()
                viewModel.isPresentingHelp = false
            }
        }
        .onAppear {
            if !MacHowTo.hasSeen {
                viewModel.isPresentingHelp = true
            }
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
            #if os(macOS)
            .toolbarBackground(Color.quantumCanvas, for: .windowToolbar)
            #endif
            .toolbar {
                logoToolbarItem
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
        HStack(spacing: 12) {
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
        .toolbarBackground(Color.quantumCanvas, for: .tabBar)
        .toolbarBackground(.visible, for: .tabBar)
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
                .toolbarBackground(Color.quantumCanvas, for: .navigationBar)
                .toolbarBackground(.visible, for: .navigationBar)
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
                Button("How to Use", systemImage: "questionmark.circle") {
                    viewModel.isPresentingHelp = true
                }
                Button("Open…", systemImage: "folder") {
                    viewModel.presentOpen()
                }
                Button("Save…", systemImage: "square.and.arrow.down") {
                    viewModel.presentSave()
                }
                Button("Save to My Circuits", systemImage: "tray.and.arrow.down") {
                    viewModel.presentSaveToLibrary()
                }
                Button("Export Histogram…", systemImage: "tablecells") {
                    viewModel.presentHistogramExport()
                }
                .disabled(!viewModel.canExportHistogram)
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
    private var logoToolbarItem: some ToolbarContent {
        if #available(macOS 26.0, iOS 26.0, *) {
            ToolbarItem(placement: .navigation) {
                QuantumKitMark(size: 22)
                    .help("QuantumKit")
            }
            .sharedBackgroundVisibility(.hidden)
        } else {
            ToolbarItem(placement: .navigation) {
                QuantumKitMark(size: 22)
                    .help("QuantumKit")
            }
        }
    }

    @ToolbarContentBuilder
    private var desktopToolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .automatic) {
            Button {
                viewModel.isPresentingHelp = true
            } label: {
                Label("How to Use", systemImage: "questionmark.circle")
            }
            .delayedHoverHint("How to use QuantumKit on this Mac", activeHint: $toolbarHint)
            .help("How to use QuantumKit on this Mac")

            Button {
                viewModel.presentOpen()
            } label: {
                Label("Open…", systemImage: "folder")
            }
            .delayedHoverHint("Open an OpenQASM file", activeHint: $toolbarHint)
            .help("Open an OpenQASM file")

            Button {
                viewModel.presentSave()
            } label: {
                Label("Save…", systemImage: "square.and.arrow.down")
            }
            .delayedHoverHint("Export OpenQASM to a file", activeHint: $toolbarHint)
            .help("Export OpenQASM to a file")

            Button {
                viewModel.presentSaveToLibrary()
            } label: {
                Label("Save Circuit", systemImage: "tray.and.arrow.down")
            }
            .delayedHoverHint("Save to My Circuits in the sidebar", activeHint: $toolbarHint)
            .help("Save to My Circuits in the sidebar")
        }

        ToolbarItemGroup(placement: .primaryAction) {
            Button {
                viewModel.parse()
            } label: {
                Label("Parse", systemImage: "text.alignleft")
            }
            .disabled(viewModel.isBusy)
            .delayedHoverHint("Check OpenQASM without running", activeHint: $toolbarHint)
            .help("Check OpenQASM without running")

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
            .delayedHoverHint("Simulate the circuit (⌘R)", activeHint: $toolbarHint)
            .help("Simulate the circuit (⌘R)")
        }
    }

    @ViewBuilder
    private var errorBanners: some View {
        if let banner = viewModel.runBanner {
            ErrorBannerView(title: banner.title, message: banner.message)
                .padding(.horizontal, 12)
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
