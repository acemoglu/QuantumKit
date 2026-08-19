import SwiftUI

#if canImport(AppKit)
import AppKit
#endif

@main
struct QuantumKitPlaygroundApp: App {
    @StateObject private var viewModel = PlaygroundViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
                .background(Color.quantumCanvas.ignoresSafeArea())
        }
        #if os(macOS)
        .defaultSize(width: 1280, height: 820)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open…") {
                    viewModel.presentOpen()
                }
                .keyboardShortcut("o", modifiers: .command)

                Button("Save…") {
                    viewModel.presentSave()
                }
                .keyboardShortcut("s", modifiers: .command)

                Button("Export Histogram…") {
                    viewModel.presentHistogramExport()
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
                .disabled(!viewModel.canExportHistogram)
            }
            CommandGroup(after: .help) {
                Button("How to Use QuantumKit") {
                    viewModel.isPresentingHelp = true
                }
                .keyboardShortcut("?", modifiers: .command)

                Button("QuantumKit on GitHub") {
                    NSWorkspace.shared.open(PlaygroundChrome.githubURL)
                }
            }
            CommandGroup(after: .sidebar) {
                Button("Parse") {
                    viewModel.parse()
                }
                .disabled(viewModel.isBusy)

                Button("Run") {
                    viewModel.run()
                }
                .keyboardShortcut("r", modifiers: .command)
                .disabled(viewModel.isBusy)
            }
            CommandGroup(after: .undoRedo) {
                Button("Undo Circuit") {
                    viewModel.undoCanvas()
                }
                .keyboardShortcut("z", modifiers: .command)
                .disabled(viewModel.centerPane != .circuit || !viewModel.canUndoCanvas)

                Button("Move Gate Left") {
                    viewModel.moveSelectedGate(by: -1)
                }
                .keyboardShortcut("[", modifiers: .command)
                .disabled(viewModel.centerPane != .circuit || !viewModel.canMoveSelectedGateLeft)

                Button("Move Gate Right") {
                    viewModel.moveSelectedGate(by: 1)
                }
                .keyboardShortcut("]", modifiers: .command)
                .disabled(viewModel.centerPane != .circuit || !viewModel.canMoveSelectedGateRight)
            }
        }
        #endif
    }
}
