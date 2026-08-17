import SwiftUI

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
        }
        #endif
    }
}
