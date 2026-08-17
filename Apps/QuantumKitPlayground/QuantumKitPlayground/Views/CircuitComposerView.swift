import SwiftUI

/// Palette + canvas only. OpenQASM lives on the Code pane; settings live with Results.
struct CircuitComposerView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            GatePaletteView()
            CircuitCanvasView()
        }
    }
}

#Preview {
    CircuitComposerView()
        .environmentObject(PlaygroundViewModel.previewFactory())
        .padding()
}
