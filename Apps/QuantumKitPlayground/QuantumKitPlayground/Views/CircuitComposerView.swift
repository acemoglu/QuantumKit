import SwiftUI

/// Palette + canvas only. OpenQASM lives on the Code pane; settings live with Results.
struct CircuitComposerView: View {
    @Environment(\.isPhoneLayout) private var isPhoneLayout

    var body: some View {
        VStack(alignment: .leading, spacing: isPhoneLayout ? 0 : 8) {
            GatePaletteView()
            if isPhoneLayout {
                Divider()
            }
            CircuitCanvasView()
        }
    }
}

#Preview {
    CircuitComposerView()
        .environmentObject(PlaygroundViewModel.previewFactory())
        .padding()
}
