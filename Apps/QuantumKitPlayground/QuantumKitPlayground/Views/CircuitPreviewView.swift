import SwiftUI

struct CircuitPreviewView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "point.3.connected.trianglepath.dotted")
                    .foregroundStyle(.secondary)
                Text("Circuit")
                    .font(.headline)
                Spacer()
                if let circuit = viewModel.parsedCircuit {
                    Text("\(circuit.qubitCount)q · \(circuit.gates.count)g")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }

            if viewModel.asciiPreview.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "waveform.path.ecg")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                    Text(viewModel.parseError == nil
                         ? "ASCII preview appears after a successful parse."
                         : "Fix the OpenQASM to preview this circuit.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView([.horizontal, .vertical]) {
                    Text(viewModel.asciiPreview)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .padding(4)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .playgroundPanel()
    }
}

#Preview {
    CircuitPreviewView()
        .environmentObject(PlaygroundViewModel.previewFactory())
        .frame(width: 420, height: 240)
        .padding()
}
