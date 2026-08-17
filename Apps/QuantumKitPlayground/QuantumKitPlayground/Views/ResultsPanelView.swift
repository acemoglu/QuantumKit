import SwiftUI

struct ResultsPanelView: View {
    @EnvironmentObject private var viewModel: PlaygroundViewModel
    @Environment(\.isPhoneLayout) private var isPhoneLayout
    var showsSettings: Bool = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if isPhoneLayout {
                    phoneStatus
                }

                if showsSettings {
                    SettingsPanelView()
                }

                if viewModel.isBusy {
                    HStack(spacing: 8) {
                        ProgressView()
                        Text("Running simulation…")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                if let output = viewModel.runOutput {
                    if let summary = viewModel.lastRunSummary {
                        Text(summary)
                            .font(.system(.headline, design: .monospaced))
                            .textSelection(.enabled)
                    }
                    MetadataSection(output: output)
                    ResultsHistogramView(bars: histogramBars(from: output))
                    if output.histogram.isEmpty, output.result.execution != nil {
                        ExecutionSection(output: output)
                    }
                } else if let circuit = viewModel.parsedCircuit {
                    ParsedSummarySection(qubitCount: circuit.qubitCount, gateCount: circuit.gates.count)
                    ResultsHistogramView(bars: [])
                } else {
                    PlaceholderSection()
                }
            }
            .padding(isPhoneLayout ? 16 : 16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background {
            if !isPhoneLayout {
                RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous)
                    .fill(.regularMaterial)
            }
        }
        .overlay {
            if !isPhoneLayout {
                RoundedRectangle(cornerRadius: PlaygroundChrome.cornerRadius, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.12))
            }
        }
    }

    private var phoneStatus: some View {
        Text(viewModel.statusMessage)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func histogramBars(from output: PlaygroundRunOutput) -> [HistogramBar] {
        output.bitstringHistogram.map { row in
            HistogramBar(label: row.label, count: row.count)
        }
    }
}

private struct MetadataSection: View {
    let output: PlaygroundRunOutput

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                labeled("Method", output.metadata.method.rawValue, systemImage: "cpu")
                if let device = output.metadata.deviceName {
                    labeled("Device", device, systemImage: "memorychip")
                }
                labeled("Wall", String(format: "%.2f ms", output.wallClockMilliseconds), systemImage: "timer")
                labeled("Qubits", "\(output.metadata.qubitCount)", systemImage: "circle.grid.2x1")
                labeled("Gates", "\(output.metadata.gateCount)", systemImage: "square.stack.3d.up")
                if let seed = output.metadata.seed {
                    labeled("Seed", "\(seed)", systemImage: "number")
                }
                if let shots = output.result.shotCounts?.shots {
                    labeled("Shots", "\(shots)", systemImage: "chart.bar")
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Run Metadata", systemImage: "info.circle")
        }
    }

    private func labeled(_ title: String, _ value: String, systemImage: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Label(title, systemImage: systemImage)
                .foregroundStyle(.secondary)
                .frame(width: 92, alignment: .leading)
            Text(value)
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
        .font(.callout)
    }
}

private struct ExecutionSection: View {
    let output: PlaygroundRunOutput

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                if let applied = output.result.execution?.appliedGateCount {
                    Text("Applied instructions: \(applied)")
                        .font(.system(.body, design: .monospaced))
                }
                if let memory = output.result.memorySlots, !memory.isEmpty {
                    Text("Classical memory: \(memory.map(String.init).joined(separator: ", "))")
                        .font(.system(.body, design: .monospaced))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("State Evolution", systemImage: "arrow.triangle.branch")
        }
    }
}

private struct ParsedSummarySection: View {
    let qubitCount: Int
    let gateCount: Int

    var body: some View {
        GroupBox {
            Text("\(qubitCount) qubit(s), \(gateCount) gate(s). Press Run to simulate.")
                .foregroundStyle(.secondary)
        } label: {
            Label("Parsed Circuit", systemImage: "checkmark.circle")
        }
    }
}

private struct PlaceholderSection: View {
    var body: some View {
        VStack(spacing: 12) {
            QuantumKitMark(size: 40)
            Text("No Results Yet")
                .font(.title3.weight(.semibold))
            Text("Parse OpenQASM to preview the circuit, then Run to simulate with QuantumKit.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, minHeight: 180)
        .padding()
    }
}

#Preview {
    ResultsPanelView()
        .environmentObject(PlaygroundViewModel.previewFactory())
}
