import Charts
import SwiftUI

struct HistogramBar: Identifiable {
    var id: String { label }
    let label: String
    let count: Int
}

struct ResultsHistogramView: View {
    let bars: [HistogramBar]
    var canExport = false
    var onExport: () -> Void = {}

    var body: some View {
        GroupBox {
            if bars.isEmpty {
                emptyState
            } else {
                Chart(bars) { bar in
                    BarMark(
                        x: .value("Bitstring", bar.label),
                        y: .value("Counts", bar.count)
                    )
                    .foregroundStyle(Color.accentColor)
                }
                .chartXAxis {
                    AxisMarks { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel {
                            if let label = value.as(String.self) {
                                Text(label)
                                    .font(.system(.caption2, design: .monospaced))
                            }
                        }
                    }
                }
                .chartYAxisLabel("Counts")
                .frame(minHeight: 180)
                .padding(.top, 4)
            }
        } label: {
            HStack(spacing: 8) {
                Label("Measurement Histogram", systemImage: "chart.bar.fill")
                Spacer(minLength: 8)
                if canExport {
                    Button("Export CSV") {
                        onExport()
                    }
                    .buttonStyle(.borderless)
                    .help("Save bitstring counts as a CSV file")
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "chart.bar.xaxis")
                .font(.title2)
                .foregroundStyle(.secondary)
            Text("No shot histogram")
                .font(.headline)
            Text("Run a circuit with measurements to plot MSB bitstring counts.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, minHeight: 140)
        .padding(.vertical, 8)
    }
}

#Preview {
    ResultsHistogramView(
        bars: [
            HistogramBar(label: "00", count: 1024),
            HistogramBar(label: "11", count: 1024),
        ]
    )
    .padding()
}
