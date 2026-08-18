import SwiftUI

struct MacHowToView: View {
    var onDismiss: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    intro
                    section("Start here", systemImage: "1.circle") {
                        labeled("Samples", "Bell, Toffoli, GHZ-4. These finish in well under a second on Automatic.")
                        labeled("New Circuit", newCircuitBody)
                        labeled("Run", runBody)
                    }
                    section(layoutTitle, systemImage: layoutSymbol) {
                        ForEach(layoutRows, id: \.title) { row in
                            labeled(row.title, row.body)
                        }
                    }
                    section("If it feels slow", systemImage: "gauge.with.dots.needle.67percent") {
                        labeled("Width, not shots", "Time is almost entirely qubit count × gates. Raising shots on a noiseless circuit barely changes wall time. Dropping from 30 qubits to 4 does.")
                        labeled("Stay small in the app", "The visual canvas caps at 8 qubits. CPU tops out at 16. Metal can go higher, but 20+ qubits is seconds; ~30 qubits can be tens of seconds and allocates gigabytes. That is expected, not a hang.")
                        labeled("Automatic", "Leave Device on Automatic. The first Metal run after launch pays GPU warmup; the next Run on the same width is the fair number.")
                        labeled("Do not start at 30 qubits", "Open Bell, run it, confirm the histogram, then grow. A 30-qubit OpenQASM paste is a stress test, not everyday use.")
                    }
                    section("Limits", systemImage: "square.stack.3d.up") {
                        Text(limitsBody)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(24)
            }
            Divider()
            HStack {
                Spacer()
                Button("Got it") {
                    onDismiss()
                }
                .keyboardShortcut(.defaultAction)
                .controlSize(.large)
            }
            .padding(16)
        }
        #if os(macOS)
        .frame(minWidth: 520, idealWidth: 560, minHeight: 560, idealHeight: 640)
        #endif
        .background(Color.quantumCanvas)
    }

    private var header: some View {
        HStack(spacing: 12) {
            QuantumKitMark(size: 36)
            VStack(alignment: .leading, spacing: 2) {
                Text("How to Use QuantumKit")
                    .font(.title2.weight(.semibold))
                Text(headerSubtitle)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(20)
    }

    private var intro: some View {
        Text(introBody)
            .font(.body)
            .foregroundStyle(Color.quantumInk)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func section(_ title: String, systemImage: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            content()
        }
    }

    private func labeled(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.subheadline.weight(.semibold))
            Text(body)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    #if os(iOS)
    private var headerSubtitle: String { "Keep runs small, then scale." }
    private var introBody: String {
        "QuantumKit runs the circuit on this device. Small samples are instant. Wide statevectors are heavy by physics — use the app that way and it feels fast."
    }
    private var newCircuitBody: String {
        "Samples → New Circuit for a blank 2-qubit canvas, then Save Circuit to keep it under My Circuits."
    }
    private var runBody: String {
        "Tap Run in the navigation bar. Errors only show after Run, not while you type."
    }
    private var layoutTitle: String { "The tabs" }
    private var layoutSymbol: String { "square.split.2x1" }
    private var layoutRows: [(title: String, body: String)] {
        [
            ("Circuit", "Tap a gate, then a qubit wire (CNOT/CZ/SWAP need two taps). Samples and Run sit in the navigation bar."),
            ("Code", "OpenQASM. Open and Save live in the overflow menu. On iPad you can also drop a .qasm file onto the editor."),
            ("Results", "Device, shots, seed, then the histogram after Run."),
        ]
    }
    private var limitsBody: String {
        "This app composes, parses, runs, and histograms circuits locally. It is not a cloud backend or a pulse IDE."
    }
    #else
    private var headerSubtitle: String { "Keep runs small, then scale." }
    private var introBody: String {
        "QuantumKit runs the circuit on this Mac. Small samples are instant. Wide statevectors are heavy by physics — use the app that way and it feels fast."
    }
    private var newCircuitBody: String {
        "Sidebar → New Circuit for a blank 2-qubit canvas, then Save Circuit to keep it under My Circuits."
    }
    private var runBody: String {
        "⌘R. Errors only show after Run, not while you type."
    }
    private var layoutTitle: String { "The window" }
    private var layoutSymbol: String { "rectangle.split.3x1" }
    private var layoutRows: [(title: String, body: String)] {
        [
            ("Left", "Bundled samples and your saved circuits. Save Circuit is the in-app library. Save… exports a .qasm file."),
            ("Center", "Circuit: click a gate, then a qubit wire (CNOT/CZ/SWAP need two clicks). Code: OpenQASM. Drag a .qasm file onto the editor to load it."),
            ("Right", "Device, shots, seed, then the histogram after Run."),
        ]
    }
    private var limitsBody: String {
        "This app composes, parses, runs, and histograms circuits locally. It is not a cloud backend or a pulse IDE."
    }
    #endif
}

enum MacHowTo {
    #if os(iOS)
    static let seenKey = "quantumkit.iosHowToSeen"
    #else
    static let seenKey = "quantumkit.macHowToSeen"
    #endif

    static var hasSeen: Bool {
        UserDefaults.standard.bool(forKey: seenKey)
    }

    static func markSeen() {
        UserDefaults.standard.set(true, forKey: seenKey)
    }
}

#Preview {
    MacHowToView(onDismiss: {})
}
