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
                    section("How to use Circuit", systemImage: "point.3.connected.trianglepath.dotted") {
                        ForEach(circuitHelpRows, id: \.title) { row in
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
                Link("Source on GitHub", destination: PlaygroundChrome.githubURL)
                    .font(.callout)
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
        .frame(minWidth: 520, idealWidth: 560, minHeight: 620, idealHeight: 720)
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
            ("Circuit", "Build on the canvas. See How to use Circuit below."),
            ("Code", "OpenQASM. Open and Save live in the overflow menu. On iPad you can also drop a .qasm file onto the editor."),
            ("Results", "Device, shots, seed, then the histogram after Run. Export CSV from the histogram to save counts."),
        ]
    }
    private var circuitHelpRows: [(title: String, body: String)] {
        [
            ("Place", "Tap a gate, or drag it onto a qubit wire. CNOT, CZ, and SWAP need two taps (control then target). The dashed + at the right appends. The small + between columns inserts in the middle."),
            ("Insert", "Tap an existing gate to select it. The next gate you place goes before that one. Tap a thin line between columns to set the insert point."),
            ("Blocks", "Bell, GHZ-3, H all, Meas all. Tap the block, then a qubit. Bell/GHZ start on that wire and add qubits if needed."),
            ("Qubits", "Touch and hold a q label to insert a qubit there or below, or to remove that wire if it has no gates."),
            ("Edit", "Move left/right reorders the selected gate. Delete removes it. Undo walks back canvas edits. The Code tab stays in sync."),
        ]
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
            ("Center", "Circuit canvas or OpenQASM. Drag a .qasm file onto the editor to load it. How to use Circuit is below."),
            ("Right", "Device, shots, seed, then the histogram after Run. Export CSV saves bitstring counts."),
        ]
    }
    private var circuitHelpRows: [(title: String, body: String)] {
        [
            ("Place", "Click a gate in the palette, then a qubit wire — or drag the gate onto the wire. CNOT, CZ, and SWAP: first click is control (or first qubit), second is the other. The dashed + at the right appends."),
            ("Insert", "Click an existing gate to select it. The next gate you place goes before that one. Click a thin line between columns to drop something in the middle."),
            ("Blocks", "Bell, GHZ-3, H all, Meas all. Click the block, then a qubit. Bell/GHZ start on that wire and add qubits if needed."),
            ("Qubits", "Right-click a q label to insert a qubit there or below, or to remove that wire if it has no gates. Add/Remove in the header still grow from the last wire."),
            ("Keyboard", "Click the circuit first. ⌫ deletes the selected gate. ⌘Z undoes the last canvas edit. ⌘[ / ⌘] move the selected gate. ⌘R runs. ⌘? opens this sheet."),
        ]
    }
    #endif

    private var limitsBody: String {
        "This app composes, parses, runs, and histograms circuits locally. It is not a cloud backend or a pulse IDE."
    }
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
