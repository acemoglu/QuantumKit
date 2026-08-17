import Foundation
import QuantumKit

@MainActor
final class PlaygroundViewModel: ObservableObject {
    @Published var sourceText: String {
        didSet {
            guard !suppressSourceSideEffects else { return }
            scheduleDebouncedParse()
        }
    }
    @Published var settings = PlaygroundSettings()
    @Published private(set) var parsedCircuit: QuantumCircuit?
    @Published private(set) var asciiPreview: String = ""
    @Published private(set) var runOutput: PlaygroundRunOutput?
    @Published private(set) var statusMessage: String = "Edit OpenQASM, then Parse or Run."
    @Published private(set) var isBusy = false
    @Published private(set) var parseError: String?
    @Published private(set) var runError: String?
    @Published var selectedSampleID: SampleCircuit.ID?
    @Published var isPresentingOpen = false
    @Published var isPresentingSave = false
    @Published var selectedCompactTab: PlaygroundTab = .circuit
    @Published var centerPane: CenterPane = .circuit
    @Published var selectedPaletteTool: PaletteTool?
    @Published var pendingPlacement: PendingGatePlacement?
    @Published var selectedGateIndex: Int?

    var editableCircuit: QuantumCircuit? { parsedCircuit }

    var lastRunSummary: String? {
        guard let output = runOutput else { return nil }
        var parts = [output.metadata.method.rawValue]
        if let device = output.metadata.deviceName, !device.isEmpty {
            parts.append(device)
        }
        parts.append(String(format: "%.2f ms", output.wallClockMilliseconds))
        if let shots = output.result.shotCounts?.shots {
            parts.append("\(shots) shots")
        }
        return parts.joined(separator: " · ")
    }

    var canAddQubit: Bool {
        (parsedCircuit?.qubitCount ?? 2) < Self.maxPlaygroundQubits
    }

    var canRemoveQubit: Bool {
        guard let circuit = parsedCircuit else { return false }
        return circuit.qubitCount > 1
    }

    var canUndoCanvas: Bool { !canvasUndoStack.isEmpty }

    var selectedSampleName: String? {
        guard let selectedSampleID else { return nil }
        return SampleCircuit.bundled.first(where: { $0.id == selectedSampleID })?.name
    }

    var suggestedFilename: String {
        if let id = selectedSampleID,
           let sample = SampleCircuit.bundled.first(where: { $0.id == id }) {
            return sample.filename
        }
        return "circuit.qasm"
    }

    var exportDocument: QASMFileDocument {
        QASMFileDocument(text: sourceText)
    }

    private let service = SimulationService()
    private var suppressSourceSideEffects = false
    private var debounceTask: Task<Void, Never>?
    private var parseGeneration = UUID()
    private var runGeneration = UUID()
    private var canvasUndoStack: [QuantumCircuit] = []
    private static let maxPlaygroundQubits = 8

    init() {
        sourceText = SampleCircuit.bundled.first?.loadSource()
            ?? Self.fallbackSource
        selectedSampleID = SampleCircuit.bundled.first?.id
        Task { await self.performParse() }
    }

    func loadSample(_ sample: SampleCircuit) {
        selectedSampleID = sample.id
        if let text = sample.loadSource() {
            replaceSource(text, clearSampleSelection: false)
            statusMessage = "Loaded sample “\(sample.name)”."
            parse()
        } else {
            parseError = "Sample “\(sample.name)” is missing from the app bundle."
            statusMessage = "Sample missing."
        }
    }

    func parse() {
        debounceTask?.cancel()
        Task { await performParse() }
    }

    func run() {
        guard !isBusy else { return }
        debounceTask?.cancel()
        isBusy = true
        runError = nil

        let source = sourceText
        let currentSettings = settings
        let service = service
        let generation = UUID()
        runGeneration = generation

        Task { [weak self] in
            let outcome = await Task.detached(priority: .userInitiated) {
                Self.detachedRun(source: source, settings: currentSettings, service: service)
            }.value

            guard let self, self.runGeneration == generation else { return }
            self.isBusy = false
            self.applyRunOutcome(outcome)
        }
    }

    func resetToSelectedSample() {
        if let id = selectedSampleID,
           let sample = SampleCircuit.bundled.first(where: { $0.id == id }) {
            loadSample(sample)
        } else if let first = SampleCircuit.bundled.first {
            loadSample(first)
        }
    }

    func presentOpen() {
        isPresentingOpen = true
    }

    func presentSave() {
        isPresentingSave = true
    }

    func handleOpenResult(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            loadFile(at: url)
        case .failure(let error):
            parseError = Self.errorDescription(error)
            statusMessage = "Open failed."
        }
    }

    func handleSaveResult(_ result: Result<URL, Error>) {
        switch result {
        case .success:
            statusMessage = "Saved OpenQASM."
        case .failure(let error):
            parseError = Self.errorDescription(error)
            statusMessage = "Save failed."
        }
    }

    func loadFile(at url: URL) {
        do {
            let text = try QASMFileIO.load(from: url)
            applyOpenedSource(text)
            statusMessage = "Opened \(url.lastPathComponent)."
        } catch {
            parseError = Self.errorDescription(error)
            statusMessage = "Could not load file."
        }
    }

    func applyOpenedSource(_ text: String) {
        replaceSource(text, clearSampleSelection: true)
        parse()
    }

    func selectPaletteTool(_ tool: PaletteTool) {
        if selectedPaletteTool == tool, pendingPlacement == nil {
            selectedPaletteTool = nil
            return
        }
        selectedPaletteTool = tool
        pendingPlacement = nil
        statusMessage = tool.qubitCount == 1
            ? "Click or drop onto a qubit wire to place \(tool.title)."
            : tool.help
    }

    func cancelPendingPlacement() {
        pendingPlacement = nil
        statusMessage = "Placement cancelled."
    }

    func handleQubitTap(_ qubit: Int) {
        if let pending = pendingPlacement {
            placePaletteTool(pending.tool, on: qubit)
            return
        }
        if let tool = selectedPaletteTool {
            placePaletteTool(tool, on: qubit)
        }
    }

    func placePaletteTool(_ tool: PaletteTool, on qubit: Int) {
        do {
            var picked = pendingPlacement?.tool == tool ? pendingPlacement?.pickedQubits ?? [] : []
            if picked.contains(qubit) {
                statusMessage = "Pick a different qubit for \(tool.title)."
                return
            }
            picked.append(qubit)
            selectedPaletteTool = tool

            if picked.count < tool.qubitCount {
                pendingPlacement = PendingGatePlacement(tool: tool, pickedQubits: picked)
                statusMessage = tool.qubitCount == 2
                    ? "Now click the second qubit for \(tool.title)."
                    : "Pick \(tool.qubitCount - picked.count) more qubit(s) for \(tool.title)."
                return
            }

            guard let gate = tool.makeGate(qubits: picked) else {
                parseError = "Could not build \(tool.title)."
                return
            }
            let base = try currentOrBlankCircuit()
            var next = try rebuildCircuit(from: base, gates: base.gates + [gate])
            // Measure needs a classical bit for this qubit.
            if case .measure = gate, next.classicalRegisters.first?.bitCount ?? 0 < next.qubitCount {
                next = try rebuildCircuit(from: next, qubitCount: next.qubitCount, gates: next.gates)
            }
            pendingPlacement = nil
            selectedGateIndex = next.gates.count - 1
            try commitVisual(next, message: "Placed \(tool.title).")
        } catch {
            parseError = Self.errorDescription(error)
            statusMessage = "Could not place gate."
        }
    }

    func selectGate(at index: Int) {
        if selectedGateIndex == index {
            deleteSelectedGate()
            return
        }
        selectedGateIndex = index
        statusMessage = "Gate \(index) selected — click again or Delete to remove."
    }

    func deleteSelectedGate() {
        guard let circuit = parsedCircuit, let index = selectedGateIndex,
              circuit.gates.indices.contains(index) else { return }
        do {
            var gates = circuit.gates
            gates.remove(at: index)
            let next = try rebuildCircuit(from: circuit, gates: gates)
            selectedGateIndex = nil
            try commitVisual(next, message: "Removed gate.")
        } catch {
            parseError = Self.errorDescription(error)
        }
    }

    func addQubit() {
        guard let circuit = try? currentOrBlankCircuit() else { return }
        guard circuit.qubitCount < Self.maxPlaygroundQubits else { return }
        do {
            let next = try rebuildCircuit(from: circuit, qubitCount: circuit.qubitCount + 1, gates: circuit.gates)
            try commitVisual(next, message: "Added qubit q\(next.qubitCount - 1).")
        } catch {
            parseError = Self.errorDescription(error)
        }
    }

    func removeLastQubit() {
        guard let circuit = parsedCircuit, circuit.qubitCount > 1 else { return }
        let last = circuit.qubitCount - 1
        if circuit.gates.contains(where: { $0.affectedQubits.contains(last) }) {
            parseError = "Remove gates on q\(last) before deleting that qubit."
            statusMessage = "Qubit still in use."
            return
        }
        do {
            let next = try rebuildCircuit(from: circuit, qubitCount: last, gates: circuit.gates)
            try commitVisual(next, message: "Removed q\(last).")
        } catch {
            parseError = Self.errorDescription(error)
        }
    }

    func clearCanvas() {
        do {
            let n = parsedCircuit?.qubitCount ?? 2
            let next = try Self.blankCircuit(qubitCount: n)
            selectedGateIndex = nil
            pendingPlacement = nil
            try commitVisual(next, message: "Cleared circuit.")
        } catch {
            parseError = Self.errorDescription(error)
        }
    }

    func undoCanvas() {
        guard let previous = canvasUndoStack.popLast() else { return }
        selectedGateIndex = nil
        pendingPlacement = nil
        parsedCircuit = previous
        asciiPreview = previous.asciiDiagram()
        parseError = nil
        runOutput = nil
        suppressSourceSideEffects = true
        sourceText = (try? previous.openQASM2()) ?? sourceText
        suppressSourceSideEffects = false
        statusMessage = "Undo."
    }

    private func currentOrBlankCircuit() throws -> QuantumCircuit {
        if let parsedCircuit { return parsedCircuit }
        return try Self.blankCircuit(qubitCount: 2)
    }

    private func commitVisual(_ circuit: QuantumCircuit, message: String) throws {
        if let current = parsedCircuit {
            canvasUndoStack.append(current)
            if canvasUndoStack.count > 40 {
                canvasUndoStack.removeFirst(canvasUndoStack.count - 40)
            }
        }
        parsedCircuit = circuit
        asciiPreview = circuit.asciiDiagram()
        parseError = nil
        runError = nil
        runOutput = nil
        selectedSampleID = nil
        suppressSourceSideEffects = true
        if let qasm = try? circuit.openQASM2() {
            sourceText = qasm
        } else {
            sourceText = try circuit.openQASM()
        }
        suppressSourceSideEffects = false
        statusMessage = message
    }

    private func rebuildCircuit(
        from template: QuantumCircuit,
        qubitCount: Int? = nil,
        gates: [Gate]
    ) throws -> QuantumCircuit {
        let n = qubitCount ?? template.qubitCount
        var next = try Self.blankCircuit(qubitCount: n)
        for gate in gates {
            try next.apply(gate)
        }
        return next
    }

    private static func blankCircuit(qubitCount: Int) throws -> QuantumCircuit {
        try QuantumCircuit(
            qubitCount: qubitCount,
            classicalRegisters: [try ClassicalRegisterSpec(bitCount: qubitCount)]
        )
    }

    func applyDroppedProviders(_ providers: [NSItemProvider]) -> Bool {
        guard QASMFileIO.canConsume(providers) else { return false }
        Task { [weak self] in
            do {
                let text = try await QASMFileIO.loadDropped(from: providers)
                self?.applyOpenedSource(text)
                self?.statusMessage = "Loaded dropped OpenQASM."
            } catch {
                self?.parseError = Self.errorDescription(error)
                self?.statusMessage = "Drop failed."
            }
        }
        return true
    }

    func selectSample(id: SampleCircuit.ID?) {
        guard let id, let sample = SampleCircuit.bundled.first(where: { $0.id == id }) else { return }
        if selectedSampleID == id, sourceText == sample.loadSource() {
            return
        }
        loadSample(sample)
    }

    private func scheduleDebouncedParse() {
        debounceTask?.cancel()
        debounceTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            await self?.performParse()
        }
    }

    private func performParse() async {
        let source = sourceText
        let service = service
        let generation = UUID()
        parseGeneration = generation

        let outcome = await Task.detached(priority: .utility) {
            Result { () -> (QuantumCircuit, String) in
                let circuit = try service.parse(source: source)
                let ascii = circuit.asciiDiagram()
                return (circuit, ascii)
            }
        }.value

        guard parseGeneration == generation else { return }

        switch outcome {
        case .success(let (circuit, ascii)):
            parsedCircuit = circuit
            asciiPreview = ascii
            parseError = nil
            selectedGateIndex = nil
            pendingPlacement = nil
            canvasUndoStack.removeAll()
            statusMessage = "Parsed \(circuit.qubitCount) qubit(s), \(circuit.gates.count) gate(s)."
        case .failure(let error):
            parsedCircuit = nil
            asciiPreview = ""
            parseError = Self.errorDescription(error)
            statusMessage = "Parse failed."
        }
    }

    private func applyRunOutcome(_ outcome: DetachedRunOutcome) {
        switch outcome {
        case .success(let output):
            parsedCircuit = output.circuit
            asciiPreview = output.asciiDiagram
            parseError = nil
            runOutput = output
            runError = nil
            statusMessage = "Run finished (\(output.metadata.method.rawValue))."
        case .parseFailure(let message):
            parsedCircuit = nil
            asciiPreview = ""
            parseError = message
            statusMessage = "Parse failed."
        case .runFailure(let circuit, let ascii, let message):
            parsedCircuit = circuit
            asciiPreview = ascii
            parseError = nil
            runError = message
            statusMessage = "Run failed."
        }
    }

    private func replaceSource(_ text: String, clearSampleSelection: Bool) {
        suppressSourceSideEffects = true
        sourceText = text
        suppressSourceSideEffects = false
        if clearSampleSelection {
            selectedSampleID = nil
        }
        runOutput = nil
        runError = nil
        parseError = nil
        selectedGateIndex = nil
        pendingPlacement = nil
    }

    nonisolated private static func detachedRun(
        source: String,
        settings: PlaygroundSettings,
        service: SimulationService
    ) -> DetachedRunOutcome {
        let circuit: QuantumCircuit
        let ascii: String
        do {
            circuit = try service.parse(source: source)
            ascii = circuit.asciiDiagram()
        } catch {
            return .parseFailure(errorDescription(error))
        }
        do {
            let output = try service.run(circuit: circuit, settings: settings)
            return .success(output)
        } catch {
            return .runFailure(circuit: circuit, ascii: ascii, message: errorDescription(error))
        }
    }

    nonisolated static func errorDescription(_ error: Error) -> String {
        if let localized = error as? LocalizedError,
           let description = localized.errorDescription,
           !description.isEmpty {
            return description
        }
        return error.localizedDescription
    }

    private static let fallbackSource = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q -> c;
    """
}

enum CenterPane: String, Hashable, CaseIterable, Identifiable {
    case circuit
    case code

    var id: String { rawValue }

    var title: String {
        switch self {
        case .circuit: return "Circuit"
        case .code: return "Code"
        }
    }

    var systemImage: String {
        switch self {
        case .circuit: return "point.3.connected.trianglepath.dotted"
        case .code: return "chevron.left.forwardslash.chevron.right"
        }
    }
}

enum PlaygroundTab: String, Hashable, CaseIterable, Identifiable {
    case circuit
    case editor
    case results

    var id: String { rawValue }

    var title: String {
        switch self {
        case .circuit: return "Circuit"
        case .editor: return "Code"
        case .results: return "Results"
        }
    }

    var systemImage: String {
        switch self {
        case .circuit: return "point.3.connected.trianglepath.dotted"
        case .editor: return "chevron.left.forwardslash.chevron.right"
        case .results: return "chart.bar"
        }
    }
}

private enum DetachedRunOutcome: Sendable {
    case success(PlaygroundRunOutput)
    case parseFailure(String)
    case runFailure(circuit: QuantumCircuit, ascii: String, message: String)
}

#if DEBUG
extension PlaygroundViewModel {
    static func previewFactory() -> PlaygroundViewModel {
        PlaygroundViewModel()
    }
}
#endif
