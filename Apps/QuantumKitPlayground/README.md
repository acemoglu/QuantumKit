# QuantumKit Playground

A SwiftUI app for editing OpenQASM, previewing ASCII circuits, and running simulations against the local **QuantumKit** package.

## Open in Xcode

1. Open `Apps/QuantumKitPlayground/QuantumKitPlayground.xcodeproj`.
2. Wait for Xcode to resolve the local Swift package at `../../` (repo root / `Package.swift`).
3. Select the **QuantumKitPlayground** scheme and a destination (**My Mac** or an iOS simulator).
4. Press **Run** (⌘R in Xcode). In the running app, **⌘R** also runs the circuit.

## What it does

- Load bundled `.qasm` samples from `QuantumKitPlayground/Resources/Samples/` (Bell, Toffoli, teleport, Grover 2q, parametric, GHZ-4).
- Edit OpenQASM in the source pane, **or build visually**: drag gates from the palette onto qubit wires (or click a gate, then a qubit). CNOT/CZ/SWAP ask for a second qubit. The ASCII preview and OpenQASM stay in sync.
- The ASCII circuit preview updates automatically after a short debounce; **Parse** still forces an immediate refresh.
- **Open…** (⌘O) and **Save…** (⌘S) on macOS load and write `.qasm` files. Drag a `.qasm` or `.txt` file onto the editor (macOS and iPad) to replace the source; files larger than 1 MB are rejected.
- **Run** (toolbar play button, ⌘R) simulates via `QuantumBackendFactory.makeRecommended` on a background thread. The UI stays responsive and a second Run is disabled while busy.
- Choose **Automatic / Metal / CPU**, shots (1…100 000, default 2048), and a numeric seed (default 7, or random).
- Results show method, device name, wall time, qubit/gate counts, and a Swift Charts histogram of MSB bitstrings.

## Layout

- **Regular width (Mac / iPad):** sample sidebar; **Circuit | Code** in the center (visual builder is the default); results + run settings on the right.
- **Compact width (iPhone / narrow iPad):** **Circuit | Code | Results** tabs. Settings sit on Circuit; histogram on Results.

## Requirements

- Xcode 16+ (Swift 6.2 toolchain matching `Package.swift`)
- macOS 14+ or iOS 16+ (app target; QuantumKit itself still supports macOS 13)

## Package link

The app depends on the local package product `QuantumKit` via:

```
Apps/QuantumKitPlayground/QuantumKitPlayground.xcodeproj  →  ../../  (repo root)
```

No changes to `Package.swift` or library sources are required.

## Notes

- Simulation uses `SimulationPolicy.devicePreference` (Automatic / Metal / CPU). The app never constructs Metal types itself.
- Parse errors and run errors are shown in separate red banners (`LocalizedError.errorDescription`).
- This app is intentionally uncommitted until reviewed.
