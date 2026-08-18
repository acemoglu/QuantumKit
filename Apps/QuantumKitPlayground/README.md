# QuantumKit (app)

SwiftUI apps for editing OpenQASM, composing circuits visually, and running simulations against the local **QuantumKit** package. Both apps install as **QuantumKit.app** (Dock / home screen / menu bar).

Two targets share the same sources:

| Scheme | Platform | Bundle ID |
|---|---|---|
| **QuantumKitPlayground** | macOS 14+ | `com.quantumkit.playground` |
| **QuantumKitPlayground iOS** | iPhone / iPad (iOS 16+) | `com.quantumkit.playground.ios` |

## Open in Xcode

1. Open `Apps/QuantumKitPlayground/QuantumKitPlayground.xcodeproj`.
2. Wait for Xcode to resolve the local Swift package at `../../` (repo root / `Package.swift`).
3. Pick a scheme:
   - **QuantumKitPlayground** → destination **My Mac**
   - **QuantumKitPlayground iOS** → an **iPhone** (or iPad) simulator / device
4. Press **Run** (⌘R in Xcode). In the Mac app, **⌘R** also runs the circuit.

## What it does

- Load bundled `.qasm` samples from `QuantumKitPlayground/Resources/Samples/` (Bell, Toffoli, teleport, Grover 2q, parametric, GHZ-4).
- Edit OpenQASM in the source pane, **or build visually**: drag gates from the palette onto qubit wires (or click a gate, then a qubit). CNOT/CZ/SWAP ask for a second qubit. The ASCII preview and OpenQASM stay in sync.
- The ASCII circuit preview updates automatically after a short debounce; **Parse** still forces an immediate refresh.
- **Open…** and **Save…** load and write `.qasm` files. Drag a `.qasm` or `.txt` file onto the editor (macOS and iPad) to replace the source; files larger than 1 MB are rejected.
- **Run** simulates via `QuantumBackendFactory.makeRecommended` on a background thread.
- Choose **Automatic / Metal / CPU**, shots, and a numeric seed (or random).
- Results show method, device name, wall time, qubit/gate counts, and a histogram of MSB bitstrings.

## Layout

- **Mac:** sample sidebar; **Circuit | Code** in the center; results + run settings on the right. First launch opens **How to Use** (also **⌘?** / toolbar **?**). Keep everyday runs on the bundled samples (Bell, GHZ-4); 20+ qubits is seconds by design, ~30 qubits can be tens of seconds.
- **iOS (iPhone and iPad):** bottom tab bar — **Circuit**, **Code**, **Results**. Samples and Run sit in the navigation bar; Open / Save / Parse are in the overflow menu. On iPhone, tap a palette gate then a qubit wire. Drag-and-drop still works on iPad. Run jumps to Results.

## Requirements

- Xcode 16+ (Swift 6.2 toolchain matching `Package.swift`)
- macOS 14+ or iOS 16+ (app targets; QuantumKit itself still supports macOS 13)

## Package link

```
Apps/QuantumKitPlayground/QuantumKitPlayground.xcodeproj  →  ../../  (repo root)
```

No changes to `Package.swift` or library sources are required to build the apps.

## Notes

- Simulation uses `SimulationPolicy.devicePreference` (Automatic / Metal / CPU). The apps never construct Metal types themselves.
- Parse errors and run errors are shown in separate red banners (`LocalizedError.errorDescription`).
- On iPhone, large statevector jobs are limited by device memory; start with the bundled samples.
