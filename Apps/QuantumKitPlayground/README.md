<p align="center">
  <img src="QuantumKitPlayground/Resources/Assets.xcassets/AppIcon.appiconset/AppIcon-1024.png" width="120" alt="QuantumKit">
</p>

# QuantumKit (app)

Circuit editor for Mac and iPhone. Build on the canvas or in OpenQASM, then run the simulation on this device. The Swift package is in the [repo README](../../README.md).

<p align="center">
  <a href="https://apps.apple.com/app/quantumkit"><img src="https://developer.apple.com/assets/elements/badges/download-on-the-app-store.svg" alt="Download on the App Store" width="160"></a>
</p>

## How to use

Start with a sample. **Bell**, **Toffoli**, and **GHZ-4** finish in well under a second. Run, check the histogram, then grow.

### Build a circuit

On the **Circuit** canvas, pick a gate from the palette and drop it on a qubit wire (or tap the gate, then the wire). CNOT, CZ, and SWAP need a second qubit.

**New Circuit** gives a blank 2-qubit canvas. **Save Circuit** keeps it under My Circuits.

The visual circuit, the ASCII preview, and the OpenQASM stay in sync.

### Code

**Code** is the OpenQASM editor. Open or save a `.qasm` file, or drop one onto the editor (Mac and iPad). Files larger than 1 MB are rejected. **Parse** refreshes the circuit immediately.

### Run

**Run** (⌘R on Mac) simulates locally. Set device (leave **Automatic** unless you have a reason), shots, seed, and optionally how often to renormalize (default every 50 gates; 0 off).

Results: method, device, wall time, qubit/gate counts, and an MSB histogram. Errors show after Run, not while you type.

The first GPU run after launch warms shaders. The next run at the same width is the one that counts. Raising shots on a noiseless circuit barely changes time; qubit count and gate count do.

## Samples

| Sample | What it is |
| --- | --- |
| Bell State | H + CNOT, then measure |
| Toffoli | Three-qubit CCX, then measure |
| Teleport | Bell pair, measure, classically controlled X and Z |
| Grover (2 qubits) | Two-qubit amplitude amplification |
| Rotations | RX, RY, RZ with explicit angles |
| GHZ (4 qubits) | Four-qubit GHZ, then measure |

## Tips

The visual canvas caps at 8 qubits. Stay on the small samples for everyday use. 20+ qubits is seconds; ~30 qubits can be tens of seconds and uses gigabytes — that is expected, not a hang. On iPhone, start with Bell or GHZ-4.

How to Use is also in the app (⌘? on Mac, **?** in the toolbar).
