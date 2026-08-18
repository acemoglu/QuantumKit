# ``QuantumKit``

A Swift and Metal quantum simulator for native Apple apps.

QuantumKit evolves circuits on-device: Metal GPU statevector and density-matrix engines on Apple silicon, with host CPU fallbacks, trajectory unravelling, and opt-in stabilizer and matrix-product-state backends. Version ``QuantumKitInfo/version`` is `1.0.0`. Platforms are **macOS 13+** and **iOS 16+**.

It is for Swift developers who want a local simulator in the same process as their app — not a Python stack, not NVIDIA / CUDA, and not a cloud backend. One SPM product: `QuantumKit`.

The SwiftUI app under `Apps/QuantumKitPlayground` composes and runs circuits against this library. Depend on the library, not the playground UI.

Source: [github.com/acemoglu/QuantumKit](https://github.com/acemoglu/QuantumKit.git).

Metal statevector width is ``StateVector/maxQubitCount`` (31). At **n = 30** the amplitude buffers are about **8 GB**. CPU statevector stops at ``CPUStateVector/maxQubitCount`` (16). Host prepared-sampling copies a `2ⁿ` CDF only for **n ≤** ``ShotExecutionPolicy/hostPreparedSamplingMaxQubitCount`` (20); wider Metal shots keep that map on the GPU and return a histogram. Noiseless terminal shots use evolve-once: wall time is width × gates, not shot count. Mid-circuit ``Gate/measure(_:)`` and ``Gate/c_if(classicalRegister:expectedValue:gate:)`` stay serial. The first Metal run compiles and warms pipelines.

## How to Use

Add the package, run a Bell pair, then pick a backend.

### Add the package

**Swift Package Manager.** In `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/acemoglu/QuantumKit.git", from: "1.0.0")
]
```

Then depend on the `QuantumKit` product:

```swift
.target(
    name: "MyApp",
    dependencies: [
        .product(name: "QuantumKit", package: "QuantumKit")
    ]
)
```

`from: "1.0.0"` needs a git tag with that name, matching ``QuantumKitInfo/version``. Until the tag exists, Xcode can still add the package from `main`.

**Xcode.** File → Add Package Dependencies…, paste `https://github.com/acemoglu/QuantumKit.git`, and add the `QuantumKit` library to your target. Import with `import QuantumKit`.

The package declares **macOS 13** and **iOS 16**. There is one library product; do not look for extra modules.

### Bell pair in Swift

Build the circuit with ``QuantumCircuit``, sample with ``QuantumBackendFactory/makeRecommended(circuit:noise:policy:renormalizationInterval:)``, and pass shots plus an optional seed on ``QuantumRunOptions``. Histogram keys on ``QuantumResult/bitstringCounts`` use ``QubitBitOrdering/bitstringMSB`` (leftmost character is the highest-index qubit).

```swift
import QuantumKit

var circuit = try QuantumCircuit(qubitCount: 2)
try circuit.h(0)
try circuit.cx(0, 1)
try circuit.measure(0)
try circuit.measure(1)

let backend = try QuantumBackendFactory.makeRecommended(circuit: circuit)
let result = try backend.run(
    circuit: circuit,
    options: QuantumRunOptions(seed: 1, shots: 1024)
)
print(result.bitstringCounts ?? [:])
// Typical support: "00" and "11"
```

Trailing ``QuantumCircuit/measure(_:)`` instructions are terminal measures. They stay eligible for evolve-once sampling. The factory returns `any QuantumBackend`, so pass ``QuantumRunOptions`` explicitly — the protocol has no default `options` argument.

### Same circuit from OpenQASM

``QuantumCircuit/init(openQASM:options:)`` detects OpenQASM 2 or 3 from the header and lowers to the same IR.

```swift
import QuantumKit

let source = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""

let circuit = try QuantumCircuit(openQASM: source)
let backend = try QuantumBackendFactory.makeRecommended(circuit: circuit)
let result = try backend.run(
    circuit: circuit,
    options: QuantumRunOptions(seed: 1, shots: 1024)
)
print(result.bitstringCounts ?? [:])
```

Export with ``QuantumCircuit/openQASM(options:)`` (default OpenQASM 3) or ``QuantumCircuit/openQASM2()``. Coverage and known edges are in **OpenQASM** below.

### Automatic vs CPU vs Metal

``SimulationPolicy/devicePreference`` chooses the device. Default is ``SimulationDevicePreference/automatic``: Metal when ``MetalRuntime/isAvailable``, otherwise CPU.

```swift
import QuantumKit

var circuit = try QuantumCircuit(qubitCount: 2)
try circuit.h(0)
try circuit.cx(0, 1)

let automatic = try QuantumBackendFactory.makeRecommended(circuit: circuit)

let cpu = try QuantumBackendFactory.makeRecommended(
    circuit: circuit,
    policy: SimulationPolicy(devicePreference: .cpu)
)

let metal = try QuantumBackendFactory.makeRecommended(
    circuit: circuit,
    policy: SimulationPolicy(devicePreference: .metal)
)
_ = (automatic, cpu, metal)
```

`.metal` fails if no GPU is present (``QuantumEngineError/deviceNotFound``). `.cpu` never uses Metal. ``SimulationPrecision/float64`` is CPU-only; requesting it with `.metal` throws ``SimulationPrecisionError/metalFloat64Unsupported``.

You can also pin a method without the recommender: ``QuantumBackendFactory/makeStatevector(renormalizationInterval:devicePreference:qubitCount:policy:)``, ``QuantumBackendFactory/makeDensityMatrix(renormalizationInterval:devicePreference:qubitCount:policy:)``, and so on. Details are in **Backends** below.

### Fast vs slow

Time is **width × gates**, not shot count, for noiseless terminal sampling. First Metal run after process launch compiles shaders and warms pipelines; the next run at the same width is the fair number.

| Width | Role | What to expect |
| --- | --- | --- |
| 2–8 qubits | Everyday circuits and the playground canvas | Sub-second on Automatic after warmup |
| ≤ 16 | CPU statevector cap (``CPUStateVector/maxQubitCount``) | CPU cannot go wider; Metal continues |
| ~20 | Wide Metal statevector | Seconds, not milliseconds |
| ~30 | Metal stress (n = 30 ≈ 8 GB) | Tens of seconds and gigabytes of GPU buffers |

Do not start at 30 qubits. Confirm a Bell histogram, then grow.

### Evolve-once vs mid-circuit measure

When ``SampleCountOptions/preferPreparedSampling`` is `true` (the default) and the circuit is a unitary prefix plus **trailing** measures, the backend evolves **once** and draws all shots from the Born distribution. Wall time does not scale with `shots`.

That path **does not** apply when the circuit has:

- Projective **mid-circuit** ``Gate/measure(_:)`` (a measure with gates after it)
- ``Gate/c_if(classicalRegister:expectedValue:gate:)`` or ``Gate/while_c(classicalRegister:expectedValue:body:maxIterations:)``
- ``Gate/reset(qubit:)`` or ``Gate/initialize(qubits:amplitudes:)``
- Evolution-time noise on the statevector path (depolarizing, damping, …) — those shots re-execute

Those cases stay **serial**. Mid-circuit measure and classical control cannot share one state or one RNG across threads (``ShotExecutionPolicy/mustSerial(circuit:noise:)``).

```swift
import QuantumKit

var options = QuantumRunOptions(seed: 1, shots: 1024)
options.sampleOptions.preferPreparedSampling = false // force per-shot re-execution
```

`Apps/QuantumKitPlayground` is a SwiftUI composer on the same APIs. Ship and import **`QuantumKit`**.

## Backends

``QuantumBackendFactory`` is the usual entry point. It recommends a ``QuantumSimulationMethod`` from width, optional ``NoiseModel``, and ``SimulationPolicy``, then builds a ``QuantumBackend``. Stabilizer and MPS are **never** chosen by the width-only recommender; you construct those yourself (or opt in to Clifford recommendation).

``QuantumBackendFactory/makeRecommended(circuit:noise:policy:renormalizationInterval:)`` inspects the circuit. ``QuantumBackendFactory/makeRecommended(qubitCount:noise:policy:renormalizationInterval:)`` uses width only.

Default ``SimulationPolicy``:

| Situation | Method | Backend |
| --- | --- | --- |
| Noiseless, width ≤ Metal SV cap (31) or CPU SV cap (16) | ``QuantumSimulationMethod/statevector`` | ``StatevectorBackend`` or ``CPUStatevectorBackend`` |
| Noise present, DM still fits, ``SimulationPolicy/preferDensityMatrixWhenNoisy`` (default `true`) | ``QuantumSimulationMethod/densityMatrix`` | ``DensityMatrixBackend`` (Metal max 14) or ``CPUDensityMatrixBackend`` (CPU max 8) |
| Noise present, DM too wide, noise is trajectory-compatible, SV still fits | ``QuantumSimulationMethod/trajectory`` | ``TrajectoryBackend`` wrapping an SV engine |
| Clifford+measure, noiseless, and ``SimulationPolicy/preferStabilizerWhenClifford`` is `true` | ``QuantumSimulationMethod/stabilizer`` | ``StabilizerBackend`` (circuit-aware recommender only) |

Device comes from ``SimulationPolicy/devicePreference``. CPU width caps are ``SimulationPolicy/cpuStatevectorQubitLimit`` (≤ 16) and ``SimulationPolicy/cpuDensityMatrixQubitLimit`` (≤ 8).

```swift
import QuantumKit

var circuit = try QuantumCircuit(qubitCount: 4)
try circuit.h(0)

let method = try QuantumBackendFactory.recommendMethod(circuit: circuit)
let estimate = try QuantumBackendFactory.estimateResources(
    qubitCount: circuit.qubitCount,
    gateCount: circuit.gates.count
)
let backend = try QuantumBackendFactory.makeRecommended(circuit: circuit)
_ = (method, estimate.recommendedDevice, backend.method)
```

``ResourceEstimate/estimatedRuntimeHintNanoseconds`` is an order-of-magnitude heuristic, not a guarantee.

If the footprint would exceed ``SimulationPolicy/maxPeakMemoryBytes``, factory construction throws ``SimulationPolicyError/estimatedMemoryExceedsBudget(estimated:budget:)``. Direct ``StatevectorBackend`` / ``CPUStatevectorBackend`` inits are **not** budgeted; the factory + policy path is.

**Statevector.** Exact pure-state evolution. Metal: ``StateVector/maxQubitCount`` **31** (n = 30 ≈ 8 GB). CPU: ``CPUStateVector/maxQubitCount`` **16**.

```swift
let metalSV = try QuantumBackendFactory.makeStatevector(devicePreference: .metal)
let cpuSV = try QuantumBackendFactory.makeStatevector(devicePreference: .cpu)
```

``StatevectorBackend`` is backed by ``QuantumEngine``. First Metal `init` compiles `.metalsrc` shaders.

**Density matrix.** Exact mixed-state evolution. Metal ``DensityMatrix/maxQubitCount`` is **14**; CPU ``CPUDensityMatrix/maxQubitCount`` is **8**. Localized gate noise and ``MeasurementMode/dephasingOnly`` **require** DM. Statevector and trajectory throw ``QuantumEngineError/localizedNoiseRequiresDensityMatrixBackend`` (or the CPU equivalent).

**Trajectory.** Monte-Carlo statevector unravelling. ``TrajectoryBackend/run(circuit:options:)`` **requires** ``QuantumRunOptions/shots``. The factory selects it only when DM does not fit, ``SimulationPolicy/preferTrajectoryWhenDensityMatrixTooWide`` is true, and ``NoiseModel/supportsTrajectorySimulation`` is true.

**Stabilizer.** Construct with ``QuantumBackendFactory/makeStabilizer(maxQubitCount:)``, or set ``SimulationPolicy/preferStabilizerWhenClifford`` and use the circuit-aware recommender. Width-only recommend never returns stabilizer. Noise is rejected. Non-Clifford gates throw.

**MPS.** ``QuantumBackendFactory/makeMPS(configuration:)`` only — never auto-selected. Bond dimension is ``MPSConfiguration/maxBondDimension``.

``QuantumEngine`` is safe to share across threads. Do not mutate one ``StateVector`` concurrently.

## Measurement

Request shots on ``QuantumRunOptions/shots`` and read ``QuantumResult/bitstringCounts``. Integer keys on ``ShotCounts/counts`` are ``QubitBitOrdering/engineLSB`` (bit *k* is qubit *k*, qubit 0 = LSB). Display bitstrings default to ``QubitBitOrdering/bitstringMSB`` (leftmost character is qubit *n−1*).

```swift
let msb = result.bitstringCounts // e.g. ["00": 512, "11": 512]
let lsbIndex = result.shotCounts?.counts // e.g. [0: 512, 3: 512] for |00⟩ and |11⟩
```

On two qubits, index `1` is qubit 0 excited → bitstring `"01"` under MSB. Convert with ``QubitBitOrdering/bitstring(forIndex:qubitCount:)`` / ``QubitBitOrdering/index(fromBitstring:qubitCount:)``.

``QuantumResult/hexCounts`` keys the packed engine-LSB index (`"0x3"` for bitstring `11` on 2 qubits). Without `shots`, ``QuantumBackend/run(circuit:options:)`` returns evolved state in ``QuantumResult/execution`` (SV / DM). Trajectory still requires shots.

``QuantumMeasurement/sampleCounts(state:engine:shots:)`` draws the full computational basis. ``QuantumMeasurement/partialProbabilities(state:engine:qubits:)`` marginalizes a subset.

A **trailing** block of measures (nothing but measure / barrier / delay after the first measure) is still terminal. A measure with **unitary work after it** is mid-circuit and forces serial re-execution.

Metal implements projective collapse on GPU, including a **1-qubit** mid-circuit measure path. ``QuantumEngine/executePartialMeasurementCollapse(on:qubits:rng:noise:)`` samples a marginal of up to 8 qubits on device; wider simultaneous measures fold on the host.

Host CDF sampling is capped at **n ≤ 20**. Above that, Metal keeps the probability map on the GPU and returns only the histogram. Shot count does not shrink the `2ⁿ` working set.

CPU independent shots and Metal sequential measurement RNGs are **not** bit-identical under the same seed for batchable circuits — see ``SampleCountOptions/batchSize``.

## OpenQASM

QuantumKit lowers OpenQASM 2 and a core OpenQASM 3 subset into ``QuantumCircuit``. The façade is ``OpenQASM``; ``QuantumCircuit/init(openQASM:options:)`` is the convenience entry.

Engine **LSB = qubit 0**. OpenQASM `q[0]` is engine qubit `0`. Multiple `qreg` / `qubit` declarations concatenate in declaration order. Classical registers stay separate ``ClassicalRegisterSpec`` entries. Display histograms still default to MSB.

```swift
import QuantumKit

let qasm2 = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""

let fromText = try QuantumCircuit(openQASM: qasm2)
let viaFacade = try OpenQASM.importCircuit(qasm2)
let version = try OpenQASM.detectVersion(from: qasm2)
_ = (fromText, viaFacade, version)
```

Missing `OPENQASM` header defaults to OpenQASM 2. `OPENQASM 3` / `3.0` selects ``OpenQASM3Importer``.

**OpenQASM 2.** Language core plus embedded `include "qelib1.inc"` (no filesystem search). ``OpenQASMQelib1/mappedGateNames`` lower directly to ``Gate``. Remaining qelib1 names expand from the embedded catalog. `if (creg == imm)` becomes ``Gate/c_if(classicalRegister:expectedValue:gate:)``. User `gate` declarations inline. Rejected: `opaque`, non-qelib1 `include`, OpenQASM 3 `qubit`/`bit` in a v2 program.

**OpenQASM 3.** `qubit` / `bit` (omit size → 1), plus compatibility `qreg` / `creg`. `include "stdgates.inc"` is the same embedded catalog. Whole-register broadcast and basic `ctrl@` / `inv@` / `pow(n)@` modifiers are supported. Bounded `while` only: `// @quantumkit.max_while_iterations N` immediately before the `while`, or ``OpenQASM3ImporterOptions/defaultWhileMaxIterations``. Unbounded `while` throws ``OpenQASMError/unsupported(line:column:feature:message:)``.

Default export is OpenQASM 3 (`qubit` / `bit`, no `include`).

```swift
let qasm3 = try circuit.openQASM()
let qasm2Again = try circuit.openQASM2()
let explicit = try OpenQASM.export(circuit, options: OpenQASMExportOptions(version: .v2))
_ = (qasm3, qasm2Again, explicit)
```

Only ``QFloatExpr/literal(_:)`` angles export. Symbolic parameters throw. ``Gate/while_c(classicalRegister:expectedValue:body:maxIterations:)`` exports as OpenQASM 3 `while` plus the max-iteration pragma. OpenQASM 2 export rejects `while_c`.

The parser rejects a catalog of keywords up front (``OpenQASMUnsupported/parserRejectedKeywords``): `defcal`, `cal`, `extern`, `delay`, `box`, `for`, `switch`, `def`, typed classical decls, `gphase`, `else`, `break` / `continue` / `return`, and more.

Export also refuses IR that has no QASM spelling: ``Gate/unitary1(matrix:target:)``, ``Gate/customUnitary(matrix:qubits:)``, ``Gate/initialize(qubits:amplitudes:)``, ``Gate/delay(duration:qubit:)``, ``Gate/mcx(controls:target:)`` / ``Gate/mcz(controls:target:)``, ECR / iSWAP / DCX / `ryy`. `rxx` / `rzz` export as qelib1 names.

This is an ingest/export front-end for QuantumKit’s gate IR, not a full OpenQASM 3 compiler or pulse toolchain.

## Noise

Attach noise on ``QuantumRunOptions/noise``. Localized gate noise needs a density-matrix backend. ``MeasurementMode/dephasingOnly`` is density-matrix only.

```swift
import QuantumKit

var circuit = try QuantumCircuit(qubitCount: 2)
try circuit.h(0)
try circuit.cx(0, 1)

let noise = NoiseModel(depolarizingProbability: 0.01)
let backend = try QuantumBackendFactory.makeRecommended(circuit: circuit, noise: noise)
let result = try backend.run(
    circuit: circuit,
    options: QuantumRunOptions(noise: noise, seed: 1, shots: 512)
)
print(result.metadata.method)
```

With default ``SimulationPolicy``, noisy circuits that still fit DM width become ``QuantumSimulationMethod/densityMatrix``. Wider, trajectory-compatible noise can become ``QuantumSimulationMethod/trajectory``.

Global channels: ``NoiseModel/depolarizingProbability`` (1-qubit random non-identity Pauli; 2-qubit `cx`/`cz`/`swap`: one of 15 two-qubit Paulis), amplitude damping or T1+gateTime, phase damping or T2, readout flips / confusion matrix, reset/prep flips, idle on ``Gate/delay(duration:qubit:)``, measurement dephasing. These are trajectory-compatible when ``MeasurementMode`` stays ``MeasurementMode/projective``.

| Path | What it is | Use when |
| --- | --- | --- |
| Density matrix | Exact mixed state, `4ⁿ` elements | Default noisy pick while width ≤ Metal 14 / CPU 8 |
| Trajectory | Monte-Carlo SV unravelling; shots required | DM too wide, noise still ``NoiseModel/supportsTrajectorySimulation`` |
| Statevector + noise | Per-shot unravelling on SV | ``SimulationPolicy/preferDensityMatrixWhenNoisy`` is `false` and SV fits |

```swift
import QuantumKit

let localized = NoiseModel().adding(
    .depolarizing(probability: 0.02),
    for: .gate(.cx)
)
```

Statevector engines throw ``QuantumEngineError/localizedNoiseRequiresDensityMatrixBackend``. The factory will not recommend trajectory for localized gate noise.

## Sampler and estimator

``Sampler`` and ``Estimator`` sit on top of ``QuantumBackend``. They do not replace ``QuantumBackend/run(circuit:options:)``.

Without shots, ``Sampler/run(circuit:backend:options:)`` returns exact Born-rule (or DM diagonal) probabilities in ``SamplerResult/quasiProbabilities``. With shots, empirical frequencies. **Stabilizer is not supported.**

```swift
import QuantumKit

var circuit = try QuantumCircuit(qubitCount: 2)
try circuit.h(0)
try circuit.cx(0, 1)

let backend = try QuantumBackendFactory.makeRecommended(circuit: circuit)
let sampled = try Sampler().run(
    circuit: circuit,
    backend: backend,
    options: QuantumRunOptions(seed: 1, shots: 1024)
)
print(sampled.quasiProbabilities)
```

``Estimator/run(circuit:hamiltonian:backend:options:estimatorOptions:)`` evaluates ⟨ψ|H|ψ⟩ or Tr(ρH). Default ``EstimatorOptions/exact``: one evolve, then exact Pauli expectations. Trajectory requires shots.

```swift
import QuantumKit

var circuit = try QuantumCircuit(qubitCount: 2)
try circuit.h(0)
try circuit.cx(0, 1)

let hamiltonian = Hamiltonian(try PauliTerm(coefficient: 1, label: "Z0 Z1"))
let backend = try QuantumBackendFactory.makeRecommended(circuit: circuit)
let estimated = try Estimator().run(
    circuit: circuit,
    hamiltonian: hamiltonian,
    backend: backend
)
print(estimated.value)
```

``Hamiltonian`` is also spelled ``SparsePauliOp``. ``Transpiler/transpile(_:targetBasis:)-(_,BasisGateSet)`` is a circuit transform; it does not pick a simulator.

## Topics

### Run a circuit

- ``QuantumBackendFactory``
- ``QuantumBackend``
- ``QuantumRunOptions``
- ``QuantumResult``
- ``SimulationPolicy``
- ``QuantumSimulationMethod``
- ``SimulationDevicePreference``
- ``ResourceEstimate``

### Simulation backends

- ``StatevectorBackend``
- ``DensityMatrixBackend``
- ``TrajectoryBackend``
- ``CPUStatevectorBackend``
- ``CPUDensityMatrixBackend``
- ``StabilizerBackend``
- ``MPSBackend``

### Circuits

- ``QuantumCircuit``
- ``Gate``
- ``GateSequence``
- ``MeasureSpec``
- ``ClassicalRegisterSpec``
- ``QubitBitOrdering``

### State and engines

- ``StateVector``
- ``DensityMatrix``
- ``QuantumEngine``
- ``DensityMatrixEngine``
- ``MetalRuntime``
- ``CPUStateVector``
- ``CPUDensityMatrix``

### Measurement and shots

- ``QuantumMeasurement``
- ``ShotCounts``
- ``SampleCountOptions``
- ``ShotExecutionPolicy``
- ``Sampler``
- ``Estimator``
- ``Hamiltonian``

### OpenQASM

- ``OpenQASM``
- ``OpenQASMImporter``
- ``OpenQASMExporter``
- ``OpenQASMUnsupportedFeature``

### Noise models

- ``NoiseModel``
- ``QuantumChannel``
- ``LocalizedNoiseRule``
- ``NoiseTarget``
- ``MeasurementMode``

### Compile

- ``Transpiler``
- ``PassManager``
- ``CompilerPass``
- ``BasisGateSet``

### Package

- ``QuantumKitInfo``
