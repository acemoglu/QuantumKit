import Foundation

/// Versioning and public-API stability policy for QuantumKit (H4 / I9).
///
/// This type is a DocC / SPM documentation anchor only. It has no runtime behavior.
///
/// ## Supported public API
///
/// Clients should depend on these surfaces; they are covered by Semantic Versioning:
///
/// - **Backends:** ``QuantumBackend``, ``QuantumBackendFactory``, ``QuantumRunOptions``,
///   ``QuantumResult``, simulation method / policy types, and the concrete backends
///   those factories return (statevector, density-matrix, trajectory, CPU / MPS /
///   stabilizer where exposed).
/// - **Circuit IR:** ``QuantumCircuit``, ``Gate``, ``GateSequence``, related circuit
///   composition / binding APIs, and versioned IR helpers (``CircuitIRSchema``).
/// - **Primitives:** ``Sampler``, ``Estimator``, and their options / result types.
/// - **Noise:** ``NoiseModel`` and the public noise / channel catalog used with it.
/// - **Transpiler entrypoints:** ``Transpiler``, ``PassManager``, ``TranspileOptions``,
///   ``CompilerPass`` / ``DAGCompilerPass``, the shipped pass types, and opt-in
///   ``CompilerPassRegistry`` / ``CompilerPassFactory`` (B14 lite named discovery —
///   not a marketplace or dylib loader).
///
/// Supporting types that appear in these APIs (errors, metadata, shot counts, observables,
/// layouts, coupling maps, and so on) are part of the same stability contract when they
/// are `public`.
///
/// Package identity / reproducibility metadata: ``QuantumKitInfo/version``.
///
/// ## Packaging (H11 lite)
///
/// SPM ships **one** library product and target: `QuantumKit`. Clients keep
/// `import QuantumKit`. A multi-product split (Core / Simulator+Metal resources /
/// Transpiler / Algorithms + umbrella re-exports) is deferred until an explicit
/// inter-target import graph exists; see the H11 comments in `Package.swift`.
///
/// ## Metal encapsulation (H6c / H7b)
///
/// Prefer device-free APIs. The only public ``MTLDevice`` entry point is
/// ``MetalRuntime/sharedDevice()`` (advanced interop / diagnostics via ``deviceName``).
/// Explicit state / batch `device:` inits and ``DensityMatrix/device`` were **removed in H6c**.
/// Public ``MTLBuffer`` accessors and the `outputBuffer:` probability-kernel shim were
/// **removed in H7b**. Inventory lives on ``MetalRuntime``.
///
/// ## `0.2.0` breaking surface (same pre-1.0 MINOR bump)
///
/// In addition to H6c / H7b removals above, ``0.2.0`` also made these former public symbols
/// **package-internal** (breaking for clients that referenced them):
///
/// - `Pipelines` (Metal compute-pipeline cache on ``QuantumEngine``; was a public struct)
/// - `TRNGCollapse` (hardware-entropy helpers; prefer ``QuantumRNG/hardware``)
///
/// Those changes together are reflected in ``QuantumKitInfo/version`` **0.2.0**. Any further
/// removal of supported public API **must** bump MAJOR (post-1.0) or MINOR (while still `0.y.z`)
/// before the next tagged release — do not ship a tag that still claims `0.1.0` after H6c/H7b
/// or the Pipelines / TRNGCollapse visibility change.
///
/// ## Semantic Versioning
///
/// QuantumKit follows [SemVer](https://semver.org):
///
/// - **MAJOR** — breaking changes to supported public API (including the H6c device-API and
///   H7b buffer / kernel shim removals, and the `0.2.0` demotion of `Pipelines` /
///   `TRNGCollapse` to package-internal).
/// - **MINOR** — additive APIs and **deprecations** of existing APIs (no removal). While
///   still pre-1.0 (`0.y.z`), MINOR also records breaking removals (e.g. `0.1.0` → `0.2.0`).
/// - **PATCH** — bug fixes and documentation that do not change the public contract.
///
/// Pre-1.0 releases still use the same MAJOR/MINOR/PATCH rules once a version is tagged;
/// ``QuantumKitInfo/version`` is the single in-tree source of truth for the library version
/// string stamped into run metadata.
public enum QuantumKitAPIPolicy: Sendable {}
