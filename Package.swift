// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// MARK: - Packaging
//
// Choice: **A** — single library product / target `QuantumKit`.
//
// Why not B (multi-product split) in this session:
// - Sources are one Swift module today; Circuit / Transpiler / Engine / CPU / Backend /
//   Primitives / Algorithms / Noise freely share types with no internal import graph.
// - Metal `.metalsrc` shaders are loaded at runtime via `Bundle.module` from
//   `QuantumEngine` / `DensityMatrixEngine`. Splitting those types off the resource
//   target would break shader discovery unless resources move with them and every
//   loader stays on that same target.
// - A half-split (folders moved, products declared, umbrella incomplete) risks circular
//   target deps and broken Xcode/CLI resource bundling. Honesty over fake modularization.
//
// Follow-up split plan (when the graph is made explicit — do not half-do this):
// 1. Extract `QuantumKitCore` first: Gate, QuantumCircuit, IR, parameters, public errors,
//    QuantumKitInfo — no Metal, no Bundle.module.
// 2. Then `QuantumKitSimulator`: Engine/, CPU/, Backend/, Measure/, StateVector,
//    DensityMatrix*, MetalRuntime, BatchExecution, plus **all** `.metalsrc` resources
//    on that target only (loaders must stay here).
// 3. Then `QuantumKitTranspiler` (depends on Core) and `QuantumKitAlgorithms`
//    (depends on Core + Simulator and/or Primitives as needed).
// 4. Keep product `QuantumKit` as an umbrella target that `@_exported import`s the
//    pieces so existing `import QuantumKit` and `QuantumKitTests` stay one entry product.
// 5. Re-validate macOS 13 / iOS 16 platforms and Metal resource copy rules after any split.
//
// Product aliases / documentation-only modules remain optional later; they are not required
// for a single-target package while the single target remains the shipped surface.

let package = Package(
    name: "QuantumKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        // Single shipped library. Future multi-target layout (if any) must keep this
        // product name and continue to expose the full public API via one entry module
        // or an umbrella that re-exports the split modules.
        .library(
            name: "QuantumKit",
            targets: ["QuantumKit"]
        ),
    ],
    targets: [
        .target(
            name: "QuantumKit",
            resources: [
                // Shaders are bundled as raw text via the `.metalsrc` extension (not `.metal`) and
                // compiled at runtime. Xcode's build rule compiles any bundled `.metal` file into a
                // `default.metallib` and drops the raw source, regardless of `.process` vs `.copy`.
                // That breaks `loadPreciseLibrary`, which must recompile the shaders from source
                // with fast-math disabled to preserve the compensated (Kahan) summation in the
                // scan/collapse path. Using a non-`.metal` extension keeps the raw text in the
                // bundle on every toolchain, so CLI and Xcode behave identically.
                //
                // These resources MUST remain on whichever target owns QuantumEngine /
                // DensityMatrixEngine (today: this single QuantumKit target).
                .copy("Metal/Gates.metalsrc"),
                .copy("Metal/GateKernels.metalsrc"),
                .copy("Metal/Renormalization.metalsrc"),
                .copy("Metal/DensityMatrixKernels.metalsrc"),
                .copy("Metal/Trajectories.metalsrc")
            ]
        ),
        .testTarget(
            name: "QuantumKitTests",
            dependencies: ["QuantumKit"],
            resources: [
                // Frozen analytic oracles for OracleConformanceTests / ShotStatisticsTests (no Aer/Stim).
                .copy("Resources")
            ]
        ),
    ]
)
