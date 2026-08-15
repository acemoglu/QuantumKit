// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "QuantumKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
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
