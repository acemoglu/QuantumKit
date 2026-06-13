// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "QuantumKit",
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
                .process("Metal/Gates.metal")
            ]
        ),
        .testTarget(
            name: "QuantumKitTests",
            dependencies: ["QuantumKit"]
        ),
    ]
)
