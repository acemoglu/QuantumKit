import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    // MARK: - CompilerPassRegistry (B14 lite)

    func testCompilerPassRegistryStartsEmptyWithNoDefaultFactories() {
        let registry = CompilerPassRegistry()
        XCTAssertTrue(registry.isEmpty)
        XCTAssertEqual(registry.registeredIDs, [])
        XCTAssertFalse(registry.contains(id: CompilerPassRegistry.identityPassID))
    }

    func testCompilerPassRegistryIdentityPassRunsThroughPassManager() throws {
        let registry = CompilerPassRegistry()
        try registry.register(ClosureCompilerPassFactory.identity)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.x(0)

        let manager = try registry.makePassManager(ids: [CompilerPassRegistry.identityPassID])
        let out = try manager.run(on: circuit)

        XCTAssertEqual(out.gates, circuit.gates)
        XCTAssertEqual(out.qubitCount, circuit.qubitCount)
        XCTAssertEqual(registry.registeredIDs, [CompilerPassRegistry.identityPassID])
        XCTAssertFalse(registry.isEmpty)
    }

    func testCompilerPassRegistryResolvesOrderedPassesWithoutCrossRegistryLeakage() throws {
        struct AppendZPass: CompilerPass {
            func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
                var output = try QuantumCircuit(qubitCount: circuit.qubitCount)
                for gate in circuit.gates { try output.apply(gate) }
                try output.z(0)
                return output
            }
        }

        let registry = CompilerPassRegistry()
        let other = CompilerPassRegistry()
        try registry.register(ClosureCompilerPassFactory.identity)
        try registry.register(
            ClosureCompilerPassFactory(id: "test.append-z") { AppendZPass() }
        )

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)

        let out = try registry
            .makePassManager(ids: [
                CompilerPassRegistry.identityPassID,
                "test.append-z",
            ])
            .run(on: circuit)

        XCTAssertEqual(out.gates, [.h(target: 0), .z(target: 0)])
        XCTAssertTrue(other.isEmpty)
        XCTAssertFalse(other.contains(id: CompilerPassRegistry.identityPassID))
    }

    func testCompilerPassRegistryDuplicateAndUnknownIDs() throws {
        let registry = CompilerPassRegistry()
        try registry.register(ClosureCompilerPassFactory.identity)

        XCTAssertThrowsError(try registry.register(ClosureCompilerPassFactory.identity)) { error in
            guard case CompilerPassRegistryError.duplicateID(let id) = error else {
                return XCTFail("expected duplicateID, got \(error)")
            }
            XCTAssertEqual(id, CompilerPassRegistry.identityPassID)
        }

        XCTAssertThrowsError(try registry.makePass(id: "missing")) { error in
            guard case CompilerPassRegistryError.unknownID(let id) = error else {
                return XCTFail("expected unknownID, got \(error)")
            }
            XCTAssertEqual(id, "missing")
        }

        XCTAssertThrowsError(
            try registry.register(ClosureCompilerPassFactory(id: "") { IdentityCompilerPass() })
        ) { error in
            XCTAssertEqual(error as? CompilerPassRegistryError, .emptyID)
        }
    }

    func testExistingPassManagerCallSiteStillWorksWithoutRegistry() throws {
        let original = try QuantumCircuit(qubitCount: 1)
        let result = try PassManager(passes: [IdentityCompilerPass()]).run(on: original)
        XCTAssertEqual(result.gates, original.gates)
        XCTAssertEqual(result.qubitCount, 1)
    }
}
