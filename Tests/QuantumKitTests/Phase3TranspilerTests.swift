import XCTest
@testable import QuantumKit

extension QuantumKitTests {

    func testPassManagerAppliesPassesInOrder() throws {
        struct AppendHPass: CompilerPass {
            let target: Int
            func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
                var output = try QuantumCircuit(qubitCount: circuit.qubitCount)
                for gate in circuit.gates { try output.apply(gate) }
                try output.h(target)
                return output
            }
        }

        let original = try QuantumCircuit(qubitCount: 1)
        let result = try PassManager(passes: [AppendHPass(target: 0)]).run(on: original)
        XCTAssertEqual(result.gates.count, 1)
        XCTAssertEqual(result.gates[0], .h(target: 0))
    }

    func testTranspileHadamardAndCCXToNativeBasis() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.ccx(0, 1, 2)

        let targetBasis = BasisGateSet.ibmEagle
        let transpiled = try Transpiler.transpile(circuit, targetBasis: targetBasis)

        XCTAssertEqual(transpiled.qubitCount, circuit.qubitCount)
        XCTAssertFalse(transpiled.gates.isEmpty)

        for gate in transpiled.gates {
            guard let kind = BasisGateKind(gate: gate) else {
                XCTFail("Unexpected gate outside target basis: \(gate)")
                return
            }
            XCTAssertTrue(targetBasis.contains(kind))
        }

        XCTAssertTrue(
            try assertTranspiledEquivalent(circuit, transpiled),
            "Transpiled circuit must preserve measurement statistics on all basis inputs"
        )
    }

    func testTranspileBellCircuitPreservesUnitary() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let transpiled = try Transpiler.transpile(circuit, targetBasis: .ibmEagle)

        for gate in transpiled.gates {
            XCTAssertNotNil(BasisGateKind(gate: gate))
        }

        XCTAssertTrue(try assertTranspiledEquivalent(circuit, transpiled))
    }

    func testHadamardDecompositionMatchesDirectUnitary() throws {
        guard makeDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }

        var hadamardCircuit = try QuantumCircuit(qubitCount: 1)
        try hadamardCircuit.h(0)

        var decomposedCircuit = try QuantumCircuit(qubitCount: 1)
        let halfPi = QFloat(Double.pi / 2)
        try decomposedCircuit.rz(theta: halfPi, 0)
        try decomposedCircuit.sx(0)
        try decomposedCircuit.rz(theta: halfPi, 0)

        XCTAssertTrue(try assertTranspiledEquivalent(hadamardCircuit, decomposedCircuit))
    }

    func testTranspilerRejectsUnsupportedMultiControlGate() throws {
        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.mcx(controls: [0, 1, 2], target: 3)

        XCTAssertThrowsError(
            try Transpiler.transpile(circuit, targetBasis: .ibmEagle)
        ) { error in
            XCTAssertEqual(error as? TranspilerError, .unsupportedGate(.mcx(controls: [0, 1, 2], target: 3)))
        }
    }

    private func assertTranspiledEquivalent(
        _ original: QuantumCircuit,
        _ transpiled: QuantumCircuit
    ) throws -> Bool {
        if try CircuitUnitary.areEquivalent(original, transpiled, tolerance: 1e-4) {
            return true
        }

        let engine = try QuantumEngine()
        return try CircuitEquivalence.haveIdenticalAction(
            original,
            transpiled,
            engine: engine,
            tolerance: 1e-4
        )
    }
}
