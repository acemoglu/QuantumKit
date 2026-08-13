import Foundation

/// Remaps logical qubit indices onto physical wires according to a fixed ``Layout``.
///
/// Does not insert SWAPs. Use ``BasicSwapRoutingPass`` when the coupling map is sparse.
public struct InitialLayoutPass: CompilerPass, Sendable {
    public let layout: Layout
    public let physicalQubitCount: Int
    /// When `true`, copies source instruction metadata onto remapped gates (1:1).
    public let preserveInstructionMetadata: Bool

    public init(
        layout: Layout,
        physicalQubitCount: Int? = nil,
        preserveInstructionMetadata: Bool = false
    ) {
        self.layout = layout
        self.physicalQubitCount = physicalQubitCount ?? max(layout.physicalQubitCount, layout.logicalQubitCount)
        self.preserveInstructionMetadata = preserveInstructionMetadata
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        // Allow extending a narrower layout when the circuit grew (e.g. ancilla unroll).
        let resolvedLayout: Layout
        if layout.logicalQubitCount == circuit.qubitCount {
            resolvedLayout = layout
        } else if layout.logicalQubitCount < circuit.qubitCount {
            resolvedLayout = try layout.extended(
                toLogicalCount: circuit.qubitCount,
                physicalCount: physicalQubitCount
            )
        } else {
            throw TranspilerError.invalidLayout(
                reason: "layout logical width \(layout.logicalQubitCount) exceeds circuit width \(circuit.qubitCount)"
            )
        }
        guard physicalQubitCount >= resolvedLayout.physicalQubitCount else {
            throw TranspilerError.circuitWiderThanDevice(
                circuitQubits: resolvedLayout.physicalQubitCount,
                deviceQubits: physicalQubitCount
            )
        }
        guard physicalQubitCount >= circuit.qubitCount else {
            throw TranspilerError.circuitWiderThanDevice(
                circuitQubits: circuit.qubitCount,
                deviceQubits: physicalQubitCount
            )
        }

        var remapped = try QuantumCircuit(
            qubitCount: physicalQubitCount,
            classicalRegisters: circuit.classicalRegisters
        )

        for (index, gate) in circuit.gates.enumerated() {
            let mapped = try gate.remappingQubits { logical in
                try resolvedLayout.physical(forLogical: logical)
            }
            let meta = preserveInstructionMetadata ? circuit.metadata(at: index) : nil
            try remapped.apply(mapped, metadata: meta)
        }

        return remapped
    }
}
