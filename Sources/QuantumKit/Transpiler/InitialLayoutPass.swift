import Foundation

/// Remaps logical qubit indices onto physical wires according to a fixed ``Layout``.
///
/// Does not insert SWAPs. Use ``BasicSwapRoutingPass`` when the coupling map is sparse.
public struct InitialLayoutPass: CompilerPass, Sendable {
    public let layout: Layout
    public let physicalQubitCount: Int

    public init(layout: Layout, physicalQubitCount: Int? = nil) {
        self.layout = layout
        self.physicalQubitCount = physicalQubitCount ?? max(layout.physicalQubitCount, layout.logicalQubitCount)
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        guard layout.logicalQubitCount == circuit.qubitCount else {
            throw TranspilerError.invalidLayout(
                reason: "layout logical width \(layout.logicalQubitCount) does not match circuit width \(circuit.qubitCount)"
            )
        }
        guard physicalQubitCount >= layout.physicalQubitCount else {
            throw TranspilerError.circuitWiderThanDevice(
                circuitQubits: layout.physicalQubitCount,
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

        for gate in circuit.gates {
            let mapped = try gate.remappingQubits { logical in
                try layout.physical(forLogical: logical)
            }
            try remapped.apply(mapped)
        }

        return remapped
    }
}
