import Foundation

/// Inserts ``Gate/swap`` gates so every two-qubit interaction lies on a ``CouplingMap`` edge.
///
/// Algorithm (basic shortest-path swap):
/// 1. Start from `initialLayout` (default: identity).
/// 2. For each gate, map logical qubits through the current layout.
/// 3. If a two-qubit gate is not adjacent, SWAP along the shortest coupling path until it is,
///    updating the layout after each SWAP.
/// 4. Emit the gate on the resulting physical wires.
///
/// Multi-qubit (>2) gates are rejected; run ``UnrollMultiQubitPass`` first.
public struct BasicSwapRoutingPass: CompilerPass, Sendable {
    public let couplingMap: CouplingMap
    public let initialLayout: Layout?

    public init(couplingMap: CouplingMap, initialLayout: Layout? = nil) {
        self.couplingMap = couplingMap
        self.initialLayout = initialLayout
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        guard circuit.qubitCount <= couplingMap.qubitCount else {
            throw TranspilerError.circuitWiderThanDevice(
                circuitQubits: circuit.qubitCount,
                deviceQubits: couplingMap.qubitCount
            )
        }

        let layout: Layout
        if let initialLayout {
            guard initialLayout.logicalQubitCount == circuit.qubitCount else {
                throw TranspilerError.invalidLayout(
                    reason: "layout logical width \(initialLayout.logicalQubitCount) does not match circuit width \(circuit.qubitCount)"
                )
            }
            guard initialLayout.physicalQubitCount <= couplingMap.qubitCount else {
                throw TranspilerError.circuitWiderThanDevice(
                    circuitQubits: initialLayout.physicalQubitCount,
                    deviceQubits: couplingMap.qubitCount
                )
            }
            layout = initialLayout
        } else {
            layout = try Layout.identity(qubitCount: circuit.qubitCount)
        }

        var mutable = MutableLayout(layout)
        var routed = try QuantumCircuit(
            qubitCount: couplingMap.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )

        for gate in circuit.gates {
            try appendRouted(gate, layout: &mutable, into: &routed)
        }

        return routed
    }

    private func appendRouted(
        _ gate: Gate,
        layout: inout MutableLayout,
        into circuit: inout QuantumCircuit
    ) throws {
        if case .c_if(let classicalRegister, let expectedValue, let inner) = gate {
            // SWAPs inserted while placing the body must stay unconditional so the
            // evolving layout remains consistent regardless of the classical bit.
            var body = try QuantumCircuit(qubitCount: couplingMap.qubitCount)
            try appendRouted(inner, layout: &layout, into: &body)
            for piece in body.gates {
                if case .swap = piece {
                    try circuit.apply(piece)
                } else {
                    try circuit.apply(
                        .c_if(
                            classicalRegister: classicalRegister,
                            expectedValue: expectedValue,
                            gate: piece
                        )
                    )
                }
            }
            return
        }

        // Measure / initialize / barrier / delay act per qubit; they do not need pairwise adjacency.
        if case .measure = gate {
            let mapped = try gate.remappingQubits { logical in
                layout.physical(forLogical: logical)
            }
            try circuit.apply(mapped)
            return
        }
        if case .initialize = gate {
            let mapped = try gate.remappingQubits { logical in
                layout.physical(forLogical: logical)
            }
            try circuit.apply(mapped)
            return
        }
        if case .barrier = gate {
            let mapped = try gate.remappingQubits { logical in
                layout.physical(forLogical: logical)
            }
            try circuit.apply(mapped)
            return
        }
        if case .delay = gate {
            let mapped = try gate.remappingQubits { logical in
                layout.physical(forLogical: logical)
            }
            try circuit.apply(mapped)
            return
        }

        var orderedUnique: [Int] = []
        var seen = Set<Int>()
        for qubit in gate.affectedQubits where seen.insert(qubit).inserted {
            orderedUnique.append(qubit)
        }

        if orderedUnique.count <= 1 {
            let mapped = try gate.remappingQubits { logical in
                layout.physical(forLogical: logical)
            }
            try circuit.apply(mapped)
            return
        }

        guard orderedUnique.count == 2 else {
            throw TranspilerError.routingRequiresTwoQubitGates(gate)
        }

        try bringAdjacent(orderedUnique[0], orderedUnique[1], layout: &layout, into: &circuit)

        let mapped = try gate.remappingQubits { logical in
            layout.physical(forLogical: logical)
        }

        let physicals = mapped.affectedQubits
        if physicals.count >= 2 {
            let p0 = physicals[0]
            let p1 = physicals[1]
            guard couplingMap.areAdjacent(p0, p1) else {
                throw TranspilerError.qubitsNotConnected(p0, p1)
            }
        }

        try circuit.apply(mapped)
    }

    private func bringAdjacent(
        _ logicalA: Int,
        _ logicalB: Int,
        layout: inout MutableLayout,
        into circuit: inout QuantumCircuit
    ) throws {
        let physicalA = layout.physical(forLogical: logicalA)
        let physicalB = layout.physical(forLogical: logicalB)

        if couplingMap.areAdjacent(physicalA, physicalB) {
            return
        }

        guard let path = couplingMap.shortestPath(from: physicalA, to: physicalB) else {
            throw TranspilerError.qubitsNotConnected(physicalA, physicalB)
        }

        // Walk logical A toward B along the path until the pair is adjacent.
        for index in 0..<(path.count - 2) {
            let left = path[index]
            let right = path[index + 1]
            try circuit.apply(.swap(q1: left, q2: right))
            layout.applyPhysicalSwap(left, right)
        }

        let finalA = layout.physical(forLogical: logicalA)
        let finalB = layout.physical(forLogical: logicalB)
        guard couplingMap.areAdjacent(finalA, finalB) else {
            throw TranspilerError.qubitsNotConnected(finalA, finalB)
        }
    }
}
