import Foundation

/// Inserts ``Gate/swap`` gates so every two-qubit interaction lies on a ``CouplingMap`` edge.
///
/// Algorithm (basic shortest-path swap):
/// 1. Start from `initialLayout` (default: identity), or a seeded random layout when `seed` is set.
/// 2. For each gate, map logical qubits through the current layout.
/// 3. If a two-qubit gate is not adjacent, SWAP along the shortest coupling path until it is,
///    updating the layout after each SWAP. With a seed, the walker may move either endpoint.
/// 4. Emit the gate on the resulting physical wires.
///
/// Multi-qubit (>2) gates are rejected; run ``UnrollMultiQubitPass`` first.
public struct BasicSwapRoutingPass: CompilerPass, Sendable {
    public let couplingMap: CouplingMap
    public let initialLayout: Layout?
    public let seed: UInt64?
    /// When `true`, copies source instruction metadata onto remapped original gates.
    /// Inserted SWAPs always get `nil` metadata.
    public let preserveInstructionMetadata: Bool

    public init(
        couplingMap: CouplingMap,
        initialLayout: Layout? = nil,
        seed: UInt64? = nil,
        preserveInstructionMetadata: Bool = false
    ) {
        self.couplingMap = couplingMap
        self.initialLayout = initialLayout
        self.seed = seed
        self.preserveInstructionMetadata = preserveInstructionMetadata
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        guard circuit.qubitCount <= couplingMap.qubitCount else {
            throw TranspilerError.circuitWiderThanDevice(
                circuitQubits: circuit.qubitCount,
                deviceQubits: couplingMap.qubitCount
            )
        }

        var rng: QuantumRNG? = seed.map { .seeded($0) }

        let layout: Layout
        if let initialLayout {
            // Ancilla growth (V-chain unroll) may widen the circuit past a user/pre-ancilla
            // layout — extend onto unused physicals instead of failing with a stale width.
            if initialLayout.logicalQubitCount > circuit.qubitCount {
                throw TranspilerError.invalidLayout(
                    reason: "layout logical width \(initialLayout.logicalQubitCount) exceeds circuit width \(circuit.qubitCount)"
                )
            }
            layout = try initialLayout.extended(
                toLogicalCount: circuit.qubitCount,
                physicalCount: couplingMap.qubitCount
            )
        } else if seed != nil, var localRNG = rng {
            layout = try Self.seededLayout(
                logicalCount: circuit.qubitCount,
                physicalCount: couplingMap.qubitCount,
                rng: &localRNG
            )
            rng = localRNG
        } else {
            layout = try Layout.identity(qubitCount: circuit.qubitCount)
        }

        var mutable = MutableLayout(layout)
        var routed = try QuantumCircuit(
            qubitCount: couplingMap.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )

        for (index, gate) in circuit.gates.enumerated() {
            let meta = preserveInstructionMetadata ? circuit.metadata(at: index) : nil
            try appendRouted(gate, layout: &mutable, into: &routed, rng: &rng, sourceMetadata: meta)
        }

        return routed
    }

    /// Deterministic random injective layout `logical → physical`.
    public static func seededLayout(
        logicalCount: Int,
        physicalCount: Int,
        rng: inout QuantumRNG
    ) throws -> Layout {
        guard logicalCount > 0, physicalCount >= logicalCount else {
            throw TranspilerError.invalidLayout(
                reason: "need physicalCount >= logicalCount > 0 for seeded layout"
            )
        }
        var pool = Array(0..<physicalCount)
        for i in 0..<logicalCount {
            let j = i + rng.nextInt(upperBound: physicalCount - i)
            pool.swapAt(i, j)
        }
        return try Layout(logicalToPhysical: Array(pool.prefix(logicalCount)))
    }

    private func appendRouted(
        _ gate: Gate,
        layout: inout MutableLayout,
        into circuit: inout QuantumCircuit,
        rng: inout QuantumRNG?,
        sourceMetadata: InstructionMetadata? = nil
    ) throws {
        if case .c_if(let classicalRegister, let expectedValue, let inner) = gate {
            var body = try QuantumCircuit(qubitCount: couplingMap.qubitCount)
            try appendRouted(inner, layout: &layout, into: &body, rng: &rng, sourceMetadata: nil)
            for piece in body.gates {
                if case .swap = piece {
                    try circuit.apply(piece)
                } else {
                    try circuit.apply(
                        .c_if(
                            classicalRegister: classicalRegister,
                            expectedValue: expectedValue,
                            gate: piece
                        ),
                        metadata: sourceMetadata
                    )
                }
            }
            return
        }

        switch gate {
        case .measure, .initialize, .barrier, .delay:
            let mapped = try gate.remappingQubits { logical in
                layout.physical(forLogical: logical)
            }
            try circuit.apply(mapped, metadata: sourceMetadata)
            return
        default:
            break
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
            try circuit.apply(mapped, metadata: sourceMetadata)
            return
        }

        guard orderedUnique.count == 2 else {
            throw TranspilerError.routingRequiresTwoQubitGates(gate)
        }

        try bringAdjacent(
            orderedUnique[0],
            orderedUnique[1],
            layout: &layout,
            into: &circuit,
            rng: &rng
        )

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

        try circuit.apply(mapped, metadata: sourceMetadata)
    }

    private func bringAdjacent(
        _ logicalA: Int,
        _ logicalB: Int,
        layout: inout MutableLayout,
        into circuit: inout QuantumCircuit,
        rng: inout QuantumRNG?
    ) throws {
        let physicalA = layout.physical(forLogical: logicalA)
        let physicalB = layout.physical(forLogical: logicalB)

        if couplingMap.areAdjacent(physicalA, physicalB) {
            return
        }

        // Seeded choice: move A toward B, or B toward A (different SWAP sequences when both work).
        let fromLogical: Int
        let toLogical: Int
        if var localRNG = rng {
            if localRNG.nextInt(upperBound: 2) == 0 {
                fromLogical = logicalA
                toLogical = logicalB
            } else {
                fromLogical = logicalB
                toLogical = logicalA
            }
            rng = localRNG
        } else {
            fromLogical = logicalA
            toLogical = logicalB
        }

        let fromPhysical = layout.physical(forLogical: fromLogical)
        let toPhysical = layout.physical(forLogical: toLogical)

        guard let path = couplingMap.shortestPath(from: fromPhysical, to: toPhysical) else {
            throw TranspilerError.qubitsNotConnected(fromPhysical, toPhysical)
        }

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
