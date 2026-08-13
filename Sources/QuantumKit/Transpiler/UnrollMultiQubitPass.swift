import Foundation

/// Expands gates that act on more than two qubits into 1q/2q sequences.
///
/// Routing only understands pairwise connectivity, so Toffoli / MCX / MCZ must be
/// unrolled before ``BasicSwapRoutingPass``. Measurement, reset, and already 1q/2q
/// gates are left unchanged.
///
/// When ``controlledSynthesis`` is ``ControlledGateSynthesisStrategy/vChainAncilla``,
/// set ``enableAncillaAllocation`` so scratch qubits are allocated and reused.
/// Expanding always strips instruction metadata (indices no longer align 1:1).
public struct UnrollMultiQubitPass: CompilerPass, Sendable {
    public let controlledSynthesis: ControlledGateSynthesisStrategy
    public let enableAncillaAllocation: Bool
    /// When `true`, each MCX allocates fresh ancillas without releasing (width upper bound).
    public let disableAncillaReuse: Bool

    public init(
        controlledSynthesis: ControlledGateSynthesisStrategy = .ancillaFree,
        enableAncillaAllocation: Bool = false,
        disableAncillaReuse: Bool = false
    ) {
        self.controlledSynthesis = controlledSynthesis
        self.enableAncillaAllocation = enableAncillaAllocation
        self.disableAncillaReuse = disableAncillaReuse
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        var allocator = AncillaAllocator(originalQubitCount: circuit.qubitCount)
        var pending: [Gate] = []

        for gate in circuit.gates {
            let pieces = try expandUntilTwoQubit(
                gate,
                allocator: &allocator
            )
            pending.append(contentsOf: pieces)
        }

        var output = try QuantumCircuit(
            qubitCount: max(circuit.qubitCount, allocator.qubitCount),
            classicalRegisters: circuit.classicalRegisters
        )
        for gate in pending {
            try output.apply(gate)
        }
        return output
    }

    private func expandUntilTwoQubit(
        _ gate: Gate,
        allocator: inout AncillaAllocator
    ) throws -> [Gate] {
        switch gate {
        case .measure, .reset, .initialize, .unitary1, .barrier, .delay, .id:
            return [gate]

        case .c_if(let classicalRegister, let expectedValue, let inner):
            let expandedInner = try expandUntilTwoQubit(inner, allocator: &allocator)
            if expandedInner.count == 1, expandedInner[0] == inner {
                return [gate]
            }
            return expandedInner.map { replacement in
                .c_if(
                    classicalRegister: classicalRegister,
                    expectedValue: expectedValue,
                    gate: replacement
                )
            }

        case .customUnitary(_, let qubits) where qubits.count > 2:
            throw TranspilerError.routingRequiresTwoQubitGates(gate)

        case .mcx(let controls, let target):
            if controlledSynthesis == .vChainAncilla {
                try requireAncillaAllocationEnabled()
                if controls.count >= 3 {
                    return try expandVChainMCX(
                        controls: controls,
                        target: target,
                        allocator: &allocator
                    )
                }
            }
            return try expandAncillaFree(gate, allocator: &allocator)

        case .mcz(let controls, let target):
            if controlledSynthesis == .vChainAncilla {
                try requireAncillaAllocationEnabled()
                if controls.count >= 3 {
                    // MCZ ≡ H · MCX · H; use the same V-chain path as MCX (no silent downgrade).
                    var pieces: [Gate] = [.h(target: target)]
                    pieces.append(contentsOf: try GateDecomposition.expandMCXWithAllocator(
                        controls: controls,
                        target: target,
                        strategy: .vChainAncilla,
                        allocator: &allocator,
                        reuseAncillas: !disableAncillaReuse
                    ))
                    pieces.append(.h(target: target))
                    var result: [Gate] = []
                    for piece in pieces {
                        result.append(contentsOf: try expandUntilTwoQubit(piece, allocator: &allocator))
                    }
                    return result
                }
            }
            return try expandAncillaFree(gate, allocator: &allocator)

        case .ccx, .cswap:
            return try expandAncillaFree(gate, allocator: &allocator)

        default:
            if gate.affectedQubits.count <= 2 {
                return [gate]
            }
            throw TranspilerError.routingRequiresTwoQubitGates(gate)
        }
    }

    private func requireAncillaAllocationEnabled() throws {
        guard enableAncillaAllocation else {
            throw AncillaAllocationError.ancillaAllocationDisabled(strategy: .vChainAncilla)
        }
    }

    private func expandVChainMCX(
        controls: [Int],
        target: Int,
        allocator: inout AncillaAllocator
    ) throws -> [Gate] {
        let pieces = try GateDecomposition.expandMCXWithAllocator(
            controls: controls,
            target: target,
            strategy: .vChainAncilla,
            allocator: &allocator,
            reuseAncillas: !disableAncillaReuse
        )
        var result: [Gate] = []
        for piece in pieces {
            result.append(contentsOf: try expandUntilTwoQubit(piece, allocator: &allocator))
        }
        return result
    }

    private func expandAncillaFree(
        _ gate: Gate,
        allocator: inout AncillaAllocator
    ) throws -> [Gate] {
        var context = GateDecomposition.Context(
            controlledSynthesis: .ancillaFree,
            ancillaAllocator: nil
        )
        var result: [Gate] = []
        for piece in try GateDecomposition.expand(gate, context: &context) {
            result.append(contentsOf: try expandUntilTwoQubit(piece, allocator: &allocator))
        }
        return result
    }
}
