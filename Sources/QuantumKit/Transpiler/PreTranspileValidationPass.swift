import Foundation

/// Early structural checks before routing / basis translation (A9).
///
/// Validates optional device width against a ``CouplingMap``, an optional gate-count
/// depth budget, and rejects circuits that still contain unbound parameters.
/// Does **not** require two-qubit adjacency — that is routing's responsibility.
public struct PreTranspileValidationPass: CompilerPass, Sendable {
    public let couplingMap: CouplingMap?
    public let maxDepth: Int?

    public init(couplingMap: CouplingMap? = nil, maxDepth: Int? = nil) {
        self.couplingMap = couplingMap
        self.maxDepth = maxDepth
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        if let couplingMap {
            guard circuit.qubitCount <= couplingMap.qubitCount else {
                throw TranspilerError.circuitWiderThanDevice(
                    circuitQubits: circuit.qubitCount,
                    deviceQubits: couplingMap.qubitCount
                )
            }
        }

        var unbound = Set<String>()
        for gate in circuit.gates where gate.containsUnboundParameters {
            unbound.formUnion(gate.referencedParameters)
        }
        if !unbound.isEmpty {
            throw TranspilerError.unboundParameters(unbound.sorted())
        }

        if let maxDepth {
            let depth = circuit.gates.count
            guard depth <= maxDepth else {
                throw TranspilerError.circuitExceedsMaxDepth(depth: depth, maxDepth: maxDepth)
            }
        }

        return circuit
    }
}
