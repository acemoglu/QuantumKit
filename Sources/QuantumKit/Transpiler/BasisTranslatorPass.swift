import Foundation

/// Replaces every gate outside the configured target basis with an equivalent native sequence.
public struct BasisTranslatorPass: CompilerPass, Sendable {
    public let targetBasis: BasisGateSet

    public init(targetBasis: BasisGateSet) {
        self.targetBasis = targetBasis
    }

    public init(targetBasis: BasisGateKind...) {
        self.targetBasis = BasisGateSet(kinds: Set(targetBasis))
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        var translated = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )

        for gate in circuit.gates {
            if targetBasis.contains(gate) {
                try translated.apply(gate)
                continue
            }

            switch gate {
            case .measure, .reset, .c_if, .barrier, .delay:
                try translated.apply(gate)
            default:
                let replacements = try GateDecomposition.expandRecursively(gate, into: targetBasis)
                for replacement in replacements {
                    try translated.apply(replacement)
                }
            }
        }

        return translated
    }
}
