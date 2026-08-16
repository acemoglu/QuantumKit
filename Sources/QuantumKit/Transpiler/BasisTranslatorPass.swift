import Foundation

/// Replaces every gate outside the configured target basis with an equivalent native sequence.
public struct BasisTranslatorPass: CompilerPass, Sendable {
    public let targetBasis: BasisGateSet
    /// When `true`, keeps metadata on gates that pass through without expansion.
    /// Expanded replacements always receive `nil` metadata.
    public let preserveInstructionMetadata: Bool

    public init(targetBasis: BasisGateSet, preserveInstructionMetadata: Bool = false) {
        self.targetBasis = targetBasis
        self.preserveInstructionMetadata = preserveInstructionMetadata
    }

    public init(targetBasis: BasisGateKind..., preserveInstructionMetadata: Bool = false) {
        self.targetBasis = BasisGateSet(kinds: Set(targetBasis))
        self.preserveInstructionMetadata = preserveInstructionMetadata
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        var translated = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )

        for (index, gate) in circuit.gates.enumerated() {
            let meta = preserveInstructionMetadata ? circuit.metadata(at: index) : nil
            if targetBasis.contains(gate) {
                try translated.apply(gate, metadata: meta)
                continue
            }

            switch gate {
            case .measure, .reset, .c_if, .while_c, .barrier, .delay:
                try translated.apply(gate, metadata: meta)
            default:
                // Expansion breaks 1:1 index alignment — strip metadata for replacements.
                let replacements = try GateDecomposition.expandRecursively(gate, into: targetBasis)
                for replacement in replacements {
                    try translated.apply(replacement)
                }
            }
        }

        return translated
    }
}
