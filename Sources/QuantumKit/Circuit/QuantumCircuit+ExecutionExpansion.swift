import Foundation

extension QuantumCircuit {

    /// Returns a copy where every gate is recursively expanded via
    /// ``QuantumEngine/expandForExecution(_:)`` (drops `.id`, unrolls `.iswap` / `.ecr` /
    /// `.rxx` / …). Useful for IR-level equivalence checks against the same rewrite the
    /// CPU / Metal engines apply before kernel dispatch.
    public func expandedForExecution() throws -> QuantumCircuit {
        var output = try QuantumCircuit(
            qubitCount: qubitCount,
            classicalRegisters: classicalRegisters
        )
        for gate in gates {
            for piece in try QuantumEngine.expandForExecution(gate) {
                try output.apply(piece)
            }
        }
        return output
    }
}
