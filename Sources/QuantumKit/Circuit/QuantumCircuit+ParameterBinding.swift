import Foundation

extension QuantumCircuit {

    /// `true` when any gate still contains symbolic parameters.
    public var containsUnboundParameters: Bool {
        gates.contains { $0.containsUnboundParameters }
    }

    /// All symbolic parameter names referenced by this circuit.
    public var referencedParameters: Set<String> {
        gates.reduce(into: Set<String>()) { result, gate in
            result.formUnion(gate.referencedParameters)
        }
    }

    /// Replaces symbolic parameters with concrete ``QFloat`` values and returns a fully bound circuit.
    public func bind(parameters: [String: QFloat]) throws -> QuantumCircuit {
        var bound = try QuantumCircuit(qubitCount: qubitCount, classicalRegisters: classicalRegisters)
        for gate in gates {
            try bound.apply(try gate.bound(using: parameters))
        }
        return bound
    }

    /// Throws when symbolic parameters remain and the circuit cannot be executed.
    public func requireFullyBound() throws {
        let unbound = referencedParameters
        guard unbound.isEmpty else {
            throw ParameterBindingError.circuitContainsUnboundParameters(unbound)
        }
    }
}

/// Replaces symbolic gate angles with concrete values. Intended for use in a transpiler pipeline
/// immediately before execution backends that require fully resolved angles.
public struct ParameterBindingPass: CompilerPass, Sendable {
    public let parameters: [String: QFloat]

    public init(parameters: [String: QFloat]) {
        self.parameters = parameters
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        try circuit.bind(parameters: parameters)
    }
}
