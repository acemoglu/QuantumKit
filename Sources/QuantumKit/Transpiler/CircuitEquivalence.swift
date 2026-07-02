import Foundation

enum CircuitEquivalence {

    /// Verifies two unitary-only circuits induce identical Born-rule statistics on every
    /// computational basis input (equivalent up to input-dependent diagonal phases).
    static func haveIdenticalAction(
        _ lhs: QuantumCircuit,
        _ rhs: QuantumCircuit,
        engine: QuantumEngine,
        tolerance: QFloat = 1e-4
    ) throws -> Bool {
        guard lhs.qubitCount == rhs.qubitCount else { return false }
        guard lhs.isUnitaryOnly, rhs.isUnitaryOnly else {
            throw QuantumCircuitError.circuitNotUnitary
        }

        let dimension = 1 << lhs.qubitCount
        for basisIndex in 0..<dimension {
            let leftProbabilities = try outputProbabilities(
                circuit: lhs,
                basisIndex: basisIndex,
                engine: engine
            )
            let rightProbabilities = try outputProbabilities(
                circuit: rhs,
                basisIndex: basisIndex,
                engine: engine
            )

            for index in 0..<dimension {
                if abs(leftProbabilities[index] - rightProbabilities[index]) > tolerance {
                    return false
                }
            }
        }
        return true
    }

    private static func outputProbabilities(
        circuit: QuantumCircuit,
        basisIndex: Int,
        engine: QuantumEngine
    ) throws -> [QFloat] {
        let state = try StateVector(qubitCount: circuit.qubitCount)
        try prepareBasisState(basisIndex, on: state, engine: engine)
        try engine.execute(circuit, on: state)
        return try QuantumMeasurement.probabilities(state: state, engine: engine)
    }

    private static func prepareBasisState(
        _ basisIndex: Int,
        on state: StateVector,
        engine: QuantumEngine
    ) throws {
        for qubit in 0..<state.qubitCount where ((basisIndex >> qubit) & 1) == 1 {
            try engine.executeUnitaryGate(.x(target: qubit), on: state)
        }
    }
}
