import Foundation

/// Evaluates a sequence of bound circuits against a Hamiltonian, reusing GPU state buffers
/// and batching unitary evolution when the workload permits.
enum BatchExpectationExecutor {

    static func evaluate(
        circuits: [QuantumCircuit],
        hamiltonian: Hamiltonian,
        engine: QuantumEngine,
        options: QuantumRunOptions,
        batchSize: Int
    ) throws -> [QFloat] {
        guard !circuits.isEmpty else { return [] }

        let qubitCount = circuits[0].qubitCount
        guard circuits.allSatisfy({ $0.qubitCount == qubitCount }) else {
            throw QuantumEngineError.qubitCountMismatch(circuit: qubitCount, state: circuits[0].qubitCount)
        }

        let gateNoise = options.noise?.hasGateNoise == true
        let canBatch = !gateNoise && circuits.allSatisfy(\.isUnitaryOnly)
            && !circuits.contains(where: \.containsHostAppliedUnitaryGates)
        let effectiveBatchSize = canBatch ? min(batchSize, circuits.count) : 1

        var rng = makePrimitiveRNG(seed: options.seed)
        var expectations: [QFloat] = []
        expectations.reserveCapacity(circuits.count)

        if canBatch {
            let pool = try StateVectorBatch(qubitCount: qubitCount, capacity: effectiveBatchSize)
            var completed = 0

            while completed < circuits.count {
                let activeCount = min(effectiveBatchSize, circuits.count - completed)
                let slice = circuits[completed..<(completed + activeCount)]
                let activeStates = Array(pool.states.prefix(activeCount))

                for (index, circuit) in slice.enumerated() {
                    activeStates[index].resetToZero()
                    try engine.executeUnitaryBatch(circuit, on: [activeStates[index]])
                    expectations.append(
                        try hamiltonian.expectation(state: activeStates[index], engine: engine)
                    )
                }

                completed += activeCount
            }
        } else {
            let state = try StateVector(qubitCount: qubitCount)
            for circuit in circuits {
                state.resetToZero()
                _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: options.noise)
                expectations.append(try hamiltonian.expectation(state: state, engine: engine))
            }
        }

        return expectations
    }

    static func evaluate(
        circuits: [QuantumCircuit],
        hamiltonian: Hamiltonian,
        engine: DensityMatrixEngine,
        options: QuantumRunOptions
    ) throws -> [QFloat] {
        guard !circuits.isEmpty else { return [] }

        let density = try DensityMatrix(qubitCount: circuits[0].qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        var expectations: [QFloat] = []
        expectations.reserveCapacity(circuits.count)

        for circuit in circuits {
            _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: options.noise)
            expectations.append(try hamiltonian.expectation(density: density, engine: engine))
        }

        return expectations
    }
}
