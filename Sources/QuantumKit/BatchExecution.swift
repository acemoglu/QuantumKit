import Foundation
import Metal

/// Options for repeated circuit execution (sampling).
public struct SampleCountOptions: Sendable, Equatable {

    /// How many independent runs to group per GPU submission. Ignored when evolution noise,
    /// mid-circuit measure/reset, or host-applied unitaries require sequential execution.
    public var batchSize: Int

    public init(batchSize: Int = 32) {
        self.batchSize = max(batchSize, 1)
    }
}

/// Pre-allocated state vectors reused across batched shots.
public final class StateVectorBatch: @unchecked Sendable {

    public let qubitCount: Int
    public let capacity: Int
    public let states: [StateVector]

    public convenience init(qubitCount: Int, capacity: Int) throws {
        try self.init(
            qubitCount: qubitCount,
            device: MetalRuntime.sharedDevice(),
            capacity: capacity
        )
    }

    public init(qubitCount: Int, device: MTLDevice, capacity: Int) throws {
        guard capacity > 0 else {
            throw BatchExecutionError.invalidBatchSize(capacity)
        }

        self.qubitCount = qubitCount
        self.capacity = capacity
        self.states = try (0..<capacity).map { _ in
            try StateVector(qubitCount: qubitCount, device: device)
        }
    }
}

public enum BatchExecutionError: Error {
    case invalidBatchSize(Int)
}

extension QuantumCircuit {

    /// True when the circuit contains only unitary gates (no mid-circuit measure or reset).
    ///
    /// ``Gate/barrier`` and ``Gate/delay`` are identity on the unitary (noise-off) and do
    /// **not** disqualify a circuit; engines may still apply idle noise on delay.
    public var isUnitaryOnly: Bool {
        gates.allSatisfy { gate in
            switch gate {
            case .measure, .reset, .c_if:
                return false
            case .initialize:
                return false
            default:
                return true
            }
        }
    }

    /// `true` when density-matrix evolution is deterministic for `noise`, so terminal shots can
    /// be drawn from a single prepared ρ without re-executing the circuit.
    ///
    /// Projective mid-circuit measures (and classical control that depends on them) make each
    /// run stochastic; ``MeasurementMode/dephasingOnly`` remains deterministic on ρ.
    public func allowsPreparedDensityShotBatching(noise: NoiseModel? = nil) -> Bool {
        let projective = (noise?.measurementMode ?? .projective) == .projective
        for gate in gates {
            switch gate {
            case .measure where projective:
                return false
            case .c_if:
                return false
            default:
                continue
            }
        }
        return true
    }

    /// `true` when the circuit contains at least one ``Gate/delay``.
    var containsDelay: Bool {
        gates.contains { gate in
            if case .delay = gate { return true }
            return false
        }
    }

    var containsHostAppliedUnitaryGates: Bool {
        gates.contains { gate in
            switch gate {
            case .unitary1:
                return true
            case .customUnitary(_, let qubits) where qubits.count > 1:
                return true
            default:
                return false
            }
        }
    }
}

enum BatchSampleExecutor {

    /// Noise that must be applied during circuit evolution (not only at terminal readout).
    static func requiresEvolutionNoise(_ noise: NoiseModel?, circuit: QuantumCircuit) -> Bool {
        guard let noise else { return false }
        if noise.hasGateNoise || noise.hasPreparationNoise || noise.hasMeasurementChannelNoise {
            return true
        }
        return noise.hasIdleNoise && circuit.containsDelay
    }

    static func runSampleCountsRNG(
        circuit: QuantumCircuit,
        engine: QuantumEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        options: SampleCountOptions,
        cancellationCheck: (() throws -> Void)? = nil
    ) throws -> ShotCounts {
        let device = engine.device
        guard shots > 0 else {
            throw QuantumMeasurementError.invalidShotCount(shots)
        }

        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, circuit.qubitCount > 0 ? 1 << circuit.qubitCount : 1))

        let evolutionNoise = requiresEvolutionNoise(noise, circuit: circuit)
        let canBatch = circuit.isUnitaryOnly && !evolutionNoise && !circuit.containsHostAppliedUnitaryGates
        let batchSize = canBatch ? min(options.batchSize, shots) : 1

        let pool = try StateVectorBatch(qubitCount: circuit.qubitCount, device: device, capacity: batchSize)
        var completedShots = 0

        while completedShots < shots {
            try cancellationCheck?()
            let activeCount = min(batchSize, shots - completedShots)
            let activeStates = Array(pool.states.prefix(activeCount))

            if canBatch {
                for state in activeStates {
                    state.resetToZero()
                }
                // No per-gate host samples: one GPU submit for the batch. Phase `sample`
                // wraps this call; ``SimulationProfile/gateTimings`` stays nil.
                try engine.executeUnitaryBatch(circuit, on: activeStates)

                for state in activeStates {
                    let outcome = try engine.executeMeasurementCollapse(on: state, rng: &rng, noise: noise)
                    histogram[outcome, default: 0] += 1
                }
            } else {
                for state in activeStates {
                    try cancellationCheck?()
                    state.resetToZero()
                    try engine.executeRNG(
                        circuit,
                        on: state,
                        rng: &rng,
                        noise: evolutionNoise ? noise : nil,
                        cancellationCheck: cancellationCheck
                    )
                    let outcome = try engine.executeMeasurementCollapse(on: state, rng: &rng, noise: noise)
                    histogram[outcome, default: 0] += 1
                }
            }

            completedShots += activeCount
        }

        return ShotCounts(shots: shots, counts: histogram)
    }
}
