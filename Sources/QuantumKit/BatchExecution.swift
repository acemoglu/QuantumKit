import Foundation
import Metal

/// Options for repeated circuit execution (sampling).
public struct SampleCountOptions: Sendable, Equatable {

    /// How many independent runs to group per GPU submission. Ignored when gate noise or
    /// mid-circuit measure/reset requires sequential execution.
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
    public var isUnitaryOnly: Bool {
        gates.allSatisfy { gate in
            switch gate {
            case .measure, .reset, .c_if, .barrier, .delay:
                return false
            case .initialize:
                return false
            default:
                return true
            }
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

        let gateNoise = noise?.hasGateNoise == true
        let canBatch = circuit.isUnitaryOnly && !gateNoise && !circuit.containsHostAppliedUnitaryGates
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
                        noise: gateNoise ? noise : nil,
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
