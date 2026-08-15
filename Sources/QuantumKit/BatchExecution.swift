import Foundation
import Metal

/// Options for repeated circuit execution (sampling).
public struct SampleCountOptions: Sendable, Equatable {

    /// Requested grouping for independent shots. See ``ShotExecutionPolicy``.
    ///
    /// - Metal: ``StateVectorBatch`` capacity when
    ///   ``ShotExecutionPolicy/canUseMetalUnitaryBatch(circuit:noise:)``; otherwise `1`
    ///   (evolution noise, host-applied unitaries, reset/initialize, and
    ///   ``ShotExecutionPolicy/mustSerial(circuit:noise:)`` stay serial — never share one
    ///   RNG or one ``StateVector`` across threads). Unitary Metal batches keep **one
    ///   sequential** measurement ``QuantumRNG`` (batch size does not change that schedule).
    /// - CPU: worker-pool size when ``ShotExecutionPolicy/canBatch(circuit:noise:)``.
    ///   Independent CPU shots **always** use ``QuantumRNG/independentShotStream(seed:shotIndex:)``
    ///   (or hardware entropy when no seed is available) — including when `batchSize == 1`.
    ///   Stream root: ``QuantumRunOptions/seed`` if set, otherwise the initial state of a
    ///   backend-local ``QuantumRNG/seeded`` value, otherwise hardware per shot. That RNG is
    ///   **not advanced** on the independent path. `batchSize` only controls concurrency and
    ///   must **not** change seeded histograms *within that per-shot-stream contract*.
    ///   ``ShotExecutionPolicy/mustSerial`` forces pool size `1` and one sequential ``QuantumRNG``.
    ///
    /// **Seed contract (breaking vs pre-shot-parallel CPU):** for ``canBatch`` circuits, a
    /// fixed ``QuantumRunOptions/seed`` no longer matches (1) legacy single-stream sequential
    /// CPU consumption or (2) Metal’s sequential measurement RNG. Prefer matching goldens to
    /// ``independentShotStream``, or pin ``mustSerial`` circuits if you need one shared stream.
    public var batchSize: Int

    public init(batchSize: Int = 32) {
        self.batchSize = max(batchSize, 1)
    }
}

/// Pre-allocated state vectors reused across batched shots.
///
/// Prefer ``init(qubitCount:capacity:)`` (resolves ``MetalRuntime``). Engine pairing uses
/// package-`internal` ``init(qubitCount:on:capacity:)``.
public final class StateVectorBatch: @unchecked Sendable {

    public let qubitCount: Int
    public let capacity: Int
    public let states: [StateVector]

    /// Creates a batch via ``MetalRuntime/sharedDevice()`` (no caller Metal imports).
    public convenience init(qubitCount: Int, capacity: Int) throws {
        try self.init(
            qubitCount: qubitCount,
            on: MetalRuntime.sharedDevice(),
            capacity: capacity
        )
    }

    /// Package-internal designated initializer; always honors `device` for buffer correctness.
    init(qubitCount: Int, on device: MTLDevice, capacity: Int) throws {
        guard capacity > 0 else {
            throw BatchExecutionError.invalidBatchSize(capacity)
        }

        self.qubitCount = qubitCount
        self.capacity = capacity
        self.states = try (0..<capacity).map { _ in
            try StateVector(qubitCount: qubitCount, on: device)
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
    /// Same coupling rule as ``ShotExecutionPolicy/mustSerial(circuit:noise:)``: projective
    /// mid-circuit measures (and `c_if`) make each run stochastic;
    /// ``MeasurementMode/dephasingOnly`` remains deterministic on ρ.
    public func allowsPreparedDensityShotBatching(noise: NoiseModel? = nil) -> Bool {
        ShotExecutionPolicy.canBatch(circuit: self, noise: noise)
    }

    /// `true` when the circuit contains at least one ``Gate/delay`` (including nested
    /// ``Gate/c_if`` bodies).
    var containsDelay: Bool {
        gates.contains { Self.gateContainsDelay($0) }
    }

    /// `true` when the circuit contains at least one ``Gate/measure`` (including nested
    /// ``Gate/c_if`` bodies). Used for evolution-noise charging of measurement channels.
    var containsMeasure: Bool {
        gates.contains { Self.gateContainsMeasure($0) }
    }

    private static func gateContainsDelay(_ gate: Gate) -> Bool {
        switch gate {
        case .delay:
            return true
        case .c_if(_, _, let inner):
            return gateContainsDelay(inner)
        default:
            return false
        }
    }

    private static func gateContainsMeasure(_ gate: Gate) -> Bool {
        switch gate {
        case .measure:
            return true
        case .c_if(_, _, let inner):
            return gateContainsMeasure(inner)
        default:
            return false
        }
    }

    var containsHostAppliedUnitaryGates: Bool {
        gates.contains { Self.gateContainsHostAppliedUnitary($0) }
    }

    private static func gateContainsHostAppliedUnitary(_ gate: Gate) -> Bool {
        switch gate {
        case .unitary1:
            return true
        case .customUnitary(_, let qubits) where qubits.count > 1:
            return true
        case .c_if(_, _, let inner):
            return gateContainsHostAppliedUnitary(inner)
        default:
            return false
        }
    }
}

enum BatchSampleExecutor {

    /// Noise that must be applied during circuit evolution (not only at terminal readout).
    static func requiresEvolutionNoise(_ noise: NoiseModel?, circuit: QuantumCircuit) -> Bool {
        ShotExecutionPolicy.requiresEvolutionNoise(noise, circuit: circuit)
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

        let evolutionNoise = ShotExecutionPolicy.requiresEvolutionNoise(noise, circuit: circuit)
        let metalBatch = ShotExecutionPolicy.canUseMetalUnitaryBatch(circuit: circuit, noise: noise)
        let batchSize = ShotExecutionPolicy.metalUnitaryBatchSize(
            circuit: circuit,
            noise: noise,
            requested: options.batchSize,
            shots: shots
        )

        let pool = try StateVectorBatch(qubitCount: circuit.qubitCount, on: device, capacity: batchSize)
        var completedShots = 0

        while completedShots < shots {
            try cancellationCheck?()
            let activeCount = min(batchSize, shots - completedShots)
            let activeStates = Array(pool.states.prefix(activeCount))

            if metalBatch {
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
                // Coupled shots, evolution noise, or host unitaries: one state, one sequential
                // RNG. Do not parallelize by sharing this stream or buffer.
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
