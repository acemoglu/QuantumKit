import Foundation

/// Shot independence for sampling. **One** predicate, used by Metal ``StateVectorBatch`` and
/// CPU concurrent workers (distinct states, never a shared buffer or shared RNG).
///
/// **`canBatch` (legal to parallel / GPU-batch):** shots are independent — no projective
/// mid-circuit measure, no classical control (``Gate/c_if`` / ``Gate/while_c``), and any
/// noise is per-shot unraveling (or terminal readout). Each shot must have its own state
/// and, when RNG is consumed during evolution, its own stream.
///
/// **`mustSerial`:** projective mid-circuit measure, classical control (`c_if` / `while_c`),
/// or any path that would require sharing one ``StateVector`` / ``CPUStateVector`` or one
/// ``QuantumRNG`` across threads. Forced serial even if a future scheduler could
/// theoretically isolate shots.
///
/// Metal ``QuantumEngine/executeUnitaryBatch`` is a stricter subset: independent **and**
/// unitary-only with no host-applied unitaries and no evolution-time RNG. Global depolarizing
/// (and other evolution noise) therefore stays **serial on Metal** (batch size 1) and may
/// run **concurrently on CPU** with ``QuantumRNG/independentShotStream(seed:shotIndex:)``.
///
/// **RNG asymmetry:** Metal unitary batches (and Metal serial shot loops) consume one
/// sequential ``QuantumRNG`` for measurement / unraveling. CPU ``canBatch`` paths always
/// derive a fresh stream per `shotIndex` from ``QuantumRunOptions/seed``. Do not expect
/// CPU and Metal histograms to match under the same seed for independent circuits.
public enum ShotExecutionPolicy: Sendable {

    /// Independent shots may be GPU-batched (Metal unitary) or run on concurrent CPU workers.
    public static func canBatch(circuit: QuantumCircuit, noise: NoiseModel? = nil) -> Bool {
        !mustSerial(circuit: circuit, noise: noise)
    }

    /// Projective mid-circuit measure or classical control (`c_if` / `while_c`) (or anything
    /// that would share one RNG/state across threads). Callers **must** keep `batchSize = 1`
    /// and a single sequential stream.
    public static func mustSerial(circuit: QuantumCircuit, noise: NoiseModel? = nil) -> Bool {
        let projective = (noise?.measurementMode ?? .projective) == .projective
        for gate in circuit.gates {
            switch gate {
            case .measure where projective:
                return true
            case .c_if, .while_c:
                return true
            default:
                continue
            }
        }
        return false
    }

    /// Metal fused ``StateVectorBatch`` / ``QuantumEngine/executeUnitaryBatch`` eligibility.
    ///
    /// Requires ``canBatch(circuit:noise:)``, a unitary-only circuit, no host-applied
    /// unitaries, and no evolution-time RNG. Global depolarizing and other unraveling
    /// noise therefore stay serial on Metal.
    public static func canUseMetalUnitaryBatch(
        circuit: QuantumCircuit,
        noise: NoiseModel? = nil
    ) -> Bool {
        canBatch(circuit: circuit, noise: noise)
            && circuit.isUnitaryOnly
            && !circuit.containsHostAppliedUnitaryGates
            && !requiresEvolutionNoise(noise, circuit: circuit)
    }

    /// Noise that must be applied during circuit evolution (not only at terminal readout).
    ///
    /// Measurement-channel noise (``NoiseModel/measurementDephasingProbability``, non-projective
    /// mode) applies only at mid-circuit ``Gate/measure`` — not at terminal collapse — so it
    /// is charged only when `circuit` contains a measure (including nested ``Gate/c_if`` /
    /// ``Gate/while_c`` bodies).
    /// Without a circuit (factory heuristic), it is **not** charged (same pattern as idle-on-delay).
    public static func requiresEvolutionNoise(_ noise: NoiseModel?, circuit: QuantumCircuit?) -> Bool {
        guard let noise else { return false }
        if noise.hasGateNoise || noise.hasPreparationNoise {
            return true
        }
        if noise.hasMeasurementChannelNoise {
            if let circuit {
                return circuit.containsMeasure
            }
            return false
        }
        if noise.hasIdleNoise {
            if let circuit {
                return circuit.containsDelay
            }
            // Factory estimates have no circuit: idle-on-delay is not charged as evolution
            // noise (same as the historical ``SimulationPolicy`` trajectory heuristic).
            return false
        }
        return false
    }

    /// ``SampleCountOptions/batchSize`` realized as Metal ``StateVectorBatch`` capacity (`1` when
    /// ``canUseMetalUnitaryBatch(circuit:noise:)`` is false).
    public static func metalUnitaryBatchSize(
        circuit: QuantumCircuit,
        noise: NoiseModel?,
        requested: Int,
        shots: Int
    ) -> Int {
        guard shots > 0, canUseMetalUnitaryBatch(circuit: circuit, noise: noise) else { return 1 }
        return min(max(requested, 1), shots)
    }

    /// Requested CPU worker-pool size for independent shots (`1` when
    /// ``mustSerial(circuit:noise:)``). Does **not** cap on core count — see
    /// ``cpuLiveStateCopies(circuit:noise:requested:shots:)`` for the allocated high-water.
    public static func cpuWorkerPoolSize(
        circuit: QuantumCircuit,
        noise: NoiseModel?,
        requested: Int,
        shots: Int
    ) -> Int {
        guard shots > 0, canBatch(circuit: circuit, noise: noise) else { return 1 }
        return min(max(requested, 1), shots)
    }

    /// Actual CPU live state copies: ``cpuWorkerPoolSize`` capped by
    /// `ProcessInfo.processInfo.activeProcessorCount` (matches ``CPUShotSampler`` allocation).
    public static func cpuLiveStateCopies(
        circuit: QuantumCircuit,
        noise: NoiseModel?,
        requested: Int,
        shots: Int
    ) -> Int {
        let pool = cpuWorkerPoolSize(
            circuit: circuit,
            noise: noise,
            requested: requested,
            shots: shots
        )
        return min(pool, max(ProcessInfo.processInfo.activeProcessorCount, 1), max(shots, 1))
    }

    /// Live state copies for peak-memory estimates (batch pool or CPU worker pool + not a
    /// shared single buffer). CPU concurrent shots allocate this many distinct states.
    static func liveStateCopies(
        circuit: QuantumCircuit,
        noise: NoiseModel?,
        shots: Int?,
        batchSize: Int,
        isCPU: Bool
    ) -> Int {
        guard let shots, shots > 0 else { return 1 }
        if isCPU {
            return cpuLiveStateCopies(
                circuit: circuit,
                noise: noise,
                requested: batchSize,
                shots: shots
            )
        }
        return metalUnitaryBatchSize(
            circuit: circuit,
            noise: noise,
            requested: batchSize,
            shots: shots
        )
    }

    /// Factory heuristic when no circuit is available: Metal unitary-batch only if evolution
    /// noise would not block ``executeUnitaryBatch``.
    static func metalUnitaryBatchAllowedWithoutCircuit(noise: NoiseModel?) -> Bool {
        !requiresEvolutionNoise(noise, circuit: nil)
    }
}
