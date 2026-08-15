import Foundation
import Metal

extension QuantumEngine {
    /// Runs `shots` independent stochastic trajectories and aggregates final bitstring counts.
    ///
    /// Metal is resolved via this engine’s ``MetalRuntime``-backed device.
    ///
    /// - Important: For ideal/noise-free circuits this fully bypasses trajectory overhead and reuses
    ///   the existing fast batching path.
    public func executeTrajectorySampleCounts(
        _ circuit: QuantumCircuit,
        shots: Int,
        noise: NoiseModel? = nil,
        options: SampleCountOptions = SampleCountOptions()
    ) throws -> ShotCounts {
        var rng: QuantumRNG = .hardware
        return try executeTrajectorySampleCountsRNG(
            circuit,
            shots: shots,
            rng: &rng,
            noise: noise,
            options: options
        )
    }

    /// RNG-injectable variant of `executeTrajectorySampleCounts`.
    public func executeTrajectorySampleCountsRNG(
        _ circuit: QuantumCircuit,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil,
        options: SampleCountOptions = SampleCountOptions()
    ) throws -> ShotCounts {
        try BatchSampleExecutor.runSampleCountsRNG(
            circuit: circuit,
            engine: self,
            shots: shots,
            rng: &rng,
            noise: noise,
            options: options
        )
    }
}
