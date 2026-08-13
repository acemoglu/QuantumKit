import Foundation

/// Density-matrix terminal shot sampling without avoidable full-circuit re-execution.
enum DensityMatrixShotSampler {

    /// Samples computational-basis shots from a prepared Metal density matrix.
    ///
    /// When ``QuantumCircuit/allowsPreparedDensityShotBatching(noise:)`` is true, evolves once
    /// and draws all shots from `diag(ρ)` (plus optional readout flips). Otherwise re-executes
    /// the circuit per shot (projective mid-circuit measure / classical control).
    static func runSampleCountsRNG(
        circuit: QuantumCircuit,
        engine: DensityMatrixEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?
    ) throws -> ShotCounts {
        guard shots > 0 else {
            throw QuantumMeasurementError.invalidShotCount(shots)
        }

        if circuit.allowsPreparedDensityShotBatching(noise: noise) {
            let density = try DensityMatrix(qubitCount: circuit.qubitCount, device: engine.device)
            _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: noise)
            let probabilities = engine.probabilities(of: density)
            return try sampleTerminalShots(
                probabilities: probabilities,
                shots: shots,
                measuredQubitCount: circuit.qubitCount,
                rng: &rng,
                noise: noise
            )
        }

        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, 1 << circuit.qubitCount))
        for _ in 0..<shots {
            let density = try DensityMatrix(qubitCount: circuit.qubitCount, device: engine.device)
            _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: noise)
            let probabilities = engine.probabilities(of: density)
            let single = try sampleTerminalShots(
                probabilities: probabilities,
                shots: 1,
                measuredQubitCount: circuit.qubitCount,
                rng: &rng,
                noise: noise
            )
            for (outcome, count) in single.counts {
                histogram[outcome, default: 0] += count
            }
        }
        return ShotCounts(shots: shots, counts: histogram)
    }

    /// CPU density-matrix counterpart of ``runSampleCountsRNG``.
    static func runSampleCountsRNG(
        circuit: QuantumCircuit,
        engine: CPUDensityMatrixEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?
    ) throws -> ShotCounts {
        guard shots > 0 else {
            throw QuantumMeasurementError.invalidShotCount(shots)
        }

        if circuit.allowsPreparedDensityShotBatching(noise: noise) {
            let density = try CPUDensityMatrix(qubitCount: circuit.qubitCount)
            _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: noise)
            return try sampleTerminalShots(
                probabilities: density.probabilities(),
                shots: shots,
                measuredQubitCount: circuit.qubitCount,
                rng: &rng,
                noise: noise
            )
        }

        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, 1 << circuit.qubitCount))
        for _ in 0..<shots {
            let density = try CPUDensityMatrix(qubitCount: circuit.qubitCount)
            _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: noise)
            let single = try sampleTerminalShots(
                probabilities: density.probabilities(),
                shots: 1,
                measuredQubitCount: circuit.qubitCount,
                rng: &rng,
                noise: noise
            )
            for (outcome, count) in single.counts {
                histogram[outcome, default: 0] += count
            }
        }
        return ShotCounts(shots: shots, counts: histogram)
    }

    /// Draws shots from an already-prepared probability vector, applying readout flips last.
    static func sampleTerminalShots(
        probabilities: [QFloat],
        shots: Int,
        measuredQubitCount: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?
    ) throws -> ShotCounts {
        let base = try QuantumMeasurement.buildHistogram(
            from: probabilities,
            shots: shots,
            rng: &rng
        )
        guard let noise, noise.appliesReadoutError else {
            return base
        }

        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(base.counts.count)
        for (outcome, count) in base.counts {
            for _ in 0..<count {
                let flipped = noise.flipReadoutOutcome(
                    outcome,
                    measuredQubitCount: measuredQubitCount,
                    rng: &rng
                )
                histogram[flipped, default: 0] += 1
            }
        }
        return ShotCounts(shots: shots, counts: histogram)
    }

    /// Estimates ⟨Z…⟩ (or a Pauli string after basis change) from terminal shot counts.
    static func zExpectation(from counts: ShotCounts, qubits: [Int]) -> QFloat {
        guard counts.shots > 0 else { return 0 }
        var sum: QFloat = 0
        for (outcome, count) in counts.counts {
            var parity = 0
            for qubit in qubits {
                parity ^= (outcome >> qubit) & 1
            }
            sum += (parity == 0 ? QFloat(1) : QFloat(-1)) * QFloat(count)
        }
        return sum / QFloat(counts.shots)
    }
}
