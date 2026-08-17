//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation
import Metal

public struct QuantumMeasurement {

    /// The full complex state vector (amplitude index uses ``QubitBitOrdering/engineLSB``:
    /// bit `k` ↔ qubit `k`, qubit 0 = LSB). Do not reinterpret without an explicit conversion.
    ///
    /// Reads GPU-resident amplitudes from unified memory; values are current as long as any
    /// preceding ``QuantumEngine`` execution has completed.
    public static func amplitudes(state: StateVector) -> [ComplexAmplitude] {
        let realPointer = state.metalRealBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = state.metalImagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        return (0..<state.stateCount).map { index in
            ComplexAmplitude(real: realPointer[index], imaginary: imagPointer[index])
        }
    }

    public static func measure(state: StateVector, engine: QuantumEngine, noise: NoiseModel? = nil) throws -> [Int] {
        var rng: QuantumRNG = .hardware
        return try measureRNG(state: state, engine: engine, rng: &rng, noise: noise)
    }

    public static func measureRNG(
        state: StateVector,
        engine: QuantumEngine,
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil
    ) throws -> [Int] {
        let collapsedIndex = try engine.executeMeasurementCollapse(on: state, rng: &rng, noise: noise)
        return toBitArray(value: collapsedIndex, qubitCount: state.qubitCount)
    }

    /// Born-rule probabilities for each computational basis state
    /// (``QubitBitOrdering/engineLSB`` index).
    public static func probabilities(state: StateVector, engine: QuantumEngine) throws -> [QFloat] {
        let byteCount = state.stateCount * MemoryLayout<QFloat>.stride
        // Allocate on the engine's device so the probability kernel writes a buffer it can bind;
        // the normal device-free path uses the same shared MetalRuntime device for state and engine.
        guard let buffer = engine.device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw StateVectorError.bufferAllocationFailed(requiredBytes: byteCount)
        }

        try engine.executeProbabilityKernel(on: state, into: buffer)

        let pointer = buffer.contents().assumingMemoryBound(to: QFloat.self)
        return Array(UnsafeBufferPointer(start: pointer, count: state.stateCount))
    }

    /// Draws `shots` independent samples from the current state without collapsing it.
    public static func sampleCounts(
        state: StateVector,
        engine: QuantumEngine,
        shots: Int
    ) throws -> ShotCounts {
        var rng: QuantumRNG = .hardware
        return try sampleCountsRNG(state: state, engine: engine, shots: shots, rng: &rng)
    }

    public static func sampleCountsRNG(
        state: StateVector,
        engine: QuantumEngine,
        shots: Int,
        rng: inout QuantumRNG
    ) throws -> ShotCounts {
        guard shots > 0 else {
            throw QuantumMeasurementError.invalidShotCount(shots)
        }

        if ShotExecutionPolicy.usesDevicePreparedSampling(qubitCount: state.qubitCount) {
            return try engine.sampleComputationalBasisCounts(on: state, shots: shots, rng: &rng)
        }

        let distribution = try probabilities(state: state, engine: engine)
        return try buildHistogram(from: distribution, shots: shots, rng: &rng)
    }

    /// Marginal Born-rule probabilities for a subset of qubits.
    public static func partialProbabilities(
        state: StateVector,
        engine: QuantumEngine,
        qubits: [Int]
    ) throws -> [QFloat] {
        try validateQubits(qubits, qubitCount: state.qubitCount)

        let fullDistribution = try probabilities(state: state, engine: engine)

        // Accumulate in Double so summing 2ⁿ sub-ulp float probabilities does not lose mass to
        // Float32 cancellation, then narrow the (bounded, small) marginal back to QFloat.
        var marginal = [Double](repeating: 0, count: 1 << qubits.count)
        for (stateIndex, probability) in fullDistribution.enumerated() {
            let outcome = partialOutcomeIndex(stateIndex: stateIndex, qubits: qubits)
            marginal[outcome] += Double(probability)
        }

        return marginal.map { QFloat($0) }
    }

    public static func sampleCounts(
        state: StateVector,
        engine: QuantumEngine,
        qubits: [Int],
        shots: Int
    ) throws -> ShotCounts {
        var rng: QuantumRNG = .hardware
        return try sampleCountsRNG(state: state, engine: engine, qubits: qubits, shots: shots, rng: &rng)
    }

    public static func sampleCountsRNG(
        state: StateVector,
        engine: QuantumEngine,
        qubits: [Int],
        shots: Int,
        rng: inout QuantumRNG
    ) throws -> ShotCounts {
        guard shots > 0 else {
            throw QuantumMeasurementError.invalidShotCount(shots)
        }

        let distribution = try partialProbabilities(state: state, engine: engine, qubits: qubits)
        return try buildHistogram(from: distribution, shots: shots, rng: &rng)
    }

    /// Runs the circuit `shots` times from |0…0⟩, collapsing each run before the next.
    ///
    /// Independent unitary circuits use Metal ``StateVectorBatch`` when
    /// ``ShotExecutionPolicy/canUseMetalUnitaryBatch(circuit:noise:)`` (see
    /// ``SampleCountOptions/batchSize``). Coupled / noisy-evolution paths stay serial.
    /// Metal keeps one sequential measurement RNG; CPU ``canBatch`` uses per-shot streams
    /// — same `seed` does not imply matching histograms across backends.
    public static func runSampleCounts(
        circuit: QuantumCircuit,
        engine: QuantumEngine,
        shots: Int,
        noise: NoiseModel? = nil,
        options: SampleCountOptions = SampleCountOptions()
    ) throws -> ShotCounts {
        var rng: QuantumRNG = .hardware
        return try runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise,
            options: options
        )
    }

    public static func runSampleCountsRNG(
        circuit: QuantumCircuit,
        engine: QuantumEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil,
        options: SampleCountOptions = SampleCountOptions(),
        cancellationCheck: (() throws -> Void)? = nil
    ) throws -> ShotCounts {
        try BatchSampleExecutor.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise,
            options: options,
            cancellationCheck: cancellationCheck
        )
    }

    static func pauliZEigenvalue(stateIndex: Int, qubit: Int) -> QFloat {
        ((stateIndex >> qubit) & 1) == 0 ? 1 : -1
    }

    static func buildHistogram(
        from distribution: [QFloat],
        shots: Int,
        rng: inout QuantumRNG
    ) throws -> ShotCounts {
        // Build the cumulative distribution once, then sample each shot with an O(log n) binary
        // search instead of re-scanning the whole distribution per shot. This drops histogram
        // building from O(shots · 2ⁿ) to O(2ⁿ + shots · log 2ⁿ) and is bit-for-bit identical to the
        // old linear scan for any given roll (same CDF, same first-exceeding index).
        let cumulative = cumulativeDistribution(distribution)

        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, distribution.count))

        for _ in 0..<shots {
            let roll = rng.nextUnitDouble()
            let outcome = sampleIndex(roll: roll, cumulative: cumulative)
            histogram[outcome, default: 0] += 1
        }

        return ShotCounts(shots: shots, counts: histogram)
    }

    static func validateQubits(_ qubits: [Int], qubitCount: Int) throws {
        guard !qubits.isEmpty else {
            throw QuantumMeasurementError.emptyQubitSelection
        }

        for index in qubits where index < 0 || index >= qubitCount {
            throw QuantumMeasurementError.qubitIndexOutOfBounds(index: index, qubitCount: qubitCount)
        }
    }

    static func partialOutcomeIndex(stateIndex: Int, qubits: [Int]) -> Int {
        var outcome = 0
        for (position, qubit) in qubits.enumerated() {
            let bit = (stateIndex >> qubit) & 1
            outcome |= bit << position
        }
        return outcome
    }

    /// Inclusive CDF accumulated in `Double`.
    ///
    /// At high qubit counts a Float32 CDF quantizes away the tail of the distribution (summing 2ⁿ
    /// sub-ulp probabilities loses mass to cancellation), so the running total is carried in `Double`
    /// — mirroring the compensated GPU collapse path.
    static func cumulativeDistribution(_ distribution: [QFloat]) -> [Double] {
        var cumulative = [Double](repeating: 0, count: distribution.count)
        var running = 0.0
        for (index, probability) in distribution.enumerated() {
            running += Double(probability)
            cumulative[index] = running
        }
        return cumulative
    }

    /// Smallest index `i` with `roll < cumulative[i]` — exactly the index the linear scan would
    /// return for the same roll. Because the CDF is non-decreasing the predicate is monotonic, so a
    /// binary search finds it in O(log n); the final index is returned when the roll lands in the
    /// rounding gap above the total mass.
    static func sampleIndex(roll: Double, cumulative: [Double]) -> Int {
        var low = 0
        var high = cumulative.count - 1

        while low < high {
            let mid = (low + high) / 2
            if roll < cumulative[mid] {
                high = mid
            } else {
                low = mid + 1
            }
        }

        return low
    }

    static func toBitArray(value: Int, qubitCount: Int) -> [Int] {
        // ``QubitBitOrdering/bitstringMSB`` array: index 0 = highest qubit.
        (try? QubitBitOrdering.bitstringMSB.bits(forIndex: value, qubitCount: qubitCount))
            ?? Array(repeating: 0, count: qubitCount)
    }
    
}
