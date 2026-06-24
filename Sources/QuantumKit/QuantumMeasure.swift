//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation
import Metal

public enum QuantumMeasurementError: Error {
    case invalidShotCount(Int)
}

public struct QuantumMeasurement {

    public static func measure(state: StateVector, engine: QuantumEngine) throws -> [Int] {
        var rng: QuantumRNG = .hardware
        return try measure(state: state, engine: engine, rng: &rng)
    }

    public static func measure(
        state: StateVector,
        engine: QuantumEngine,
        rng: inout QuantumRNG
    ) throws -> [Int] {
        let diceRoll = rng.nextUnitFloat()
        let collapsedIndex = try engine.executeMeasurementCollapse(on: state, diceRoll: diceRoll)
        return toBitArray(value: collapsedIndex, qubitCount: state.qubitCount)
    }

    /// Born-rule probabilities for each computational basis state (index `i` ↔ bitstring of `i`).
    public static func probabilities(state: StateVector, engine: QuantumEngine) throws -> [QFloat] {
        let byteCount = state.stateCount * MemoryLayout<QFloat>.stride
        guard let buffer = state.realBuffer.device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw StateVectorError.bufferAllocationFailed(requiredBytes: byteCount)
        }

        try engine.executeProbabilityKernel(on: state, outputBuffer: buffer)

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
        return try sampleCounts(state: state, engine: engine, shots: shots, rng: &rng)
    }

    public static func sampleCounts(
        state: StateVector,
        engine: QuantumEngine,
        shots: Int,
        rng: inout QuantumRNG
    ) throws -> ShotCounts {
        guard shots > 0 else {
            throw QuantumMeasurementError.invalidShotCount(shots)
        }

        let distribution = try probabilities(state: state, engine: engine)
        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, distribution.count))

        for _ in 0..<shots {
            let outcome = sampleIndex(from: distribution, rng: &rng)
            histogram[outcome, default: 0] += 1
        }

        return ShotCounts(shots: shots, counts: histogram)
    }

    private static func sampleIndex(from distribution: [QFloat], rng: inout QuantumRNG) -> Int {
        let roll = rng.nextUnitFloat()
        var cumulative: QFloat = 0

        for (index, probability) in distribution.enumerated() {
            cumulative += probability
            if roll < cumulative {
                return index
            }
        }

        return distribution.count - 1
    }

    private static func toBitArray(value: Int, qubitCount: Int) -> [Int] {
        var bits = [Int](repeating: 0, count: qubitCount)
        for i in 0..<qubitCount {
            bits[qubitCount - 1 - i] = (value & (1 << i)) != 0 ? 1 : 0
        }
        
        return bits
    }
    
}
