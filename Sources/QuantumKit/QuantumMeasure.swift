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
    case emptyQubitSelection
    case qubitIndexOutOfBounds(index: Int, qubitCount: Int)
}

public struct QuantumMeasurement {

    public static func measure(state: StateVector, engine: QuantumEngine) throws -> [Int] {
        var rng: QuantumRNG = .hardware
        return try measureRNG(state: state, engine: engine, rng: &rng)
    }

    public static func measureRNG(
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
        var marginal = [QFloat](repeating: 0, count: 1 << qubits.count)

        for (stateIndex, probability) in fullDistribution.enumerated() {
            let outcome = partialOutcomeIndex(stateIndex: stateIndex, qubits: qubits)
            marginal[outcome] += probability
        }

        return marginal
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

    /// ⟨Z⟩ for a single qubit in the computational basis.
    public static func expectationZ(
        state: StateVector,
        engine: QuantumEngine,
        qubit: Int
    ) throws -> QFloat {
        try expectationPauliZ(state: state, engine: engine, qubits: [qubit])
    }

    /// ⟨Z_a Z_b⟩ for two qubits.
    public static func expectationZZ(
        state: StateVector,
        engine: QuantumEngine,
        qubitA: Int,
        qubitB: Int
    ) throws -> QFloat {
        try expectationPauliZ(state: state, engine: engine, qubits: [qubitA, qubitB])
    }

    /// ⟨Z_{i_0} Z_{i_1} …⟩ as the product of Pauli-Z on the listed qubits.
    public static func expectationPauliZ(
        state: StateVector,
        engine: QuantumEngine,
        qubits: [Int]
    ) throws -> QFloat {
        try validateQubits(qubits, qubitCount: state.qubitCount)

        let distribution = try probabilities(state: state, engine: engine)
        var expectation: QFloat = 0

        for (stateIndex, probability) in distribution.enumerated() {
            var eigenvalue: QFloat = 1
            for qubit in qubits {
                eigenvalue *= pauliZEigenvalue(stateIndex: stateIndex, qubit: qubit)
            }
            expectation += probability * eigenvalue
        }

        return expectation
    }

    /// Runs the circuit `shots` times from |0…0⟩, collapsing each run before the next.
    public static func runSampleCounts(
        circuit: QuantumCircuit,
        engine: QuantumEngine,
        device: MTLDevice,
        shots: Int
    ) throws -> ShotCounts {
        var rng: QuantumRNG = .hardware
        return try runSampleCountsRNG(circuit: circuit, engine: engine, device: device, shots: shots, rng: &rng)
    }

    public static func runSampleCountsRNG(
        circuit: QuantumCircuit,
        engine: QuantumEngine,
        device: MTLDevice,
        shots: Int,
        rng: inout QuantumRNG
    ) throws -> ShotCounts {
        guard shots > 0 else {
            throw QuantumMeasurementError.invalidShotCount(shots)
        }

        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, circuit.qubitCount > 0 ? 1 << circuit.qubitCount : 1))

        for _ in 0..<shots {
            let state = try StateVector(qubitCount: circuit.qubitCount, device: device)
            try engine.execute(circuit, on: state)
            let diceRoll = rng.nextUnitFloat()
            let outcome = try engine.executeMeasurementCollapse(on: state, diceRoll: diceRoll)
            histogram[outcome, default: 0] += 1
        }

        return ShotCounts(shots: shots, counts: histogram)
    }

    private static func pauliZEigenvalue(stateIndex: Int, qubit: Int) -> QFloat {
        ((stateIndex >> qubit) & 1) == 0 ? 1 : -1
    }

    private static func buildHistogram(
        from distribution: [QFloat],
        shots: Int,
        rng: inout QuantumRNG
    ) throws -> ShotCounts {
        var histogram: [Int: Int] = [:]
        histogram.reserveCapacity(min(shots, distribution.count))

        for _ in 0..<shots {
            let outcome = sampleIndex(from: distribution, rng: &rng)
            histogram[outcome, default: 0] += 1
        }

        return ShotCounts(shots: shots, counts: histogram)
    }

    private static func validateQubits(_ qubits: [Int], qubitCount: Int) throws {
        guard !qubits.isEmpty else {
            throw QuantumMeasurementError.emptyQubitSelection
        }

        for index in qubits where index < 0 || index >= qubitCount {
            throw QuantumMeasurementError.qubitIndexOutOfBounds(index: index, qubitCount: qubitCount)
        }
    }

    private static func partialOutcomeIndex(stateIndex: Int, qubits: [Int]) -> Int {
        var outcome = 0
        for (position, qubit) in qubits.enumerated() {
            let bit = (stateIndex >> qubit) & 1
            outcome |= bit << position
        }
        return outcome
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
