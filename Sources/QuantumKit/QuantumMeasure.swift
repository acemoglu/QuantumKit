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
    case invalidPauliString(String)
}

/// A single-qubit Pauli operator (identity included) used to build tensor-product observables.
public enum Pauli: Equatable, Sendable {
    case i, x, y, z
}

/// A single complex amplitude of a state vector.
public struct ComplexAmplitude: Equatable, Sendable {
    public let real: QFloat
    public let imaginary: QFloat

    public init(real: QFloat, imaginary: QFloat) {
        self.real = real
        self.imaginary = imaginary
    }
}

public struct QuantumMeasurement {

    /// The full complex state vector (amplitude `i` ↔ bitstring of `i`, qubit 0 = LSB).
    ///
    /// Reads `realBuffer`/`imagBuffer` directly from unified memory; the buffers are
    /// current as long as any preceding ``QuantumEngine`` execution has completed.
    public static func amplitudes(state: StateVector) -> [ComplexAmplitude] {
        let realPointer = state.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = state.imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

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
        var expectation = 0.0

        for (stateIndex, probability) in distribution.enumerated() {
            var eigenvalue: QFloat = 1
            for qubit in qubits {
                eigenvalue *= pauliZEigenvalue(stateIndex: stateIndex, qubit: qubit)
            }
            expectation += Double(probability) * Double(eigenvalue)
        }

        return QFloat(expectation)
    }

    /// Runs the circuit `shots` times from |0…0⟩, collapsing each run before the next.
    ///
    /// Unitary-only circuits without gate noise are executed in batches on the GPU
    /// (see ``SampleCountOptions/batchSize``).
    public static func runSampleCounts(
        circuit: QuantumCircuit,
        engine: QuantumEngine,
        device: MTLDevice,
        shots: Int,
        noise: NoiseModel? = nil,
        options: SampleCountOptions = SampleCountOptions()
    ) throws -> ShotCounts {
        var rng: QuantumRNG = .hardware
        return try runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            device: device,
            shots: shots,
            rng: &rng,
            noise: noise,
            options: options
        )
    }

    public static func runSampleCountsRNG(
        circuit: QuantumCircuit,
        engine: QuantumEngine,
        device: MTLDevice,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil,
        options: SampleCountOptions = SampleCountOptions()
    ) throws -> ShotCounts {
        try BatchSampleExecutor.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            device: device,
            shots: shots,
            rng: &rng,
            noise: noise,
            options: options
        )
    }

    /// ⟨X⟩ for a single qubit in the computational basis.
    public static func expectationX(
        state: StateVector,
        engine: QuantumEngine,
        qubit: Int
    ) throws -> QFloat {
        try validateQubits([qubit], qubitCount: state.qubitCount)

        let mask = 1 << qubit
        let realPointer = state.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = state.imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        var expectation = 0.0
        for index in 0..<state.stateCount {
            let flipped = index ^ mask
            let realProduct = realPointer[index] * realPointer[flipped]
            let imagProduct = imagPointer[index] * imagPointer[flipped]
            expectation += Double(realProduct + imagProduct)
        }

        return QFloat(expectation)
    }

    /// ⟨ψ|P|ψ⟩ for an arbitrary Pauli tensor product `P`.
    ///
    /// `paulis` maps a qubit index to its Pauli factor; any qubit absent from the map (or mapped
    /// to `.i`) acts as identity. Computed analytically on the CPU in O(2ⁿ) by reading amplitudes
    /// directly — no extra circuit or state clone is required.
    ///
    /// `P|j⟩ = phase(j)·|j ⊕ flipMask⟩`, where `flipMask` collects the X/Y qubits and
    /// `phase(j) = Π_{Y}(bit=0 → +i, bit=1 → −i) · Π_{Z}((−1)^bit)`. The `Π_Y` magnitude is the
    /// global factor `i^{#Y}`, and the per-state sign comes from the bits set under the Y and Z
    /// qubits. The expectation `Σⱼ conj(a_{j⊕flipMask})·phase(j)·aⱼ` is real, so its real part is returned.
    public static func expectation(
        state: StateVector,
        engine: QuantumEngine,
        paulis: [Int: Pauli]
    ) throws -> QFloat {
        var flipMask = 0
        var yMask = 0
        var zMask = 0
        var yCount = 0

        for (qubit, pauli) in paulis {
            guard qubit >= 0, qubit < state.qubitCount else {
                throw QuantumMeasurementError.qubitIndexOutOfBounds(index: qubit, qubitCount: state.qubitCount)
            }
            let bit = 1 << qubit
            switch pauli {
            case .i:
                continue
            case .x:
                flipMask |= bit
            case .y:
                flipMask |= bit
                yMask |= bit
                yCount += 1
            case .z:
                zMask |= bit
            }
        }

        // i^{#Y}: a global complex factor shared by every term.
        let phaseBaseReal: QFloat
        let phaseBaseImag: QFloat
        switch yCount & 3 {
        case 0: (phaseBaseReal, phaseBaseImag) = (1, 0)
        case 1: (phaseBaseReal, phaseBaseImag) = (0, 1)
        case 2: (phaseBaseReal, phaseBaseImag) = (-1, 0)
        default: (phaseBaseReal, phaseBaseImag) = (0, -1)
        }

        let realPointer = state.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = state.imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        let signMask = yMask | zMask
        var expectation = 0.0

        for j in 0..<state.stateCount {
            let k = j ^ flipMask
            let negative = ((j & signMask).nonzeroBitCount & 1) == 1
            let pr = negative ? -phaseBaseReal : phaseBaseReal
            let pi = negative ? -phaseBaseImag : phaseBaseImag

            let rj = realPointer[j]
            let ij = imagPointer[j]
            // phase(j) · a_j
            let xr = pr * rj - pi * ij
            let xi = pr * ij + pi * rj
            // Re[ conj(a_k) · phase(j) · a_j ], accumulated in Double to avoid Float32 cancellation.
            expectation += Double(realPointer[k] * xr + imagPointer[k] * xi)
        }

        return QFloat(expectation)
    }

    /// ⟨ψ|P|ψ⟩ for a Pauli label such as `"XYZ"` or `"IXZ"`.
    ///
    /// The label is MSB-first to match ``measure`` bit-array ordering: the leftmost character is
    /// the highest-index qubit. Its length must equal `state.qubitCount`.
    public static func expectation(
        state: StateVector,
        engine: QuantumEngine,
        pauliString: String
    ) throws -> QFloat {
        let paulis = try parsePauliString(pauliString, qubitCount: state.qubitCount)
        return try expectation(state: state, engine: engine, paulis: paulis)
    }

    static func parsePauliString(_ string: String, qubitCount: Int) throws -> [Int: Pauli] {
        let characters = Array(string.uppercased())
        guard characters.count == qubitCount else {
            throw QuantumMeasurementError.invalidPauliString(string)
        }

        var paulis: [Int: Pauli] = [:]
        for (position, character) in characters.enumerated() {
            let qubit = qubitCount - 1 - position
            let pauli: Pauli
            switch character {
            case "I": pauli = .i
            case "X": pauli = .x
            case "Y": pauli = .y
            case "Z": pauli = .z
            default:
                throw QuantumMeasurementError.invalidPauliString(string)
            }
            if pauli != .i {
                paulis[qubit] = pauli
            }
        }
        return paulis
    }

    private static func pauliZEigenvalue(stateIndex: Int, qubit: Int) -> QFloat {
        ((stateIndex >> qubit) & 1) == 0 ? 1 : -1
    }

    private static func buildHistogram(
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

    /// Inclusive CDF accumulated in `Double`.
    ///
    /// At high qubit counts a Float32 CDF quantizes away the tail of the distribution (summing 2ⁿ
    /// sub-ulp probabilities loses mass to cancellation), so the running total is carried in `Double`
    /// — mirroring the compensated GPU collapse path.
    private static func cumulativeDistribution(_ distribution: [QFloat]) -> [Double] {
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
    private static func sampleIndex(roll: Double, cumulative: [Double]) -> Int {
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

    private static func toBitArray(value: Int, qubitCount: Int) -> [Int] {
        var bits = [Int](repeating: 0, count: qubitCount)
        for i in 0..<qubitCount {
            bits[qubitCount - 1 - i] = (value & (1 << i)) != 0 ? 1 : 0
        }
        
        return bits
    }
    
}
