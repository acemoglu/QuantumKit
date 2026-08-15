import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

    func testBellStateShotCounts() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try engine.execute(circuit, on: state)

        var rng: QuantumRNG = .seeded(42)
        let result = try QuantumMeasurement.sampleCountsRNG(state: state, engine: engine, shots: 1_000, rng: &rng)

        XCTAssertEqual(result.shots, 1_000)
        let bitstrings = result.bitstringCounts(qubitCount: 2)
        XCTAssertEqual(bitstrings.keys.sorted(), ["00", "11"])
        XCTAssertEqual(bitstrings.values.reduce(0, +), 1_000)
        XCTAssertNil(bitstrings["01"])
        XCTAssertNil(bitstrings["10"])

        var replayRNG: QuantumRNG = .seeded(42)
        let replay = try QuantumMeasurement.sampleCountsRNG(state: state, engine: engine, shots: 1_000, rng: &replayRNG)
        XCTAssertEqual(replay, result)
    }

    func testPartialMeasurementOnBellState() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: [0])
        XCTAssertEqual(marginal.count, 2)
        XCTAssertEqual(marginal[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(marginal[1], 0.5, accuracy: 1e-5)

        var rng: QuantumRNG = .seeded(7)
        let result = try QuantumMeasurement.sampleCountsRNG(
            state: state,
            engine: engine,
            qubits: [0],
            shots: 1_000,
            rng: &rng
        )

        let bitstrings = result.bitstringCounts(qubits: [0])
        XCTAssertEqual(bitstrings.keys.sorted(), ["0", "1"])
        XCTAssertEqual(bitstrings.values.reduce(0, +), 1_000)
    }

    func testExpectationZOnPlusState() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 0, accuracy: 1e-5)
    }

    func testExpectationZOnOneState() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5)
    }

    func testExpectationZZOnBellState() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try engine.execute(circuit, on: state)

        let z0 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        let zz = try QuantumMeasurement.expectationZZ(state: state, engine: engine, qubitA: 0, qubitB: 1)

        XCTAssertEqual(z0, 0, accuracy: 1e-5)
        XCTAssertEqual(z1, 0, accuracy: 1e-5)
        XCTAssertEqual(zz, 1, accuracy: 1e-5)
    }

    func testRunSampleCountsOnBellState() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        var rng: QuantumRNG = .seeded(99)
        let result = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 500,
            rng: &rng
        )

        XCTAssertEqual(result.shots, 500)
        let bitstrings = result.bitstringCounts(qubitCount: 2)
        XCTAssertEqual(bitstrings.keys.sorted(), ["00", "11"])
        XCTAssertEqual(bitstrings.values.reduce(0, +), 500)
    }

    func testBatchRunSampleCountsMatchesSequentialRNG() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        var sequentialRNG: QuantumRNG = .seeded(42)
        let sequential = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 256,
            rng: &sequentialRNG,
            options: SampleCountOptions(batchSize: 1)
        )

        var batchedRNG: QuantumRNG = .seeded(42)
        let batched = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            shots: 256,
            rng: &batchedRNG,
            options: SampleCountOptions(batchSize: 32)
        )

        XCTAssertEqual(sequential, batched)
    }

    func testExecuteUnitaryBatchMatchesSequential() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let sequentialState = try StateVector(qubitCount: 2)
        try engine.execute(circuit, on: sequentialState)

        let batchStateA = try StateVector(qubitCount: 2)
        let batchStateB = try StateVector(qubitCount: 2)
        try engine.executeUnitaryBatch(circuit, on: [batchStateA, batchStateB])

        let sequentialProbabilities = try QuantumMeasurement.probabilities(state: sequentialState, engine: engine)
        let batchProbabilities = try QuantumMeasurement.probabilities(state: batchStateA, engine: engine)

        XCTAssertEqual(sequentialProbabilities.count, batchProbabilities.count)
        for index in 0..<sequentialProbabilities.count {
            XCTAssertEqual(sequentialProbabilities[index], batchProbabilities[index], accuracy: 1e-5)
        }
        XCTAssertEqual(
            try QuantumMeasurement.probabilities(state: batchStateB, engine: engine),
            batchProbabilities
        )
    }

    func testMidCircuitMeasureAndReset() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)
        try circuit.reset(0)

        var rng: QuantumRNG = .seeded(123)
        let execution = try engine.executeRNG(circuit, on: state, rng: &rng)

        XCTAssertEqual(execution.measurementOutcomes.count, 1)
        XCTAssertTrue(execution.measurementOutcomes[0] == [0] || execution.measurementOutcomes[0] == [1])

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)
    }

    func testMidCircuitMeasureOnBellState() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try circuit.measure(0)

        var rng: QuantumRNG = .seeded(5)
        let execution = try engine.executeRNG(circuit, on: state, rng: &rng)

        XCTAssertEqual(execution.measurementOutcomes.count, 1)
        let measuredQubit0 = execution.measurementOutcomes[0][0]

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, measuredQubit0 == 0 ? 1 : -1, accuracy: 1e-5)
    }

    /// Exercises the GPU multi-qubit marginal path (k ≥ 2) with a deterministic basis state. Prepares
    /// |bit0=1, bit1=0, bit2=1, bit3=0⟩ and measures a *reordered, non-contiguous* qubit list, so the
    /// recorded bits must follow the measured-list order — catching any bin bit-ordering bug in the
    /// `marginal_leaf_histogram` kernel.
    func testMultiQubitPartialMeasurementBasisStateBitOrdering() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 4)
        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.x(0)
        try circuit.x(2)
        try circuit.measure(qubits: [2, 0, 3])

        var rng: QuantumRNG = .seeded(7)
        let execution = try engine.executeRNG(circuit, on: state, rng: &rng)

        // Bits, in measured-list order [q2, q0, q3] = [1, 1, 0].
        XCTAssertEqual(execution.measurementOutcomes, [[1, 1, 0]])
    }

    /// Exercises the GPU marginal path (k = 3) on a GHZ state: measuring all three qubits at once must
    /// collapse to a fully correlated all-zeros or all-ones outcome, never a mixed bitstring.
    func testGHZThreeQubitPartialMeasurementIsFullyCorrelated() throws {
        let engine = try QuantumEngine()
        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        var sawZeros = false
        var sawOnes = false
        for seed in UInt64(0)..<24 {
            let state = try StateVector(qubitCount: 3)
            var circuit = try QuantumCircuit(qubitCount: 3)
            try circuit.h(0)
            try circuit.cx(0, 1)
            try circuit.cx(0, 2)
            try circuit.measure(qubits: [0, 1, 2])

            var rng: QuantumRNG = .seeded(seed)
            let execution = try engine.executeRNG(circuit, on: state, rng: &rng)
            let bits = execution.measurementOutcomes[0]
            XCTAssertTrue(bits == [0, 0, 0] || bits == [1, 1, 1], "GHZ collapse must be correlated, got \(bits)")
            if bits == [0, 0, 0] { sawZeros = true }
            if bits == [1, 1, 1] { sawOnes = true }
        }
        // Sanity that the sampler isn't pinned to one branch across many seeds.
        XCTAssertTrue(sawZeros && sawOnes, "expected both GHZ branches across seeds")
    }
}
