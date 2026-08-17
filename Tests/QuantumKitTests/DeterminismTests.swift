import XCTest
@testable import QuantumKit

/// Seeded reproducibility contract (item 42 lite).
///
/// **Claimed path:** same circuit + same ``QuantumRunOptions/seed`` + same backend method
/// → identical ``shotCounts`` (sampling) or identical CPU SV amplitudes (exact evolution).
///
/// **Intentional exceptions (do not “fix”):**
/// - With ``SampleCountOptions/preferPreparedSampling`` `false`, CPU ``canBatch`` uses
///   ``QuantumRNG/independentShotStream`` while Metal trajectory uses one sequential
///   ``QuantumRNG``. Same seed does **not** imply CPU ≡ Metal for that opt-out path.
/// - Default prepared sampling (evolve-once + multinomial) uses one sequential measurement
///   stream on both CPU and Metal — same seed ⇒ matching histograms for noiseless unitaries.
/// - ``ShotExecutionPolicy/mustSerial`` (mid-circuit measure / `c_if`) keeps one shared
///   sequential stream on CPU — distinct from the independent-shot schedule.
///
/// See also ``SampleCountOptions/batchSize`` and ``QuantumRNG/independentShotStream``.
extension QuantumKitTests {

    // MARK: - Exact amplitudes (CPU SV)

    func testCPUStatevectorAmplitudesDeterministicAcrossRuns() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.cx(0, 1)
        try circuit.rz(theta: QFloat(0.37), 1)
        try circuit.cx(1, 2)
        try circuit.ry(theta: QFloatExpr(QFloat(-0.81)), 2)

        let engine = CPUStatevectorEngine()
        let a = try CPUStateVector(qubitCount: 3)
        let b = try CPUStateVector(qubitCount: 3)
        _ = try engine.execute(circuit, on: a)
        _ = try engine.execute(circuit, on: b)

        XCTAssertEqual(a.real, b.real)
        XCTAssertEqual(a.imag, b.imag)
    }

    // MARK: - Seeded shotCounts (CPU SV)

    func testCPUStatevectorSeededShotCountsAreBitIdentical() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let options = QuantumRunOptions(
            seed: 99_001,
            shots: 256,
            sampleOptions: SampleCountOptions(batchSize: 8)
        )
        let backend = CPUStatevectorBackend()
        let first = try backend.run(circuit: circuit, options: options)
        let second = try backend.run(circuit: circuit, options: options)

        XCTAssertEqual(first.shotCounts, second.shotCounts)
        XCTAssertNotNil(first.shotCounts)
    }

    func testCPUStatevectorSeededShotCountsIndependentOfBatchSize() throws {
        // canBatch + independentShotStream: batchSize must not change the seeded schedule.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let seed: UInt64 = 55
        let shots = 128
        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: circuit, noise: nil))

        let serial = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 1)
            )
        )
        let parallel = try CPUStatevectorBackend().run(
            circuit: circuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 16)
            )
        )
        XCTAssertEqual(serial.shotCounts, parallel.shotCounts)
    }

    // MARK: - Metal (skip if unavailable)

    func testMetalStatevectorSeededShotCountsAreBitIdenticalWhenAvailable() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let options = QuantumRunOptions(
            seed: 77_777,
            shots: 128,
            sampleOptions: SampleCountOptions(batchSize: 1)
        )
        let backend = try StatevectorBackend()
        let first = try backend.run(circuit: circuit, options: options)
        let second = try backend.run(circuit: circuit, options: options)

        XCTAssertEqual(first.shotCounts, second.shotCounts)
        XCTAssertNotNil(first.shotCounts)
    }

    func testMetalStatevectorAmplitudesDeterministicWhenAvailable() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let engine = try QuantumEngine()
        let a = try StateVector(qubitCount: 2)
        let b = try StateVector(qubitCount: 2)
        _ = try engine.execute(circuit, on: a)
        _ = try engine.execute(circuit, on: b)

        let pa = try QuantumMeasurement.probabilities(state: a, engine: engine)
        let pb = try QuantumMeasurement.probabilities(state: b, engine: engine)
        XCTAssertEqual(pa.count, pb.count)
        for index in pa.indices {
            XCTAssertEqual(pa[index], pb[index], accuracy: 1e-5)
        }
    }

    // MARK: - Documented exceptions (assert the contract, do not unify schedules)

    func testCPUIndependentShotStreamDivergesFromMetalSequentialUnderSameSeed() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: circuit, noise: nil))

        let seed: UInt64 = 42
        let shots = 128
        let options = QuantumRunOptions(
            seed: seed,
            shots: shots,
            sampleOptions: SampleCountOptions(batchSize: 8, preferPreparedSampling: false)
        )

        let cpu = try CPUStatevectorBackend().run(circuit: circuit, options: options)
        let metal = try StatevectorBackend().run(circuit: circuit, options: options)

        // Documented asymmetry on the trajectory opt-out path.
        XCTAssertNotEqual(
            cpu.shotCounts,
            metal.shotCounts,
            "CPU canBatch and Metal sequential schedules must remain distinct under the same seed"
        )
    }

    func testPreparedSamplingCPUMatchesMetalUnderSameSeed() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        let options = QuantumRunOptions(
            seed: 42,
            shots: 128,
            sampleOptions: SampleCountOptions(batchSize: 8)
        )
        let cpu = try CPUStatevectorBackend().run(circuit: circuit, options: options)
        let metal = try StatevectorBackend().run(circuit: circuit, options: options)
        XCTAssertEqual(cpu.shotCounts, metal.shotCounts)
    }

    func testMustSerialUsesSequentialStreamDistinctFromIndependentCanBatch() throws {
        // mustSerial: one shared seeded stream. canBatch Bell under the same seed uses
        // independentShotStream — histograms are not required to match (and typically will not).
        let creg = try ClassicalRegisterSpec(bitCount: 1)
        var serialCircuit = try QuantumCircuit(qubitCount: 2, classicalRegisters: [creg])
        try serialCircuit.h(0)
        try serialCircuit.measure(qubits: [0], classicalRegister: 0)
        try serialCircuit.c_if(classicalRegister: 0, equals: 1, x: 1)
        XCTAssertTrue(ShotExecutionPolicy.mustSerial(circuit: serialCircuit, noise: nil))

        var independentCircuit = try QuantumCircuit(qubitCount: 2)
        try independentCircuit.applyBellState()
        XCTAssertTrue(ShotExecutionPolicy.canBatch(circuit: independentCircuit, noise: nil))

        let seed: UInt64 = 19
        let shots = 64
        let serial = try CPUStatevectorBackend().run(
            circuit: serialCircuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 1)
            )
        )
        let independent = try CPUStatevectorBackend().run(
            circuit: independentCircuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 1)
            )
        )
        // Different circuits ⇒ different counts expected; the policy point is mustSerial
        // is still reproducible under the sequential contract:
        let serialAgain = try CPUStatevectorBackend().run(
            circuit: serialCircuit,
            options: QuantumRunOptions(
                seed: seed,
                shots: shots,
                sampleOptions: SampleCountOptions(batchSize: 32) // forced serial internally
            )
        )
        XCTAssertEqual(serial.shotCounts, serialAgain.shotCounts)
        XCTAssertNotNil(independent.shotCounts)
    }
}
