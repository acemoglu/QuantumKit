import XCTest
import Metal
@testable import QuantumKit

final class QuantumKitTests: XCTestCase {

    private func makeDevice() -> MTLDevice? {
        MTLCreateSystemDefaultDevice()
    }
    
    func testMassiveGHZStateGPUPerformance() throws {
            let engine = try QuantumEngine()

            guard let device = makeDevice() else {
                XCTFail("Apple Silicon GPU not found!")
                return
            }

            // 24 Kübit = Yaklaşık 16.7 Milyon Paralel Durum (State)
            let qubitCount = 28
            let state = try StateVector(qubitCount: qubitCount, device: device)
            var circuit = try QuantumCircuit(qubitCount: qubitCount)

            // 1. Evreni tam ortadan iki ihtimale bölüyoruz
            try circuit.h(0)
            
            // 2. Tüm kübitleri birbirine "Domino Taşı" gibi dolanık hale getiriyoruz
            // Bu işlem GPU'yu tam kapasite çalıştıracak devasa bir zincirdir.
            for i in 0..<(qubitCount - 1) {
                try circuit.cx(i, i + 1)
            }

            let startTime = CFAbsoluteTimeGetCurrent()
            
            // 16.7 milyon durumu Metal'de hesapla
            try engine.execute(circuit, on: state)
            
            // Parallel Prefix Sum ve GPU Binary Search ile ölçüm yap
            let result = try QuantumMeasurement.measure(state: state, engine: engine)
            
            let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
            
            print("🌌 28-QUBIT SUPER-ENTANGLEMENT COLLAPSE [Süre: \(String(format: "%.4f", timeElapsed)) saniye]")
            print("Sonuç dizisi: \(result)")

            // Kusursuz dolanıklık kanıtı: Evren ya tamamen 0 ya da tamamen 1 çökmeli!
            let isAllZeros = result.allSatisfy { $0 == 0 }
            let isAllOnes = result.allSatisfy { $0 == 1 }

            XCTAssertTrue(isAllZeros || isAllOnes, "Kuantum zinciri koptu! Sistem fiziğe aykırı davrandı.")
        }

    func testBellStateShotCounts() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
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

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
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

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 0, accuracy: 1e-5)
    }

    func testExpectationZOnOneState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5)
    }

    func testExpectationZZOnBellState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
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
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        var rng: QuantumRNG = .seeded(99)
        let result = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            device: device,
            shots: 500,
            rng: &rng
        )

        XCTAssertEqual(result.shots, 500)
        let bitstrings = result.bitstringCounts(qubitCount: 2)
        XCTAssertEqual(bitstrings.keys.sorted(), ["00", "11"])
        XCTAssertEqual(bitstrings.values.reduce(0, +), 500)
    }

    func testMidCircuitMeasureAndReset() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
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

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
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

    func testZeroDepolarizingNoisePreservesBellState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let idealState = try StateVector(qubitCount: 2, device: device)
        var idealCircuit = try QuantumCircuit(qubitCount: 2)
        try idealCircuit.applyBellState()
        try engine.execute(idealCircuit, on: idealState)
        let idealZZ = try QuantumMeasurement.expectationZZ(state: idealState, engine: engine, qubitA: 0, qubitB: 1)

        let noisyState = try StateVector(qubitCount: 2, device: device)
        var noisyCircuit = try QuantumCircuit(qubitCount: 2)
        try noisyCircuit.applyBellState()

        var rng: QuantumRNG = .seeded(42)
        let noise = NoiseModel(depolarizingProbability: 0)
        _ = try engine.executeRNG(noisyCircuit, on: noisyState, rng: &rng, noise: noise)

        let noisyZZ = try QuantumMeasurement.expectationZZ(state: noisyState, engine: engine, qubitA: 0, qubitB: 1)
        XCTAssertEqual(noisyZZ, idealZZ, accuracy: 1e-5)
    }

    func testDepolarizingNoiseCanFlipQubitWithPauliX() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        var rng: QuantumRNG = .seeded(11)
        let noise = NoiseModel(depolarizingProbability: 1)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)
    }

    func testAmplitudeDampingResetsExcitedQubit() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        var rng: QuantumRNG = .seeded(11)
        let noise = NoiseModel(amplitudeDampingProbability: 1)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)
    }

    func testPhaseDampingDephasesSuperposition() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)

        var rng: QuantumRNG = .seeded(17)
        let noise = NoiseModel(phaseDampingProbability: 1)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        let expectation = try QuantumMeasurement.expectationX(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5)
    }

    func testReadoutErrorFlipsClassicalOutcomeNotState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)
        try engine.execute(circuit, on: state)

        var rng: QuantumRNG = .seeded(99)
        let noise = NoiseModel(readoutErrorProbability: 1)
        let measured = try QuantumMeasurement.measureRNG(state: state, engine: engine, rng: &rng, noise: noise)

        let zAfter = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(zAfter, -1, accuracy: 1e-5)
        XCTAssertEqual(measured, [0])
    }

    func testZeroNoisePreservesBellStateWithAllChannelsDisabled() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let idealState = try StateVector(qubitCount: 2, device: device)
        var idealCircuit = try QuantumCircuit(qubitCount: 2)
        try idealCircuit.applyBellState()
        try engine.execute(idealCircuit, on: idealState)
        let idealZZ = try QuantumMeasurement.expectationZZ(state: idealState, engine: engine, qubitA: 0, qubitB: 1)

        let noisyState = try StateVector(qubitCount: 2, device: device)
        var noisyCircuit = try QuantumCircuit(qubitCount: 2)
        try noisyCircuit.applyBellState()

        var rng: QuantumRNG = .seeded(42)
        let noise = NoiseModel()
        _ = try engine.executeRNG(noisyCircuit, on: noisyState, rng: &rng, noise: noise)

        let noisyZZ = try QuantumMeasurement.expectationZZ(state: noisyState, engine: engine, qubitA: 0, qubitB: 1)
        XCTAssertEqual(noisyZZ, idealZZ, accuracy: 1e-5)
    }

    func testBellStateEntanglement() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }
        let state = try StateVector(qubitCount: 2, device: device)

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🚀 QUANTUM COLLAPSE RESULT: \(result)")

        let isZeroZero = (result[0] == 0 && result[1] == 0)
        let isOneOne = (result[0] == 1 && result[1] == 1)

        XCTAssertTrue(isZeroZero || isOneOne, "Entanglement broken! Collapsed into an impossible state: \(result)")
    }

    func testQuantumFourierTransform() throws {
        let qubitCount = 3
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }
        let state = try StateVector(qubitCount: qubitCount, device: device)
        var circuit = try QuantumCircuit(qubitCount: qubitCount)

        try circuit.applyQFT()

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🌊 QFT COLLAPSE RESULT (3 Qubit): \(result)")

        XCTAssertEqual(result.count, qubitCount)
    }

    func testCCXGate() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 3, device: device)
        var circuit = try QuantumCircuit(qubitCount: 3)

        try circuit.x(0)
        try circuit.x(1)
        try circuit.ccx(0, 1, 2)

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🔺 CCX COLLAPSE RESULT: \(result)")

        XCTAssertEqual(result, [1, 1, 1], "CCX should flip target when both controls are |1>")
    }

    func testSwapGate() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)

        try circuit.x(0)
        try circuit.applySwap(q1: 0, q2: 1)

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🔄 SWAP COLLAPSE RESULT: \(result)")

        XCTAssertEqual(result, [1, 0], "SWAP should exchange qubit amplitudes")
    }

    func testModularExponentiationScaffold() throws {
        var circuit = try QuantumCircuit(qubitCount: 24)

        XCTAssertNoThrow(try circuit.applyModularExponentiation(
            a: 3,
            modulus: 7,
            controlRegister: 0...2,
            targetRegister: 3...7,
            ancillaRegister: 8...23
        ))
    }

    func testModularExponentiationRejectsInvalidParameters() throws {
        var circuit = try QuantumCircuit(qubitCount: 24)

        XCTAssertThrowsError(try circuit.applyModularExponentiation(
            a: 2,
            modulus: 1,
            controlRegister: 0...1,
            targetRegister: 2...5,
            ancillaRegister: 6...23
        )) { error in
            XCTAssertTrue(error is QuantumCircuitError)
        }

        XCTAssertThrowsError(try circuit.applyModularExponentiation(
            a: 2,
            modulus: 5,
            controlRegister: 0...3,
            targetRegister: 2...5,
            ancillaRegister: 6...23
        )) { error in
            XCTAssertTrue(error is QuantumCircuitError)
        }

        XCTAssertThrowsError(try circuit.applyModularExponentiation(
            a: -1,
            modulus: 5,
            controlRegister: 0...1,
            targetRegister: 2...5,
            ancillaRegister: 6...23
        )) { error in
            XCTAssertTrue(error is QuantumCircuitError)
        }
    }

    func testQuantumAdder() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 6, device: device)
        var circuit = try QuantumCircuit(qubitCount: 6)

        // Encode a = 1 (binary 01): set qubit 0 (LSB of registerA)
        try circuit.x(0)
        // Encode b = 2 (binary 10): set qubit 3 (MSB of registerB)
        try circuit.x(3)

        // registerA = [0, 1], registerB = [2, 3], carryIn = 4, carryOut = 5
        // Expected: 1 + 2 = 3 (binary 11) stored in registerB [2, 3]
        try circuit.applyQuantumAdd(
            registerA: [0, 1],
            registerB: [2, 3],
            carryIn: 4,
            carryOut: 5
        )

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        // Big-endian result layout for 6 qubits: result[k] = state of qubit(5 - k)
        //   result[0] = qubit 5 (carryOut)
        //   result[1] = qubit 4 (carryIn)
        //   result[2] = qubit 3 (registerB MSB)
        //   result[3] = qubit 2 (registerB LSB)
        //   result[4] = qubit 1 (registerA MSB)
        //   result[5] = qubit 0 (registerA LSB)
        print("➕ QUANTUM ADDER RESULT (1 + 2 = 3): \(result)")

        // Sum 3 = binary 11 in registerB (qubits 2 and 3)
        XCTAssertEqual(result[2], 1, "registerB MSB (qubit 3) should be 1 — sum = 3")
        XCTAssertEqual(result[3], 1, "registerB LSB (qubit 2) should be 1 — sum = 3")

        // Ancilla qubits restored
        XCTAssertEqual(result[0], 0, "carryOut (qubit 5) should be 0 — no overflow for 1+2")
        XCTAssertEqual(result[1], 0, "carryIn (qubit 4) should remain 0")

        // registerA preserved (a = 1 = binary 01: qubit 0 = 1, qubit 1 = 0)
        XCTAssertEqual(result[4], 0, "registerA MSB (qubit 1) should be restored to 0")
        XCTAssertEqual(result[5], 1, "registerA LSB (qubit 0) should be restored to 1")
    }

    func testQuantumSubtract() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 8, device: device)
        var circuit = try QuantumCircuit(qubitCount: 8)

        // Encode A = 2 (binary 010): set qubit 1 (middle bit of registerA)
        try circuit.x(1)
        // Encode B = 5 (binary 101): set qubits 3 (LSB) and 5 (MSB of registerB)
        try circuit.x(3)
        try circuit.x(5)

        // registerA = [0, 1, 2], registerB = [3, 4, 5], carryIn = 6, carryOut = 7
        // Expected: 5 - 2 = 3 (binary 011) stored in registerB [3, 4, 5]
        try circuit.applyQuantumSubtract(
            registerA: [0, 1, 2],
            registerB: [3, 4, 5],
            carryIn: 6,
            carryOut: 7
        )

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        // Big-endian result layout for 8 qubits: result[k] = state of qubit(7 - k)
        //   result[0] = qubit 7 (carryOut)
        //   result[1] = qubit 6 (carryIn)
        //   result[2] = qubit 5 (registerB MSB)
        //   result[3] = qubit 4 (registerB middle)
        //   result[4] = qubit 3 (registerB LSB)
        //   result[5] = qubit 2 (registerA MSB)
        //   result[6] = qubit 1 (registerA middle)
        //   result[7] = qubit 0 (registerA LSB)
        print("➖ QUANTUM SUBTRACT RESULT (5 - 2 = 3): \(result)")

        // Difference 3 = binary 011 in registerB (qubits 3, 4, 5)
        XCTAssertEqual(result[2], 0, "registerB MSB (qubit 5) should be 0 — difference = 3")
        XCTAssertEqual(result[3], 1, "registerB middle (qubit 4) should be 1 — difference = 3")
        XCTAssertEqual(result[4], 1, "registerB LSB (qubit 3) should be 1 — difference = 3")

        // carryIn restored
        XCTAssertEqual(result[1], 0, "carryIn (qubit 6) should remain 0")

        // registerA preserved (A = 2 = binary 010)
        XCTAssertEqual(result[5], 0, "registerA MSB (qubit 2) should be restored to 0")
        XCTAssertEqual(result[6], 1, "registerA middle (qubit 1) should be restored to 1")
        XCTAssertEqual(result[7], 0, "registerA LSB (qubit 0) should be restored to 0")
    }

    func testModularAdd() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 13, device: device)
        var circuit = try QuantumCircuit(qubitCount: 13)

        // Encode x = 4 (binary 0100): set qubit 2 (bit 2 of 4-bit registerX)
        try circuit.x(2)

        // registerX = [0, 1, 2, 3] (4-bit — holds intermediate x + a before reduction)
        // ancillaRegister = [4...12] → constantReg | carryInAdd | carryOutAdd | carryInSub | carryOutSub | c3xAncilla
        // Expected: (4 + 5) % 7 = 2 (binary 0010) in registerX
        try circuit.applyModularAdd(
            a: 5,
            modulus: 7,
            registerX: [0, 1, 2, 3],
            ancillaRegister: [4, 5, 6, 7, 8, 9, 10, 11, 12]
        )

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        // Big-endian result layout for 13 qubits: result[k] = state of qubit(12 - k)
        //   result[0]  = qubit 12 (c3xAncilla)
        //   result[1]  = qubit 11 (carryOutSub)
        //   result[2]  = qubit 10 (carryInSub)
        //   result[3]  = qubit 9  (carryOutAdd)
        //   result[4]  = qubit 8  (carryInAdd)
        //   result[5]  = qubit 7  (constantReg MSB)
        //   result[6]  = qubit 6  (constantReg bit 2)
        //   result[7]  = qubit 5  (constantReg bit 1)
        //   result[8]  = qubit 4  (constantReg LSB)
        //   result[9]  = qubit 3  (registerX MSB)
        //   result[10] = qubit 2  (registerX bit 2)
        //   result[11] = qubit 1  (registerX bit 1)
        //   result[12] = qubit 0  (registerX LSB)
        print("🔢 MODULAR ADD RESULT ((4 + 5) % 7 = 2): \(result)")

        // Result 2 = binary 0010 in registerX (qubits 0, 1, 2, 3)
        XCTAssertEqual(result[9], 0, "registerX MSB (qubit 3) should be 0 — (4 + 5) % 7 = 2")
        XCTAssertEqual(result[10], 0, "registerX bit 2 (qubit 2) should be 0 — (4 + 5) % 7 = 2")
        XCTAssertEqual(result[11], 1, "registerX bit 1 (qubit 1) should be 1 — (4 + 5) % 7 = 2")
        XCTAssertEqual(result[12], 0, "registerX LSB (qubit 0) should be 0 — (4 + 5) % 7 = 2")

        // Ancilla qubits restored (carryOutSub may hold subtract carry flag)
        XCTAssertEqual(result[0], 0, "c3xAncilla (qubit 12) should be restored to 0")
        XCTAssertEqual(result[2], 0, "carryInSub (qubit 10) should remain 0")
        XCTAssertEqual(result[4], 0, "carryInAdd (qubit 8) should remain 0")
        XCTAssertEqual(result[5], 0, "constantReg MSB (qubit 7) should be restored to 0")
        XCTAssertEqual(result[6], 0, "constantReg bit 2 (qubit 6) should be restored to 0")
        XCTAssertEqual(result[7], 0, "constantReg bit 1 (qubit 5) should be restored to 0")
        XCTAssertEqual(result[8], 0, "constantReg LSB (qubit 4) should be restored to 0")
    }

    func testCCXRejectsOutOfBoundsIndex() throws {
        var circuit = try QuantumCircuit(qubitCount: 3)

        XCTAssertThrowsError(try circuit.ccx(0, 1, 5)) { error in
            guard case QuantumCircuitError.qubitIndexOutOfBounds = error else {
                XCTFail("Expected qubitIndexOutOfBounds")
                return
            }
        }
    }

    func testSwapRejectsOutOfBoundsIndex() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)

        XCTAssertThrowsError(try circuit.applySwap(q1: 0, q2: 4)) { error in
            guard case QuantumCircuitError.qubitIndexOutOfBounds = error else {
                XCTFail("Expected qubitIndexOutOfBounds")
                return
            }
        }
    }
}
