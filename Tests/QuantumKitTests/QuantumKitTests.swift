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

    /// Kahan/double-single CDF — yüksek kübitte Float32'nin "plateau" (yutulma) çöküşünü aşıyor.
    ///
    /// 25 kübitte her durumun olasılığı ≈ 2⁻²⁵'tir. Kümülatif toplam (CDF) 0.5'i aştığında
    /// [0.5, 1.0) aralığının Float32 ulp'si 2⁻²⁴ olur; eklenecek 2⁻²⁵'lik olasılıklar bu ulp'nin
    /// yarısı kadar olduğu için yuvarlanarak YUTULUR. Sonuç: ardışık CDF değerleri birbirinin aynısı
    /// çıkar (plateau), toplam 1.0'a ulaşamaz ve binary-search bu durumları asla seçemez.
    ///
    /// Kahan (compensated) toplama, her adımda Float32'nin attığı düşük-mertebe bitleri bir telafi
    /// değişkeni `c`'de biriktirip bir sonraki adımda geri ekler. Böylece toplam, neredeyse Double
    /// doğruluğunda 1.0'a yürür. Bu test, GPU'nun GERÇEK olasılıklarını alıp aynı veriyi üç farklı
    /// şekilde toplayarak hem çöküşü hem de kurtarışı ölçer, ve gerçek bir ölçümün çökmediğini
    /// (zeroStateNorm fırlatmadığını) doğrular.
    func testKahanCDFSurvivesFloat32PlateauAt25Qubits() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let qubitCount = 25
        let stateCount = 1 << qubitCount

        // Tüm kübitlere H → yoğun, neredeyse uniform süperpozisyon (durum başına olasılık ≈ 2⁻²⁵).
        func makeUniformState() throws -> StateVector {
            let state = try StateVector(qubitCount: qubitCount, device: device)
            var circuit = try QuantumCircuit(qubitCount: qubitCount)
            for qubit in 0..<qubitCount {
                try circuit.h(qubit)
            }
            try engine.execute(circuit, on: state)
            return state
        }

        // GPU'nun kendi ürettiği Float32 olasılıkları (amplitüd yuvarlamasını da içerir) — ortak girdi.
        let probabilities = try QuantumMeasurement.probabilities(state: try makeUniformState(), engine: engine)

        // (1) NAİF Float32 toplamı: tek bir Float akümülatör. Plateau tam burada doğar.
        // (2) KAHAN Float32 toplamı: GPU çekirdeğindeki register-telafisinin host eşdeğeri; `c`
        //     yutulan bitleri taşır. Swift, Float işlemlerini IEEE'ye sadık tuttuğu için (fast-math
        //     yeniden-ilişkilendirmesi yok) burada `volatile`'a gerek kalmaz.
        // (3) DOUBLE toplamı: "yer gerçeği" referansı (≈ 1.0).
        var naiveFloat: Float = 0
        var kahanSum: Float = 0
        var kahanComp: Float = 0   // telafi değişkeni — kavramsal olarak bir register'da yaşar
        var doubleTotal = 0.0
        var plateauCount = 0
        var previousNaive: Float = 0

        for index in 0..<stateCount {
            let p = probabilities[index]

            // (1) naif birikim + plateau tespiti: yeni eklenen olasılık tamamen yutulduysa
            // kümülatif değer hiç değişmez → ardışık CDF değerleri eşit (düzlük).
            previousNaive = naiveFloat
            naiveFloat += p
            if index > 0 && p > 0 && naiveFloat == previousNaive {
                plateauCount += 1
            }

            // (2) Kahan: y = düzeltilmiş katkı, t = yeni toplam, c = bu adımda yutulan kalıntı.
            let y = p - kahanComp
            let t = kahanSum + y
            kahanComp = (t - kahanSum) - y
            kahanSum = t

            // (3) referans
            doubleTotal += Double(p)
        }

        let naiveDeficit = doubleTotal - Double(naiveFloat)
        let kahanDeficit = abs(doubleTotal - Double(kahanSum))

        print("🧮 25-QUBIT CDF PRECISION")
        print("   Double toplam (referans): \(doubleTotal)")
        print("   Naïf Float32 toplam      : \(naiveFloat)  (eksik: \(naiveDeficit))")
        print("   Kahan  Float32 toplam    : \(kahanSum)  (eksik: \(kahanDeficit))")
        print("   Plateau (yutulan) nokta sayısı: \(plateauCount)")

        // Referans gerçekten ~1.0 (üniter devre normu korur).
        XCTAssertEqual(doubleTotal, 1.0, accuracy: 1e-3, "H^⊗25 normu ≈ 1.0 olmalı.")

        // ÇÖKÜŞ: Float32 toplamı yutulma yüzünden 1.0'ın belirgin biçimde altında kalır ve
        // binlerce plateau noktası üretir.
        XCTAssertGreaterThan(plateauCount, 1000,
            "25 kübitte naif Float32 CDF'nin çok sayıda plateau üretmesi beklenir; senaryo aksi halde önemsizdir.")
        XCTAssertGreaterThan(naiveDeficit, 1e-3,
            "Naif Float32 toplamı, yutulma nedeniyle 1.0'dan kayda değer biçimde eksik kalmalı.")

        // KURTARIŞ: Kahan toplamı, naif Float32'den en az iki kat daha doğru ve Double'a çok yakın.
        XCTAssertLessThan(kahanDeficit, 1e-5,
            "Kahan toplamı Double referansına ~tam doğrulukta ulaşmalı.")
        XCTAssertLessThan(kahanDeficit, naiveDeficit / 100,
            "Kahan, naif Float32'ye göre büyük bir doğruluk sıçraması sağlamalı.")

        // UÇTAN UCA: Aynı plateau bölgesinde, GPU'nun double-single CDF + 53-bit dice ile yaptığı
        // gerçek collapse, Double-doğrulukta referans CDF ile BİREBİR aynı durumu seçmeli (saf
        // Float32 olsaydı plateau yüzünden komşuya kayardı).
        func firstIndexAbove(_ dice: Double, in cdf: [Double]) -> Int {
            var low = 0, high = cdf.count - 1, result = cdf.count - 1
            while low <= high {
                let mid = (low + high) / 2
                if dice < cdf[mid] { result = mid; if mid == 0 { break }; high = mid - 1 }
                else { low = mid + 1 }
            }
            return result
        }
        var cdfDouble = [Double](repeating: 0, count: stateCount)
        var running = 0.0
        for index in 0..<stateCount { running += Double(probabilities[index]); cdfDouble[index] = running }

        for dice in [0.25, 0.5, 0.75, 0.99, 0.999999] as [Double] {
            let expected = firstIndexAbove(dice, in: cdfDouble)
            let collapsed = try engine.executeMeasurementCollapse(on: try makeUniformState(), dice: dice)
            XCTAssertEqual(collapsed, expected,
                "Telafili GPU collapse (dice \(dice)), Double-doğrulukta CDF ile aynı durumu vermeli.")
        }

        // SAĞLAMLIK: Gerçek bir ölçüm bu yoğun süperpozisyonda çökmemeli (zeroStateNorm yok),
        // ve geçerli bir 25-bitlik sonuç dönmeli.
        var rng: QuantumRNG = .seeded(2026)
        let measured = try QuantumMeasurement.measureRNG(state: try makeUniformState(), engine: engine, rng: &rng)
        XCTAssertEqual(measured.count, qubitCount)
        XCTAssertTrue(measured.allSatisfy { $0 == 0 || $0 == 1 })
    }

    /// CDF/scan aşamasının donanımsal maliyeti: naif Float32 vs Kahan (double-single).
    ///
    /// Prefix-sum bant-genişliği sınırlı (memory-bound) bir işlemdir; baskın maliyet 2²⁵ elemanlık
    /// tamponları okuyup yazmaktır, aritmetik değil. Telafili sürüm fazladan birkaç toplama (register
    /// içinde) ve bir `lo` telafi tamponu trafiği ekler. Bu test her iki scan'i aynı uniform girdi
    /// üzerinde 10'ar kez çalıştırıp ortalama süreleri ms cinsinden karşılaştırır.
    func testPrefixSumScanBenchmarkNaiveVsKahan() throws {
        let engine = try QuantumEngine()

        guard makeDevice() != nil else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let result = try engine.benchmarkPrefixSumScan(qubitCount: 25, iterations: 10)

        print("⏱️  CDF/SCAN BENCHMARK (25 qubit, \(result.iterations) iterasyon)")
        print(String(format: "   Naïf  Float32      : %.4f ms (ortalama)", result.naiveMillisecondsAverage))
        print(String(format: "   Kahan double-single: %.4f ms (ortalama)", result.compensatedMillisecondsAverage))
        print(String(format: "   Ek maliyet         : %.2f %%", result.overheadPercent))

        // Süreler anlamlı (ölçüm gerçekten çalıştı).
        XCTAssertGreaterThan(result.naiveMillisecondsAverage, 0)
        XCTAssertGreaterThan(result.compensatedMillisecondsAverage, 0)
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

    func testBatchRunSampleCountsMatchesSequentialRNG() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()

        var sequentialRNG: QuantumRNG = .seeded(42)
        let sequential = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            device: device,
            shots: 256,
            rng: &sequentialRNG,
            options: SampleCountOptions(batchSize: 1)
        )

        var batchedRNG: QuantumRNG = .seeded(42)
        let batched = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            device: device,
            shots: 256,
            rng: &batchedRNG,
            options: SampleCountOptions(batchSize: 32)
        )

        XCTAssertEqual(sequential, batched)
    }

    func testExecuteUnitaryBatchMatchesSequential() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.cx(0, 1)

        let sequentialState = try StateVector(qubitCount: 2, device: device)
        try engine.execute(circuit, on: sequentialState)

        let batchStateA = try StateVector(qubitCount: 2, device: device)
        let batchStateB = try StateVector(qubitCount: 2, device: device)
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

    func testPhaseDampingMatchesPhaseFlipChannel() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // Phase damping of strength λ is exactly the phase-flip channel, which decays the
        // coherence (⟨X⟩) of |+⟩ by a factor √(1 - λ) in the ensemble average.
        let lambda: QFloat = 0.5
        let expectedMeanX = (1 - lambda).squareRoot()

        let trajectories = 4000
        var rng: QuantumRNG = .seeded(123_456)
        let noise = NoiseModel(phaseDampingProbability: lambda)

        var accumulatedX: QFloat = 0
        for _ in 0..<trajectories {
            let state = try StateVector(qubitCount: 1, device: device)
            var circuit = try QuantumCircuit(qubitCount: 1)
            try circuit.h(0)
            _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
            accumulatedX += try QuantumMeasurement.expectationX(state: state, engine: engine, qubit: 0)
        }

        let meanX = accumulatedX / QFloat(trajectories)
        XCTAssertEqual(meanX, expectedMeanX, accuracy: 0.05)
    }

    func testPhaseDampingFullStrengthRemovesCoherence() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let trajectories = 4000
        var rng: QuantumRNG = .seeded(98_765)
        let noise = NoiseModel(phaseDampingProbability: 1)

        var accumulatedX: QFloat = 0
        for _ in 0..<trajectories {
            let state = try StateVector(qubitCount: 1, device: device)
            var circuit = try QuantumCircuit(qubitCount: 1)
            try circuit.h(0)
            _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)
            accumulatedX += try QuantumMeasurement.expectationX(state: state, engine: engine, qubit: 0)
        }

        let meanX = accumulatedX / QFloat(trajectories)
        XCTAssertEqual(meanX, 0, accuracy: 0.05)
    }

    func testAmplitudeDampingPreservesClassicalCorrelationOfBellState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // On the Bell state (|00⟩ + |11⟩)/√2, full amplitude damping on qubit 0 must always
        // drive qubit 0 to |0⟩ while leaving qubit 1 in a definite basis state (|0⟩ if no jump,
        // |1⟩ if a jump occurred). The old additive kernel incorrectly left qubit 1 in a
        // coherent superposition (⟨Z₁⟩ ≈ 0); the correct σ⁻ jump keeps |⟨Z₁⟩| = 1.
        let noise = NoiseModel(amplitudeDampingProbability: 1)

        for seed in UInt64(1)...UInt64(8) {
            let state = try StateVector(qubitCount: 2, device: device)
            var circuit = try QuantumCircuit(qubitCount: 2)
            try circuit.applyBellState(control: 0, target: 1)

            var rng: QuantumRNG = .seeded(seed)
            _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

            let z0 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
            let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)

            XCTAssertEqual(z0, 1, accuracy: 1e-5, "qubit 0 must relax to |0⟩ under full amplitude damping")
            XCTAssertEqual(abs(z1), 1, accuracy: 1e-5, "qubit 1 must remain in a definite basis state")
        }
    }

    func testT1GateTimeAmplitudeDampingProbability() {
        let gateTime = QFloat(0.69314718)
        let noise = NoiseModel(t1: 1, gateTime: gateTime)
        XCTAssertTrue(noise.usesT1TimeModel)
        XCTAssertEqual(noise.effectiveAmplitudeDampingProbability, 0.5, accuracy: 1e-5)
    }

    func testT1GateTimeResetsExcitedQubit() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        var rng: QuantumRNG = .seeded(11)
        let noise = NoiseModel(t1: 1, gateTime: 10)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)
    }

    func testAsymmetricReadoutErrorFlipsDirectionally() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let oneState = try StateVector(qubitCount: 1, device: device)
        var oneCircuit = try QuantumCircuit(qubitCount: 1)
        try oneCircuit.x(0)
        try engine.execute(oneCircuit, on: oneState)

        var rngOne: QuantumRNG = .seeded(42)
        let flipOneToZero = NoiseModel(readoutFlip0To1: 0, readoutFlip1To0: 1)
        let measuredOne = try QuantumMeasurement.measureRNG(
            state: oneState,
            engine: engine,
            rng: &rngOne,
            noise: flipOneToZero
        )
        XCTAssertEqual(measuredOne, [0])

        let zeroState = try StateVector(qubitCount: 1, device: device)
        var rngZero: QuantumRNG = .seeded(42)
        let flipZeroToOne = NoiseModel(readoutFlip0To1: 1, readoutFlip1To0: 0)
        let measuredZero = try QuantumMeasurement.measureRNG(
            state: zeroState,
            engine: engine,
            rng: &rngZero,
            noise: flipZeroToOne
        )
        XCTAssertEqual(measuredZero, [1])
    }

    func testMidCircuitMeasureWithReadoutFlipPreservesCollapsedState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)

        var rng: QuantumRNG = .seeded(123)
        let noise = NoiseModel(readoutFlip0To1: 0, readoutFlip1To0: 1)
        let execution = try engine.executeRNG(circuit, on: state, rng: &rng, noise: noise)

        XCTAssertEqual(execution.measurementOutcomes.count, 1)
        XCTAssertEqual(execution.measurementOutcomes[0][0], 0)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        let trueBit = expectation > 0 ? 0 : 1
        XCTAssertEqual(expectation, trueBit == 0 ? 1 : -1, accuracy: 1e-5)
        if trueBit == 1 {
            XCTAssertEqual(execution.measurementOutcomes[0][0], 0)
        }
    }

    func testRunSampleCountsWithReadoutNoiseOnly() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.x(0)

        var rng: QuantumRNG = .seeded(7)
        let noise = NoiseModel(readoutFlip0To1: 0, readoutFlip1To0: 1)
        let counts = try QuantumMeasurement.runSampleCountsRNG(
            circuit: circuit,
            engine: engine,
            device: device,
            shots: 1,
            rng: &rng,
            noise: noise
        )

        XCTAssertEqual(counts.shots, 1)
        XCTAssertEqual(counts.counts[0], 1)
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

        // Deterministic 1→0 readout flip: the classical bit must flip while the |1⟩ state is left
        // untouched. (A symmetric readoutErrorProbability of 1 is a 50/50 coin per bit, so asserting
        // a specific measured value would depend on exact RNG draws rather than on the channel.)
        var rng: QuantumRNG = .seeded(99)
        let noise = NoiseModel(readoutFlip1To0: 1)
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

    func testRyPiRotatesZeroToOne() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.ry(theta: QFloat(Double.pi), 0)

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)
        XCTAssertEqual(result, [1], "RY(pi) should rotate |0> to |1>")
    }

    func testSGateMatchesRzPiOverTwo() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let sState = try StateVector(qubitCount: 1, device: device)
        var sCircuit = try QuantumCircuit(qubitCount: 1)
        try sCircuit.h(0)
        try sCircuit.s(0)
        try engine.execute(sCircuit, on: sState)

        let rzState = try StateVector(qubitCount: 1, device: device)
        var rzCircuit = try QuantumCircuit(qubitCount: 1)
        try rzCircuit.h(0)
        try rzCircuit.rz(theta: QFloat(Double.pi / 2.0), 0)
        try engine.execute(rzCircuit, on: rzState)

        let sProbabilities = try QuantumMeasurement.probabilities(state: sState, engine: engine)
        let rzProbabilities = try QuantumMeasurement.probabilities(state: rzState, engine: engine)

        XCTAssertEqual(sProbabilities.count, rzProbabilities.count)
        for index in 0..<sProbabilities.count {
            XCTAssertEqual(sProbabilities[index], rzProbabilities[index], accuracy: 1e-5)
        }
    }

    func testTGateMatchesRzPiOverFour() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let tState = try StateVector(qubitCount: 1, device: device)
        var tCircuit = try QuantumCircuit(qubitCount: 1)
        try tCircuit.h(0)
        try tCircuit.t(0)
        try engine.execute(tCircuit, on: tState)

        let rzState = try StateVector(qubitCount: 1, device: device)
        var rzCircuit = try QuantumCircuit(qubitCount: 1)
        try rzCircuit.h(0)
        try rzCircuit.rz(theta: QFloat(Double.pi / 4.0), 0)
        try engine.execute(rzCircuit, on: rzState)

        let tProbabilities = try QuantumMeasurement.probabilities(state: tState, engine: engine)
        let rzProbabilities = try QuantumMeasurement.probabilities(state: rzState, engine: engine)

        XCTAssertEqual(tProbabilities.count, rzProbabilities.count)
        for index in 0..<tProbabilities.count {
            XCTAssertEqual(tProbabilities[index], rzProbabilities[index], accuracy: 1e-5)
        }
    }

    func testTSquaredEqualsS() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let ttState = try StateVector(qubitCount: 1, device: device)
        var ttCircuit = try QuantumCircuit(qubitCount: 1)
        try ttCircuit.h(0)
        try ttCircuit.t(0)
        try ttCircuit.t(0)
        try engine.execute(ttCircuit, on: ttState)

        let sState = try StateVector(qubitCount: 1, device: device)
        var sCircuit = try QuantumCircuit(qubitCount: 1)
        try sCircuit.h(0)
        try sCircuit.s(0)
        try engine.execute(sCircuit, on: sState)

        let ttProbabilities = try QuantumMeasurement.probabilities(state: ttState, engine: engine)
        let sProbabilities = try QuantumMeasurement.probabilities(state: sState, engine: engine)

        XCTAssertEqual(ttProbabilities.count, sProbabilities.count)
        for index in 0..<ttProbabilities.count {
            XCTAssertEqual(ttProbabilities[index], sProbabilities[index], accuracy: 1e-5)
        }
    }

    // MARK: - Extended gate set (Wave A)

    func testSDaggerInvertsS() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.s(0)
        try circuit.sdg(0)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5, "S·S† = I should return |0⟩")
    }

    func testTDaggerInvertsT() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.t(0)
        try circuit.tdg(0)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, 1, accuracy: 1e-5, "T·T† = I should return |0⟩")
    }

    func testSXSquaredEqualsX() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.sx(0)
        try circuit.sx(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5, "SX·SX = X should map |0⟩ → |1⟩")
    }

    func testSingleSXProducesEqualSuperposition() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.sx(0)
        try engine.execute(circuit, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probabilities[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 0.5, accuracy: 1e-5)
    }

    func testPhaseGatePiActsAsZ() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.p(theta: QFloat(Double.pi), 0)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5, "H·P(π)·H = X should map |0⟩ → |1⟩")
    }

    func testControlledZActsAsZWhenControlIsOne() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)            // control = |1⟩
        try circuit.h(1)            // target = |+⟩
        try circuit.cz(0, 1)        // acts as Z on target
        try circuit.h(1)            // H·Z·H = X → target = |1⟩
        try engine.execute(circuit, on: state)

        let z0 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z0, -1, accuracy: 1e-5)
        XCTAssertEqual(z1, -1, accuracy: 1e-5)
    }

    func testControlledZLeavesTargetWhenControlIsZero() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(1)            // target = |+⟩, control stays |0⟩
        try circuit.cz(0, 1)        // no-op since control = |0⟩
        try circuit.h(1)            // H·I·H = I → target back to |0⟩
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, 1, accuracy: 1e-5)
    }

    func testNativeSwapExchangesQubits() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)            // qubit0 = |1⟩, qubit1 = |0⟩
        try circuit.swap(0, 1)      // → qubit0 = |0⟩, qubit1 = |1⟩
        try engine.execute(circuit, on: state)

        let z0 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z0, 1, accuracy: 1e-5)
        XCTAssertEqual(z1, -1, accuracy: 1e-5)
    }

    func testUniversalGateReproducesPauliX() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.u(theta: QFloat(Double.pi), phi: 0, lambda: QFloat(Double.pi), 0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5, "U(π,0,π) = X should map |0⟩ → |1⟩")
    }

    func testUniversalGateReproducesHadamard() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.u(theta: QFloat(Double.pi / 2), phi: 0, lambda: QFloat(Double.pi), 0)
        try engine.execute(circuit, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probabilities[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 0.5, accuracy: 1e-5)

        let expectationX = try QuantumMeasurement.expectationX(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectationX, 1, accuracy: 1e-5, "U(π/2,0,π) = H should produce |+⟩")
    }

    func testUniversalGateMatchesPhaseGate() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        // U(0,0,λ) = P(λ): H · U(0,0,π) · H should behave like H · Z · H = X.
        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.u(theta: 0, phi: 0, lambda: QFloat(Double.pi), 0)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, -1, accuracy: 1e-5, "U(0,0,π) = P(π) = Z")
    }

    // MARK: - Extended gate set (Wave B: controlled & multi-controlled)

    func testControlledRYRotatesTargetWhenControlIsOne() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.cry(theta: QFloat(Double.pi), control: 0, target: 1)
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, -1, accuracy: 1e-5, "CRY(π) with control |1⟩ should map target |0⟩ → |1⟩")
    }

    func testControlledRYLeavesTargetWhenControlIsZero() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.cry(theta: QFloat(Double.pi), control: 0, target: 1)
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, 1, accuracy: 1e-5, "CRY with control |0⟩ should leave the target untouched")
    }

    func testControlledRXExcitesTargetWhenControlIsOne() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.crx(theta: QFloat(Double.pi), control: 0, target: 1)
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, -1, accuracy: 1e-5, "CRX(π) with control |1⟩ should fully excite the target")
    }

    func testControlledRZActsAsZWhenControlIsOne() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.h(1)
        try circuit.crz(theta: QFloat(Double.pi), control: 0, target: 1)
        try circuit.h(1)
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, -1, accuracy: 1e-5, "H·CRZ(π)·H with control |1⟩ should flip the target")
    }

    func testControlledPhasePiMatchesControlledZ() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.x(0)
        try circuit.h(1)
        try circuit.cp(theta: QFloat(Double.pi), control: 0, target: 1)
        try circuit.h(1)
        try engine.execute(circuit, on: state)

        let z1 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 1)
        XCTAssertEqual(z1, -1, accuracy: 1e-5, "CP(π) should behave like CZ")
    }

    func testMultiControlledXFlipsTargetWhenAllControlsAreOne() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 4, device: device)
        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.x(0)
        try circuit.x(1)
        try circuit.x(2)
        try circuit.mcx(controls: [0, 1, 2], target: 3)
        try engine.execute(circuit, on: state)

        let z3 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 3)
        XCTAssertEqual(z3, -1, accuracy: 1e-5, "MCX should flip the target when all controls are |1⟩")
    }

    func testMultiControlledXLeavesTargetWhenAControlIsZero() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 4, device: device)
        var circuit = try QuantumCircuit(qubitCount: 4)
        try circuit.x(0)
        try circuit.x(1)
        // qubit 2 stays |0⟩
        try circuit.mcx(controls: [0, 1, 2], target: 3)
        try engine.execute(circuit, on: state)

        let z3 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 3)
        XCTAssertEqual(z3, 1, accuracy: 1e-5, "MCX must not flip the target when a control is |0⟩")
    }

    func testMultiControlledZActsAsZWhenAllControlsAreOne() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else { XCTFail("Apple Silicon GPU not found!"); return }

        let state = try StateVector(qubitCount: 3, device: device)
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.x(0)
        try circuit.x(1)
        try circuit.h(2)
        try circuit.mcz(controls: [0, 1], target: 2)
        try circuit.h(2)
        try engine.execute(circuit, on: state)

        let z2 = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 2)
        XCTAssertEqual(z2, -1, accuracy: 1e-5, "H·MCZ·H with all controls |1⟩ should flip the target")
    }

    // MARK: - Algebraic pre-compiler

    func testAlgebraicPreCompilerCancelsDoubleHadamard() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.h(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.originalGateCount, 2)
        XCTAssertEqual(result.optimizedGateCount, 0)
        XCTAssertTrue(result.gates.isEmpty)
    }

    func testAlgebraicPreCompilerSSBecomesZ() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.s(0)
        try circuit.s(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.z(target: 0)])
    }

    func testAlgebraicPreCompilerTTBecomesS() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.t(0)
        try circuit.t(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.s(target: 0)])
    }

    func testAlgebraicPreCompilerMergesAdjacentRotations() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        let quarter = QFloat(Double.pi / 4.0)
        try circuit.rx(theta: quarter, 0)
        try circuit.rx(theta: quarter, 0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.rx(theta: QFloat(Double.pi / 2.0), target: 0)])
    }

    func testAlgebraicPreCompilerCancelsSWithSDagger() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.s(0)
        try circuit.sdg(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 0, "S·S† should cancel completely")
    }

    func testAlgebraicPreCompilerCancelsTWithTDagger() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.t(0)
        try circuit.tdg(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 0, "T·T† should cancel completely")
    }

    func testAlgebraicPreCompilerMergesPhaseGatesIntoS() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        let quarter = QFloat(Double.pi / 4.0)
        try circuit.p(theta: quarter, 0)
        try circuit.p(theta: quarter, 0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.s(target: 0)], "P(π/4)·P(π/4) = P(π/2) = S")
    }

    func testAlgebraicPreCompilerMergesTWithPhaseIntoSDagger() throws {
        // T (π/4) followed by P(-3π/4) sums to -π/2 → S†.
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.t(0)
        try circuit.p(theta: QFloat(-3.0 * Double.pi / 4.0), 0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.sdg(target: 0)])
    }

    func testAlgebraicPreCompilerCancelsDoubleCZ() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.cz(0, 1)
        try circuit.cz(0, 1)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 0, "CZ·CZ should cancel completely")
    }

    func testAlgebraicPreCompilerCancelsDoubleSwap() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.swap(0, 1)
        try circuit.swap(0, 1)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 0, "SWAP·SWAP should cancel completely")
    }

    func testAlgebraicPreCompilerSlidesZAxisGatesThroughCZ() throws {
        // S(0), CZ(0,1), S(0): the two S gates commute through CZ and fold into Z.
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.s(0)
        try circuit.cz(0, 1)
        try circuit.s(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 2)
        // Z and CZ commute, so either ordering is a valid optimization.
        XCTAssertTrue(result.gates.contains(.z(target: 0)))
        XCTAssertTrue(result.gates.contains(.cz(control: 0, target: 1)))
    }

    func testCommutationSlidesHadamardThroughDisjointPauli() throws {
        let gates: [Gate] = [.h(target: 0), .x(target: 1), .h(target: 0)]

        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.x(1)
        try circuit.h(0)

        let slid = AlgebraicPreCompiler.slideGates(gates)
        XCTAssertEqual(slid, [.x(target: 1), .h(target: 0), .h(target: 0)])

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.x(target: 1)])
    }

    func testCommutationDoesNotCancelSameQubitHadamardSandwich() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.x(0)
        try circuit.h(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 3)
    }

    func testCommutationDoesNotSlidePastMeasurement() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.h(0)
        try circuit.measure(1)
        try circuit.h(0)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 3)
        XCTAssertEqual(result.gates, circuit.gates)
    }

    func testCommutationSlidesZThroughCXOnTarget() throws {
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.z(1)
        try circuit.cx(0, 1)
        try circuit.z(1)

        let result = AlgebraicPreCompiler.optimize(gates: circuit.gates)
        XCTAssertEqual(result.optimizedGateCount, 1)
        XCTAssertEqual(result.gates, [.cx(control: 0, target: 1)])
    }

    func testAlgebraicPreCompilerPreservesBellStateProbabilities() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        var original = try QuantumCircuit(qubitCount: 2)
        try original.applyBellState()

        var redundant = try QuantumCircuit(qubitCount: 2)
        try redundant.applyBellState()
        try redundant.h(0)
        try redundant.h(0)
        try redundant.cx(0, 1)
        try redundant.cx(0, 1)

        let optimized = try redundant.algebraicallyOptimized()

        let originalState = try StateVector(qubitCount: 2, device: device)
        try engine.execute(original, on: originalState)

        let optimizedState = try StateVector(qubitCount: 2, device: device)
        try engine.execute(optimized, on: optimizedState)

        let originalProbabilities = try QuantumMeasurement.probabilities(state: originalState, engine: engine)
        let optimizedProbabilities = try QuantumMeasurement.probabilities(state: optimizedState, engine: engine)

        XCTAssertLessThan(optimized.gates.count, redundant.gates.count)
        XCTAssertEqual(originalProbabilities.count, optimizedProbabilities.count)
        for index in 0..<originalProbabilities.count {
            XCTAssertEqual(originalProbabilities[index], optimizedProbabilities[index], accuracy: 1e-5)
        }
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

    // MARK: - Shor accuracy (N = 15 = 3 × 5)

    func testShorFactors15Accuracy() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let modulus = 15
        let base = 7
        let expectedPeriod = try ShorClassical.multiplicativeOrder(base: base, modulus: modulus)
        XCTAssertEqual(expectedPeriod, 4, "Precondition: multiplicative order of 7 mod 15 is 4")

        // 4 counting qubits suffice for r = 4 (phase peaks at k/4); more controls explode gate depth.
        let controlQubitCount = 4
        let targetQubitCount = 4
        let ancillaQubitCount = (targetQubitCount * 2) + 6
        let qubitCount = controlQubitCount + targetQubitCount + ancillaQubitCount

        let controlRange = 0..<controlQubitCount
        let targetRange = controlQubitCount..<(controlQubitCount + targetQubitCount)
        let ancillaRange = (controlQubitCount + targetQubitCount)..<qubitCount

        var circuit = try QuantumCircuit(qubitCount: qubitCount)

        // |y⟩ = |1⟩
        try circuit.x(targetRange.lowerBound)

        for control in controlRange {
            try circuit.h(control)
        }

        try circuit.applyModularExponentiation(
            a: base,
            modulus: modulus,
            controlRegister: controlRange.lowerBound...controlRange.upperBound - 1,
            targetRegister: targetRange.lowerBound...targetRange.upperBound - 1,
            ancillaRegister: ancillaRange.lowerBound...ancillaRange.upperBound - 1
        )

        try circuit.applyInverseQFT(qubits: controlRange)

        let state = try StateVector(qubitCount: qubitCount, device: device)
        try engine.execute(circuit, on: state)

        let shots = 512
        var rng: QuantumRNG = .seeded(0x5100_0015)
        let counts = try QuantumMeasurement.sampleCountsRNG(
            state: state,
            engine: engine,
            qubits: Array(controlRange),
            shots: shots,
            rng: &rng
        )

        let analysis = ShorClassical.analyze(
            counts: counts,
            controlQubitCount: controlQubitCount,
            base: base,
            modulus: modulus
        )

        print("🔐 SHOR N=15: recovered periods \(analysis.recoveredPeriods.sorted()), factors \(analysis.foundFactors.sorted())")

        XCTAssertTrue(
            analysis.recoveredPeriods.contains(expectedPeriod),
            "Expected to recover classical period r=\(expectedPeriod) from QFT peaks; got \(analysis.recoveredPeriods)"
        )
        XCTAssertEqual(
            analysis.foundFactors,
            [3, 5],
            "Shor post-processing should factor 15 into 3 and 5"
        )
    }

    // MARK: - Task 1: amplitudes()

    func testAmplitudesOnPlusState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let amplitudes = QuantumMeasurement.amplitudes(state: state)
        let invSqrt2 = QFloat(1.0 / 2.0.squareRoot())

        XCTAssertEqual(amplitudes.count, 2)
        XCTAssertEqual(amplitudes[0].real, invSqrt2, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[0].imaginary, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[1].real, invSqrt2, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[1].imaginary, 0, accuracy: 1e-5)
    }

    func testAmplitudesOnBellState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try engine.execute(circuit, on: state)

        let amplitudes = QuantumMeasurement.amplitudes(state: state)
        let invSqrt2 = QFloat(1.0 / 2.0.squareRoot())

        XCTAssertEqual(amplitudes.count, 4)
        XCTAssertEqual(amplitudes[0].real, invSqrt2, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[0].imaginary, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[1].real, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[1].imaginary, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[2].real, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[2].imaginary, 0, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[3].real, invSqrt2, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[3].imaginary, 0, accuracy: 1e-5)
    }

    // MARK: - Task 1: sxdg gate

    func testSXDaggerInvertsSX() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.sx(0)
        try circuit.sxdg(0)
        try engine.execute(circuit, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probabilities[0], 1, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 0, accuracy: 1e-5)
    }

    func testSXDaggerSquaredIsPauliX() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // SX†·SX† = X (up to global phase): |0> → |1>.
        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.sxdg(0)
        try circuit.sxdg(0)
        try engine.execute(circuit, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probabilities[0], 0, accuracy: 1e-5)
        XCTAssertEqual(probabilities[1], 1, accuracy: 1e-5)
    }

    // MARK: - Task 1: inverse()

    func testInverseReturnsToZeroState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // A mixed circuit exercising self-inverse, dagger-paired, and parametrized gates.
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.t(1)
        try circuit.sx(2)
        try circuit.rx(theta: 0.7, 0)
        try circuit.ry(theta: -1.3, 1)
        try circuit.rz(theta: 2.1, 2)
        try circuit.cx(0, 1)
        try circuit.cz(1, 2)
        try circuit.crx(theta: 0.9, control: 0, target: 2)
        try circuit.cp(theta: 1.1, control: 1, target: 0)
        try circuit.s(2)
        try circuit.u(theta: 0.5, phi: 1.2, lambda: -0.8, 1)
        try circuit.ccx(0, 1, 2)

        let inverse = try circuit.inverse()

        let state = try StateVector(qubitCount: 3, device: device)
        try engine.execute(circuit, on: state)
        try engine.execute(inverse, on: state)

        let probabilities = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probabilities[0], 1, accuracy: 1e-4)
        for index in 1..<probabilities.count {
            XCTAssertEqual(probabilities[index], 0, accuracy: 1e-4)
        }
    }

    func testInverseGateOrderIsReversedAndAdjointed() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.s(0)
        try circuit.t(0)
        try circuit.sx(0)

        let inverse = try circuit.inverse()

        XCTAssertEqual(inverse.gates, [.sxdg(target: 0), .tdg(target: 0), .sdg(target: 0)])
    }

    func testInverseThrowsForNonUnitaryCircuit() throws {
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try circuit.measure(0)

        XCTAssertThrowsError(try circuit.inverse()) { error in
            guard case QuantumCircuitError.circuitNotUnitary = error else {
                XCTFail("Expected circuitNotUnitary")
                return
            }
        }
    }

    // MARK: - Task 2: general Pauli expectation

    func testPauliExpectationXOnPlusState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let expectation = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .x])
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)

        // Cross-validate against the dedicated single-qubit ⟨X⟩.
        let reference = try QuantumMeasurement.expectationX(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, reference, accuracy: 1e-5)
    }

    func testPauliExpectationZOnZeroState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 1, device: device)
        // |0⟩: no gates applied.
        let expectation = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .z])
        XCTAssertEqual(expectation, 1, accuracy: 1e-5)

        let reference = try QuantumMeasurement.expectationZ(state: state, engine: engine, qubit: 0)
        XCTAssertEqual(expectation, reference, accuracy: 1e-5)
    }

    func testPauliExpectationOnBellState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try engine.execute(circuit, on: state)

        let xx = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .x, 1: .x])
        let yy = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .y, 1: .y])
        let zz = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .z, 1: .z])

        XCTAssertEqual(xx, 1, accuracy: 1e-5)
        XCTAssertEqual(yy, -1, accuracy: 1e-5)
        XCTAssertEqual(zz, 1, accuracy: 1e-5)

        // ⟨Z₀Z₁⟩ must agree with the existing ZZ helper.
        let referenceZZ = try QuantumMeasurement.expectationZZ(state: state, engine: engine, qubitA: 0, qubitB: 1)
        XCTAssertEqual(zz, referenceZZ, accuracy: 1e-5)
    }

    func testPauliExpectationIdentityFactorsAreIgnored() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let state = try StateVector(qubitCount: 2, device: device)
        var circuit = try QuantumCircuit(qubitCount: 2)
        try circuit.applyBellState()
        try engine.execute(circuit, on: state)

        // ⟨Z₀⟩ on the Bell state is 0; an explicit identity on qubit 1 changes nothing.
        let withIdentity = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .z, 1: .i])
        let withoutIdentity = try QuantumMeasurement.expectation(state: state, engine: engine, paulis: [0: .z])

        XCTAssertEqual(withIdentity, 0, accuracy: 1e-5)
        XCTAssertEqual(withIdentity, withoutIdentity, accuracy: 1e-5)
    }

    func testPauliExpectationStringMatchesDictionary() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // |+⟩ on qubit 0, |1⟩ on qubit 1, |0⟩ on qubit 2.
        let state = try StateVector(qubitCount: 3, device: device)
        var circuit = try QuantumCircuit(qubitCount: 3)
        try circuit.h(0)
        try circuit.x(1)
        try engine.execute(circuit, on: state)

        // MSB-first label "ZZX" → qubit2=Z, qubit1=Z, qubit0=X.
        let fromString = try QuantumMeasurement.expectation(state: state, engine: engine, pauliString: "ZZX")
        let fromDict = try QuantumMeasurement.expectation(
            state: state,
            engine: engine,
            paulis: [2: .z, 1: .z, 0: .x]
        )

        XCTAssertEqual(fromString, fromDict, accuracy: 1e-5)
        // ⟨Z₂⟩=+1 (|0⟩), ⟨Z₁⟩=−1 (|1⟩), ⟨X₀⟩=+1 (|+⟩) ⇒ product −1.
        XCTAssertEqual(fromString, -1, accuracy: 1e-5)
    }

    func testPauliExpectationRejectsWrongLengthString() throws {
        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let engine = try QuantumEngine()
        let state = try StateVector(qubitCount: 2, device: device)

        XCTAssertThrowsError(try QuantumMeasurement.expectation(state: state, engine: engine, pauliString: "XYZ")) { error in
            guard case QuantumMeasurementError.invalidPauliString = error else {
                XCTFail("Expected invalidPauliString")
                return
            }
        }
    }

    // MARK: - Task 3: Deutsch-Jozsa

    func testDeutschJozsaConstantOracleYieldsAllZeros() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let inputs = [0, 1, 2]
        let ancilla = 3
        let state = try StateVector(qubitCount: 4, device: device)
        var circuit = try QuantumCircuit(qubitCount: 4)

        // Constant f(x) = 0: the oracle does nothing at all.
        try circuit.applyDeutschJozsa(inputQubits: inputs, ancilla: ancilla) { _ in }
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: inputs)
        XCTAssertEqual(marginal[0], 1, accuracy: 1e-5, "Constant function must collapse inputs to all-zeros")
    }

    func testDeutschJozsaConstantOneOracleYieldsAllZeros() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let inputs = [0, 1, 2]
        let ancilla = 3
        let state = try StateVector(qubitCount: 4, device: device)
        var circuit = try QuantumCircuit(qubitCount: 4)

        // Constant f(x) = 1: flip the ancilla unconditionally (global phase only).
        try circuit.applyDeutschJozsa(inputQubits: inputs, ancilla: ancilla) { c in
            try c.x(ancilla)
        }
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: inputs)
        XCTAssertEqual(marginal[0], 1, accuracy: 1e-5, "Constant function must collapse inputs to all-zeros")
    }

    func testDeutschJozsaBalancedOracleYieldsNonZero() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let inputs = [0, 1, 2]
        let ancilla = 3
        let state = try StateVector(qubitCount: 4, device: device)
        var circuit = try QuantumCircuit(qubitCount: 4)

        // Balanced f(x) = parity(x): CNOT each input into the ancilla.
        try circuit.applyDeutschJozsa(inputQubits: inputs, ancilla: ancilla) { c in
            for input in inputs {
                try c.cx(input, ancilla)
            }
        }
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: inputs)
        // Parity oracle collapses every input qubit to |1⟩ → outcome 0b111 = 7.
        XCTAssertEqual(marginal[7], 1, accuracy: 1e-5, "Balanced parity function must yield all-ones")
        XCTAssertEqual(marginal[0], 0, accuracy: 1e-5, "Balanced function must not yield all-zeros")
    }

    // MARK: - Task 3: Grover

    func testGroverOptimalIterations() {
        XCTAssertEqual(QuantumCircuit.groverOptimalIterations(searchSpaceSize: 8, markedCount: 1), 2)
        XCTAssertEqual(QuantumCircuit.groverOptimalIterations(searchSpaceSize: 16, markedCount: 1), 3)
        XCTAssertEqual(QuantumCircuit.groverOptimalIterations(searchSpaceSize: 4, markedCount: 1), 1)
    }

    func testGroverFindsMarkedState() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        let qubits = [0, 1, 2]
        let target = 5 // 0b101: qubit0 = 1, qubit1 = 0, qubit2 = 1
        let state = try StateVector(qubitCount: 3, device: device)
        var circuit = try QuantumCircuit(qubitCount: 3)

        // Phase oracle marking |101⟩: open-control with X on the zero bits, MCZ, then undo the X's.
        let markOracle: (inout QuantumCircuit) throws -> Void = { c in
            try c.x(1)
            try c.mcz(controls: [0, 1], target: 2)
            try c.x(1)
        }

        let iterations = QuantumCircuit.groverOptimalIterations(searchSpaceSize: 8, markedCount: 1)
        try circuit.applyGrover(qubits: qubits, iterations: iterations, oracle: markOracle)
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: qubits)
        let peak = marginal.enumerated().max(by: { $0.element < $1.element })!
        XCTAssertEqual(peak.offset, target, "Grover should amplify the marked state |101⟩ = 5")
        XCTAssertGreaterThan(marginal[target], 0.9, "Marked state should dominate the distribution")
    }

    // MARK: - Task 3: Quantum Phase Estimation

    func testPhaseEstimationRecoversTGatePhase() throws {
        let engine = try QuantumEngine()

        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }

        // 3 counting qubits [0,1,2]; eigenstate qubit 3 prepared in |1⟩, eigenvalue of T = e^{2πi·(1/8)}.
        let counting = [0, 1, 2]
        let eigenstate = 3
        let state = try StateVector(qubitCount: 4, device: device)
        var circuit = try QuantumCircuit(qubitCount: 4)

        try circuit.x(eigenstate) // |1⟩ eigenstate of T

        // Controlled-T^power on the eigenstate qubit ≡ controlled phase of (π/4)·power on |1⟩.
        let controlledPower: (inout QuantumCircuit, Int, Int) throws -> Void = { c, control, power in
            let theta = QFloat(Double.pi / 4.0 * Double(power))
            try c.cp(theta: theta, control: control, target: eigenstate)
        }

        try circuit.applyPhaseEstimation(countingRegister: counting, controlledUnitaryPower: controlledPower)
        try engine.execute(circuit, on: state)

        let marginal = try QuantumMeasurement.partialProbabilities(state: state, engine: engine, qubits: counting)
        // φ = 1/8 with t = 3 → exact estimate m = 1.
        XCTAssertEqual(marginal[1], 1, accuracy: 1e-4, "QPE should read m = 1 (phase 1/8) deterministically")
    }
}
