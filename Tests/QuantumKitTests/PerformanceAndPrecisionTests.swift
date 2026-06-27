import XCTest
import Metal
@testable import QuantumKit

extension QuantumKitTests {

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
}
