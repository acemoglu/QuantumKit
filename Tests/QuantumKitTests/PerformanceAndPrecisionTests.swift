import XCTest
import Metal
import Foundation
@testable import QuantumKit

extension QuantumKitTests {
    private enum DispatchProfile: String, CaseIterable, Codable {
        case legacyMaxThreadgroup = "legacy-max"
        case refactoredSimdAligned = "refactored-simd"
    }

    private struct MicrobenchmarkResult: Codable {
        let gate: String
        let profile: DispatchProfile
        let qubitCount: Int
        let pairCount: Int
        let averageMilliseconds: Double
        let estimatedBandwidthGBps: Double
        let threadgroupWidth: Int
        let simdGroupsPerThreadgroup: Int
    }

    private struct BenchmarkExportPayload: Codable {
        let timestamp: String
        let deviceName: String
        let scales: [Int]
        let iterationsByScale: [String: Int]
        let bytesPerPair: Double
        let results: [MicrobenchmarkResult]
    }

    private struct BenchmarkKey: Hashable {
        let gate: String
        let profile: DispatchProfile
        let qubitCount: Int
    }

    private func threadgroupWidth(
        profile: DispatchProfile,
        pipeline: MTLComputePipelineState,
        pairCount: Int
    ) -> Int {
        switch profile {
        case .legacyMaxThreadgroup:
            return min(pipeline.maxTotalThreadsPerThreadgroup, max(pairCount, 1))
        case .refactoredSimdAligned:
            let simdWidth = max(1, pipeline.threadExecutionWidth)
            let maxWidth = pipeline.maxTotalThreadsPerThreadgroup
            let clamped = min(maxWidth, max(pairCount, simdWidth))
            let aligned = (clamped / simdWidth) * simdWidth
            return max(simdWidth, min(pairCount, aligned == 0 ? simdWidth : aligned))
        }
    }

    private func averageKernelTimeMs(
        engine: QuantumEngine,
        state: StateVector,
        pipeline: MTLComputePipelineState,
        pairCount: Int,
        profile: DispatchProfile,
        iterations: Int,
        setArguments: (MTLComputeCommandEncoder) -> Void
    ) throws -> (ms: Double, threadgroupWidth: Int) {
        let tgWidth = threadgroupWidth(profile: profile, pipeline: pipeline, pairCount: pairCount)
        let grid = MTLSize(width: pairCount, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        var samples: [Double] = []
        var fallbackCount = 0
        let warmupIterations = 2

        for sampleIndex in 0..<(iterations + warmupIterations) {
            guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw QuantumEngineError.commandBufferCreationFailed
            }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(state.realBuffer, offset: 0, index: 0)
            encoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
            setArguments(encoder)
            var pairCountValue = UInt32(pairCount)
            encoder.setBytes(
                &pairCountValue,
                length: MemoryLayout<UInt32>.stride,
                index: QuantumEngine.pairCountBufferIndex
            )
            encoder.setThreadgroupMemoryLength(
                MemoryLayout<SIMD4<Float>>.stride * tgWidth,
                index: 0
            )
            encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
            encoder.endEncoding()

            let wallStart = CFAbsoluteTimeGetCurrent()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
            }

            let gpuTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
            let measuredSeconds: Double
            if gpuTime.isFinite, gpuTime > 0 {
                measuredSeconds = gpuTime
            } else {
                measuredSeconds = CFAbsoluteTimeGetCurrent() - wallStart
                fallbackCount += 1
            }
            if sampleIndex >= warmupIterations {
                samples.append(measuredSeconds)
            }
        }

        if fallbackCount > 0 {
            print("ℹ️ GPU timestamp fallback count: \(fallbackCount) for profile \(profile.rawValue)")
        }
        guard !samples.isEmpty else {
            return (0.0, tgWidth)
        }
        let sorted = samples.sorted()
        let mid = sorted.count / 2
        let median = sorted.count.isMultiple(of: 2)
            ? (sorted[mid - 1] + sorted[mid]) * 0.5
            : sorted[mid]
        return (median * 1_000.0, tgWidth)
    }

    private func benchmarkOutputDirectory() throws -> URL {
        let fm = FileManager.default
        let outputDir: URL
        if let customDir = ProcessInfo.processInfo.environment["QUANTUMKIT_BENCHMARK_OUTPUT_DIR"], !customDir.isEmpty {
            outputDir = URL(fileURLWithPath: customDir, isDirectory: true)
        } else {
            outputDir = URL(fileURLWithPath: fm.currentDirectoryPath)
                .appendingPathComponent(".build", isDirectory: true)
                .appendingPathComponent("benchmark-artifacts", isDirectory: true)
        }
        try fm.createDirectory(at: outputDir, withIntermediateDirectories: true)
        return outputDir
    }

    private func mostRecentBaselinePayload(
        in outputDirectory: URL
    ) throws -> (payload: BenchmarkExportPayload, url: URL)? {
        let fm = FileManager.default
        let files = try fm.contentsOfDirectory(at: outputDirectory, includingPropertiesForKeys: [.contentModificationDateKey])
            .filter { $0.lastPathComponent.hasPrefix("pairwise-gate-bench-") && $0.pathExtension == "json" }

        guard !files.isEmpty else { return nil }

        let sorted = try files.sorted { lhs, rhs in
            let lhsDate = try lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? .distantPast
            let rhsDate = try rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? .distantPast
            return lhsDate > rhsDate
        }
        guard let latest = sorted.first else { return nil }
        let data = try Data(contentsOf: latest)
        let payload = try JSONDecoder().decode(BenchmarkExportPayload.self, from: data)
        return (payload, latest)
    }

    private func writeBenchmarkArtifacts(
        _ payload: BenchmarkExportPayload,
        into outputDir: URL
    ) throws -> (csv: URL, json: URL) {

        let stamp = payload.timestamp.replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: ".", with: "-")
        let csvURL = outputDir.appendingPathComponent("pairwise-gate-bench-\(stamp).csv")
        let jsonURL = outputDir.appendingPathComponent("pairwise-gate-bench-\(stamp).json")

        let csvHeader = "timestamp,device,gate,profile,qubits,pair_count,avg_ms,bandwidth_gbps,threadgroup_width,simd_groups_per_threadgroup\n"
        let csvRows = payload.results.map { result in
            "\(payload.timestamp),\(payload.deviceName),\(result.gate),\(result.profile.rawValue),\(result.qubitCount),\(result.pairCount),\(String(format: "%.6f", result.averageMilliseconds)),\(String(format: "%.6f", result.estimatedBandwidthGBps)),\(result.threadgroupWidth),\(result.simdGroupsPerThreadgroup)"
        }.joined(separator: "\n")
        try (csvHeader + csvRows + "\n").write(to: csvURL, atomically: true, encoding: .utf8)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(payload).write(to: jsonURL)
        return (csvURL, jsonURL)
    }

    private func renderBandwidthTable(
        results: [MicrobenchmarkResult],
        scales: [Int]
    ) {
        let sorted = results.sorted {
            if $0.gate != $1.gate { return $0.gate < $1.gate }
            if $0.profile.rawValue != $1.profile.rawValue { return $0.profile.rawValue < $1.profile.rawValue }
            return $0.qubitCount < $1.qubitCount
        }
        let groups = Dictionary(grouping: sorted, by: { "\($0.gate)|\($0.profile.rawValue)" })
        let orderedKeys = groups.keys.sorted()
        let scaleColumns = scales.map(String.init)
        let header = ["Gate", "Profile"] + scaleColumns + ["Q26/Q24"]
        let colWidths = [6, 16] + Array(repeating: 10, count: scaleColumns.count) + [10]
        let separator = colWidths.map { String(repeating: "-", count: $0) }.joined(separator: " ")

        func row(_ cells: [String]) -> String {
            zip(cells, colWidths).map { value, width in
                value.count >= width ? String(value.prefix(width)) : value + String(repeating: " ", count: width - value.count)
            }.joined(separator: " ")
        }

        print("📊 BANDWIDTH TABLE (GB/s)")
        print(row(header))
        print(separator)

        for key in orderedKeys {
            guard let rows = groups[key], let sample = rows.first else { continue }
            var scaleValues: [String] = []
            for scale in scales {
                if let metric = rows.first(where: { $0.qubitCount == scale }) {
                    scaleValues.append(String(format: "%.2f", metric.estimatedBandwidthGBps))
                } else {
                    scaleValues.append("n/a")
                }
            }
            let q24 = rows.first(where: { $0.qubitCount == 24 })?.estimatedBandwidthGBps
            let q26 = rows.first(where: { $0.qubitCount == 26 })?.estimatedBandwidthGBps
            let cliff = (q24 != nil && q26 != nil && q24! > 0) ? String(format: "%.3f", q26! / q24!) : "n/a"
            print(row([sample.gate, sample.profile.rawValue] + scaleValues + [cliff]))
        }
    }

    private func renderLatencyTable(
        results: [MicrobenchmarkResult],
        scales: [Int]
    ) {
        let sorted = results.sorted {
            if $0.gate != $1.gate { return $0.gate < $1.gate }
            if $0.profile.rawValue != $1.profile.rawValue { return $0.profile.rawValue < $1.profile.rawValue }
            return $0.qubitCount < $1.qubitCount
        }
        let groups = Dictionary(grouping: sorted, by: { "\($0.gate)|\($0.profile.rawValue)" })
        let orderedKeys = groups.keys.sorted()
        let scaleColumns = scales.map(String.init)
        let header = ["Gate", "Profile"] + scaleColumns + ["Q26/Q24"]
        let colWidths = [6, 16] + Array(repeating: 10, count: scaleColumns.count) + [10]
        let separator = colWidths.map { String(repeating: "-", count: $0) }.joined(separator: " ")

        func row(_ cells: [String]) -> String {
            zip(cells, colWidths).map { value, width in
                value.count >= width ? String(value.prefix(width)) : value + String(repeating: " ", count: width - value.count)
            }.joined(separator: " ")
        }

        print("⏱️ LATENCY TABLE (ms)")
        print(row(header))
        print(separator)

        for key in orderedKeys {
            guard let rows = groups[key], let sample = rows.first else { continue }
            var scaleValues: [String] = []
            for scale in scales {
                if let metric = rows.first(where: { $0.qubitCount == scale }) {
                    scaleValues.append(String(format: "%.4f", metric.averageMilliseconds))
                } else {
                    scaleValues.append("n/a")
                }
            }
            let q24 = rows.first(where: { $0.qubitCount == 24 })?.averageMilliseconds
            let q26 = rows.first(where: { $0.qubitCount == 26 })?.averageMilliseconds
            let slope = (q24 != nil && q26 != nil && q24! > 0) ? String(format: "%.3f", q26! / q24!) : "n/a"
            print(row([sample.gate, sample.profile.rawValue] + scaleValues + [slope]))
        }
    }

    private func regressionThresholdMultiplier() -> Double {
        if let raw = ProcessInfo.processInfo.environment["QUANTUMKIT_BENCHMARK_REGRESSION_THRESHOLD"],
           let parsed = Double(raw),
           parsed > 0.0, parsed <= 1.0 {
            return parsed
        }
        return 0.90
    }

    private func minBaselineLatencyForRegressionMs() -> Double {
        if let raw = ProcessInfo.processInfo.environment["QUANTUMKIT_BENCHMARK_MIN_BASELINE_LATENCY_MS"],
           let parsed = Double(raw),
           parsed >= 0 {
            return parsed
        }
        return 0.10
    }

    private func latencyAbsoluteToleranceMs() -> Double {
        if let raw = ProcessInfo.processInfo.environment["QUANTUMKIT_BENCHMARK_LATENCY_ABS_TOLERANCE_MS"],
           let parsed = Double(raw),
           parsed >= 0 {
            return parsed
        }
        return 0.10
    }

    private func bandwidthAbsoluteToleranceGBps() -> Double {
        if let raw = ProcessInfo.processInfo.environment["QUANTUMKIT_BENCHMARK_BANDWIDTH_ABS_TOLERANCE_GBPS"],
           let parsed = Double(raw),
           parsed >= 0 {
            return parsed
        }
        return 15.0
    }

    private func minQubitForRegressionCheck() -> Int {
        if let raw = ProcessInfo.processInfo.environment["QUANTUMKIT_BENCHMARK_REGRESSION_MIN_QUBIT"],
           let parsed = Int(raw),
           parsed >= 0 {
            return parsed
        }
        return 24
    }

    private func shouldEnforceRegressionFailures() -> Bool {
        if let raw = ProcessInfo.processInfo.environment["QUANTUMKIT_BENCHMARK_ENFORCE_REGRESSION"],
           ["1", "true", "yes", "on"].contains(raw.lowercased()) {
            return true
        }
        return false
    }

    private func validateRegressions(
        baseline: BenchmarkExportPayload,
        current: BenchmarkExportPayload,
        threshold: Double
    ) -> [String] {
        var failures: [String] = []
        let baselineMap = Dictionary(
            uniqueKeysWithValues: baseline.results.map {
                (BenchmarkKey(gate: $0.gate, profile: $0.profile, qubitCount: $0.qubitCount), $0)
            }
        )

        for metric in current.results {
            let key = BenchmarkKey(gate: metric.gate, profile: metric.profile, qubitCount: metric.qubitCount)
            guard let previous = baselineMap[key] else { continue }
            if metric.qubitCount < minQubitForRegressionCheck() {
                continue
            }

            // Skip extremely short baseline kernels where timer jitter dominates relative deltas.
            if previous.averageMilliseconds < minBaselineLatencyForRegressionMs() {
                continue
            }

            let minBandwidth = max(
                0.0,
                (previous.estimatedBandwidthGBps * threshold) - bandwidthAbsoluteToleranceGBps()
            )
            if metric.estimatedBandwidthGBps < minBandwidth {
                failures.append(
                    String(
                        format: "[REGRESSION DETECTED] %@/%@ at Q%d bandwidth: Expected >= %.2f GB/s ((baseline %.2f * %.2f) - %.2f), got %.2f GB/s",
                        metric.gate,
                        metric.profile.rawValue,
                        metric.qubitCount,
                        minBandwidth,
                        previous.estimatedBandwidthGBps,
                        threshold,
                        bandwidthAbsoluteToleranceGBps(),
                        metric.estimatedBandwidthGBps
                    )
                )
            }

            let maxLatency = (previous.averageMilliseconds / threshold) + latencyAbsoluteToleranceMs()
            if metric.averageMilliseconds > maxLatency {
                failures.append(
                    String(
                        format: "[REGRESSION DETECTED] %@/%@ at Q%d latency: Expected <= %.4f ms ((baseline %.4f / %.2f) + %.4f), got %.4f ms",
                        metric.gate,
                        metric.profile.rawValue,
                        metric.qubitCount,
                        maxLatency,
                        previous.averageMilliseconds,
                        threshold,
                        latencyAbsoluteToleranceMs(),
                        metric.averageMilliseconds
                    )
                )
            }
        }
        return failures
    }

    /// Multi-scale dispatch-level microbenchmark for pairwise gate kernels.
    ///
    /// Features:
    /// - Gate set includes H, CX, CCX, CRY, MCX, U
    /// - Scales across 20/22/24/26 qubits
    /// - Compares legacy-max vs refactored-simd dispatch profiles
    /// - Exports CSV + JSON artifacts for trend plotting
    func testPairwiseGateMicrobenchmarkDispatchProfiles() throws {
        let engine = try QuantumEngine()
        guard let device = makeDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }
        let outputDir = try benchmarkOutputDirectory()
        let baseline = try mostRecentBaselinePayload(in: outputDir)

        let scales = [20, 22, 24, 26]
        let iterationsByScale = [20: 8, 22: 6, 24: 4, 26: 2]
        let bytesPerPair = 32.0 // read 16B (r0,i0,r1,i1) + write 16B

        typealias CaseDef = (
            name: String,
            pipeline: MTLComputePipelineState,
            setArgs: (MTLComputeCommandEncoder) -> Void
        )

        let cases: [CaseDef] = [
            (
                name: "H",
                pipeline: engine.pipelines.hadamard,
                setArgs: { encoder in
                    var target: UInt32 = 5
                    encoder.setBytes(&target, length: MemoryLayout<UInt32>.stride, index: 2)
                }
            ),
            (
                name: "CX",
                pipeline: engine.pipelines.cnot,
                setArgs: { encoder in
                    var packed = SIMD2<UInt32>(x: UInt32(1) << 2, y: 5)
                    encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                }
            ),
            (
                name: "CCX",
                pipeline: engine.pipelines.ccx,
                setArgs: { encoder in
                    let mask = (UInt32(1) << 1) | (UInt32(1) << 2)
                    var packed = SIMD2<UInt32>(x: mask, y: 5)
                    encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                }
            ),
            (
                name: "CRY",
                pipeline: engine.pipelines.cRotY,
                setArgs: { encoder in
                    var packed = SIMD2<UInt32>(x: UInt32(1) << 2, y: 5)
                    encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                    var theta: Float = .pi / 3
                    encoder.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 3)
                }
            ),
            (
                name: "MCX",
                pipeline: engine.pipelines.mcx,
                setArgs: { encoder in
                    let mask = (UInt32(1) << 1) | (UInt32(1) << 2) | (UInt32(1) << 3) | (UInt32(1) << 4)
                    var packed = SIMD2<UInt32>(x: mask, y: 6)
                    encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                }
            ),
            (
                name: "U",
                pipeline: engine.pipelines.universal,
                setArgs: { encoder in
                    var target: UInt32 = 5
                    encoder.setBytes(&target, length: MemoryLayout<UInt32>.stride, index: 2)
                    var angles = SIMD3<Float>(x: .pi / 2, y: .pi / 5, z: .pi / 7)
                    encoder.setBytes(&angles, length: MemoryLayout<SIMD3<Float>>.stride, index: 3)
                }
            ),
        ]

        var results: [MicrobenchmarkResult] = []

        for qubitCount in scales {
            let iterations = iterationsByScale[qubitCount] ?? 4
            let state = try StateVector(qubitCount: qubitCount, device: device)
            let pairCount = state.stateCount / 2

            // Build a moderately non-trivial state so controlled-path arithmetic and phase paths are active.
            var prep = try QuantumCircuit(qubitCount: qubitCount)
            try prep.h(0)
            try prep.h(1)
            try prep.h(2)
            try prep.h(3)
            try engine.execute(prep, on: state)

            for benchmarkCase in cases {
                for profile in DispatchProfile.allCases {
                    let timing = try averageKernelTimeMs(
                        engine: engine,
                        state: state,
                        pipeline: benchmarkCase.pipeline,
                        pairCount: pairCount,
                        profile: profile,
                        iterations: iterations,
                        setArguments: benchmarkCase.setArgs
                    )
                    let seconds = timing.ms / 1_000.0
                    let totalBytes = Double(pairCount) * bytesPerPair
                    let bandwidthGBps = (totalBytes / seconds) / 1_000_000_000.0
                    let simdGroups = max(1, timing.threadgroupWidth / max(1, benchmarkCase.pipeline.threadExecutionWidth))
                    results.append(
                        MicrobenchmarkResult(
                            gate: benchmarkCase.name,
                            profile: profile,
                            qubitCount: qubitCount,
                            pairCount: pairCount,
                            averageMilliseconds: timing.ms,
                            estimatedBandwidthGBps: bandwidthGBps,
                            threadgroupWidth: timing.threadgroupWidth,
                            simdGroupsPerThreadgroup: simdGroups
                        )
                    )
                }
            }
        }

        print("📈 PAIRWISE GATE MICROBENCHMARK SWEEP")
        print("   scales=\(scales), bytes/pair=\(Int(bytesPerPair)), device=\(device.name)")
        renderBandwidthTable(results: results, scales: scales)
        renderLatencyTable(results: results, scales: scales)

        let timestamp = ISO8601DateFormatter().string(from: Date())
        let payload = BenchmarkExportPayload(
            timestamp: timestamp,
            deviceName: device.name,
            scales: scales,
            iterationsByScale: Dictionary(uniqueKeysWithValues: iterationsByScale.map { (String($0.key), $0.value) }),
            bytesPerPair: bytesPerPair,
            results: results
        )
        let artifacts = try writeBenchmarkArtifacts(payload, into: outputDir)
        print("🗂️ benchmark-csv: \(artifacts.csv.path)")
        print("🗂️ benchmark-json: \(artifacts.json.path)")

        let threshold = regressionThresholdMultiplier()
        print(String(format: "🛡️ regression-threshold: %.2f", threshold))
        print(
            String(
                format: "🛡️ regression-noise-floor: min-qubit=%d, baseline-latency>=%.3f ms, bandwidth-abs-tol=%.2f GB/s, latency-abs-tol=%.3f ms, enforce=%@",
                minQubitForRegressionCheck(),
                minBaselineLatencyForRegressionMs(),
                bandwidthAbsoluteToleranceGBps(),
                latencyAbsoluteToleranceMs(),
                shouldEnforceRegressionFailures() ? "yes" : "no"
            )
        )
        if let baseline {
            print("📎 baseline-json: \(baseline.url.path)")
            if baseline.payload.deviceName == payload.deviceName {
                let regressions = validateRegressions(
                    baseline: baseline.payload,
                    current: payload,
                    threshold: threshold
                )
                if regressions.isEmpty {
                    print("✅ regression-check: no regressions detected")
                } else {
                    for message in regressions {
                        print(message)
                    }
                    if shouldEnforceRegressionFailures() {
                        XCTFail("Benchmark regression guard failed with \(regressions.count) issue(s).")
                    } else {
                        print("⚠️ regression-check: regressions detected but not enforced (set QUANTUMKIT_BENCHMARK_ENFORCE_REGRESSION=1 to fail)")
                    }
                }
            } else {
                print("⚠️ regression-check skipped: baseline device '\(baseline.payload.deviceName)' != current '\(payload.deviceName)'")
            }
        } else {
            print("⚠️ regression-check skipped: no baseline JSON artifact found")
        }

        // Guard rails: benchmark must produce finite positive values.
        XCTAssertFalse(results.isEmpty)
        for result in results {
            XCTAssertGreaterThan(result.averageMilliseconds, 0)
            XCTAssert(result.estimatedBandwidthGBps.isFinite)
            XCTAssertGreaterThan(result.estimatedBandwidthGBps, 0)
        }
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
}
