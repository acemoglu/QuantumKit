import Foundation
import XCTest
@testable import QuantumKit

/// I5-lite perf / memory **canaries** for CI — not hard SLAs or regression gates.
///
/// Soft thresholds: exceed → ``XCTSkip`` (or loosen via env). Only absurd hang bounds fail.
/// Override with `QUANTUMKIT_PERF_CANARY_LOOSE=1` (widen soft→loose) or
/// `QUANTUMKIT_SKIP_PERF_CANARIES=1` (skip all canary asserts).
enum PerfCanaryHarness {

    /// Wall-time bands (seconds). Soft = healthy host; loose = slow CI / thermal throttle.
    struct WallBudget: Sendable {
        let softSeconds: Double
        let looseSeconds: Double
        /// Absolute hang guard — only this fails the suite (not an SLA).
        let hangSeconds: Double

        static let cpuSV_n8_shallow = WallBudget(softSeconds: 0.25, looseSeconds: 4.0, hangSeconds: 30.0)
        static let cpuSV_n10_depth8 = WallBudget(softSeconds: 0.75, looseSeconds: 8.0, hangSeconds: 45.0)
        static let cpuSV_n12_shallow = WallBudget(softSeconds: 1.5, looseSeconds: 12.0, hangSeconds: 60.0)
    }

    /// Memory canary: estimated peak must stay within `maxMultiple` of the theoretical
    /// Double SV footprint (`2^n × 2 × 8` real+imag), with soft host headroom skip.
    struct MemoryBudget: Sendable {
        let qubitCount: Int
        /// Soft upper multiple of theoretical state bytes (estimate, not RSS).
        let maxMultipleOfStateBytes: Double
        /// Fail only if estimate exceeds this absurd multiple (corruption / policy bug).
        let hangMultipleOfStateBytes: Double

        static let cpuSV_n14 = MemoryBudget(
            qubitCount: 14,
            maxMultipleOfStateBytes: 8.0,
            hangMultipleOfStateBytes: 64.0
        )
    }

    static var skipAll: Bool {
        guard let raw = ProcessInfo.processInfo.environment["QUANTUMKIT_SKIP_PERF_CANARIES"] else {
            return false
        }
        return raw == "1" || raw.lowercased() == "true"
    }

    static var forceLoose: Bool {
        if let raw = ProcessInfo.processInfo.environment["QUANTUMKIT_PERF_CANARY_LOOSE"],
           raw == "1" || raw.lowercased() == "true" {
            return true
        }
        // Few logical cores → treat as slow host (widen soft to loose).
        return ProcessInfo.processInfo.activeProcessorCount <= 2
    }

    static func secondsSince(_ start: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
    }

    /// Theoretical CPU SV state buffer bytes (real + imag Doubles).
    static func theoreticalStateBytes(qubitCount: Int) -> Int {
        (1 << qubitCount) * 2 * MemoryLayout<Double>.stride
    }

    /// Soft-skip when physical RAM cannot cover estimated peak with 4× headroom.
    static func requireHostMemory(peakBytes: Int, label: String) throws {
        let available = ProcessInfo.processInfo.physicalMemory
        let required = UInt64(max(peakBytes, 1)) * 4
        if available < required {
            throw XCTSkip("\(label): need ~\(required) B headroom, physicalMemory=\(available)")
        }
    }

    /// Assert wall canary. Soft/loose exceed → skip (canary trip). Hang exceed → fail.
    static func assertWallCanary(
        elapsedSeconds: Double,
        budget: WallBudget,
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        if skipAll {
            throw XCTSkip("QUANTUMKIT_SKIP_PERF_CANARIES: \(label)")
        }
        XCTAssertGreaterThanOrEqual(elapsedSeconds, 0, file: file, line: line)
        XCTAssertLessThan(
            elapsedSeconds,
            budget.hangSeconds,
            "\(label): hang canary \(elapsedSeconds)s ≥ \(budget.hangSeconds)s (not an SLA)",
            file: file,
            line: line
        )
        let soft = forceLoose ? budget.looseSeconds : budget.softSeconds
        let loose = budget.looseSeconds
        if elapsedSeconds > loose {
            throw XCTSkip(
                "\(label): wall canary \(elapsedSeconds)s > loose \(loose)s — host slow; not an SLA"
            )
        }
        if elapsedSeconds > soft {
            throw XCTSkip(
                "\(label): wall canary \(elapsedSeconds)s > soft \(soft)s — loosen/skip; not an SLA"
            )
        }
    }

    /// Assert estimated peak memory stays in a soft band vs theoretical SV size.
    static func assertMemoryEstimateCanary(
        peakBytes: Int,
        budget: MemoryBudget,
        label: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        if skipAll {
            throw XCTSkip("QUANTUMKIT_SKIP_PERF_CANARIES: \(label)")
        }
        let state = theoreticalStateBytes(qubitCount: budget.qubitCount)
        XCTAssertGreaterThan(peakBytes, 0, file: file, line: line)
        let hangCap = Int(Double(state) * budget.hangMultipleOfStateBytes)
        XCTAssertLessThanOrEqual(
            peakBytes,
            hangCap,
            "\(label): peak \(peakBytes) ≫ hang cap \(hangCap) (policy bug?)",
            file: file,
            line: line
        )
        let softCap = Int(Double(state) * budget.maxMultipleOfStateBytes)
        if peakBytes > softCap {
            throw XCTSkip(
                "\(label): peak estimate \(peakBytes) > soft \(softCap) (×\(budget.maxMultipleOfStateBytes) of state); canary only"
            )
        }
    }

    /// Fixed shallow layered circuit for wall canaries (1Q rotations + adjacent CX).
    static func makeShallowLayeredCircuit(qubitCount n: Int, depth: Int, seed: UInt64) throws -> QuantumCircuit {
        var rng = QuantumRNG.seeded(seed)
        var circuit = try QuantumCircuit(qubitCount: n)
        for layer in 0..<depth {
            for q in 0..<n {
                switch (Int(seed) &+ layer &+ q) % 4 {
                case 0: try circuit.h(q)
                case 1: try circuit.rx(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
                case 2: try circuit.ry(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
                default: try circuit.rz(theta: QFloat(rng.nextUnitDouble() * Double.pi), q)
                }
            }
            for q in 0..<(n - 1) {
                try circuit.cx(q, q + 1)
            }
        }
        return circuit
    }
}
