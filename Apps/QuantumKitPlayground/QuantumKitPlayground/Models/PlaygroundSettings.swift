import Foundation
import QuantumKit

struct PlaygroundSettings: Equatable, Sendable {
    var shots: Int = 2048
    var seed: UInt64 = 7
    var useRandomSeed: Bool = false
    var devicePreference: SimulationDevicePreference = .automatic

    /// Seed passed to `QuantumRunOptions`. `nil` when the random-seed toggle is on.
    var effectiveSeed: UInt64? { useRandomSeed ? nil : seed }

    static let shotsRange = 1...100_000
    static let defaultShots = 2048
    static let defaultSeed: UInt64 = 7

    mutating func clampShots() {
        shots = min(max(shots, Self.shotsRange.lowerBound), Self.shotsRange.upperBound)
    }
}
