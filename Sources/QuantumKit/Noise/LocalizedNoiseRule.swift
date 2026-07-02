import Foundation

/// A single localized noise rule: apply `channel` when `target` matches a executed gate.
public struct LocalizedNoiseRule: Sendable, Equatable, Codable {
    public let target: NoiseTarget
    public let channel: QuantumChannel

    public init(target: NoiseTarget, channel: QuantumChannel) {
        self.target = target
        self.channel = channel
    }
}
