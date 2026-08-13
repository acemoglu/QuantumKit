import Foundation

extension NoiseModel {

    /// Builds a noise model from hardware calibration data.
    ///
    /// Per-qubit T1/T2 are mapped to localized amplitude and phase damping on every gate
    /// touching that qubit. Gate-specific error rates become localized depolarizing channels.
    /// Readout asymmetry is stored in the global readout fields (applied at measurement).
    public static func from(calibration: DeviceCalibration) -> NoiseModel {
        var rules: [LocalizedNoiseRule] = []
        rules.reserveCapacity(calibration.qubits.count * 2 + calibration.gateErrors.count)

        for index in calibration.qubits.indices {
            let props = calibration.qubits[index]
            if props.t1 > 0, calibration.gateTime > 0 {
                rules.append(
                    LocalizedNoiseRule(
                        target: .allGatesOnQubit(index),
                        channel: .amplitudeDamping(t1: props.t1, gateTime: calibration.gateTime)
                    )
                )
            }
            if props.t2 > 0, calibration.gateTime > 0 {
                rules.append(
                    LocalizedNoiseRule(
                        target: .allGatesOnQubit(index),
                        channel: .phaseDamping(
                            t1: props.t1,
                            t2: props.t2,
                            gateTime: calibration.gateTime
                        )
                    )
                )
            }
        }

        for gateError in calibration.gateErrors where gateError.errorRate > 0 {
            rules.append(
                LocalizedNoiseRule(
                    target: gateError.noiseTarget,
                    channel: .depolarizing(probability: gateError.errorRate)
                )
            )
        }

        var readoutFlip0To1: QFloat = 0
        var readoutFlip1To0: QFloat = 0
        if !calibration.qubits.isEmpty {
            readoutFlip0To1 = calibration.qubits.map(\.readoutError0To1).max() ?? 0
            readoutFlip1To0 = calibration.qubits.map(\.readoutError1To0).max() ?? 0
        }

        return NoiseModel(
            gateTime: calibration.gateTime,
            readoutFlip0To1: readoutFlip0To1,
            readoutFlip1To0: readoutFlip1To0,
            localizedRules: rules
        )
    }

    /// Returns a copy with an additional localized noise rule.
    public func adding(_ channel: QuantumChannel, for target: NoiseTarget) -> NoiseModel {
        adding(LocalizedNoiseRule(target: target, channel: channel))
    }

    /// Returns a copy with an additional localized noise rule.
    public func adding(_ rule: LocalizedNoiseRule) -> NoiseModel {
        var copy = self
        copy.localizedRules.append(rule)
        return copy
    }

    /// Adds bidirectional nearest-neighbor spectator crosstalk (C5): a gate on `a` applies
    /// `channel` on `b` and vice versa, for every undirected edge of `couplingMap`.
    public func addingNearestNeighborCrosstalk(
        couplingMap: CouplingMap,
        channel: QuantumChannel
    ) -> NoiseModel {
        var copy = self
        for (a, b) in couplingMap.edges {
            copy = copy
                .adding(channel, for: .crosstalk(driven: a, spectator: b))
                .adding(channel, for: .crosstalk(driven: b, spectator: a))
        }
        return copy
    }

    /// Convenience: nearest-neighbor spectator depolarizing with probability `probability`.
    public func addingNearestNeighborCrosstalk(
        couplingMap: CouplingMap,
        probability: QFloat
    ) -> NoiseModel {
        addingNearestNeighborCrosstalk(
            couplingMap: couplingMap,
            channel: .depolarizing(probability: probability)
        )
    }

    /// `true` when any localized rule applies gate-time stochastic channels.
    public var hasLocalizedGateNoise: Bool {
        localizedRules.contains { $0.channel.isGateChannel }
    }

    /// Localized rules that match `gate` after execution at `gateIndex`.
    func matchingLocalizedRules(
        for gate: Gate,
        affectedQubits: [Int],
        gateIndex: Int
    ) -> [LocalizedNoiseRule] {
        localizedRules.filter {
            $0.target.matches(gate: gate, affectedQubits: affectedQubits, gateIndex: gateIndex)
        }
    }
}
