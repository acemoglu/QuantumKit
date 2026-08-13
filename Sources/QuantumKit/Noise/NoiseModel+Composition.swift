import Foundation

extension NoiseModel {

    /// Builds a noise model from hardware calibration data (C3).
    ///
    /// Per-qubit T1/T2 become localized amplitude/phase damping. Gate error rates become
    /// localized depolarizing. Per-qubit readout asymmetry is preserved as a tensor-product
    /// ``ReadoutConfusionMatrix`` (qubit 0 = LSB). Optional coupling-map crosstalk is attached
    /// when ``DeviceCalibration/nearestNeighborCrosstalkProbability`` is positive.
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

        var model = NoiseModel(
            gateTime: calibration.gateTime,
            localizedRules: rules
        )

        if calibration.qubitCount > 0 {
            let pairs: [(p01: QFloat, p10: QFloat)] = (0..<calibration.qubitCount).map { index in
                let props = calibration[qubit: index]
                return (p01: props.readoutError0To1, p10: props.readoutError1To0)
            }
            if let confusion = try? ReadoutConfusionMatrix.product(of: pairs) {
                model.readoutConfusion = confusion
            }
            // Keep max globals as fallback when a measure width ≠ calibration qubitCount
            // (confusion only applies on exact width match).
            model.readoutFlip0To1 = pairs.map(\.p01).max() ?? 0
            model.readoutFlip1To0 = pairs.map(\.p10).max() ?? 0
        }

        if let map = calibration.couplingMap,
           calibration.nearestNeighborCrosstalkProbability > 0 {
            model = model.addingNearestNeighborCrosstalk(
                couplingMap: map,
                probability: calibration.nearestNeighborCrosstalkProbability
            )
        }

        return model
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

    /// Layered composition (C1): concatenate localized rules (`self` then `other`) and take
    /// the element-wise max of shared global probabilities. `other`'s non-default
    /// measurement mode / confusion matrix wins when set.
    public func merging(_ other: NoiseModel) -> NoiseModel {
        var copy = self
        copy.depolarizingProbability = max(depolarizingProbability, other.depolarizingProbability)
        copy.amplitudeDampingProbability = max(amplitudeDampingProbability, other.amplitudeDampingProbability)
        copy.phaseDampingProbability = max(phaseDampingProbability, other.phaseDampingProbability)
        copy.t1 = max(t1, other.t1)
        copy.t2 = max(t2, other.t2)
        copy.gateTime = max(gateTime, other.gateTime)
        copy.readoutFlip0To1 = max(readoutFlip0To1, other.readoutFlip0To1)
        copy.readoutFlip1To0 = max(readoutFlip1To0, other.readoutFlip1To0)
        copy.resetErrorProbability = max(resetErrorProbability, other.resetErrorProbability)
        copy.preparationErrorProbability = max(preparationErrorProbability, other.preparationErrorProbability)
        copy.thermalRelaxationOnDelay = thermalRelaxationOnDelay || other.thermalRelaxationOnDelay
        copy.measurementDephasingProbability = max(
            measurementDephasingProbability,
            other.measurementDephasingProbability
        )
        if other.measurementMode != .projective {
            copy.measurementMode = other.measurementMode
        }
        if let confusion = other.readoutConfusion {
            copy.readoutConfusion = confusion
        } else if copy.readoutConfusion == nil {
            copy.readoutConfusion = readoutConfusion
        }
        copy.localizedRules = localizedRules + other.localizedRules
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
