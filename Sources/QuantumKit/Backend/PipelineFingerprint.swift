import Foundation

/// Stable fingerprint of a circuit + run configuration for reproducibility metadata (D4).
///
/// Portable FNV-1a 64-bit hash folded into a 16-character hex string. Uses a canonical
/// string form (not raw `JSONEncoder` defaults) so identical inputs always match.
public enum PipelineFingerprint {
    /// Hex digest over version, method, seed, shots, circuit structure, and noise snapshot.
    /// ``QuantumRunOptions/profiling`` is omitted: telemetry must not change reproducibility identity.
    public static func hash(
        circuit: QuantumCircuit,
        method: QuantumSimulationMethod,
        options: QuantumRunOptions
    ) -> String {
        var hasher = FNV64()
        hasher.combine(QuantumKitInfo.version)
        hasher.combine(method.rawValue)
        if let seed = options.seed {
            hasher.combine("seed:\(seed)")
        }
        if let shots = options.shots {
            hasher.combine("shots:\(shots)")
        }
        hasher.combine("qubits:\(circuit.qubitCount)")
        hasher.combine("cregs:\(circuit.classicalRegisters.map(\.bitCount))")
        for (index, gate) in circuit.gates.enumerated() {
            hasher.combine("\(index):\(String(describing: gate))")
        }
        if let noise = options.noise {
            hasher.combine(canonicalNoise(noise))
        }
        return String(format: "%016llx", hasher.finalize())
    }

    private static func canonicalNoise(_ noise: NoiseModel) -> String {
        var parts: [String] = [
            "dep:\(noise.depolarizingProbability)",
            "amp:\(noise.amplitudeDampingProbability)",
            "t1:\(noise.t1)",
            "t2:\(noise.t2)",
            "gt:\(noise.gateTime)",
            "phase:\(noise.phaseDampingProbability)",
            "p01:\(noise.readoutFlip0To1)",
            "p10:\(noise.readoutFlip1To0)",
            "reset:\(noise.resetErrorProbability)",
            "prep:\(noise.preparationErrorProbability)",
            "idleDelay:\(noise.thermalRelaxationOnDelay)",
            "mDeph:\(noise.measurementDephasingProbability)",
            "mMode:\(noise.measurementMode.rawValue)",
            "rules:\(noise.localizedRules.count)",
        ]
        if let matrix = noise.readoutConfusion {
            parts.append("conf:\(matrix.qubitCount):\(matrix.probabilities)")
        }
        for rule in noise.localizedRules {
            parts.append("rule:\(String(describing: rule.target)):\(String(describing: rule.channel))")
        }
        return parts.joined(separator: "|")
    }
}

private struct FNV64 {
    private var value: UInt64 = 0xcbf29ce484222325

    mutating func combine(_ string: String) {
        for byte in string.utf8 {
            value ^= UInt64(byte)
            value = value &* 0x100000001b3
        }
    }

    func finalize() -> UInt64 { value }
}
