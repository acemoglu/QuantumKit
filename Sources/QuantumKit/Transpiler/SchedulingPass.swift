import Foundation

/// Non-pulse gate duration table for ASAP/ALAP scheduling.
public struct GateDurationTable: Sendable, Equatable, Codable {
    /// Default duration used when a kind has no explicit entry.
    public var defaultDuration: QFloat
    public var durations: [GateKind: QFloat]

    public init(defaultDuration: QFloat = 1, durations: [GateKind: QFloat] = [:]) {
        self.defaultDuration = max(defaultDuration, 0)
        self.durations = durations.mapValues { max($0, 0) }
    }

    public func duration(for gate: Gate) -> QFloat {
        if case .delay(let duration, _) = gate {
            return max(duration, 0)
        }
        return durations[gate.kind] ?? defaultDuration
    }

    /// Uniform duration for every gate kind (except explicit ``Gate/delay`` values).
    public static func uniform(_ duration: QFloat) -> GateDurationTable {
        GateDurationTable(defaultDuration: duration)
    }
}

/// Timeline placement policy for inserting ``Gate/delay`` fillers.
public enum SchedulingMethod: String, Sendable, Equatable, Codable, CaseIterable {
    /// Push gates as early as qubit timelines allow.
    case asap
    /// Push gates as late as possible against the global makespan.
    case alap
}

/// Inserts ``Gate/delay`` ops so idle gaps match a coupling-aware (per-qubit) timeline.
///
/// Explicitly **not** a pulse / OpenPulse engine — only discrete duration metadata that
/// feeds ``NoiseModel/thermalRelaxationOnDelay`` idle channels.
public struct SchedulingPass: CompilerPass, Sendable {
    public let durations: GateDurationTable
    public let method: SchedulingMethod

    public init(durations: GateDurationTable = .uniform(1), method: SchedulingMethod = .asap) {
        self.durations = durations
        self.method = method
    }

    public func run(on circuit: QuantumCircuit) throws -> QuantumCircuit {
        switch method {
        case .asap:
            return try scheduleASAP(circuit)
        case .alap:
            return try scheduleALAP(circuit)
        }
    }

    private func scheduleASAP(_ circuit: QuantumCircuit) throws -> QuantumCircuit {
        var qubitTime = Array(repeating: QFloat(0), count: circuit.qubitCount)
        var output = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )

        for gate in circuit.gates {
            if case .barrier(let qubits) = gate {
                let resolved = qubits.isEmpty ? Array(0..<circuit.qubitCount) : qubits
                let t = resolved.map { qubitTime[$0] }.max() ?? 0
                for q in resolved where q >= 0 && q < circuit.qubitCount {
                    let idle = t - qubitTime[q]
                    if idle > 0 {
                        try output.apply(.delay(duration: idle, qubit: q))
                    }
                    qubitTime[q] = t
                }
                try output.apply(gate)
                continue
            }
            if case .delay = gate {
                try output.apply(gate)
                let q = gate.affectedQubits[0]
                qubitTime[q] += durations.duration(for: gate)
                continue
            }

            let qubits = gate.affectedQubits
            guard !qubits.isEmpty else {
                try output.apply(gate)
                continue
            }

            let start = qubits.map { qubitTime[$0] }.max() ?? 0
            for q in qubits {
                let idle = start - qubitTime[q]
                if idle > 0 {
                    try output.apply(.delay(duration: idle, qubit: q))
                }
            }
            try output.apply(gate)
            let finish = start + durations.duration(for: gate)
            for q in qubits {
                qubitTime[q] = finish
            }
        }

        return output
    }

    private func scheduleALAP(_ circuit: QuantumCircuit) throws -> QuantumCircuit {
        // First compute ASAP finish times / makespan, then place delays so each gate
        // finishes as late as possible without changing order.
        var qubitTime = Array(repeating: QFloat(0), count: circuit.qubitCount)
        var starts: [QFloat] = []
        var durs: [QFloat] = []

        for gate in circuit.gates {
            let qubits = resolvedQubits(gate, width: circuit.qubitCount)
            let start = qubits.map { qubitTime[$0] }.max() ?? 0
            let dur: QFloat
            if case .barrier = gate {
                dur = 0
            } else {
                dur = durations.duration(for: gate)
            }
            starts.append(start)
            durs.append(dur)
            let finish = start + dur
            for q in qubits {
                qubitTime[q] = finish
            }
        }

        let makespan = qubitTime.max() ?? 0
        var latestFinish = Array(repeating: makespan, count: circuit.qubitCount)
        var scheduledStarts = Array(repeating: QFloat(0), count: circuit.gates.count)

        for index in stride(from: circuit.gates.count - 1, through: 0, by: -1) {
            let gate = circuit.gates[index]
            let qubits = resolvedQubits(gate, width: circuit.qubitCount)
            let dur = durs[index]
            let latest = qubits.map { latestFinish[$0] }.min() ?? makespan
            let start = max(QFloat(0), latest - dur)
            scheduledStarts[index] = start
            for q in qubits {
                latestFinish[q] = start
            }
        }

        var output = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        var cursor = Array(repeating: QFloat(0), count: circuit.qubitCount)
        for (index, gate) in circuit.gates.enumerated() {
            if case .barrier = gate {
                let qubits = resolvedQubits(gate, width: circuit.qubitCount)
                let t = scheduledStarts[index]
                for q in qubits {
                    let idle = t - cursor[q]
                    if idle > 0 {
                        try output.apply(.delay(duration: idle, qubit: q))
                    }
                    cursor[q] = t
                }
                try output.apply(gate)
                continue
            }
            let qubits = gate.affectedQubits
            let start = scheduledStarts[index]
            for q in qubits {
                let idle = start - cursor[q]
                if idle > 0 {
                    try output.apply(.delay(duration: idle, qubit: q))
                }
            }
            try output.apply(gate)
            let finish = start + durs[index]
            for q in qubits {
                cursor[q] = finish
            }
        }
        return output
    }

    private func resolvedQubits(_ gate: Gate, width: Int) -> [Int] {
        if case .barrier(let qubits) = gate {
            return qubits.isEmpty ? Array(0..<width) : qubits
        }
        return gate.affectedQubits
    }
}
