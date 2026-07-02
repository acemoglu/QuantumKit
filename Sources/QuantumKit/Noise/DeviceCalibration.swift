import Foundation

/// Per-qubit relaxation and readout calibration data.
public struct QubitCalibration: Sendable, Equatable, Codable {
    public var t1: QFloat
    public var t2: QFloat
    public var readoutError0To1: QFloat
    public var readoutError1To0: QFloat

    public init(
        t1: QFloat = 0,
        t2: QFloat = 0,
        readoutError0To1: QFloat = 0,
        readoutError1To0: QFloat = 0
    ) {
        self.t1 = max(t1, 0)
        self.t2 = max(t2, 0)
        self.readoutError0To1 = min(max(readoutError0To1, 0), 1)
        self.readoutError1To0 = min(max(readoutError1To0, 0), 1)
    }
}

/// Per-gate error calibration on specific qubits (e.g. SX on qubit 0, CX on edge 0–1).
public struct GateCalibration: Sendable, Equatable, Codable {
    public let gate: GateKind
    public let qubits: [Int]
    public var errorRate: QFloat

    public init(gate: GateKind, qubits: [Int], errorRate: QFloat) {
        self.gate = gate
        self.qubits = qubits
        self.errorRate = min(max(errorRate, 0), 1)
    }

    public var noiseTarget: NoiseTarget {
        if qubits.count == 1 {
            return .gateOnQubit(gate: gate, qubit: qubits[0])
        }
        return .gateOnQubits(gate: gate, qubits: qubits)
    }
}

/// Hardware calibration properties for a quantum device.
public struct DeviceCalibration: Sendable, Equatable, Codable {
    public let qubitCount: Int
    public var qubits: [QubitCalibration]
    public var gateErrors: [GateCalibration]
    public var gateTime: QFloat

    public init(
        qubitCount: Int,
        qubits: [QubitCalibration] = [],
        gateErrors: [GateCalibration] = [],
        gateTime: QFloat = 0
    ) {
        self.qubitCount = qubitCount
        self.qubits = qubits
        self.gateErrors = gateErrors
        self.gateTime = max(gateTime, 0)
    }

    public subscript(qubit index: Int) -> QubitCalibration {
        get {
            guard index >= 0, index < qubits.count else { return QubitCalibration() }
            return qubits[index]
        }
        set {
            guard index >= 0 else { return }
            if index >= qubits.count {
                qubits.append(contentsOf: Array(
                    repeating: QubitCalibration(),
                    count: index - qubits.count + 1
                ))
            }
            qubits[index] = newValue
        }
    }
}
