//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

public enum QuantumCircuitError: Error {
    case invalidQubitCount(Int)
    case qubitIndexOutOfBounds(index: Int, qubitCount: Int)
    case invalidAlgorithmParameter(reason: String)
    case circuitNotUnitary
}

public struct QuantumCircuit {

    public let qubitCount: Int
    public private(set) var gates: [Gate] = []

    public init(qubitCount: Int) throws {
        guard qubitCount > 0 else {
            throw QuantumCircuitError.invalidQubitCount(qubitCount)
        }

        self.qubitCount = qubitCount
    }

    private func validateQubitIndex(_ index: Int) throws {
        guard index >= 0, index < qubitCount else {
            throw QuantumCircuitError.qubitIndexOutOfBounds(index: index, qubitCount: qubitCount)
        }
    }

    mutating func applyValidated(_ gate: Gate) throws {
        switch gate {
        case .h(let target), .x(let target), .y(let target), .z(let target),
             .s(let target), .t(let target), .sdg(let target), .tdg(let target),
             .sx(let target), .sxdg(let target):
            try validateQubitIndex(target)
        case .p(_, let target), .u(_, _, _, let target):
            try validateQubitIndex(target)
        case .rx(_, let target), .ry(_, let target), .rz(_, let target):
            try validateQubitIndex(target)
        case .crx(_, let control, let target), .cry(_, let control, let target),
             .crz(_, let control, let target), .cp(_, let control, let target):
            try validateQubitIndex(control)
            try validateQubitIndex(target)
            guard control != target else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Controlled rotation requires distinct control and target qubits"
                )
            }
        case .mcx(let controls, let target), .mcz(let controls, let target):
            guard !controls.isEmpty else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Multi-controlled gate requires at least one control qubit"
                )
            }
            try validateQubitIndex(target)
            for control in controls {
                try validateQubitIndex(control)
            }
            guard Set(controls).count == controls.count else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Multi-controlled gate requires distinct control qubits"
                )
            }
            guard !controls.contains(target) else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Multi-controlled gate target must differ from all controls"
                )
            }
        case .cx(let control, let target):
            try validateQubitIndex(control)
            try validateQubitIndex(target)
            guard control != target else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "CNOT requires distinct control and target qubits"
                )
            }
        case .cz(let control, let target):
            try validateQubitIndex(control)
            try validateQubitIndex(target)
            guard control != target else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "CZ requires distinct control and target qubits"
                )
            }
        case .swap(let q1, let q2):
            try validateQubitIndex(q1)
            try validateQubitIndex(q2)
            guard q1 != q2 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "SWAP requires two distinct qubits"
                )
            }
        case .ccx(let control1, let control2, let target):
            try validateQubitIndex(control1)
            try validateQubitIndex(control2)
            try validateQubitIndex(target)
            guard control1 != control2, control1 != target, control2 != target else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "CCX requires three distinct qubits (control1, control2, target)"
                )
            }
        case .measure(let qubits):
            guard !qubits.isEmpty else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Measure requires at least one qubit"
                )
            }
            for index in qubits {
                try validateQubitIndex(index)
            }
        case .reset(let qubit):
            try validateQubitIndex(qubit)
        }

        gates.append(gate)
    }

    public mutating func apply(_ gate: Gate) throws {
        try applyValidated(gate)
    }

    /// The inverse (adjoint) circuit: applies each gate's adjoint in reverse order, so
    /// `C` followed by `C.inverse()` returns the state vector to its starting point.
    ///
    /// Throws ``QuantumCircuitError/circuitNotUnitary`` when the circuit contains
    /// non-unitary operations (`measure`/`reset`), which have no inverse.
    public func inverse() throws -> QuantumCircuit {
        guard isUnitaryOnly else {
            throw QuantumCircuitError.circuitNotUnitary
        }

        var inverted = try QuantumCircuit(qubitCount: qubitCount)
        for gate in gates.reversed() {
            try inverted.apply(gate.adjoint)
        }
        return inverted
    }
}
