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

    private mutating func applyValidated(_ gate: Gate) throws {
        switch gate {
        case .h(let target), .x(let target), .y(let target), .z(let target),
             .s(let target), .t(let target), .sdg(let target), .tdg(let target),
             .sx(let target):
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

    @discardableResult
    public mutating func h(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.h(target: target))
        return self
    }

    @discardableResult
    public mutating func cx(_ control: Int, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.cx(control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func ccx(_ control1: Int, _ control2: Int, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.ccx(control1: control1, control2: control2, target: target))
        return self
    }

    @discardableResult
    public mutating func rx(theta: QFloat, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.rx(theta: theta, target: target))
        return self
    }

    @discardableResult
    public mutating func rz(theta: QFloat, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.rz(theta: theta, target: target))
        return self
    }

    @discardableResult
    public mutating func ry(theta: QFloat, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.ry(theta: theta, target: target))
        return self
    }

    @discardableResult
    public mutating func s(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.s(target: target))
        return self
    }

    @discardableResult
    public mutating func t(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.t(target: target))
        return self
    }

    @discardableResult
    public mutating func sdg(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.sdg(target: target))
        return self
    }

    @discardableResult
    public mutating func tdg(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.tdg(target: target))
        return self
    }

    @discardableResult
    public mutating func sx(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.sx(target: target))
        return self
    }

    @discardableResult
    public mutating func p(theta: QFloat, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.p(theta: theta, target: target))
        return self
    }

    @discardableResult
    public mutating func u(theta: QFloat, phi: QFloat, lambda: QFloat, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.u(theta: theta, phi: phi, lambda: lambda, target: target))
        return self
    }

    @discardableResult
    public mutating func cz(_ control: Int, _ target: Int) throws -> QuantumCircuit {
        try applyValidated(.cz(control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func swap(_ q1: Int, _ q2: Int) throws -> QuantumCircuit {
        try applyValidated(.swap(q1: q1, q2: q2))
        return self
    }

    @discardableResult
    public mutating func crx(theta: QFloat, control: Int, target: Int) throws -> QuantumCircuit {
        try applyValidated(.crx(theta: theta, control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func cry(theta: QFloat, control: Int, target: Int) throws -> QuantumCircuit {
        try applyValidated(.cry(theta: theta, control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func crz(theta: QFloat, control: Int, target: Int) throws -> QuantumCircuit {
        try applyValidated(.crz(theta: theta, control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func cp(theta: QFloat, control: Int, target: Int) throws -> QuantumCircuit {
        try applyValidated(.cp(theta: theta, control: control, target: target))
        return self
    }

    @discardableResult
    public mutating func mcx(controls: [Int], target: Int) throws -> QuantumCircuit {
        try applyValidated(.mcx(controls: controls, target: target))
        return self
    }

    @discardableResult
    public mutating func mcz(controls: [Int], target: Int) throws -> QuantumCircuit {
        try applyValidated(.mcz(controls: controls, target: target))
        return self
    }

    @discardableResult
    public mutating func x(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.x(target: target))
        return self
    }

    @discardableResult
    public mutating func y(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.y(target: target))
        return self
    }

    @discardableResult
    public mutating func z(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.z(target: target))
        return self
    }

    @discardableResult
    public mutating func measure(qubits: [Int]) throws -> QuantumCircuit {
        try applyValidated(.measure(qubits: qubits))
        return self
    }

    @discardableResult
    public mutating func measure(_ qubit: Int) throws -> QuantumCircuit {
        try measure(qubits: [qubit])
    }

    @discardableResult
    public mutating func reset(_ qubit: Int) throws -> QuantumCircuit {
        try applyValidated(.reset(qubit: qubit))
        return self
    }
}
