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
        case .h(let target), .x(let target), .y(let target), .z(let target):
            try validateQubitIndex(target)
        case .rx(_, let target), .rz(_, let target):
            try validateQubitIndex(target)
        case .cx(let control, let target):
            try validateQubitIndex(control)
            try validateQubitIndex(target)
        case .ccx(let control1, let control2, let target):
            try validateQubitIndex(control1)
            try validateQubitIndex(control2)
            try validateQubitIndex(target)
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
}
