extension QuantumCircuit {

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
    public mutating func sxdg(_ target: Int) throws -> QuantumCircuit {
        try applyValidated(.sxdg(target: target))
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
