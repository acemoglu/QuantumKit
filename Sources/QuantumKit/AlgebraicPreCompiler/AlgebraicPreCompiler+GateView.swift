import Foundation

// MARK: - Single-qubit gate view

struct SingleQubitGate {
    enum Kind {
        case h, x, y, z, s, t, sdg, tdg, p, rx, ry, rz
    }

    let qubit: Int
    let kind: Kind

    var isZAxis: Bool {
        switch kind {
        case .z, .s, .t, .sdg, .tdg, .p, .rz:
            return true
        default:
            return false
        }
    }
}

extension Gate {

    var asSingleQubitGate: SingleQubitGate? {
        switch self {
        case .h(let target):
            return SingleQubitGate(qubit: target, kind: .h)
        case .x(let target):
            return SingleQubitGate(qubit: target, kind: .x)
        case .y(let target):
            return SingleQubitGate(qubit: target, kind: .y)
        case .z(let target):
            return SingleQubitGate(qubit: target, kind: .z)
        case .s(let target):
            return SingleQubitGate(qubit: target, kind: .s)
        case .t(let target):
            return SingleQubitGate(qubit: target, kind: .t)
        case .sdg(let target):
            return SingleQubitGate(qubit: target, kind: .sdg)
        case .tdg(let target):
            return SingleQubitGate(qubit: target, kind: .tdg)
        case .p(_, let target):
            return SingleQubitGate(qubit: target, kind: .p)
        case .rx(_, let target):
            return SingleQubitGate(qubit: target, kind: .rx)
        case .ry(_, let target):
            return SingleQubitGate(qubit: target, kind: .ry)
        case .rz(_, let target):
            return SingleQubitGate(qubit: target, kind: .rz)
        default:
            return nil
        }
    }
}

extension QuantumCircuit {

    /// Returns an algebraically simplified copy of this circuit.
    public func algebraicallyOptimized() throws -> QuantumCircuit {
        try AlgebraicPreCompiler.optimize(self)
    }
}
