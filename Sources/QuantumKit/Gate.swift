//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation

public enum Gate: Equatable, Sendable {
    
    /// Hadamard gate
    case h(target:Int)
    
    /// Pauli X, Y, Z  gates:
    case x(target:Int)
    case y(target:Int)
    case z(target:Int)

    /// Phase gate S = √Z; applies e^{iπ/2} to |1⟩.
    case s(target: Int)

    /// π/8 gate T = √S; applies e^{iπ/4} to |1⟩.
    case t(target: Int)
    
    /// Entangles two qubits. Controlled-NOT gate
    case cx(control:Int, target:Int)

    /// Toffoli gate (CCX): X on target when both controls are |1>
    case ccx(control1:Int, control2:Int, target:Int)
    
    /// Parametrized rotation around the X-axis
    case rx(theta:QFloat, target:Int)

    /// Parametrized rotation around the Y-axis
    case ry(theta: QFloat, target: Int)

    /// Parametrized rotation around the Z-axis
    case rz(theta:QFloat, target:Int)

    /// Mid-circuit computational-basis measurement on the listed qubits.
    case measure(qubits: [Int])

    /// Project a qubit onto |0⟩ and renormalize (non-unitary).
    case reset(qubit: Int)

}

extension Gate {

    /// Qubits touched by this operation (used for noise injection).
    public var affectedQubits: [Int] {
        switch self {
        case .h(let target), .x(let target), .y(let target), .z(let target),
             .s(let target), .t(let target),
             .rx(_, let target), .ry(_, let target), .rz(_, let target), .reset(let target):
            return [target]
        case .cx(let control, let target):
            return [control, target]
        case .ccx(let control1, let control2, let target):
            return [control1, control2, target]
        case .measure(let qubits):
            return qubits
        }
    }
}
