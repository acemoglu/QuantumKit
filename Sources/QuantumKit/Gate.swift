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

    /// Inverse phase gate S† ; applies e^{-iπ/2} to |1⟩.
    case sdg(target: Int)

    /// Inverse π/8 gate T† ; applies e^{-iπ/4} to |1⟩.
    case tdg(target: Int)

    /// √X gate; SX·SX = X.
    case sx(target: Int)

    /// General phase gate P(θ) = diag(1, e^{iθ}).
    case p(theta: QFloat, target: Int)

    /// Universal single-qubit gate U(θ, φ, λ) (Qiskit convention).
    case u(theta: QFloat, phi: QFloat, lambda: QFloat, target: Int)

    /// Entangles two qubits. Controlled-NOT gate
    case cx(control:Int, target:Int)

    /// Controlled-Z; applies a -1 phase to |11⟩ (symmetric in control/target).
    case cz(control: Int, target: Int)

    /// Swaps the states of two qubits.
    case swap(q1: Int, q2: Int)

    /// Toffoli gate (CCX): X on target when both controls are |1>
    case ccx(control1:Int, control2:Int, target:Int)

    /// Multi-controlled X: X on target when every control qubit is |1⟩.
    case mcx(controls: [Int], target: Int)

    /// Multi-controlled Z: -1 phase when every control qubit and the target are |1⟩.
    case mcz(controls: [Int], target: Int)
    
    /// Parametrized rotation around the X-axis
    case rx(theta:QFloat, target:Int)

    /// Parametrized rotation around the Y-axis
    case ry(theta: QFloat, target: Int)

    /// Parametrized rotation around the Z-axis
    case rz(theta:QFloat, target:Int)

    /// Controlled rotation around the X-axis (applied when control is |1⟩).
    case crx(theta: QFloat, control: Int, target: Int)

    /// Controlled rotation around the Y-axis (applied when control is |1⟩).
    case cry(theta: QFloat, control: Int, target: Int)

    /// Controlled rotation around the Z-axis (applied when control is |1⟩).
    case crz(theta: QFloat, control: Int, target: Int)

    /// Controlled phase CP(θ): e^{iθ} phase on |11⟩.
    case cp(theta: QFloat, control: Int, target: Int)

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
             .s(let target), .t(let target), .sdg(let target), .tdg(let target),
             .sx(let target), .p(_, let target), .u(_, _, _, let target),
             .rx(_, let target), .ry(_, let target), .rz(_, let target), .reset(let target):
            return [target]
        case .cx(let control, let target), .cz(let control, let target),
             .crx(_, let control, let target), .cry(_, let control, let target),
             .crz(_, let control, let target), .cp(_, let control, let target):
            return [control, target]
        case .swap(let q1, let q2):
            return [q1, q2]
        case .ccx(let control1, let control2, let target):
            return [control1, control2, target]
        case .mcx(let controls, let target), .mcz(let controls, let target):
            return controls + [target]
        case .measure(let qubits):
            return qubits
        }
    }
}
