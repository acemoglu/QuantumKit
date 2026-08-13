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

    /// Inverse √X gate (SX†); the conjugate-transpose of `sx`, so SX†·SX = I.
    case sxdg(target: Int)

    /// General phase gate P(θ) = diag(1, e^{iθ}).
    case p(theta: QFloatExpr, target: Int)

    /// Universal single-qubit gate U(θ, φ, λ) (Qiskit convention).
    case u(theta: QFloatExpr, phi: QFloatExpr, lambda: QFloatExpr, target: Int)

    /// Entangles two qubits. Controlled-NOT gate
    case cx(control:Int, target:Int)

    /// Controlled-Z; applies a -1 phase to |11⟩ (symmetric in control/target).
    case cz(control: Int, target: Int)

    /// Swaps the states of two qubits.
    case swap(q1: Int, q2: Int)

    /// Identity on a single qubit (explicit no-op).
    case id(target: Int)

    /// Ordering barrier on the listed qubits (empty = all circuit qubits at validation time).
    case barrier(qubits: [Int])

    /// Idle / delay instruction with duration metadata (simulation timing; not a pulse).
    case delay(duration: QFloat, qubit: Int)

    /// iSWAP: swaps qubits with a relative phase of +i on the odd-parity subspace.
    case iswap(q1: Int, q2: Int)

    /// Echoed cross-resonance (IBM native-style); decomposed for simulation.
    case ecr(control: Int, target: Int)

    /// Ising XX rotation exp(-i θ XX / 2).
    case rxx(theta: QFloatExpr, q1: Int, q2: Int)

    /// Ising YY rotation exp(-i θ YY / 2).
    case ryy(theta: QFloatExpr, q1: Int, q2: Int)

    /// Ising ZZ rotation exp(-i θ ZZ / 2).
    case rzz(theta: QFloatExpr, q1: Int, q2: Int)

    /// Double CNOT: CX(q1,q2) then CX(q2,q1).
    case dcx(q1: Int, q2: Int)

    /// Fredkin gate (controlled-SWAP).
    case cswap(control: Int, q1: Int, q2: Int)

    /// Toffoli gate (CCX): X on target when both controls are |1>
    case ccx(control1:Int, control2:Int, target:Int)

    /// Multi-controlled X: X on target when every control qubit is |1⟩.
    case mcx(controls: [Int], target: Int)

    /// Multi-controlled Z: -1 phase when every control qubit and the target are |1⟩.
    case mcz(controls: [Int], target: Int)
    
    /// Parametrized rotation around the X-axis
    case rx(theta: QFloatExpr, target: Int)

    /// Parametrized rotation around the Y-axis
    case ry(theta: QFloatExpr, target: Int)

    /// Parametrized rotation around the Z-axis
    case rz(theta: QFloatExpr, target: Int)

    /// Controlled rotation around the X-axis (applied when control is |1⟩).
    case crx(theta: QFloatExpr, control: Int, target: Int)

    /// Controlled rotation around the Y-axis (applied when control is |1⟩).
    case cry(theta: QFloatExpr, control: Int, target: Int)

    /// Controlled rotation around the Z-axis (applied when control is |1⟩).
    case crz(theta: QFloatExpr, control: Int, target: Int)

    /// Controlled phase CP(θ): e^{iθ} phase on |11⟩.
    case cp(theta: QFloatExpr, control: Int, target: Int)

    /// Mid-circuit computational-basis measurement on the listed qubits.
    case measure(MeasureSpec)

    /// Project a qubit onto |0⟩ and renormalize (non-unitary).
    case reset(qubit: Int)

    /// Classically conditioned gate: applies `gate` only when the classical register
    /// `classicalRegister` holds `expectedValue`. Marked `indirect` so a `Gate` can nest inside
    /// another `Gate`, giving the recursive classical-feedback representation required by the
    /// deferred-measurement principle.
    indirect case c_if(classicalRegister: Int, expectedValue: Int, gate: Gate)

    /// Arbitrary single-qubit unitary specified as a row-major 2×2 matrix.
    case unitary1(matrix: [ComplexAmplitude], target: Int)

    /// Sets the quantum state on the listed qubits from explicit, normalized amplitudes.
    case initialize(qubits: [Int], amplitudes: [ComplexAmplitude])

    /// Arbitrary unitary on the listed qubits, specified as a row-major 2^k×2^k matrix.
    case customUnitary(matrix: [ComplexAmplitude], qubits: [Int])
}

extension Gate {

    /// Qubits touched by this operation (used for noise injection).
    public var affectedQubits: [Int] {
        switch self {
        case .h(let target), .x(let target), .y(let target), .z(let target),
             .s(let target), .t(let target), .sdg(let target), .tdg(let target),
             .sx(let target), .sxdg(let target), .p(_, let target), .u(_, _, _, let target),
             .rx(_, let target), .ry(_, let target), .rz(_, let target), .reset(let target),
             .unitary1(_, let target), .id(let target), .delay(_, let target):
            return [target]
        case .cx(let control, let target), .cz(let control, let target),
             .crx(_, let control, let target), .cry(_, let control, let target),
             .crz(_, let control, let target), .cp(_, let control, let target),
             .ecr(let control, let target):
            return [control, target]
        case .swap(let q1, let q2), .iswap(let q1, let q2), .dcx(let q1, let q2),
             .rxx(_, let q1, let q2), .ryy(_, let q1, let q2), .rzz(_, let q1, let q2):
            return [q1, q2]
        case .cswap(let control, let q1, let q2):
            return [control, q1, q2]
        case .barrier(let qubits):
            return qubits
        case .ccx(let control1, let control2, let target):
            return [control1, control2, target]
        case .mcx(let controls, let target), .mcz(let controls, let target):
            return controls + [target]
        case .measure(let spec):
            return spec.qubits
        case .c_if(_, _, let gate):
            return gate.affectedQubits
        case .initialize(let qubits, _):
            return qubits
        case .customUnitary(_, let qubits):
            return qubits
        }
    }

    /// The adjoint (inverse) of this gate.
    ///
    /// Self-inverse gates (Pauli, H, CX, CZ, SWAP, CCX, MCX, MCZ) return themselves;
    /// S↔Sdg, T↔Tdg, SX↔SXdg are swapped; parametrized gates negate their angle(s).
    /// For `u`, the Qiskit convention gives U(θ,φ,λ)† = U(-θ,-λ,-φ) (φ and λ swap and negate).
    /// `measure`/`reset` are non-unitary and return themselves; guard with ``QuantumCircuit/isUnitaryOnly``.
    public var adjoint: Gate {
        switch self {
        case .h, .x, .y, .z, .cx, .cz, .swap, .ccx, .mcx, .mcz, .id, .barrier, .iswap, .dcx, .cswap, .ecr:
            return self
        case .delay(let duration, let qubit):
            return .delay(duration: duration, qubit: qubit)
        case .rxx(let theta, let q1, let q2):
            return .rxx(theta: -theta, q1: q1, q2: q2)
        case .ryy(let theta, let q1, let q2):
            return .ryy(theta: -theta, q1: q1, q2: q2)
        case .rzz(let theta, let q1, let q2):
            return .rzz(theta: -theta, q1: q1, q2: q2)
        case .s(let target):
            return .sdg(target: target)
        case .sdg(let target):
            return .s(target: target)
        case .t(let target):
            return .tdg(target: target)
        case .tdg(let target):
            return .t(target: target)
        case .sx(let target):
            return .sxdg(target: target)
        case .sxdg(let target):
            return .sx(target: target)
        case .p(let theta, let target):
            return .p(theta: -theta, target: target)
        case .rx(let theta, let target):
            return .rx(theta: -theta, target: target)
        case .ry(let theta, let target):
            return .ry(theta: -theta, target: target)
        case .rz(let theta, let target):
            return .rz(theta: -theta, target: target)
        case .crx(let theta, let control, let target):
            return .crx(theta: -theta, control: control, target: target)
        case .cry(let theta, let control, let target):
            return .cry(theta: -theta, control: control, target: target)
        case .crz(let theta, let control, let target):
            return .crz(theta: -theta, control: control, target: target)
        case .cp(let theta, let control, let target):
            return .cp(theta: -theta, control: control, target: target)
        case .u(let theta, let phi, let lambda, let target):
            return .u(theta: -theta, phi: -lambda, lambda: -phi, target: target)
        case .measure, .reset, .initialize:
            return self
        case .unitary1(let matrix, let target):
            let dagger = Self.adjoint1QMatrix(matrix)
            return .unitary1(matrix: dagger, target: target)
        case .customUnitary(let matrix, let qubits):
            let dimension = 1 << qubits.count
            let dagger = UnitaryValidation.adjoint(matrix: matrix, dimension: dimension)
            return .customUnitary(matrix: dagger, qubits: qubits)
        case .c_if(let classicalRegister, let expectedValue, let gate):
            return .c_if(classicalRegister: classicalRegister, expectedValue: expectedValue, gate: gate.adjoint)
        }
    }

    /// Wraps every rotation/phase angle into `[-π, π]` using a robust ``Double`` modulo before the
    /// value is truncated to `Float32` for the GPU.
    ///
    /// Unbounded angles (e.g. `.rz(theta: 1e7)`) that bypass the algebraic precompiler would
    /// otherwise reach Metal's `Float32` trig functions and suffer catastrophic loss of
    /// significance. Reducing the argument first — in `Double`, before the `Float32` cast — keeps
    /// the GPU's `cos`/`sin` inputs small and accurate. The reduction is exact for the `2π`-periodic
    /// `p`/`cp` gates and differs only by a (physically irrelevant) global phase for the rotation
    /// gates, so the produced state is unchanged.
    public var angleWrapped: Gate {
        switch self {
        case .p(let theta, let target):
            return .p(theta: Gate.wrapAngle(theta), target: target)
        case .u(let theta, let phi, let lambda, let target):
            return .u(
                theta: Gate.wrapAngle(theta),
                phi: Gate.wrapAngle(phi),
                lambda: Gate.wrapAngle(lambda),
                target: target
            )
        case .rx(let theta, let target):
            return .rx(theta: Gate.wrapAngle(theta), target: target)
        case .ry(let theta, let target):
            return .ry(theta: Gate.wrapAngle(theta), target: target)
        case .rz(let theta, let target):
            return .rz(theta: Gate.wrapAngle(theta), target: target)
        case .crx(let theta, let control, let target):
            return .crx(theta: Gate.wrapAngle(theta), control: control, target: target)
        case .cry(let theta, let control, let target):
            return .cry(theta: Gate.wrapAngle(theta), control: control, target: target)
        case .crz(let theta, let control, let target):
            return .crz(theta: Gate.wrapAngle(theta), control: control, target: target)
        case .cp(let theta, let control, let target):
            return .cp(theta: Gate.wrapAngle(theta), control: control, target: target)
        case .rxx(let theta, let q1, let q2):
            return .rxx(theta: Gate.wrapAngle(theta), q1: q1, q2: q2)
        case .ryy(let theta, let q1, let q2):
            return .ryy(theta: Gate.wrapAngle(theta), q1: q1, q2: q2)
        case .rzz(let theta, let q1, let q2):
            return .rzz(theta: Gate.wrapAngle(theta), q1: q1, q2: q2)
        default:
            return self
        }
    }

    /// Robust reduction of `angle` into `[-π, π]`, performed in `Double` so that huge `Float32`
    /// inputs are reduced before precision is lost. `Double.remainder(dividingBy:)` is the IEEE
    /// remainder, whose result already lies in `[-π, π]` for a divisor of `2π`.
    static func wrapAngle(_ angle: QFloat) -> QFloat {
        QFloat(Double(angle).remainder(dividingBy: 2.0 * Double.pi))
    }

    static func wrapAngle(_ angle: QFloatExpr) -> QFloatExpr {
        guard let literal = angle.literalValue else { return angle }
        return .literal(wrapAngle(literal))
    }

    static func adjoint1QMatrix(_ matrix: [ComplexAmplitude]) -> [ComplexAmplitude] {
        precondition(matrix.count == 4)
        return [
            ComplexAmplitude(real: matrix[0].real, imaginary: -matrix[0].imaginary),
            ComplexAmplitude(real: matrix[2].real, imaginary: -matrix[2].imaginary),
            ComplexAmplitude(real: matrix[1].real, imaginary: -matrix[1].imaginary),
            ComplexAmplitude(real: matrix[3].real, imaginary: -matrix[3].imaginary),
        ]
    }
}
