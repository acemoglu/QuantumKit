import Foundation

extension Gate {
    /// Returns a copy of this gate with every qubit index rewritten by `map`.
    public func remappingQubits(_ map: (Int) throws -> Int) rethrows -> Gate {
        switch self {
        case .h(let target):
            return .h(target: try map(target))
        case .x(let target):
            return .x(target: try map(target))
        case .y(let target):
            return .y(target: try map(target))
        case .z(let target):
            return .z(target: try map(target))
        case .s(let target):
            return .s(target: try map(target))
        case .t(let target):
            return .t(target: try map(target))
        case .sdg(let target):
            return .sdg(target: try map(target))
        case .tdg(let target):
            return .tdg(target: try map(target))
        case .sx(let target):
            return .sx(target: try map(target))
        case .sxdg(let target):
            return .sxdg(target: try map(target))
        case .p(let theta, let target):
            return .p(theta: theta, target: try map(target))
        case .u(let theta, let phi, let lambda, let target):
            return .u(theta: theta, phi: phi, lambda: lambda, target: try map(target))
        case .cx(let control, let target):
            return .cx(control: try map(control), target: try map(target))
        case .cz(let control, let target):
            return .cz(control: try map(control), target: try map(target))
        case .swap(let q1, let q2):
            return .swap(q1: try map(q1), q2: try map(q2))
        case .id(let target):
            return .id(target: try map(target))
        case .barrier(let qubits):
            return .barrier(qubits: try qubits.map(map))
        case .delay(let duration, let qubit):
            return .delay(duration: duration, qubit: try map(qubit))
        case .iswap(let q1, let q2):
            return .iswap(q1: try map(q1), q2: try map(q2))
        case .ecr(let control, let target):
            return .ecr(control: try map(control), target: try map(target))
        case .rxx(let theta, let q1, let q2):
            return .rxx(theta: theta, q1: try map(q1), q2: try map(q2))
        case .ryy(let theta, let q1, let q2):
            return .ryy(theta: theta, q1: try map(q1), q2: try map(q2))
        case .rzz(let theta, let q1, let q2):
            return .rzz(theta: theta, q1: try map(q1), q2: try map(q2))
        case .dcx(let q1, let q2):
            return .dcx(q1: try map(q1), q2: try map(q2))
        case .cswap(let control, let q1, let q2):
            return .cswap(
                control: try map(control),
                q1: try map(q1),
                q2: try map(q2)
            )
        case .ccx(let control1, let control2, let target):
            return .ccx(
                control1: try map(control1),
                control2: try map(control2),
                target: try map(target)
            )
        case .mcx(let controls, let target):
            return .mcx(controls: try controls.map(map), target: try map(target))
        case .mcz(let controls, let target):
            return .mcz(controls: try controls.map(map), target: try map(target))
        case .rx(let theta, let target):
            return .rx(theta: theta, target: try map(target))
        case .ry(let theta, let target):
            return .ry(theta: theta, target: try map(target))
        case .rz(let theta, let target):
            return .rz(theta: theta, target: try map(target))
        case .crx(let theta, let control, let target):
            return .crx(theta: theta, control: try map(control), target: try map(target))
        case .cry(let theta, let control, let target):
            return .cry(theta: theta, control: try map(control), target: try map(target))
        case .crz(let theta, let control, let target):
            return .crz(theta: theta, control: try map(control), target: try map(target))
        case .cp(let theta, let control, let target):
            return .cp(theta: theta, control: try map(control), target: try map(target))
        case .measure(let spec):
            return .measure(
                MeasureSpec(
                    qubits: try spec.qubits.map(map),
                    classicalRegister: spec.classicalRegister,
                    classicalBitOffset: spec.classicalBitOffset
                )
            )
        case .reset(let qubit):
            return .reset(qubit: try map(qubit))
        case .c_if(let classicalRegister, let expectedValue, let gate):
            return .c_if(
                classicalRegister: classicalRegister,
                expectedValue: expectedValue,
                gate: try gate.remappingQubits(map)
            )
        case .unitary1(let matrix, let target):
            return .unitary1(matrix: matrix, target: try map(target))
        case .initialize(let qubits, let amplitudes):
            return .initialize(qubits: try qubits.map(map), amplitudes: amplitudes)
        case .customUnitary(let matrix, let qubits):
            return .customUnitary(matrix: matrix, qubits: try qubits.map(map))
        }
    }
}
