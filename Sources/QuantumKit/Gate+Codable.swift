import Foundation

extension Gate: Codable {

    private enum CodingKeys: String, CodingKey {
        case type
        case target
        case control
        case control1
        case control2
        case q1
        case q2
        case controls
        case qubits
        case qubit
        case theta
        case phi
        case lambda
        case duration
        case classicalRegister
        case classicalBitOffset
        case expectedValue
        case gate
        case matrix
    }

    private enum GateType: String, Codable {
        case h, x, y, z, s, t, sdg, tdg, sx, sxdg, p, u
        case cx, cz, swap, id, barrier, delay
        case iswap, ecr, rxx, ryy, rzz, dcx, cswap
        case ccx, mcx, mcz
        case rx, ry, rz, crx, cry, crz, cp
        case measure, reset, c_if, unitary1, initialize, customUnitary
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(GateType.self, forKey: .type)

        switch type {
        case .h:
            self = .h(target: try container.decode(Int.self, forKey: .target))
        case .x:
            self = .x(target: try container.decode(Int.self, forKey: .target))
        case .y:
            self = .y(target: try container.decode(Int.self, forKey: .target))
        case .z:
            self = .z(target: try container.decode(Int.self, forKey: .target))
        case .s:
            self = .s(target: try container.decode(Int.self, forKey: .target))
        case .t:
            self = .t(target: try container.decode(Int.self, forKey: .target))
        case .sdg:
            self = .sdg(target: try container.decode(Int.self, forKey: .target))
        case .tdg:
            self = .tdg(target: try container.decode(Int.self, forKey: .target))
        case .sx:
            self = .sx(target: try container.decode(Int.self, forKey: .target))
        case .sxdg:
            self = .sxdg(target: try container.decode(Int.self, forKey: .target))
        case .p:
            self = .p(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .u:
            self = .u(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                phi: try container.decode(QFloatExpr.self, forKey: .phi),
                lambda: try container.decode(QFloatExpr.self, forKey: .lambda),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .cx:
            self = .cx(
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .cz:
            self = .cz(
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .swap:
            self = .swap(
                q1: try container.decode(Int.self, forKey: .q1),
                q2: try container.decode(Int.self, forKey: .q2)
            )
        case .id:
            self = .id(target: try container.decode(Int.self, forKey: .target))
        case .barrier:
            self = .barrier(qubits: try container.decode([Int].self, forKey: .qubits))
        case .delay:
            self = .delay(
                duration: try container.decode(QFloat.self, forKey: .duration),
                qubit: try container.decode(Int.self, forKey: .qubit)
            )
        case .iswap:
            self = .iswap(
                q1: try container.decode(Int.self, forKey: .q1),
                q2: try container.decode(Int.self, forKey: .q2)
            )
        case .ecr:
            self = .ecr(
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .rxx:
            self = .rxx(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                q1: try container.decode(Int.self, forKey: .q1),
                q2: try container.decode(Int.self, forKey: .q2)
            )
        case .ryy:
            self = .ryy(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                q1: try container.decode(Int.self, forKey: .q1),
                q2: try container.decode(Int.self, forKey: .q2)
            )
        case .rzz:
            self = .rzz(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                q1: try container.decode(Int.self, forKey: .q1),
                q2: try container.decode(Int.self, forKey: .q2)
            )
        case .dcx:
            self = .dcx(
                q1: try container.decode(Int.self, forKey: .q1),
                q2: try container.decode(Int.self, forKey: .q2)
            )
        case .cswap:
            self = .cswap(
                control: try container.decode(Int.self, forKey: .control),
                q1: try container.decode(Int.self, forKey: .q1),
                q2: try container.decode(Int.self, forKey: .q2)
            )
        case .ccx:
            self = .ccx(
                control1: try container.decode(Int.self, forKey: .control1),
                control2: try container.decode(Int.self, forKey: .control2),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .mcx:
            self = .mcx(
                controls: try container.decode([Int].self, forKey: .controls),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .mcz:
            self = .mcz(
                controls: try container.decode([Int].self, forKey: .controls),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .rx:
            self = .rx(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .ry:
            self = .ry(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .rz:
            self = .rz(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .crx:
            self = .crx(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .cry:
            self = .cry(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .crz:
            self = .crz(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .cp:
            self = .cp(
                theta: try container.decode(QFloatExpr.self, forKey: .theta),
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .measure:
            let qubits = try container.decode([Int].self, forKey: .qubits)
            let classicalRegister = try container.decodeIfPresent(Int.self, forKey: .classicalRegister) ?? 0
            let classicalBitOffset = try container.decodeIfPresent(Int.self, forKey: .classicalBitOffset) ?? 0
            self = .measure(
                MeasureSpec(
                    qubits: qubits,
                    classicalRegister: classicalRegister,
                    classicalBitOffset: classicalBitOffset
                )
            )
        case .reset:
            self = .reset(qubit: try container.decode(Int.self, forKey: .qubit))
        case .c_if:
            self = .c_if(
                classicalRegister: try container.decode(Int.self, forKey: .classicalRegister),
                expectedValue: try container.decode(Int.self, forKey: .expectedValue),
                gate: try container.decode(Gate.self, forKey: .gate)
            )
        case .unitary1:
            self = .unitary1(
                matrix: try container.decode([ComplexAmplitude].self, forKey: .matrix),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .initialize:
            self = .initialize(
                qubits: try container.decode([Int].self, forKey: .qubits),
                amplitudes: try container.decode([ComplexAmplitude].self, forKey: .matrix)
            )
        case .customUnitary:
            self = .customUnitary(
                matrix: try container.decode([ComplexAmplitude].self, forKey: .matrix),
                qubits: try container.decode([Int].self, forKey: .qubits)
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .h(let target):
            try container.encode(GateType.h, forKey: .type)
            try container.encode(target, forKey: .target)
        case .x(let target):
            try container.encode(GateType.x, forKey: .type)
            try container.encode(target, forKey: .target)
        case .y(let target):
            try container.encode(GateType.y, forKey: .type)
            try container.encode(target, forKey: .target)
        case .z(let target):
            try container.encode(GateType.z, forKey: .type)
            try container.encode(target, forKey: .target)
        case .s(let target):
            try container.encode(GateType.s, forKey: .type)
            try container.encode(target, forKey: .target)
        case .t(let target):
            try container.encode(GateType.t, forKey: .type)
            try container.encode(target, forKey: .target)
        case .sdg(let target):
            try container.encode(GateType.sdg, forKey: .type)
            try container.encode(target, forKey: .target)
        case .tdg(let target):
            try container.encode(GateType.tdg, forKey: .type)
            try container.encode(target, forKey: .target)
        case .sx(let target):
            try container.encode(GateType.sx, forKey: .type)
            try container.encode(target, forKey: .target)
        case .sxdg(let target):
            try container.encode(GateType.sxdg, forKey: .type)
            try container.encode(target, forKey: .target)
        case .p(let theta, let target):
            try container.encode(GateType.p, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(target, forKey: .target)
        case .u(let theta, let phi, let lambda, let target):
            try container.encode(GateType.u, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(phi, forKey: .phi)
            try container.encode(lambda, forKey: .lambda)
            try container.encode(target, forKey: .target)
        case .cx(let control, let target):
            try container.encode(GateType.cx, forKey: .type)
            try container.encode(control, forKey: .control)
            try container.encode(target, forKey: .target)
        case .cz(let control, let target):
            try container.encode(GateType.cz, forKey: .type)
            try container.encode(control, forKey: .control)
            try container.encode(target, forKey: .target)
        case .swap(let q1, let q2):
            try container.encode(GateType.swap, forKey: .type)
            try container.encode(q1, forKey: .q1)
            try container.encode(q2, forKey: .q2)
        case .id(let target):
            try container.encode(GateType.id, forKey: .type)
            try container.encode(target, forKey: .target)
        case .barrier(let qubits):
            try container.encode(GateType.barrier, forKey: .type)
            try container.encode(qubits, forKey: .qubits)
        case .delay(let duration, let qubit):
            try container.encode(GateType.delay, forKey: .type)
            try container.encode(duration, forKey: .duration)
            try container.encode(qubit, forKey: .qubit)
        case .iswap(let q1, let q2):
            try container.encode(GateType.iswap, forKey: .type)
            try container.encode(q1, forKey: .q1)
            try container.encode(q2, forKey: .q2)
        case .ecr(let control, let target):
            try container.encode(GateType.ecr, forKey: .type)
            try container.encode(control, forKey: .control)
            try container.encode(target, forKey: .target)
        case .rxx(let theta, let q1, let q2):
            try container.encode(GateType.rxx, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(q1, forKey: .q1)
            try container.encode(q2, forKey: .q2)
        case .ryy(let theta, let q1, let q2):
            try container.encode(GateType.ryy, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(q1, forKey: .q1)
            try container.encode(q2, forKey: .q2)
        case .rzz(let theta, let q1, let q2):
            try container.encode(GateType.rzz, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(q1, forKey: .q1)
            try container.encode(q2, forKey: .q2)
        case .dcx(let q1, let q2):
            try container.encode(GateType.dcx, forKey: .type)
            try container.encode(q1, forKey: .q1)
            try container.encode(q2, forKey: .q2)
        case .cswap(let control, let q1, let q2):
            try container.encode(GateType.cswap, forKey: .type)
            try container.encode(control, forKey: .control)
            try container.encode(q1, forKey: .q1)
            try container.encode(q2, forKey: .q2)
        case .ccx(let control1, let control2, let target):
            try container.encode(GateType.ccx, forKey: .type)
            try container.encode(control1, forKey: .control1)
            try container.encode(control2, forKey: .control2)
            try container.encode(target, forKey: .target)
        case .mcx(let controls, let target):
            try container.encode(GateType.mcx, forKey: .type)
            try container.encode(controls, forKey: .controls)
            try container.encode(target, forKey: .target)
        case .mcz(let controls, let target):
            try container.encode(GateType.mcz, forKey: .type)
            try container.encode(controls, forKey: .controls)
            try container.encode(target, forKey: .target)
        case .rx(let theta, let target):
            try container.encode(GateType.rx, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(target, forKey: .target)
        case .ry(let theta, let target):
            try container.encode(GateType.ry, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(target, forKey: .target)
        case .rz(let theta, let target):
            try container.encode(GateType.rz, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(target, forKey: .target)
        case .crx(let theta, let control, let target):
            try container.encode(GateType.crx, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(control, forKey: .control)
            try container.encode(target, forKey: .target)
        case .cry(let theta, let control, let target):
            try container.encode(GateType.cry, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(control, forKey: .control)
            try container.encode(target, forKey: .target)
        case .crz(let theta, let control, let target):
            try container.encode(GateType.crz, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(control, forKey: .control)
            try container.encode(target, forKey: .target)
        case .cp(let theta, let control, let target):
            try container.encode(GateType.cp, forKey: .type)
            try container.encode(theta, forKey: .theta)
            try container.encode(control, forKey: .control)
            try container.encode(target, forKey: .target)
        case .measure(let spec):
            try container.encode(GateType.measure, forKey: .type)
            try container.encode(spec.qubits, forKey: .qubits)
            if spec.classicalRegister != 0 {
                try container.encode(spec.classicalRegister, forKey: .classicalRegister)
            }
            if spec.classicalBitOffset != 0 {
                try container.encode(spec.classicalBitOffset, forKey: .classicalBitOffset)
            }
        case .reset(let qubit):
            try container.encode(GateType.reset, forKey: .type)
            try container.encode(qubit, forKey: .qubit)
        case .c_if(let classicalRegister, let expectedValue, let gate):
            try container.encode(GateType.c_if, forKey: .type)
            try container.encode(classicalRegister, forKey: .classicalRegister)
            try container.encode(expectedValue, forKey: .expectedValue)
            try container.encode(gate, forKey: .gate)
        case .unitary1(let matrix, let target):
            try container.encode(GateType.unitary1, forKey: .type)
            try container.encode(matrix, forKey: .matrix)
            try container.encode(target, forKey: .target)
        case .initialize(let qubits, let amplitudes):
            try container.encode(GateType.initialize, forKey: .type)
            try container.encode(qubits, forKey: .qubits)
            try container.encode(amplitudes, forKey: .matrix)
        case .customUnitary(let matrix, let qubits):
            try container.encode(GateType.customUnitary, forKey: .type)
            try container.encode(matrix, forKey: .matrix)
            try container.encode(qubits, forKey: .qubits)
        }
    }
}
