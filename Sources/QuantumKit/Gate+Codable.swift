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
        case classicalRegister
        case expectedValue
        case gate
    }

    private enum GateType: String, Codable {
        case h, x, y, z, s, t, sdg, tdg, sx, sxdg, p, u
        case cx, cz, swap, ccx, mcx, mcz
        case rx, ry, rz, crx, cry, crz, cp
        case measure, reset, c_if
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
                theta: try container.decode(QFloat.self, forKey: .theta),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .u:
            self = .u(
                theta: try container.decode(QFloat.self, forKey: .theta),
                phi: try container.decode(QFloat.self, forKey: .phi),
                lambda: try container.decode(QFloat.self, forKey: .lambda),
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
                theta: try container.decode(QFloat.self, forKey: .theta),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .ry:
            self = .ry(
                theta: try container.decode(QFloat.self, forKey: .theta),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .rz:
            self = .rz(
                theta: try container.decode(QFloat.self, forKey: .theta),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .crx:
            self = .crx(
                theta: try container.decode(QFloat.self, forKey: .theta),
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .cry:
            self = .cry(
                theta: try container.decode(QFloat.self, forKey: .theta),
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .crz:
            self = .crz(
                theta: try container.decode(QFloat.self, forKey: .theta),
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .cp:
            self = .cp(
                theta: try container.decode(QFloat.self, forKey: .theta),
                control: try container.decode(Int.self, forKey: .control),
                target: try container.decode(Int.self, forKey: .target)
            )
        case .measure:
            self = .measure(qubits: try container.decode([Int].self, forKey: .qubits))
        case .reset:
            self = .reset(qubit: try container.decode(Int.self, forKey: .qubit))
        case .c_if:
            self = .c_if(
                classicalRegister: try container.decode(Int.self, forKey: .classicalRegister),
                expectedValue: try container.decode(Int.self, forKey: .expectedValue),
                gate: try container.decode(Gate.self, forKey: .gate)
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
        case .measure(let qubits):
            try container.encode(GateType.measure, forKey: .type)
            try container.encode(qubits, forKey: .qubits)
        case .reset(let qubit):
            try container.encode(GateType.reset, forKey: .type)
            try container.encode(qubit, forKey: .qubit)
        case .c_if(let classicalRegister, let expectedValue, let gate):
            try container.encode(GateType.c_if, forKey: .type)
            try container.encode(classicalRegister, forKey: .classicalRegister)
            try container.encode(expectedValue, forKey: .expectedValue)
            try container.encode(gate, forKey: .gate)
        }
    }
}
