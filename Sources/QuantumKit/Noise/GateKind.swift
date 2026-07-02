import Foundation

/// Gate family used for localized noise targeting (ignores symbolic angles and qubit indices).
public enum GateKind: String, Sendable, Equatable, Codable, CaseIterable {
    case h, x, y, z
    case s, t, sdg, tdg
    case sx, sxdg
    case p, u
    case rx, ry, rz
    case cx, cz, swap
    case ccx, mcx, mcz
    case crx, cry, crz, cp
    case measure, reset, unitary1, c_if
}

extension Gate {

    /// The gate family for noise-rule matching.
    public var kind: GateKind {
        switch self {
        case .h: return .h
        case .x: return .x
        case .y: return .y
        case .z: return .z
        case .s: return .s
        case .t: return .t
        case .sdg: return .sdg
        case .tdg: return .tdg
        case .sx: return .sx
        case .sxdg: return .sxdg
        case .p: return .p
        case .u: return .u
        case .rx: return .rx
        case .ry: return .ry
        case .rz: return .rz
        case .cx: return .cx
        case .cz: return .cz
        case .swap: return .swap
        case .ccx: return .ccx
        case .mcx: return .mcx
        case .mcz: return .mcz
        case .crx: return .crx
        case .cry: return .cry
        case .crz: return .crz
        case .cp: return .cp
        case .measure: return .measure
        case .reset: return .reset
        case .unitary1: return .unitary1
        case .c_if: return .c_if
        }
    }
}
