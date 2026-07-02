import Foundation

/// Hardware-native gate identity, ignoring rotation angles and qubit indices.
public enum BasisGateKind: String, Sendable, Hashable, CaseIterable, Codable {
    case rz
    case sx
    case sxdg
    case x
    case cx
}

/// A target instruction set for basis translation.
public struct BasisGateSet: Sendable, Equatable {
    public let kinds: Set<BasisGateKind>

    public init(_ kinds: BasisGateKind...) {
        self.kinds = Set(kinds)
    }

    public init(kinds: Set<BasisGateKind>) {
        self.kinds = kinds
    }

    /// IBM-style physical basis: Rz, SX, SX†, and CX.
    public static let ibmEagle: BasisGateSet = BasisGateSet(.rz, .sx, .sxdg, .cx)

    public func contains(_ kind: BasisGateKind) -> Bool {
        kinds.contains(kind)
    }

    /// Whether `gate` is already native to this basis (parametrized angles are ignored).
    public func contains(_ gate: Gate) -> Bool {
        guard let kind = BasisGateKind(gate: gate) else { return false }
        return kinds.contains(kind)
    }
}

extension BasisGateKind {

    init?(gate: Gate) {
        switch gate {
        case .rz:
            self = .rz
        case .sx:
            self = .sx
        case .sxdg:
            self = .sxdg
        case .x:
            self = .x
        case .cx:
            self = .cx
        default:
            return nil
        }
    }
}
