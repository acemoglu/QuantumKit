import CoreTransferable
import Foundation
import QuantumKit
import UniformTypeIdentifiers

/// A placeable playground gate. Drag from the palette onto a qubit wire.
enum PaletteTool: String, Codable, CaseIterable, Identifiable, Hashable, Sendable {
    case h, x, y, z, s, t, sdg, tdg, sx
    case rx, ry, rz
    case cx, cz, swap
    case ccx
    case measure, barrier, reset

    var id: String { rawValue }

    var title: String {
        switch self {
        case .h: return "H"
        case .x: return "X"
        case .y: return "Y"
        case .z: return "Z"
        case .s: return "S"
        case .t: return "T"
        case .sdg: return "S†"
        case .tdg: return "T†"
        case .sx: return "√X"
        case .rx: return "RX"
        case .ry: return "RY"
        case .rz: return "RZ"
        case .cx: return "CNOT"
        case .cz: return "CZ"
        case .swap: return "SWAP"
        case .ccx: return "Toffoli"
        case .measure: return "Measure"
        case .barrier: return "Barrier"
        case .reset: return "Reset"
        }
    }

    var help: String {
        switch self {
        case .h: return "Hadamard"
        case .x: return "Pauli X"
        case .y: return "Pauli Y"
        case .z: return "Pauli Z"
        case .s: return "S phase"
        case .t: return "T phase"
        case .sdg: return "S dagger"
        case .tdg: return "T dagger"
        case .sx: return "Square-root X"
        case .rx: return "RX(π/2)"
        case .ry: return "RY(π/2)"
        case .rz: return "RZ(π/2)"
        case .cx: return "CNOT — drop control, then target"
        case .cz: return "CZ — drop control, then target"
        case .swap: return "SWAP — drop two qubits"
        case .ccx: return "Toffoli — control, control, target"
        case .measure: return "Measure into matching classical bit"
        case .barrier: return "Barrier on this qubit"
        case .reset: return "Reset to |0⟩"
        }
    }

    var qubitCount: Int {
        switch self {
        case .cx, .cz, .swap: return 2
        case .ccx: return 3
        default: return 1
        }
    }

    var section: PaletteSection {
        switch self {
        case .h, .x, .y, .z, .s, .t, .sdg, .tdg, .sx:
            return .oneQubit
        case .rx, .ry, .rz:
            return .rotations
        case .cx, .cz, .swap, .ccx:
            return .multiQubit
        case .measure, .barrier, .reset:
            return .ops
        }
    }

    func makeGate(qubits: [Int]) -> Gate? {
        guard qubits.count == qubitCount else { return nil }
        switch self {
        case .h: return .h(target: qubits[0])
        case .x: return .x(target: qubits[0])
        case .y: return .y(target: qubits[0])
        case .z: return .z(target: qubits[0])
        case .s: return .s(target: qubits[0])
        case .t: return .t(target: qubits[0])
        case .sdg: return .sdg(target: qubits[0])
        case .tdg: return .tdg(target: qubits[0])
        case .sx: return .sx(target: qubits[0])
        case .rx: return .rx(theta: QFloatExpr(QFloat.pi / 2), target: qubits[0])
        case .ry: return .ry(theta: QFloatExpr(QFloat.pi / 2), target: qubits[0])
        case .rz: return .rz(theta: QFloatExpr(QFloat.pi / 2), target: qubits[0])
        case .cx: return .cx(control: qubits[0], target: qubits[1])
        case .cz: return .cz(control: qubits[0], target: qubits[1])
        case .swap: return .swap(q1: qubits[0], q2: qubits[1])
        case .ccx: return .ccx(control1: qubits[0], control2: qubits[1], target: qubits[2])
        case .measure:
            return .measure(MeasureSpec(qubits: [qubits[0]], classicalRegister: 0, classicalBitOffset: qubits[0]))
        case .barrier: return .barrier(qubits: [qubits[0]])
        case .reset: return .reset(qubit: qubits[0])
        }
    }
}

enum PaletteSection: String, CaseIterable, Identifiable {
    case oneQubit
    case rotations
    case multiQubit
    case ops

    var id: String { rawValue }

    var title: String {
        switch self {
        case .oneQubit: return "1-qubit"
        case .rotations: return "Rotations"
        case .multiQubit: return "2/3-qubit"
        case .ops: return "Measure & reset"
        }
    }
}

extension PaletteTool: Transferable {
    static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .json)
        ProxyRepresentation(exporting: \.rawValue) { raw in
            PaletteTool(rawValue: raw) ?? .h
        }
    }
}

struct PendingGatePlacement: Equatable, Sendable {
    var tool: PaletteTool
    var pickedQubits: [Int]
    var insertAt: Int?
}

/// Ready-made fragments stamped onto the canvas from a starting qubit.
enum CircuitBlock: String, CaseIterable, Identifiable, Hashable, Sendable {
    case bell
    case ghz3
    case hAll
    case measureAll

    var id: String { rawValue }

    var title: String {
        switch self {
        case .bell: return "Bell"
        case .ghz3: return "GHZ-3"
        case .hAll: return "H all"
        case .measureAll: return "Meas all"
        }
    }

    var help: String {
        switch self {
        case .bell: return "H + CNOT on this qubit and the next"
        case .ghz3: return "H + two CNOTs on this qubit and the next two"
        case .hAll: return "Hadamard on every qubit"
        case .measureAll: return "Measure every qubit"
        }
    }

    func makeGates(start: Int, qubitCount: Int) -> [Gate] {
        switch self {
        case .bell:
            return [
                .h(target: start),
                .cx(control: start, target: start + 1),
            ]
        case .ghz3:
            return [
                .h(target: start),
                .cx(control: start, target: start + 1),
                .cx(control: start, target: start + 2),
            ]
        case .hAll:
            return (0..<qubitCount).map { .h(target: $0) }
        case .measureAll:
            return (0..<qubitCount).map { qubit in
                .measure(MeasureSpec(qubits: [qubit], classicalRegister: 0, classicalBitOffset: qubit))
            }
        }
    }
}
