import Foundation

// MARK: - Cells

/// One cell in the qubit/classical × moment visualization grid.
///
/// Row order is LSB-first: `q0` at row 0, then `q1` … `qN-1`, then classical bits
/// flattened in ``ClassicalRegisterSpec`` declaration order (`c0`, `c1`, …).
///
/// ## Glyph policy (``CircuitASCIIRenderer``)
/// - Controlled-X target is always ``target``(`"X"`) — not `⊕` (one convention).
/// - Control dots use ``control`` → `■`; intermediate span uses ``wire`` → `┼`.
/// - Placeholder labels: `║` barrier, `τ` delay, `IF` `c_if`, `W` `while_c`.
public enum CircuitVizCell: Equatable, Sendable {
    /// Idle horizontal wire for this moment.
    case idle
    /// Vertical connector through an intermediate qubit on a multi-qubit op.
    case wire
    /// Named gate glyph (already display-cased, e.g. `"H"`, `"RX"`).
    case gate(String)
    /// Control marker on a controlled operation.
    case control
    /// Target marker; `label` is typically `"X"`, `"Z"`, `"⊕"`, etc.
    case target(String)
    /// SWAP / iSWAP endpoint.
    case swap
    /// Measurement on a qubit wire.
    case measure
    /// Measurement sink on a classical bit row.
    case measureClassical
    /// Safe stand-in for `barrier` / `delay` / `c_if` / `while_c` (and similar).
    case placeholder(String)
}

// MARK: - Moment / layout

/// One packed time-step: independent ops share a column when their touched rows are disjoint.
public struct CircuitVizMoment: Equatable, Sendable {
    /// One entry per row (`qubitCount + classicalBitCount`).
    public let cells: [CircuitVizCell]
    /// Source gates packed into this moment (circuit order preserved).
    public let gateIndices: [Int]
}

/// Spatial grid for circuit visualization (Foundation-only; no UI frameworks).
///
/// **Moment packing:** greedy left-to-right. A gate joins the current moment when its
/// packing rows (qubit span including intermediates, plus classical bits for `measure`)
/// do not intersect rows already used in that moment; otherwise a new moment starts.
///
/// Named `CircuitVizLayout` (not `Layout` / `CircuitLayout`) to avoid colliding with the
/// transpiler logical↔physical ``Layout`` map.
public struct CircuitVizLayout: Equatable, Sendable {
    public let qubitCount: Int
    public let classicalBitCount: Int
    public let moments: [CircuitVizMoment]

    public var rowCount: Int { qubitCount + classicalBitCount }
    public var momentCount: Int { moments.count }

    /// Builds a visualization grid from a flat ``QuantumCircuit``.
    public init(circuit: QuantumCircuit) {
        self.qubitCount = circuit.qubitCount
        self.classicalBitCount = circuit.classicalRegisters.reduce(0) { $0 + $1.bitCount }
        self.moments = Self.packMoments(circuit: circuit)
    }

    /// Builds a visualization grid from a ``GateSequence`` body.
    public init(sequence: GateSequence) {
        self.init(circuit: sequence.body)
    }

    public func cell(row: Int, moment: Int) -> CircuitVizCell {
        guard moments.indices.contains(moment),
              moments[moment].cells.indices.contains(row) else {
            return .idle
        }
        return moments[moment].cells[row]
    }

    /// Row label for ASCII / SwiftUI headers (`q0`, `c0`, …).
    public func rowLabel(_ row: Int) -> String {
        if row < qubitCount {
            return "q\(row)"
        }
        return "c\(row - qubitCount)"
    }
}

// MARK: - Packing

extension CircuitVizLayout {
    fileprivate struct PreparedOp {
        let gateIndex: Int
        let packingRows: Set<Int>
        let placements: [(row: Int, cell: CircuitVizCell)]
    }

    private static func packMoments(circuit: QuantumCircuit) -> [CircuitVizMoment] {
        let qubitCount = circuit.qubitCount
        let classicalBitCount = circuit.classicalRegisters.reduce(0) { $0 + $1.bitCount }
        let rowCount = qubitCount + classicalBitCount
        let bases = classicalBitBases(circuit.classicalRegisters)

        var prepared: [PreparedOp] = []
        prepared.reserveCapacity(circuit.gates.count)
        for (index, gate) in circuit.gates.enumerated() {
            prepared.append(
                prepare(
                    gate: gate,
                    gateIndex: index,
                    qubitCount: qubitCount,
                    classicalBases: bases,
                    classicalWidths: circuit.classicalRegisters.map(\.bitCount)
                )
            )
        }

        var momentOps: [[PreparedOp]] = []
        var current: [PreparedOp] = []
        var occupied = Set<Int>()

        for op in prepared {
            if !op.packingRows.isDisjoint(with: occupied) {
                momentOps.append(current)
                current = []
                occupied = []
            }
            current.append(op)
            occupied.formUnion(op.packingRows)
        }
        if !current.isEmpty {
            momentOps.append(current)
        }

        return momentOps.map { ops in
            var cells = Array(repeating: CircuitVizCell.idle, count: rowCount)
            for op in ops {
                for placement in op.placements {
                    guard cells.indices.contains(placement.row) else { continue }
                    cells[placement.row] = placement.cell
                }
            }
            return CircuitVizMoment(
                cells: cells,
                gateIndices: ops.map(\.gateIndex)
            )
        }
    }

    private static func classicalBitBases(_ registers: [ClassicalRegisterSpec]) -> [Int] {
        var bases: [Int] = []
        var running = 0
        for spec in registers {
            bases.append(running)
            running += spec.bitCount
        }
        return bases
    }

    private static func prepare(
        gate: Gate,
        gateIndex: Int,
        qubitCount: Int,
        classicalBases: [Int],
        classicalWidths: [Int]
    ) -> PreparedOp {
        switch gate {
        case .measure(let spec):
            return prepareMeasure(
                spec: spec,
                gateIndex: gateIndex,
                qubitCount: qubitCount,
                classicalBases: classicalBases,
                classicalWidths: classicalWidths
            )

        case .barrier(let qubits):
            let resolved = qubits.isEmpty ? Array(0..<qubitCount) : qubits
            let rows = Set(resolved.filter { $0 >= 0 && $0 < qubitCount })
            let placements = rows.sorted().map { (row: $0, cell: CircuitVizCell.placeholder("║")) }
            return PreparedOp(gateIndex: gateIndex, packingRows: rows, placements: placements)

        case .delay(_, let qubit):
            let rows: Set<Int> = (qubit >= 0 && qubit < qubitCount) ? [qubit] : []
            let placements = rows.map { (row: $0, cell: CircuitVizCell.placeholder("τ")) }
            return PreparedOp(gateIndex: gateIndex, packingRows: rows, placements: Array(placements))

        case .c_if(_, _, let inner):
            return preparePlaceholder(
                label: "IF",
                qubits: inner.affectedQubits,
                gateIndex: gateIndex,
                qubitCount: qubitCount
            )

        case .while_c(_, _, let body, _):
            let qubits = body.flatMap(\.affectedQubits)
            return preparePlaceholder(
                label: "W",
                qubits: qubits,
                gateIndex: gateIndex,
                qubitCount: qubitCount
            )

        default:
            return prepareUnitaryLike(
                gate: gate,
                gateIndex: gateIndex,
                qubitCount: qubitCount
            )
        }
    }

    private static func preparePlaceholder(
        label: String,
        qubits: [Int],
        gateIndex: Int,
        qubitCount: Int
    ) -> PreparedOp {
        let unique = Array(Set(qubits.filter { $0 >= 0 && $0 < qubitCount })).sorted()
        let rows = Set(unique)
        let placements = unique.map { (row: $0, cell: CircuitVizCell.placeholder(label)) }
        return PreparedOp(gateIndex: gateIndex, packingRows: rows, placements: placements)
    }

    private static func prepareMeasure(
        spec: MeasureSpec,
        gateIndex: Int,
        qubitCount: Int,
        classicalBases: [Int],
        classicalWidths: [Int]
    ) -> PreparedOp {
        var packing = Set<Int>()
        var placements: [(row: Int, cell: CircuitVizCell)] = []

        for qubit in spec.qubits where qubit >= 0 && qubit < qubitCount {
            packing.insert(qubit)
            placements.append((qubit, .measure))
        }

        let register = spec.classicalRegister
        if register >= 0,
           register < classicalBases.count,
           register < classicalWidths.count {
            let base = classicalBases[register]
            let width = classicalWidths[register]
            for (offset, _) in spec.qubits.enumerated() {
                let bitInReg = spec.classicalBitOffset + offset
                guard bitInReg >= 0, bitInReg < width else { continue }
                let flat = base + bitInReg
                let row = qubitCount + flat
                packing.insert(row)
                placements.append((row, .measureClassical))
            }
        }

        return PreparedOp(gateIndex: gateIndex, packingRows: packing, placements: placements)
    }

    private static func prepareUnitaryLike(
        gate: Gate,
        gateIndex: Int,
        qubitCount: Int
    ) -> PreparedOp {
        let roles = qubitRoles(for: gate)
        let active = roles.keys.filter { $0 >= 0 && $0 < qubitCount }
        guard let minQ = active.min(), let maxQ = active.max() else {
            return PreparedOp(gateIndex: gateIndex, packingRows: [], placements: [])
        }

        var packing = Set<Int>()
        var placements: [(row: Int, cell: CircuitVizCell)] = []

        // Multi-qubit ops reserve the full vertical span so intermediates stay free of other gates.
        let spansVertically = active.count >= 2 || roles.values.contains(where: {
            if case .control = $0 { return true }
            if case .target = $0 { return true }
            if case .swap = $0 { return true }
            if case .wire = $0 { return true }
            return false
        })

        if spansVertically && minQ < maxQ {
            for q in minQ...maxQ {
                packing.insert(q)
                if let role = roles[q] {
                    placements.append((q, role))
                } else {
                    placements.append((q, .wire))
                }
            }
        } else {
            for q in active {
                packing.insert(q)
                if let role = roles[q] {
                    placements.append((q, role))
                }
            }
        }

        return PreparedOp(gateIndex: gateIndex, packingRows: packing, placements: placements)
    }

    /// Per-qubit display role for unitary / reset / initialize ops (no classical rows).
    private static func qubitRoles(for gate: Gate) -> [Int: CircuitVizCell] {
        switch gate {
        case .h(let t): return [t: .gate("H")]
        case .x(let t): return [t: .gate("X")]
        case .y(let t): return [t: .gate("Y")]
        case .z(let t): return [t: .gate("Z")]
        case .s(let t): return [t: .gate("S")]
        case .t(let t): return [t: .gate("T")]
        case .sdg(let t): return [t: .gate("SDG")]
        case .tdg(let t): return [t: .gate("TDG")]
        case .sx(let t): return [t: .gate("SX")]
        case .sxdg(let t): return [t: .gate("SXDG")]
        case .id(let t): return [t: .gate("I")]
        case .p(_, let t): return [t: .gate("P")]
        case .u(_, _, _, let t): return [t: .gate("U")]
        case .rx(_, let t): return [t: .gate("RX")]
        case .ry(_, let t): return [t: .gate("RY")]
        case .rz(_, let t): return [t: .gate("RZ")]
        case .reset(let t): return [t: .gate("|0⟩")]
        case .unitary1(_, let t): return [t: .gate("U1")]

        case .cx(let c, let t):
            return [c: .control, t: .target("X")]
        case .cz(let c, let t):
            return [c: .control, t: .target("Z")]
        case .ecr(let c, let t):
            return [c: .control, t: .target("ECR")]
        case .crx(_, let c, let t):
            return [c: .control, t: .target("RX")]
        case .cry(_, let c, let t):
            return [c: .control, t: .target("RY")]
        case .crz(_, let c, let t):
            return [c: .control, t: .target("RZ")]
        case .cp(_, let c, let t):
            return [c: .control, t: .target("P")]

        case .swap(let q1, let q2), .iswap(let q1, let q2):
            return [q1: .swap, q2: .swap]
        case .dcx(let q1, let q2):
            return [q1: .gate("DCX"), q2: .gate("DCX")]
        case .rxx(_, let q1, let q2):
            return [q1: .gate("RXX"), q2: .gate("RXX")]
        case .ryy(_, let q1, let q2):
            return [q1: .gate("RYY"), q2: .gate("RYY")]
        case .rzz(_, let q1, let q2):
            return [q1: .gate("RZZ"), q2: .gate("RZZ")]

        case .cswap(let c, let q1, let q2):
            return [c: .control, q1: .swap, q2: .swap]
        case .ccx(let c1, let c2, let t):
            return [c1: .control, c2: .control, t: .target("X")]
        case .mcx(let controls, let t):
            var map: [Int: CircuitVizCell] = [t: .target("X")]
            for c in controls { map[c] = .control }
            return map
        case .mcz(let controls, let t):
            var map: [Int: CircuitVizCell] = [t: .target("Z")]
            for c in controls { map[c] = .control }
            return map

        case .initialize(let qubits, _):
            var map: [Int: CircuitVizCell] = [:]
            for q in qubits { map[q] = .gate("INIT") }
            return map
        case .customUnitary(_, let qubits):
            var map: [Int: CircuitVizCell] = [:]
            for q in qubits { map[q] = .gate("U*") }
            return map

        case .barrier, .delay, .measure, .c_if, .while_c:
            return [:]
        }
    }
}
