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
        case .while_c(let classicalRegister, let expectedValue, let body, let maxIterations):
            return .while_c(
                classicalRegister: classicalRegister,
                expectedValue: expectedValue,
                body: try body.map { try $0.remappingQubits(map) },
                maxIterations: maxIterations
            )
        case .unitary1(let matrix, let target):
            return .unitary1(matrix: matrix, target: try map(target))
        case .initialize(let qubits, let amplitudes):
            return .initialize(qubits: try qubits.map(map), amplitudes: amplitudes)
        case .customUnitary(let matrix, let qubits):
            return .customUnitary(matrix: matrix, qubits: try qubits.map(map))
        }
    }

    /// Rewrites classical-register indices on ``measure`` / ``c_if`` / ``while_c`` (nested included).
    public func remappingClassicalRegisters(_ map: (Int) throws -> Int) rethrows -> Gate {
        switch self {
        case .measure(let spec):
            return .measure(
                MeasureSpec(
                    qubits: spec.qubits,
                    classicalRegister: try map(spec.classicalRegister),
                    classicalBitOffset: spec.classicalBitOffset
                )
            )
        case .c_if(let classicalRegister, let expectedValue, let gate):
            return .c_if(
                classicalRegister: try map(classicalRegister),
                expectedValue: expectedValue,
                gate: try gate.remappingClassicalRegisters(map)
            )
        case .while_c(let classicalRegister, let expectedValue, let body, let maxIterations):
            return .while_c(
                classicalRegister: try map(classicalRegister),
                expectedValue: expectedValue,
                body: try body.map { try $0.remappingClassicalRegisters(map) },
                maxIterations: maxIterations
            )
        default:
            return self
        }
    }

    /// Lifts this gate to a controlled form with the given control qubits (applied when all are |1⟩).
    ///
    /// Returns a single gate. Throws ``QuantumCircuitError/unsupportedControlledGate`` for
    /// non-unitary ops and for unitaries without a native controlled encoding in ``Gate``.
    public func controlled(by controls: [Int]) throws -> Gate {
        guard !controls.isEmpty else {
            throw QuantumCircuitError.unsupportedControlledGate(
                reason: "controlled lift requires at least one control qubit"
            )
        }
        guard Set(controls).count == controls.count else {
            throw QuantumCircuitError.unsupportedControlledGate(
                reason: "control qubits must be distinct"
            )
        }
        let targetQubits = affectedQubits
        for control in controls {
            if targetQubits.contains(control) {
                throw QuantumCircuitError.unsupportedControlledGate(
                    reason: "control qubit \(control) overlaps gate targets \(targetQubits)"
                )
            }
        }

        switch self {
        case .measure:
            throw QuantumCircuitError.unsupportedControlledGate(
                reason: "measure cannot be controlled"
            )
        case .reset:
            throw QuantumCircuitError.unsupportedControlledGate(
                reason: "reset cannot be controlled"
            )
        case .c_if:
            throw QuantumCircuitError.unsupportedControlledGate(
                reason: "c_if cannot be controlled"
            )
        case .while_c:
            throw QuantumCircuitError.unsupportedControlledGate(
                reason: "while_c cannot be controlled"
            )
        case .initialize:
            throw QuantumCircuitError.unsupportedControlledGate(
                reason: "initialize cannot be controlled"
            )

        case .id(let target):
            // Controlled identity is still identity on the target.
            return .id(target: target)

        case .barrier(let qubits):
            var combined = controls
            for q in qubits where !combined.contains(q) {
                combined.append(q)
            }
            return .barrier(qubits: combined)

        case .delay(let duration, let qubit):
            // Delay is scheduling metadata; leave it on the target qubit.
            return .delay(duration: duration, qubit: qubit)

        case .x(let target):
            return Self.controlledX(controls: controls, target: target)

        case .z(let target):
            return Self.controlledZ(controls: controls, target: target)

        case .cx(let existingControl, let target):
            return Self.controlledX(controls: controls + [existingControl], target: target)

        case .ccx(let c1, let c2, let target):
            return Self.controlledX(controls: controls + [c1, c2], target: target)

        case .mcx(let existing, let target):
            return Self.controlledX(controls: controls + existing, target: target)

        case .cz(let existingControl, let target):
            return Self.controlledZ(controls: controls + [existingControl], target: target)

        case .mcz(let existing, let target):
            return Self.controlledZ(controls: controls + existing, target: target)

        case .swap(let q1, let q2):
            guard controls.count == 1 else {
                throw QuantumCircuitError.unsupportedControlledGate(
                    reason: "SWAP supports a single control (CSWAP); got \(controls.count)"
                )
            }
            return .cswap(control: controls[0], q1: q1, q2: q2)

        case .rx(let theta, let target):
            guard controls.count == 1 else {
                throw QuantumCircuitError.unsupportedControlledGate(
                    reason: "rx supports a single control (crx); got \(controls.count)"
                )
            }
            return .crx(theta: theta, control: controls[0], target: target)

        case .ry(let theta, let target):
            guard controls.count == 1 else {
                throw QuantumCircuitError.unsupportedControlledGate(
                    reason: "ry supports a single control (cry); got \(controls.count)"
                )
            }
            return .cry(theta: theta, control: controls[0], target: target)

        case .rz(let theta, let target):
            guard controls.count == 1 else {
                throw QuantumCircuitError.unsupportedControlledGate(
                    reason: "rz supports a single control (crz); got \(controls.count)"
                )
            }
            return .crz(theta: theta, control: controls[0], target: target)

        case .p(let theta, let target):
            guard controls.count == 1 else {
                throw QuantumCircuitError.unsupportedControlledGate(
                    reason: "p supports a single control (cp); got \(controls.count)"
                )
            }
            return .cp(theta: theta, control: controls[0], target: target)

        case .h, .y, .s, .t, .sdg, .tdg, .sx, .sxdg, .u,
             .crx, .cry, .crz, .cp, .cswap, .iswap, .ecr, .rxx, .ryy, .rzz, .dcx,
             .unitary1, .customUnitary:
            throw QuantumCircuitError.unsupportedControlledGate(
                reason: "no native controlled form for \(kind.rawValue)"
            )
        }
    }

    private static func controlledX(controls: [Int], target: Int) -> Gate {
        switch controls.count {
        case 1:
            return .cx(control: controls[0], target: target)
        case 2:
            return .ccx(control1: controls[0], control2: controls[1], target: target)
        default:
            return .mcx(controls: controls, target: target)
        }
    }

    private static func controlledZ(controls: [Int], target: Int) -> Gate {
        if controls.count == 1 {
            return .cz(control: controls[0], target: target)
        }
        return .mcz(controls: controls, target: target)
    }
}
