import Foundation

public enum StabilizerError: Error, Equatable, CustomStringConvertible {
    case invalidQubitCount(Int)
    case qubitCountExceedsLimit(max: Int, requested: Int)
    case qubitCountMismatch(circuit: Int, tableau: Int)
    case nonCliffordGate(Gate)
    case noiseNotSupported
    case nonProjectiveMeasurementNotSupported

    public var description: String {
        switch self {
        case .invalidQubitCount(let n):
            return "Stabilizer tableau requires qubitCount > 0 (got \(n))."
        case .qubitCountExceedsLimit(let max, let requested):
            return "Stabilizer qubit count \(requested) exceeds limit \(max)."
        case .qubitCountMismatch(let circuit, let tableau):
            return "Circuit width \(circuit) does not match tableau width \(tableau)."
        case .nonCliffordGate(let gate):
            return "Stabilizer backend rejected non-Clifford gate: \(gate)."
        case .noiseNotSupported:
            return "Stabilizer backend does not support noise models."
        case .nonProjectiveMeasurementNotSupported:
            return "Stabilizer backend only supports projective measurement."
        }
    }
}

/// Validates that a circuit stays in the Clifford+measure group supported by
/// ``StabilizerBackend``.
public enum StabilizerCircuitValidator: Sendable {

    /// Native generators applied directly on the tableau (plus measure / classical control).
    public static let supportedCliffordGatesDescription = """
        H, S, Sdg, X, Y, Z, SX, SXdg, CX, CZ, SWAP, DCX, iSWAP, I, barrier, delay; \
        measure; reset; c_if when the body is supported. \
        Rejects T/Tdg, RX/RY/RZ and other parametrized rotations, ECR, RXX/RYY/RZZ, \
        CCX/MCX/MCZ/CSWAP, unitary1/customUnitary/initialize, and any noise model.
        """

    public static func isSupported(_ circuit: QuantumCircuit) -> Bool {
        (try? validate(circuit)) != nil
    }

    public static func validate(_ circuit: QuantumCircuit) throws {
        for gate in circuit.gates {
            try validate(gate)
        }
    }

    public static func validate(_ gate: Gate) throws {
        switch gate {
        case .h, .x, .y, .z, .s, .sdg, .sx, .sxdg,
             .cx, .cz, .swap, .dcx, .iswap, .id, .barrier, .delay,
             .measure, .reset:
            return
        case .c_if(_, _, let body):
            try validate(body)
        default:
            throw StabilizerError.nonCliffordGate(gate)
        }
    }

    /// Same Clifford generators as ``CliffordSimplificationPass`` plus SX/SXdg/DCX/iSWAP.
    public static func isCliffordUnitary(_ gate: Gate) -> Bool {
        switch gate {
        case .h, .x, .y, .z, .s, .sdg, .sx, .sxdg,
             .cx, .cz, .swap, .dcx, .iswap, .id:
            return true
        default:
            return false
        }
    }
}

/// Host-side stabilizer engine (CHP tableau). No Metal kernels.
public final class StabilizerEngine: @unchecked Sendable {
    public init() {}

    public func execute(
        _ circuit: QuantumCircuit,
        on tableau: inout StabilizerTableau,
        rng: inout QuantumRNG
    ) throws -> CircuitExecutionResult {
        guard circuit.qubitCount == tableau.qubitCount else {
            throw StabilizerError.qubitCountMismatch(
                circuit: circuit.qubitCount,
                tableau: tableau.qubitCount
            )
        }
        try circuit.requireFullyBound()
        try StabilizerCircuitValidator.validate(circuit)

        var measurementOutcomes: [[Int]] = []
        var appliedGateCount = 0
        var classicalMemory = ClassicalMemory(
            registerWidths: circuit.classicalRegisters.map(\.bitCount)
        )

        func bits(from outcome: Int, count: Int) -> [Int] {
            (0..<count).map { (outcome >> $0) & 1 }
        }

        func executeRuntimeGate(_ gate: Gate, countsTowardApplied: Bool = true) throws {
            switch gate {
            case .measure(let spec):
                let outcome = tableau.measure(qubits: spec.qubits, rng: &rng)
                measurementOutcomes.append(bits(from: outcome, count: spec.qubits.count))
                try classicalMemory.writeOutcome(
                    outcome,
                    measuredQubitCount: spec.qubits.count,
                    register: spec.classicalRegister,
                    bitOffset: spec.classicalBitOffset
                )

            case .reset(let qubit):
                tableau.reset(qubit: qubit, rng: &rng)

            case .barrier, .delay, .id:
                break

            case .c_if(let classicalRegister, let expectedValue, let conditionedGate):
                if classicalMemory.value(ofRegister: classicalRegister) == expectedValue {
                    try executeRuntimeGate(conditionedGate, countsTowardApplied: false)
                }

            case .h(let t):
                tableau.applyH(qubit: t)
            case .s(let t):
                tableau.applyS(qubit: t)
            case .sdg(let t):
                tableau.applySdg(qubit: t)
            case .x(let t):
                tableau.applyX(qubit: t)
            case .y(let t):
                tableau.applyY(qubit: t)
            case .z(let t):
                tableau.applyZ(qubit: t)
            case .sx(let t):
                tableau.applySX(qubit: t)
            case .sxdg(let t):
                tableau.applySXdg(qubit: t)
            case .cx(let c, let t):
                tableau.applyCX(control: c, target: t)
            case .cz(let c, let t):
                tableau.applyCZ(control: c, target: t)
            case .swap(let q1, let q2):
                tableau.applySWAP(q1: q1, q2: q2)
            case .dcx(let q1, let q2):
                tableau.applyCX(control: q1, target: q2)
                tableau.applyCX(control: q2, target: q1)
            case .iswap(let q1, let q2):
                // Same Clifford expansion as GateDecomposition (S/H/CX only).
                tableau.applyS(qubit: q1)
                tableau.applyS(qubit: q2)
                tableau.applyH(qubit: q1)
                tableau.applyCX(control: q1, target: q2)
                tableau.applyCX(control: q2, target: q1)
                tableau.applyH(qubit: q2)

            default:
                throw StabilizerError.nonCliffordGate(gate)
            }

            if countsTowardApplied {
                appliedGateCount += 1
            }
        }

        for gate in circuit.gates {
            try executeRuntimeGate(gate)
        }

        return CircuitExecutionResult(
            measurementOutcomes: measurementOutcomes,
            classicalMemory: classicalMemory,
            appliedGateCount: appliedGateCount,
            unitaryRenormCount: nil
        )
    }

    /// Terminal computational-basis sample of all qubits (after circuit evolution).
    func sampleTerminalOutcome(
        circuit: QuantumCircuit,
        tableau: inout StabilizerTableau,
        rng: inout QuantumRNG
    ) throws -> Int {
        _ = try execute(circuit, on: &tableau, rng: &rng)
        return tableau.measure(qubits: Array(0..<circuit.qubitCount), rng: &rng)
    }
}
