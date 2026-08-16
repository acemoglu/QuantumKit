//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

public enum QuantumCircuitError: Error, Equatable {
    case invalidQubitCount(Int)
    case qubitIndexOutOfBounds(index: Int, qubitCount: Int)
    case invalidAlgorithmParameter(reason: String)
    case circuitNotUnitary
    /// Qubit / classical-register map or width conflict during append / compose / tensor.
    case invalidComposition(reason: String)
    /// Gate cannot be lifted to a controlled form (non-unitary or no native controlled encoding).
    case unsupportedControlledGate(reason: String)
    /// ``Gate/while_c`` reached ``maxIterations`` without the classical exit condition becoming false.
    case maxLoopIterationsExceeded(maxIterations: Int)
}

public struct QuantumCircuit: Sendable {

    /// Logical width; may grow under ``append`` / ``formTensor`` when maps require it.
    public internal(set) var qubitCount: Int
    /// Classical register declarations; may grow when appending / tensoring circuits with cregs.
    public internal(set) var classicalRegisters: [ClassicalRegisterSpec]
    public private(set) var gates: [Gate] = []
    /// Parallel metadata aligned with ``gates`` (same length). Execution ignores these entries.
    public private(set) var instructionMetadata: [InstructionMetadata?] = []

    public init(qubitCount: Int, classicalRegisters: [ClassicalRegisterSpec] = []) throws {
        guard qubitCount > 0 else {
            throw QuantumCircuitError.invalidQubitCount(qubitCount)
        }

        self.qubitCount = qubitCount
        self.classicalRegisters = classicalRegisters
    }

    private func validateQubitIndex(_ index: Int) throws {
        guard index >= 0, index < qubitCount else {
            throw QuantumCircuitError.qubitIndexOutOfBounds(index: index, qubitCount: qubitCount)
        }
    }

    /// Validates gate structure against this circuit's width without appending.
    func validateGateStructure(_ gate: Gate) throws {
        switch gate {
        case .h(let target), .x(let target), .y(let target), .z(let target),
             .s(let target), .t(let target), .sdg(let target), .tdg(let target),
             .sx(let target), .sxdg(let target):
            try validateQubitIndex(target)
        case .p(_, let target), .u(_, _, _, let target):
            try validateQubitIndex(target)
        case .rx(_, let target), .ry(_, let target), .rz(_, let target):
            try validateQubitIndex(target)
        case .crx(_, let control, let target), .cry(_, let control, let target),
             .crz(_, let control, let target), .cp(_, let control, let target):
            try validateQubitIndex(control)
            try validateQubitIndex(target)
            guard control != target else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Controlled rotation requires distinct control and target qubits"
                )
            }
        case .mcx(let controls, let target), .mcz(let controls, let target):
            guard !controls.isEmpty else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Multi-controlled gate requires at least one control qubit"
                )
            }
            try validateQubitIndex(target)
            for control in controls {
                try validateQubitIndex(control)
            }
            guard Set(controls).count == controls.count else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Multi-controlled gate requires distinct control qubits"
                )
            }
            guard !controls.contains(target) else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Multi-controlled gate target must differ from all controls"
                )
            }
        case .cx(let control, let target):
            try validateQubitIndex(control)
            try validateQubitIndex(target)
            guard control != target else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "CNOT requires distinct control and target qubits"
                )
            }
        case .cz(let control, let target):
            try validateQubitIndex(control)
            try validateQubitIndex(target)
            guard control != target else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "CZ requires distinct control and target qubits"
                )
            }
        case .swap(let q1, let q2):
            try validateQubitIndex(q1)
            try validateQubitIndex(q2)
            guard q1 != q2 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "SWAP requires two distinct qubits"
                )
            }
        case .id(let target):
            try validateQubitIndex(target)
        case .barrier(let qubits):
            // Empty barrier is allowed (caller may resolve to all qubits later).
            for qubit in qubits {
                try validateQubitIndex(qubit)
            }
        case .delay(let duration, let qubit):
            guard duration >= 0 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "delay duration must be non-negative"
                )
            }
            try validateQubitIndex(qubit)
        case .iswap(let q1, let q2), .dcx(let q1, let q2):
            try validateQubitIndex(q1)
            try validateQubitIndex(q2)
            guard q1 != q2 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Two-qubit gate requires distinct qubits"
                )
            }
        case .ecr(let control, let target):
            try validateQubitIndex(control)
            try validateQubitIndex(target)
            guard control != target else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "ECR requires distinct control and target qubits"
                )
            }
        case .rxx(_, let q1, let q2), .ryy(_, let q1, let q2), .rzz(_, let q1, let q2):
            try validateQubitIndex(q1)
            try validateQubitIndex(q2)
            guard q1 != q2 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Ising rotation requires two distinct qubits"
                )
            }
        case .cswap(let control, let q1, let q2):
            try validateQubitIndex(control)
            try validateQubitIndex(q1)
            try validateQubitIndex(q2)
            guard control != q1, control != q2, q1 != q2 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "CSWAP requires three distinct qubits"
                )
            }
        case .ccx(let control1, let control2, let target):
            try validateQubitIndex(control1)
            try validateQubitIndex(control2)
            try validateQubitIndex(target)
            guard control1 != control2, control1 != target, control2 != target else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "CCX requires three distinct qubits (control1, control2, target)"
                )
            }
        case .measure(let spec):
            guard !spec.qubits.isEmpty else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Measure requires at least one qubit"
                )
            }
            for index in spec.qubits {
                try validateQubitIndex(index)
            }
            guard spec.classicalRegister >= 0 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Measurement requires a non-negative classical register index"
                )
            }
            guard spec.classicalBitOffset >= 0 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Measurement requires a non-negative classical bit offset"
                )
            }
        case .reset(let qubit):
            try validateQubitIndex(qubit)
        case .c_if(let classicalRegister, let expectedValue, let conditionedGate):
            guard classicalRegister >= 0 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Conditional gate requires a non-negative classical register index"
                )
            }
            guard expectedValue >= 0 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "Conditional gate requires a non-negative expected classical value"
                )
            }
            try validateGateStructure(conditionedGate)
        case .while_c(let classicalRegister, let expectedValue, let body, let maxIterations):
            guard classicalRegister >= 0 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "while_c requires a non-negative classical register index"
                )
            }
            guard expectedValue >= 0 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "while_c requires a non-negative expected classical value"
                )
            }
            guard maxIterations > 0 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "while_c requires maxIterations > 0 (unbounded loops are rejected)"
                )
            }
            for nested in body {
                try validateGateStructure(nested)
            }
        case .unitary1(let matrix, let target):
            guard matrix.count == 4 else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "unitary1 requires exactly four complex amplitudes"
                )
            }
            try validateQubitIndex(target)
            try UnitaryValidation.validateUnitary(matrix: matrix, dimension: 2)
        case .initialize(let qubits, let amplitudes):
            guard !qubits.isEmpty else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "initialize requires at least one qubit"
                )
            }
            guard Set(qubits).count == qubits.count else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "initialize requires distinct qubit indices"
                )
            }
            for qubit in qubits {
                try validateQubitIndex(qubit)
            }
            let expectedCount = 1 << qubits.count
            guard amplitudes.count == expectedCount else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "initialize requires \(expectedCount) amplitudes for \(qubits.count) qubits"
                )
            }
            do {
                try UnitaryValidation.validateUnitNorm(amplitudes)
            } catch let error as StateVectorInitializationError {
                if case .nonUnitNorm(let squaredNorm) = error {
                    throw QuantumCircuitError.invalidAlgorithmParameter(
                        reason: "initialize amplitudes are not normalized (squared norm \(squaredNorm))"
                    )
                }
                throw error
            }
        case .customUnitary(let matrix, let qubits):
            guard !qubits.isEmpty else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "customUnitary requires at least one qubit"
                )
            }
            guard Set(qubits).count == qubits.count else {
                throw QuantumCircuitError.invalidAlgorithmParameter(
                    reason: "customUnitary requires distinct qubit indices"
                )
            }
            for qubit in qubits {
                try validateQubitIndex(qubit)
            }
            try UnitaryValidation.validateUnitary(matrix: matrix, dimension: 1 << qubits.count)
        }
    }

    mutating func applyValidated(_ gate: Gate) throws {
        try validateGateStructure(gate)
        gates.append(gate)
        instructionMetadata.append(nil)
    }

    public mutating func apply(_ gate: Gate) throws {
        try applyValidated(gate)
    }

    /// Appends `gate` with optional non-semantic ``InstructionMetadata``.
    public mutating func apply(_ gate: Gate, metadata: InstructionMetadata?) throws {
        try applyValidated(gate)
        instructionMetadata[instructionMetadata.count - 1] = metadata
    }

    /// Sets metadata for an existing instruction index without changing the gate.
    public mutating func setInstructionMetadata(_ metadata: InstructionMetadata?, at index: Int) throws {
        guard gates.indices.contains(index) else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "instruction metadata index \(index) is out of range 0..<\(gates.count)"
            )
        }
        ensureMetadataLength()
        instructionMetadata[index] = metadata
    }

    /// Returns metadata for `index`, or `nil` when unset / out of range.
    public func metadata(at index: Int) -> InstructionMetadata? {
        guard instructionMetadata.indices.contains(index) else { return nil }
        return instructionMetadata[index]
    }

    mutating func ensureMetadataLength() {
        if instructionMetadata.count < gates.count {
            instructionMetadata.append(
                contentsOf: Array(repeating: nil, count: gates.count - instructionMetadata.count)
            )
        } else if instructionMetadata.count > gates.count {
            instructionMetadata.removeLast(instructionMetadata.count - gates.count)
        }
    }

    /// The inverse (adjoint) circuit: applies each gate's adjoint in reverse order, so
    /// `C` followed by `C.inverse()` returns the state vector to its starting point.
    ///
    /// Throws ``QuantumCircuitError/circuitNotUnitary`` when the circuit contains
    /// non-unitary operations (`measure`/`reset`), which have no inverse.
    public func inverse() throws -> QuantumCircuit {
        guard isUnitaryOnly else {
            throw QuantumCircuitError.circuitNotUnitary
        }

        var inverted = try QuantumCircuit(qubitCount: qubitCount, classicalRegisters: classicalRegisters)
        for gate in gates.reversed() {
            try inverted.apply(gate.adjoint)
        }
        return inverted
    }
}

extension QuantumCircuit: Equatable {
    public static func == (lhs: QuantumCircuit, rhs: QuantumCircuit) -> Bool {
        lhs.qubitCount == rhs.qubitCount
            && lhs.classicalRegisters == rhs.classicalRegisters
            && lhs.gates == rhs.gates
            && lhs.instructionMetadata == rhs.instructionMetadata
    }
}

extension QuantumCircuit: Codable {

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case qubitCount
        case classicalRegisters
        case gates
        case instructionMetadata
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        _ = try CircuitIRSchema.resolve(
            try container.decodeIfPresent(Int.self, forKey: .schemaVersion)
        )
        let qubitCount = try container.decode(Int.self, forKey: .qubitCount)
        let classicalRegisters = try container.decodeIfPresent(
            [ClassicalRegisterSpec].self,
            forKey: .classicalRegisters
        ) ?? []
        try self.init(qubitCount: qubitCount, classicalRegisters: classicalRegisters)

        let decodedGates = try container.decode([Gate].self, forKey: .gates)
        let decodedMetadata: [InstructionMetadata?]?
        if container.contains(.instructionMetadata) {
            let metadata = try container.decode([InstructionMetadata?].self, forKey: .instructionMetadata)
            guard metadata.count == decodedGates.count else {
                throw CircuitIRError.metadataLengthMismatch(
                    metadataCount: metadata.count,
                    gateCount: decodedGates.count
                )
            }
            decodedMetadata = metadata
        } else {
            decodedMetadata = nil
        }

        for (index, gate) in decodedGates.enumerated() {
            let maxCReg = Self.maxClassicalRegisterIndex(in: gate)
            if maxCReg >= 0, maxCReg >= classicalRegisters.count {
                throw CircuitIRError.invalidCircuit(
                    reason: "gate \(index) references classical register \(maxCReg) but circuit declares \(classicalRegisters.count) registers"
                )
            }
            do {
                if let decodedMetadata {
                    try apply(gate, metadata: decodedMetadata[index])
                } else {
                    try apply(gate)
                }
            } catch let error as CircuitIRError {
                throw error
            } catch {
                throw CircuitIRError.invalidCircuit(reason: String(describing: error))
            }
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(CircuitIRSchema.current, forKey: .schemaVersion)
        try container.encode(qubitCount, forKey: .qubitCount)
        if !classicalRegisters.isEmpty {
            try container.encode(classicalRegisters, forKey: .classicalRegisters)
        }
        try container.encode(gates, forKey: .gates)
        if instructionMetadata.contains(where: { $0 != nil }) {
            try container.encode(instructionMetadata, forKey: .instructionMetadata)
        }
    }
}

extension QuantumCircuit {
    /// Gate-count depth proxy: longest chain of overlapping qubit operations (delays count).
    public var circuitDepth: Int {
        var qubitTime = Array(repeating: 0, count: qubitCount)
        var depth = 0
        for gate in gates {
            let qubits = gate.affectedQubits
            let resolved = qubits.isEmpty ? Array(0..<qubitCount) : qubits
            let start = resolved.map { qubitTime[$0] }.max() ?? 0
            let finish = start + 1
            for q in resolved where q >= 0 && q < qubitCount {
                qubitTime[q] = finish
            }
            depth = max(depth, finish)
        }
        return depth
    }
}
