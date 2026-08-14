import Foundation

extension QuantumCircuit {

    // MARK: - Append / compose

    /// Appends `other`'s gates onto this circuit, optionally remapping qubits and classical registers.
    ///
    /// - Parameters:
    ///   - other: Circuit whose flat gate list is appended after `self.gates`.
    ///   - qubitMap: When non-`nil`, must have length `other.qubitCount`; entry `i` is the
    ///     destination qubit for `other`'s local qubit `i`. When `nil`, uses the identity map
    ///     and grows ``qubitCount`` when `other` is wider. Destination indices may extend width.
    ///   - classicalRegisterMap: When non-`nil`, remaps `other`'s classical-register indices into
    ///     this circuit's existing registers (destinations must already be declared).
    ///     When `nil`, `other.classicalRegisters` are concatenated after this circuit's registers
    ///     and measure / `c_if` indices are offset accordingly.
    ///
    /// Instruction metadata is **preserved** (see ``InstructionMetadata`` composition policy).
    public mutating func append(
        _ other: QuantumCircuit,
        qubitMap: [Int]? = nil,
        classicalRegisterMap: [Int]? = nil
    ) throws {
        let qMap = try Self.resolveQubitMap(
            otherQubitCount: other.qubitCount,
            qubitMap: qubitMap
        )
        if let maxMapped = qMap.max(), maxMapped >= qubitCount {
            qubitCount = maxMapped + 1
        } else if qubitMap == nil, other.qubitCount > qubitCount {
            qubitCount = other.qubitCount
        }

        let cMap = try resolveClassicalRegisterMap(
            other: other,
            classicalRegisterMap: classicalRegisterMap
        )

        for index in other.gates.indices {
            var gate = try other.gates[index].remappingQubits { local in
                guard local >= 0, local < qMap.count else {
                    throw QuantumCircuitError.invalidComposition(
                        reason: "qubit index \(local) out of range for map of length \(qMap.count)"
                    )
                }
                return qMap[local]
            }
            if !cMap.isEmpty || Self.maxClassicalRegisterIndex(in: gate) >= 0 {
                gate = try gate.remappingClassicalRegisters { local in
                    guard local >= 0, local < cMap.count else {
                        throw QuantumCircuitError.invalidComposition(
                            reason: "classical register \(local) out of range for map of length \(cMap.count)"
                        )
                    }
                    return cMap[local]
                }
            }
            let meta: InstructionMetadata?
            if other.instructionMetadata.indices.contains(index) {
                meta = other.instructionMetadata[index]
            } else {
                meta = nil
            }
            try apply(gate, metadata: meta)
        }
    }

    /// Non-mutating compose: returns a copy of `self` with `other` appended.
    ///
    /// Same mapping and metadata rules as ``append(_:qubitMap:classicalRegisterMap:)``.
    public func compose(
        _ other: QuantumCircuit,
        qubitMap: [Int]? = nil,
        classicalRegisterMap: [Int]? = nil
    ) throws -> QuantumCircuit {
        var copy = self
        try copy.append(other, qubitMap: qubitMap, classicalRegisterMap: classicalRegisterMap)
        return copy
    }

    // MARK: - Tensor / parallel compose

    /// Places `other` on the next disjoint qubits (`self.qubitCount..<total`), growing width.
    ///
    /// Classical registers are concatenated; `other`'s measure / `c_if` register indices are
    /// offset by `self.classicalRegisters.count`. Metadata is preserved from both sides.
    public mutating func formTensor(with other: QuantumCircuit) throws {
        let offset = qubitCount
        let qMap = Array(offset..<(offset + other.qubitCount))
        // Grow before append so identity-width checks stay consistent.
        qubitCount = offset + other.qubitCount
        try append(other, qubitMap: qMap, classicalRegisterMap: nil)
    }

    /// Parallel compose: `self` on qubits `0..<n`, `other` on `n..<n+m`.
    public func tensor(_ other: QuantumCircuit) throws -> QuantumCircuit {
        var copy = self
        try copy.formTensor(with: other)
        return copy
    }

    // MARK: - Controlled circuit

    /// Returns a controlled version of this unitary circuit.
    ///
    /// Prepends `controlCount` control qubits at indices `0..<controlCount` and shifts every
    /// original qubit by `+controlCount`. A 1-qubit X circuit with `controlCount: 1` becomes
    /// CX(0, 1).
    ///
    /// Throws ``QuantumCircuitError/unsupportedControlledGate`` for measure / reset / `c_if` /
    /// initialize and for unitaries without a native controlled ``Gate`` encoding.
    /// Metadata is preserved 1:1 (see ``InstructionMetadata``).
    public func controlled(controlCount: Int = 1) throws -> QuantumCircuit {
        guard controlCount > 0 else {
            throw QuantumCircuitError.unsupportedControlledGate(
                reason: "controlCount must be positive"
            )
        }

        let controls = Array(0..<controlCount)
        var result = try QuantumCircuit(
            qubitCount: qubitCount + controlCount,
            classicalRegisters: classicalRegisters
        )
        for index in gates.indices {
            let shifted = gates[index].remappingQubits { $0 + controlCount }
            let lifted = try shifted.controlled(by: controls)
            let meta: InstructionMetadata?
            if instructionMetadata.indices.contains(index) {
                meta = instructionMetadata[index]
            } else {
                meta = nil
            }
            try result.apply(lifted, metadata: meta)
        }
        return result
    }

    // MARK: - Map helpers

    private static func resolveQubitMap(
        otherQubitCount: Int,
        qubitMap: [Int]?
    ) throws -> [Int] {
        if let qubitMap {
            guard qubitMap.count == otherQubitCount else {
                throw QuantumCircuitError.invalidComposition(
                    reason: "qubitMap length \(qubitMap.count) must equal other.qubitCount \(otherQubitCount)"
                )
            }
            guard Set(qubitMap).count == qubitMap.count else {
                throw QuantumCircuitError.invalidComposition(
                    reason: "qubitMap entries must be distinct"
                )
            }
            for q in qubitMap where q < 0 {
                throw QuantumCircuitError.invalidComposition(
                    reason: "qubitMap contains negative index \(q)"
                )
            }
            return qubitMap
        }
        return Array(0..<otherQubitCount)
    }

    private mutating func resolveClassicalRegisterMap(
        other: QuantumCircuit,
        classicalRegisterMap: [Int]?
    ) throws -> [Int] {
        let otherDeclared = other.classicalRegisters.count
        let referencedMax = other.gates.reduce(-1) { partial, gate in
            max(partial, Self.maxClassicalRegisterIndex(in: gate))
        }
        let mapLength = max(otherDeclared, referencedMax + 1)
        if mapLength <= 0 {
            return []
        }

        if let classicalRegisterMap {
            guard classicalRegisterMap.count >= mapLength else {
                throw QuantumCircuitError.invalidComposition(
                    reason: "classicalRegisterMap length \(classicalRegisterMap.count) must be at least \(mapLength)"
                )
            }
            for dest in classicalRegisterMap.prefix(mapLength) {
                guard dest >= 0, dest < classicalRegisters.count else {
                    throw QuantumCircuitError.invalidComposition(
                        reason: "classicalRegisterMap destination \(dest) is outside declared registers 0..<\(classicalRegisters.count)"
                    )
                }
            }
            return Array(classicalRegisterMap.prefix(mapLength))
        }

        // Default: append other's declared registers and offset indices.
        let offset = classicalRegisters.count
        if !other.classicalRegisters.isEmpty {
            classicalRegisters.append(contentsOf: other.classicalRegisters)
        } else if referencedMax >= 0 {
            for index in 0...referencedMax {
                guard index < classicalRegisters.count else {
                    throw QuantumCircuitError.invalidComposition(
                        reason: "other references classical register \(index) but destination has only \(classicalRegisters.count) registers; pass classicalRegisterMap or declare registers on other"
                    )
                }
            }
            return Array(0..<mapLength)
        }
        return (0..<mapLength).map { $0 + offset }
    }

    private static func maxClassicalRegisterIndex(in gate: Gate) -> Int {
        switch gate {
        case .measure(let spec):
            return spec.classicalRegister
        case .c_if(let classicalRegister, _, let inner):
            return max(classicalRegister, maxClassicalRegisterIndex(in: inner))
        default:
            return -1
        }
    }
}
