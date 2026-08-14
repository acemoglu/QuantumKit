import Foundation

/// Reusable ordered gate list that inlines into a ``QuantumCircuit`` via composition.
///
/// ``Subcircuit`` is a typealias of this type. Execution never sees a nested IR: inlining
/// calls ``QuantumCircuit/append(_:qubitMap:classicalRegisterMap:)`` so the destination
/// remains a flat ``gates`` list.
///
/// **Metadata:** same as composition — each gate is remapped 1:1 and ``instructionMetadata``
/// is **preserved** (see ``InstructionMetadata``).
public struct GateSequence: Sendable, Equatable {

    /// Optional display / library name (e.g. `"bell"`). Not used by engines.
    public var name: String?

    /// Logical body; qubit / classical width and gate validation match ``QuantumCircuit``.
    public private(set) var body: QuantumCircuit

    public var qubitCount: Int { body.qubitCount }
    public var classicalRegisters: [ClassicalRegisterSpec] { body.classicalRegisters }
    public var gates: [Gate] { body.gates }
    public var instructionMetadata: [InstructionMetadata?] { body.instructionMetadata }

    public init(
        name: String? = nil,
        qubitCount: Int,
        classicalRegisters: [ClassicalRegisterSpec] = []
    ) throws {
        self.name = name
        self.body = try QuantumCircuit(qubitCount: qubitCount, classicalRegisters: classicalRegisters)
    }

    /// Wraps an existing circuit as a reusable sequence (copies the flat gate list).
    public init(name: String? = nil, circuit: QuantumCircuit) {
        self.name = name
        self.body = circuit
    }

    /// Circuit view used when inlining via ``QuantumCircuit/append(_:qubitMap:classicalRegisterMap:)``.
    public func asCircuit() -> QuantumCircuit { body }

    public mutating func apply(_ gate: Gate) throws {
        try body.apply(gate)
    }

    public mutating func apply(_ gate: Gate, metadata: InstructionMetadata?) throws {
        try body.apply(gate, metadata: metadata)
    }

    public mutating func setInstructionMetadata(_ metadata: InstructionMetadata?, at index: Int) throws {
        try body.setInstructionMetadata(metadata, at: index)
    }

    public func metadata(at index: Int) -> InstructionMetadata? {
        body.metadata(at: index)
    }
}

/// Named alias for ``GateSequence`` (reusable inlined block).
public typealias Subcircuit = GateSequence

extension GateSequence: Codable {
    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case name
        case body
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        _ = try CircuitIRSchema.resolve(
            try container.decodeIfPresent(Int.self, forKey: .schemaVersion)
        )
        name = try container.decodeIfPresent(String.self, forKey: .name)
        body = try container.decode(QuantumCircuit.self, forKey: .body)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(CircuitIRSchema.current, forKey: .schemaVersion)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encode(body, forKey: .body)
    }
}

extension QuantumCircuit {

    /// Inlines a ``GateSequence`` / ``Subcircuit`` using the composition append path.
    ///
    /// Unlike circuit-to-circuit ``append``, mapped qubit indices must already lie in
    /// `0..<qubitCount` — the destination width does **not** grow. Overlapping or
    /// out-of-bounds map entries throw ``QuantumCircuitError/invalidComposition``.
    ///
    /// Metadata is preserved (see ``InstructionMetadata``).
    public mutating func append(
        _ sequence: GateSequence,
        qubitMap: [Int]? = nil,
        classicalRegisterMap: [Int]? = nil
    ) throws {
        let qMap = try Self.resolveQubitMapForSequence(
            otherQubitCount: sequence.qubitCount,
            qubitMap: qubitMap
        )
        for q in qMap {
            guard q < qubitCount else {
                throw QuantumCircuitError.invalidComposition(
                    reason: "qubitMap index \(q) is out of bounds for circuit width \(qubitCount)"
                )
            }
        }
        if qubitMap == nil, sequence.qubitCount > qubitCount {
            throw QuantumCircuitError.invalidComposition(
                reason: "sequence width \(sequence.qubitCount) exceeds circuit width \(qubitCount); pass an in-bounds qubitMap"
            )
        }
        try append(
            sequence.asCircuit(),
            qubitMap: qMap,
            classicalRegisterMap: classicalRegisterMap
        )
    }

    /// Non-mutating inline of a ``GateSequence``.
    public func compose(
        _ sequence: GateSequence,
        qubitMap: [Int]? = nil,
        classicalRegisterMap: [Int]? = nil
    ) throws -> QuantumCircuit {
        var copy = self
        try copy.append(sequence, qubitMap: qubitMap, classicalRegisterMap: classicalRegisterMap)
        return copy
    }

    /// Shared map validation for sequence inlining (distinct, non-negative; length match).
    fileprivate static func resolveQubitMapForSequence(
        otherQubitCount: Int,
        qubitMap: [Int]?
    ) throws -> [Int] {
        if let qubitMap {
            guard qubitMap.count == otherQubitCount else {
                throw QuantumCircuitError.invalidComposition(
                    reason: "qubitMap length \(qubitMap.count) must equal sequence.qubitCount \(otherQubitCount)"
                )
            }
            guard Set(qubitMap).count == qubitMap.count else {
                throw QuantumCircuitError.invalidComposition(
                    reason: "qubitMap entries must be distinct (overlap is not allowed)"
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
}
