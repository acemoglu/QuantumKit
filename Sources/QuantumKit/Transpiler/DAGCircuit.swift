import Foundation

/// Stable identifier for a node in ``DAGCircuit``.
public struct DAGNodeID: Hashable, Comparable, Sendable {
    public let rawValue: UInt64

    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }

    public static func < (lhs: DAGNodeID, rhs: DAGNodeID) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// One instruction node in a ``DAGCircuit``.
public struct DAGOpNode: Sendable, Equatable {
    public let id: DAGNodeID
    public let gate: Gate
    public let metadata: InstructionMetadata?
    /// Insertion / source order used for stable topological serialization.
    public let sequence: Int

    public init(id: DAGNodeID, gate: Gate, metadata: InstructionMetadata?, sequence: Int) {
        self.id = id
        self.gate = gate
        self.metadata = metadata
        self.sequence = sequence
    }
}

public enum DAGCircuitError: Error, Equatable, Sendable {
    case cycleDetected
    case unknownNode(DAGNodeID)
    case invalidCircuit(reason: String)
}

/// Directed acyclic instruction graph for a circuit.
///
/// Nodes are gates (plus optional ``InstructionMetadata``). Edges capture qubit-wire and
/// classical-register dependencies: an edge `A → B` means `A` must execute before `B`.
///
/// **Bridge policy:** ``init(circuit:)`` / ``toQuantumCircuit()`` are lossless for the flat
/// ``Gate`` vocabulary — every gate becomes one node and flattens back in a stable
/// topological order (original sequence as tie-break). Metadata is **preserved** 1:1 on
/// surviving nodes (see ``InstructionMetadata``). Engines never execute a DAG; flatten to
/// ``QuantumCircuit`` first.
public struct DAGCircuit: Sendable {
    public let qubitCount: Int
    public let classicalRegisters: [ClassicalRegisterSpec]

    public private(set) var nodes: [DAGNodeID: DAGOpNode]
    /// Adjacency: predecessor → successors.
    public private(set) var successors: [DAGNodeID: Set<DAGNodeID>]
    public private(set) var predecessors: [DAGNodeID: Set<DAGNodeID>]

    private var nextRawID: UInt64
    private var nextSequence: Int

    public var nodeCount: Int { nodes.count }

    public init(qubitCount: Int, classicalRegisters: [ClassicalRegisterSpec] = []) throws {
        guard qubitCount > 0 else {
            throw DAGCircuitError.invalidCircuit(reason: "qubitCount must be positive")
        }
        self.qubitCount = qubitCount
        self.classicalRegisters = classicalRegisters
        self.nodes = [:]
        self.successors = [:]
        self.predecessors = [:]
        self.nextRawID = 0
        self.nextSequence = 0
    }

    /// Builds a DAG from a flat circuit (one node per gate, dependency edges on wires).
    public init(circuit: QuantumCircuit) throws {
        try self.init(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        var lastOnQubit = Array(repeating: Optional<DAGNodeID>.none, count: circuit.qubitCount)
        var lastClassicalWrite: [Int: DAGNodeID] = [:]

        for index in circuit.gates.indices {
            let gate = circuit.gates[index]
            let meta = circuit.instructionMetadata.indices.contains(index)
                ? circuit.instructionMetadata[index]
                : nil
            let id = try append(gate: gate, metadata: meta)

            var preds = Set<DAGNodeID>()
            let wireQubits = Self.dependencyQubits(for: gate, qubitCount: circuit.qubitCount)
            for q in wireQubits {
                if let pred = lastOnQubit[q] {
                    preds.insert(pred)
                }
            }
            for creg in Self.classicalReads(for: gate) {
                if let pred = lastClassicalWrite[creg] {
                    preds.insert(pred)
                }
            }
            for pred in preds {
                try addEdge(from: pred, to: id)
            }
            for q in wireQubits {
                lastOnQubit[q] = id
            }
            for creg in Self.classicalWrites(for: gate) {
                lastClassicalWrite[creg] = id
            }
        }
    }

    /// Flatten to a ``QuantumCircuit`` via stable topological order (lowest ``DAGOpNode/sequence`` first among ready nodes).
    public func toQuantumCircuit() throws -> QuantumCircuit {
        let ordered = try topologicalNodes()
        var circuit = try QuantumCircuit(
            qubitCount: qubitCount,
            classicalRegisters: classicalRegisters
        )
        for node in ordered {
            try circuit.apply(node.gate, metadata: node.metadata)
        }
        return circuit
    }

    /// Appends a node with no dependency edges (caller may add edges).
    @discardableResult
    public mutating func append(gate: Gate, metadata: InstructionMetadata? = nil) throws -> DAGNodeID {
        let id = DAGNodeID(rawValue: nextRawID)
        nextRawID += 1
        let sequence = nextSequence
        nextSequence += 1
        nodes[id] = DAGOpNode(id: id, gate: gate, metadata: metadata, sequence: sequence)
        successors[id] = []
        predecessors[id] = []
        return id
    }

    public mutating func addEdge(from predecessor: DAGNodeID, to successor: DAGNodeID) throws {
        guard nodes[predecessor] != nil else {
            throw DAGCircuitError.unknownNode(predecessor)
        }
        guard nodes[successor] != nil else {
            throw DAGCircuitError.unknownNode(successor)
        }
        guard predecessor != successor else { return }
        successors[predecessor, default: []].insert(successor)
        predecessors[successor, default: []].insert(predecessor)
    }

    /// Removes `id`, splicing predecessors directly to successors.
    public mutating func removeNode(_ id: DAGNodeID) throws {
        guard nodes[id] != nil else {
            throw DAGCircuitError.unknownNode(id)
        }
        let preds = predecessors[id] ?? []
        let succs = successors[id] ?? []
        for pred in preds {
            successors[pred]?.remove(id)
            for succ in succs {
                successors[pred, default: []].insert(succ)
                predecessors[succ, default: []].insert(pred)
            }
        }
        for succ in succs {
            predecessors[succ]?.remove(id)
            for pred in preds {
                predecessors[succ, default: []].insert(pred)
                successors[pred, default: []].insert(succ)
            }
        }
        nodes.removeValue(forKey: id)
        successors.removeValue(forKey: id)
        predecessors.removeValue(forKey: id)
    }

    /// Kahn topological sort; ties broken by ascending ``DAGOpNode/sequence``.
    public func topologicalNodes() throws -> [DAGOpNode] {
        var inDegree: [DAGNodeID: Int] = [:]
        for id in nodes.keys {
            inDegree[id] = predecessors[id]?.count ?? 0
        }
        var ready = nodes.keys.filter { (inDegree[$0] ?? 0) == 0 }
            .sorted { a, b in
                let sa = nodes[a]!.sequence
                let sb = nodes[b]!.sequence
                if sa != sb { return sa < sb }
                return a < b
            }
        var ordered: [DAGOpNode] = []
        ordered.reserveCapacity(nodes.count)

        while !ready.isEmpty {
            let id = ready.removeFirst()
            ordered.append(nodes[id]!)
            let nexts = (successors[id] ?? []).sorted { a, b in
                let sa = nodes[a]!.sequence
                let sb = nodes[b]!.sequence
                if sa != sb { return sa < sb }
                return a < b
            }
            for succ in nexts {
                let degree = (inDegree[succ] ?? 1) - 1
                inDegree[succ] = degree
                if degree == 0 {
                    ready.append(succ)
                    ready.sort { a, b in
                        let sa = nodes[a]!.sequence
                        let sb = nodes[b]!.sequence
                        if sa != sb { return sa < sb }
                        return a < b
                    }
                }
            }
        }

        guard ordered.count == nodes.count else {
            throw DAGCircuitError.cycleDetected
        }
        return ordered
    }

    // MARK: - Dependency helpers

    /// Qubits whose wire order is constrained by `gate`. Empty barrier → all qubits.
    public static func dependencyQubits(for gate: Gate, qubitCount: Int) -> [Int] {
        switch gate {
        case .barrier(let qubits):
            return qubits.isEmpty ? Array(0..<qubitCount) : qubits
        default:
            var seen = Set<Int>()
            return gate.affectedQubits.filter { seen.insert($0).inserted }
        }
    }

    public static func classicalWrites(for gate: Gate) -> [Int] {
        switch gate {
        case .measure(let spec):
            return [spec.classicalRegister]
        default:
            return []
        }
    }

    public static func classicalReads(for gate: Gate) -> [Int] {
        switch gate {
        case .c_if(let classicalRegister, _, _):
            return [classicalRegister]
        default:
            return []
        }
    }

    /// `true` when the op is a removable idle / empty barrier for ``IdleIdentityRemovalPass``.
    public static func isIdleIdentity(_ gate: Gate) -> Bool {
        switch gate {
        case .id:
            return true
        case .barrier(let qubits):
            return qubits.isEmpty
        default:
            return false
        }
    }
}
