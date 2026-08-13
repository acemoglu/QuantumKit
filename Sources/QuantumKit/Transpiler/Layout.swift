import Foundation

/// Bijection between logical circuit qubits and physical device qubits.
///
/// `logicalToPhysical[logical] == physical`. Unused physical qubits are allowed when the
/// device is wider than the circuit; every logical index must appear exactly once.
public struct Layout: Sendable, Equatable, Codable {
    public let logicalToPhysical: [Int]

    public var logicalQubitCount: Int { logicalToPhysical.count }

    public var physicalQubitCount: Int {
        (logicalToPhysical.max() ?? -1) + 1
    }

    public init(logicalToPhysical: [Int]) throws {
        guard !logicalToPhysical.isEmpty else {
            throw TranspilerError.invalidLayout(reason: "layout must map at least one logical qubit")
        }
        guard logicalToPhysical.allSatisfy({ $0 >= 0 }) else {
            throw TranspilerError.invalidLayout(reason: "physical qubit indices must be non-negative")
        }
        guard Set(logicalToPhysical).count == logicalToPhysical.count else {
            throw TranspilerError.invalidLayout(reason: "logical-to-physical mapping must be injective")
        }
        self.logicalToPhysical = logicalToPhysical
    }

    /// Identity map `i → i` for `qubitCount` qubits.
    public static func identity(qubitCount: Int) throws -> Layout {
        guard qubitCount > 0 else {
            throw TranspilerError.invalidLayout(reason: "qubitCount must be positive")
        }
        return try Layout(logicalToPhysical: Array(0..<qubitCount))
    }

    public func physical(forLogical logical: Int) throws -> Int {
        guard logical >= 0, logical < logicalToPhysical.count else {
            throw TranspilerError.invalidLayout(
                reason: "logical qubit \(logical) is outside layout range 0..<\(logicalToPhysical.count)"
            )
        }
        return logicalToPhysical[logical]
    }

    /// Physical → logical inverse for occupied physical wires. Holes are omitted.
    public func physicalToLogical() -> [Int: Int] {
        var inverse: [Int: Int] = [:]
        for (logical, physical) in logicalToPhysical.enumerated() {
            inverse[physical] = logical
        }
        return inverse
    }

    /// Extends this layout to `logicalCount` by mapping new logical qubits onto unused
    /// physical wires in `0..<physicalCount` (preferring lower indices).
    ///
    /// Used after ancilla growth (e.g. V-chain unroll) so routing does not keep a stale
    /// pre-ancilla ``initialLayout``. Throws when the device has no free physicals.
    public func extended(toLogicalCount logicalCount: Int, physicalCount: Int) throws -> Layout {
        guard logicalCount >= logicalQubitCount else {
            throw TranspilerError.invalidLayout(
                reason: "cannot shrink layout from \(logicalQubitCount) to \(logicalCount) logical qubits"
            )
        }
        if logicalCount == logicalQubitCount {
            guard physicalQubitCount <= physicalCount else {
                throw TranspilerError.circuitWiderThanDevice(
                    circuitQubits: physicalQubitCount,
                    deviceQubits: physicalCount
                )
            }
            return self
        }
        guard physicalCount >= logicalCount else {
            throw TranspilerError.circuitWiderThanDevice(
                circuitQubits: logicalCount,
                deviceQubits: physicalCount
            )
        }

        var used = Set(logicalToPhysical)
        var mapping = logicalToPhysical
        for _ in logicalQubitCount..<logicalCount {
            guard let physical = (0..<physicalCount).first(where: { !used.contains($0) }) else {
                throw TranspilerError.circuitWiderThanDevice(
                    circuitQubits: logicalCount,
                    deviceQubits: physicalCount
                )
            }
            used.insert(physical)
            mapping.append(physical)
        }
        return try Layout(logicalToPhysical: mapping)
    }
}

/// Mutable layout used while inserting SWAPs during routing.
struct MutableLayout {
    private(set) var logicalToPhysical: [Int]
    private var physicalToLogicalMap: [Int: Int]

    init(_ layout: Layout) {
        self.logicalToPhysical = layout.logicalToPhysical
        self.physicalToLogicalMap = layout.physicalToLogical()
    }

    func physical(forLogical logical: Int) -> Int {
        logicalToPhysical[logical]
    }

    mutating func applyPhysicalSwap(_ a: Int, _ b: Int) {
        let logicalA = physicalToLogicalMap[a]
        let logicalB = physicalToLogicalMap[b]

        physicalToLogicalMap[a] = logicalB
        physicalToLogicalMap[b] = logicalA

        if let logicalA {
            logicalToPhysical[logicalA] = b
        }
        if let logicalB {
            logicalToPhysical[logicalB] = a
        }
    }

    func snapshot() throws -> Layout {
        try Layout(logicalToPhysical: logicalToPhysical)
    }
}
