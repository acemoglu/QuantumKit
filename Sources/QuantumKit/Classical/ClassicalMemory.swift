import Foundation

/// Metadata for a classical register declared on a circuit.
public struct ClassicalRegisterSpec: Equatable, Sendable, Codable {
    public let bitCount: Int

    public init(bitCount: Int) throws {
        guard bitCount > 0 else {
            throw ClassicalMemoryError.invalidBitCount(bitCount)
        }
        self.bitCount = bitCount
    }
}

public enum ClassicalMemoryError: Error, Equatable {
    case invalidBitCount(Int)
    case registerIndexOutOfBounds(index: Int, count: Int)
    case bitOffsetOutOfBounds(register: Int, offset: Int, width: Int)
}

/// Runtime classical bit storage updated by mid-circuit measurements.
///
/// ``ClassicalMemory`` is a value type: copy freely, snapshot via ``Codable`` / ``snapshot()``, and
/// restore with ``init(snapshot:)`` or ``restore(from:)``. Safe to share across threads as an
/// immutable value; mutate only from one execution context at a time.
public struct ClassicalMemory: Sendable, Equatable, Codable {
    public let registerWidths: [Int]
    private(set) var registerValues: [Int]

    public init(registerWidths: [Int] = []) {
        self.registerWidths = registerWidths
        self.registerValues = Array(repeating: 0, count: registerWidths.count)
    }

    /// Restores register values from a prior snapshot (widths must match).
    public init(snapshot: ClassicalMemory) {
        self.registerWidths = snapshot.registerWidths
        self.registerValues = snapshot.registerValues
    }

    /// Value-type snapshot (identical to copying the struct).
    public func snapshot() -> ClassicalMemory {
        ClassicalMemory(snapshot: self)
    }

    /// Replaces register values from `other`. Widths must be identical.
    public mutating func restore(from other: ClassicalMemory) throws {
        guard other.registerWidths == registerWidths else {
            throw ClassicalMemoryError.registerIndexOutOfBounds(
                index: other.registerWidths.count,
                count: registerWidths.count
            )
        }
        registerValues = other.registerValues
    }

    public func value(ofRegister index: Int) -> Int {
        guard index >= 0, index < registerValues.count else { return 0 }
        return maskedValue(at: index)
    }

    /// Current classical register values (memory slots), in register-index order.
    public var memorySlots: [Int] {
        (0..<registerValues.count).map { value(ofRegister: $0) }
    }

    /// Hex representation of a register value (e.g. `"0x5"`).
    public func hexValue(ofRegister index: Int) -> String {
        String(format: "0x%x", value(ofRegister: index))
    }

    public mutating func writeMeasuredBits(
        _ bits: [Int],
        register: Int,
        bitOffset: Int
    ) throws {
        try ensureRegister(register, requiredBitCapacity: bitOffset + bits.count)

        var value = register < registerValues.count ? registerValues[register] : 0
        for (position, bit) in bits.enumerated() {
            let bitIndex = bitOffset + position
            let mask = 1 << bitIndex
            if bit == 0 {
                value &= ~mask
            } else {
                value |= mask
            }
        }
        registerValues[register] = maskedValue(value, register: register)
    }

    public mutating func writeOutcome(
        _ outcome: Int,
        measuredQubitCount: Int,
        register: Int,
        bitOffset: Int
    ) throws {
        var bits = [Int](repeating: 0, count: measuredQubitCount)
        for position in 0..<measuredQubitCount {
            bits[position] = (outcome >> position) & 1
        }
        try writeMeasuredBits(bits, register: register, bitOffset: bitOffset)
    }

    private mutating func ensureRegister(_ index: Int, requiredBitCapacity: Int) throws {
        guard index >= 0 else {
            throw ClassicalMemoryError.registerIndexOutOfBounds(index: index, count: registerValues.count)
        }

        if index >= registerValues.count {
            registerValues.append(contentsOf: Array(repeating: 0, count: index + 1 - registerValues.count))
        }

        guard index < registerWidths.count else { return }

        let width = registerWidths[index]
        guard requiredBitCapacity <= width else {
            throw ClassicalMemoryError.bitOffsetOutOfBounds(
                register: index,
                offset: requiredBitCapacity - 1,
                width: width
            )
        }
    }

    private func maskedValue(at index: Int) -> Int {
        maskedValue(registerValues[index], register: index)
    }

    private func maskedValue(_ value: Int, register: Int) -> Int {
        guard register < registerWidths.count else { return value }
        let width = registerWidths[register]
        guard width > 0 else { return value }
        let mask = (1 << width) - 1
        return value & mask
    }
}
