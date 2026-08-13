import Foundation

/// Compiler-level scratch-qubit allocator with explicit acquire / release reuse.
///
/// Logical circuit qubits `[0, originalQubitCount)` are never allocated as ancillas.
/// Freed indices are reused before growing width. Opt-in via
/// ``TranspileOptions/enableAncillaAllocation``.
public struct AncillaAllocator: Sendable, Equatable {
    public let originalQubitCount: Int
    private var freeList: [Int]
    private var highWaterMark: Int

    public init(originalQubitCount: Int) {
        precondition(originalQubitCount > 0)
        self.originalQubitCount = originalQubitCount
        self.freeList = []
        self.highWaterMark = originalQubitCount
    }

    /// Current circuit width including live and historically allocated ancillas.
    public var qubitCount: Int { highWaterMark }

    /// Number of ancilla slots currently held (not on the free list).
    public var liveAncillaCount: Int {
        highWaterMark - originalQubitCount - freeList.count
    }

    /// Peak ancilla width ever allocated (includes freed).
    public var peakAncillaCount: Int {
        highWaterMark - originalQubitCount
    }

    public mutating func acquire() -> Int {
        if let reused = freeList.popLast() {
            return reused
        }
        let index = highWaterMark
        highWaterMark += 1
        return index
    }

    public mutating func acquire(_ count: Int) -> [Int] {
        precondition(count >= 0)
        return (0..<count).map { _ in acquire() }
    }

    public mutating func release(_ qubit: Int) {
        precondition(qubit >= originalQubitCount && qubit < highWaterMark)
        precondition(!freeList.contains(qubit))
        freeList.append(qubit)
    }

    public mutating func release(_ qubits: [Int]) {
        for qubit in qubits.reversed() {
            release(qubit)
        }
    }
}

public enum AncillaAllocationError: Error, Equatable {
    case ancillaAllocationDisabled(strategy: ControlledGateSynthesisStrategy)
    case insufficientAncillas(required: Int, available: Int)
}
