import Foundation

/// Classical outcomes collected from mid-circuit measurement gates, in circuit order.
public struct CircuitExecutionResult: Sendable, Equatable {
    public let measurementOutcomes: [[Int]]
    public let classicalMemory: ClassicalMemory
    /// Number of **top-level circuit instructions** processed (including measure, reset, barrier,
    /// delay, `id`, and `c_if`). Nested ``Gate/c_if`` bodies do not increment this counter.
    /// Portable cursor for ``CircuitCheckpoint`` resume on Metal and CPU.
    public let appliedGateCount: Int
    /// Unitary-piece counter used for renormalization cadence on Metal and CPU. Measure, reset,
    /// barrier, delay, `id`, and the `c_if` wrapper do not increment this.
    public let unitaryRenormCount: Int?

    public init(
        measurementOutcomes: [[Int]],
        classicalMemory: ClassicalMemory = ClassicalMemory(),
        appliedGateCount: Int = 0,
        unitaryRenormCount: Int? = nil
    ) {
        self.measurementOutcomes = measurementOutcomes
        self.classicalMemory = classicalMemory
        self.appliedGateCount = appliedGateCount
        self.unitaryRenormCount = unitaryRenormCount
    }
}
