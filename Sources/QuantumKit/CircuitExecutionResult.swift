import Foundation

/// Classical outcomes collected from mid-circuit measurement gates, in circuit order.
public struct CircuitExecutionResult: Sendable, Equatable {
    public let measurementOutcomes: [[Int]]
    public let classicalMemory: ClassicalMemory
    /// Number of **top-level circuit instructions** processed (including measure, reset, barrier,
    /// delay, `id`, and `c_if` / `while_c`). Nested ``Gate/c_if`` / ``Gate/while_c`` bodies do not
    /// increment this counter.
    /// Portable cursor for ``CircuitCheckpoint`` resume on Metal and CPU.
    public let appliedGateCount: Int
    /// Unitary-piece counter used for renormalization cadence on Metal and CPU. Measure, reset,
    /// barrier, delay, `id`, and the `c_if` / `while_c` wrapper do not increment this.
    public let unitaryRenormCount: Int?
    /// Cumulative global phase \(\Phi\) in radians after this SV evolve slice, when tracked.
    ///
    /// See ``GlobalPhaseTracking``. Nil on density-matrix / MPS / stabilizer engines.
    public let cumulativeGlobalPhaseRadians: Double?

    public init(
        measurementOutcomes: [[Int]],
        classicalMemory: ClassicalMemory = ClassicalMemory(),
        appliedGateCount: Int = 0,
        unitaryRenormCount: Int? = nil,
        cumulativeGlobalPhaseRadians: Double? = nil
    ) {
        self.measurementOutcomes = measurementOutcomes
        self.classicalMemory = classicalMemory
        self.appliedGateCount = appliedGateCount
        self.unitaryRenormCount = unitaryRenormCount
        self.cumulativeGlobalPhaseRadians = cumulativeGlobalPhaseRadians
    }
}
