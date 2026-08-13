import Foundation

/// Classical outcomes collected from mid-circuit measurement gates, in circuit order.
public struct CircuitExecutionResult: Sendable, Equatable {
    public let measurementOutcomes: [[Int]]
    public let classicalMemory: ClassicalMemory
    /// Engine renormalization counter after this invocation (for ``CircuitCheckpoint`` resume).
    public let appliedGateCount: Int

    public init(
        measurementOutcomes: [[Int]],
        classicalMemory: ClassicalMemory = ClassicalMemory(),
        appliedGateCount: Int = 0
    ) {
        self.measurementOutcomes = measurementOutcomes
        self.classicalMemory = classicalMemory
        self.appliedGateCount = appliedGateCount
    }
}
