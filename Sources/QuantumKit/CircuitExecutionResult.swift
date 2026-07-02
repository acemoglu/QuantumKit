import Foundation

/// Classical outcomes collected from mid-circuit measurement gates, in circuit order.
public struct CircuitExecutionResult: Sendable, Equatable {
    public let measurementOutcomes: [[Int]]
    public let classicalMemory: ClassicalMemory

    public init(
        measurementOutcomes: [[Int]],
        classicalMemory: ClassicalMemory = ClassicalMemory()
    ) {
        self.measurementOutcomes = measurementOutcomes
        self.classicalMemory = classicalMemory
    }
}
