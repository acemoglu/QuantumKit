import Foundation

/// Classical outcomes collected from mid-circuit measurement gates, in circuit order.
public struct CircuitExecutionResult: Sendable, Equatable {
    public let measurementOutcomes: [[Int]]

    public init(measurementOutcomes: [[Int]]) {
        self.measurementOutcomes = measurementOutcomes
    }
}
