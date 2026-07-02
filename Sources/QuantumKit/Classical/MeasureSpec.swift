import Foundation

/// Measurement assignment from qubits to classical register bits.
public struct MeasureSpec: Equatable, Sendable, Codable {
    public let qubits: [Int]
    public let classicalRegister: Int
    public let classicalBitOffset: Int

    public init(
        qubits: [Int],
        classicalRegister: Int = 0,
        classicalBitOffset: Int = 0
    ) {
        self.qubits = qubits
        self.classicalRegister = classicalRegister
        self.classicalBitOffset = classicalBitOffset
    }
}
