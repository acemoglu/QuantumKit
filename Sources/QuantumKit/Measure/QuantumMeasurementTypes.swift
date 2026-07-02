import Foundation

public enum QuantumMeasurementError: Error {
    case invalidShotCount(Int)
    case emptyQubitSelection
    case qubitIndexOutOfBounds(index: Int, qubitCount: Int)
    case invalidPauliString(String)
}

/// A single-qubit Pauli operator (identity included) used to build tensor-product observables.
public enum Pauli: Equatable, Sendable {
    case i, x, y, z
}

/// A single complex amplitude of a state vector.
public struct ComplexAmplitude: Equatable, Sendable, Codable {
    public let real: QFloat
    public let imaginary: QFloat

    public init(real: QFloat, imaginary: QFloat) {
        self.real = real
        self.imaginary = imaginary
    }
}
