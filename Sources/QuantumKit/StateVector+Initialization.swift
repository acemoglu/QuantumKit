import Foundation

public enum StateVectorInitializationError: Error, Equatable {
    case invalidBasisIndex(index: Int, stateCount: Int)
    case amplitudeCountMismatch(expected: Int, actual: Int)
    case nonUnitNorm(squaredNorm: Double)
}

extension StateVector {

    /// Initializes the state vector to a computational basis state |index⟩.
    public func initializeComputationalBasis(index: Int) throws {
        guard index >= 0, index < stateCount else {
            throw StateVectorInitializationError.invalidBasisIndex(index: index, stateCount: stateCount)
        }

        let realPointer = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        realPointer.update(repeating: 0, count: stateCount)
        imagPointer.update(repeating: 0, count: stateCount)
        realPointer[index] = 1
    }

    /// Initializes the state vector from explicit amplitudes in computational-basis order.
    public func initialize(amplitudes: [ComplexAmplitude]) throws {
        guard amplitudes.count == stateCount else {
            throw StateVectorInitializationError.amplitudeCountMismatch(
                expected: stateCount,
                actual: amplitudes.count
            )
        }

        var squaredNorm = 0.0
        for amplitude in amplitudes {
            squaredNorm += Double(amplitude.real * amplitude.real + amplitude.imaginary * amplitude.imaginary)
        }
        guard abs(squaredNorm - 1.0) <= 1e-5 else {
            throw StateVectorInitializationError.nonUnitNorm(squaredNorm: squaredNorm)
        }

        let realPointer = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        for index in 0..<stateCount {
            realPointer[index] = amplitudes[index].real
            imagPointer[index] = amplitudes[index].imaginary
        }
    }
}
