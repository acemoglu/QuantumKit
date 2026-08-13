import Foundation

public enum StateVectorInitializationError: Error, Equatable {
    case invalidBasisIndex(index: Int, stateCount: Int)
    case amplitudeCountMismatch(expected: Int, actual: Int)
    case nonUnitNorm(squaredNorm: Double)
    case invalidQubitSubset(reason: String)
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

        try UnitaryValidation.validateUnitNorm(amplitudes)

        let realPointer = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        for index in 0..<stateCount {
            realPointer[index] = amplitudes[index].real
            imagPointer[index] = amplitudes[index].imaginary
        }
    }

    /// Initializes a normalized state on the listed qubits, tensoring with |0…0⟩ on passive qubits.
    public func initialize(qubits: [Int], amplitudes: [ComplexAmplitude]) throws {
        guard !qubits.isEmpty else {
            throw StateVectorInitializationError.invalidQubitSubset(
                reason: "initialize requires at least one qubit"
            )
        }
        guard Set(qubits).count == qubits.count else {
            throw StateVectorInitializationError.invalidQubitSubset(
                reason: "initialize requires distinct qubit indices"
            )
        }
        for qubit in qubits where qubit < 0 || qubit >= qubitCount {
            throw StateVectorInitializationError.invalidQubitSubset(
                reason: "qubit index \(qubit) is out of bounds for \(qubitCount) qubits"
            )
        }

        let expectedCount = 1 << qubits.count
        guard amplitudes.count == expectedCount else {
            throw StateVectorInitializationError.amplitudeCountMismatch(
                expected: expectedCount,
                actual: amplitudes.count
            )
        }
        try UnitaryValidation.validateUnitNorm(amplitudes)

        if qubits.count == qubitCount {
            try initialize(amplitudes: amplitudes)
            return
        }

        let embedded = QubitIndexing.embeddedAmplitudes(
            qubits: qubits,
            amplitudes: amplitudes,
            qubitCount: qubitCount
        )
        try initialize(amplitudes: embedded)
    }

    /// Writes amplitudes without renormalizing. Used for adjoint intermediate states that are
    /// not required to be unit vectors (e.g. ``H|ψ⟩``).
    func replaceAmplitudesUnchecked(_ amplitudes: [ComplexAmplitude]) throws {
        guard amplitudes.count == stateCount else {
            throw StateVectorInitializationError.amplitudeCountMismatch(
                expected: stateCount,
                actual: amplitudes.count
            )
        }

        let realPointer = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        for index in 0..<stateCount {
            realPointer[index] = amplitudes[index].real
            imagPointer[index] = amplitudes[index].imaginary
        }
    }
}
