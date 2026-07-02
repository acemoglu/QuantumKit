import Foundation

public enum DensityMatrixInitializationError: Error, Equatable {
    case invalidQubitSubset(reason: String)
    case amplitudeCountMismatch(expected: Int, actual: Int)
    case nonUnitNorm(squaredNorm: Double)
}

extension DensityMatrix {

    /// Initializes ρ to the pure state |ψ⟩⟨ψ| for a normalized amplitude vector on the full register.
    public func initialize(amplitudes: [ComplexAmplitude]) throws {
        guard amplitudes.count == stateCount else {
            throw DensityMatrixInitializationError.amplitudeCountMismatch(
                expected: stateCount,
                actual: amplitudes.count
            )
        }
        do {
            try UnitaryValidation.validateUnitNorm(amplitudes)
        } catch let error as StateVectorInitializationError {
            if case .nonUnitNorm(let squaredNorm) = error {
                throw DensityMatrixInitializationError.nonUnitNorm(squaredNorm: squaredNorm)
            }
            throw error
        }

        let real = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imag = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        real.update(repeating: 0, count: elementCount)
        imag.update(repeating: 0, count: elementCount)

        for row in 0..<stateCount {
            for column in 0..<stateCount {
                let rowAmp = amplitudes[row]
                let colAmp = amplitudes[column]
                let index = row * stateCount + column
                real[index] = (rowAmp.real * colAmp.real) + (rowAmp.imaginary * colAmp.imaginary)
                imag[index] = (rowAmp.real * colAmp.imaginary) - (rowAmp.imaginary * colAmp.real)
            }
        }
    }

    /// Initializes ρ to |ψ⟩⟨ψ| for a normalized state on the listed qubits (passive qubits in |0⟩).
    public func initialize(qubits: [Int], amplitudes: [ComplexAmplitude]) throws {
        guard !qubits.isEmpty else {
            throw DensityMatrixInitializationError.invalidQubitSubset(
                reason: "initialize requires at least one qubit"
            )
        }
        guard Set(qubits).count == qubits.count else {
            throw DensityMatrixInitializationError.invalidQubitSubset(
                reason: "initialize requires distinct qubit indices"
            )
        }
        for qubit in qubits where qubit < 0 || qubit >= qubitCount {
            throw DensityMatrixInitializationError.invalidQubitSubset(
                reason: "qubit index \(qubit) is out of bounds for \(qubitCount) qubits"
            )
        }

        let expectedCount = 1 << qubits.count
        guard amplitudes.count == expectedCount else {
            throw DensityMatrixInitializationError.amplitudeCountMismatch(
                expected: expectedCount,
                actual: amplitudes.count
            )
        }

        let embedded = QubitIndexing.embeddedAmplitudes(
            qubits: qubits,
            amplitudes: amplitudes,
            qubitCount: qubitCount
        )
        try initialize(amplitudes: embedded)
    }
}
