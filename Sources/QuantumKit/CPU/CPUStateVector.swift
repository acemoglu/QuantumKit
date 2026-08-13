import Foundation

public enum CPUEngineError: Error, Equatable {
    case qubitCountMismatch(circuit: Int, state: Int)
    case qubitCountExceedsLimit(max: Int, requested: Int)
    case invalidQubitCount(Int)
    case zeroStateNorm
    case zeroProbabilityMeasurement(qubit: Int)
    case unsupportedOnCPU(reason: String)
    case localizedNoiseRequiresDensityMatrixBackend
    case nonProjectiveMeasurementRequiresDensityMatrixBackend
}

/// Host-side statevector used by ``CPUStatevectorEngine``.
public final class CPUStateVector: @unchecked Sendable {
    public static let maxQubitCount = 16

    public let qubitCount: Int
    public let stateCount: Int
    /// Interleaved computational-basis amplitudes as (real, imag) pairs promoted to Double.
    public private(set) var real: [Double]
    public private(set) var imag: [Double]

    public init(qubitCount: Int) throws {
        guard qubitCount > 0 else { throw CPUEngineError.invalidQubitCount(qubitCount) }
        guard qubitCount <= Self.maxQubitCount else {
            throw CPUEngineError.qubitCountExceedsLimit(max: Self.maxQubitCount, requested: qubitCount)
        }
        self.qubitCount = qubitCount
        self.stateCount = 1 << qubitCount
        self.real = Array(repeating: 0, count: stateCount)
        self.imag = Array(repeating: 0, count: stateCount)
        self.real[0] = 1
    }

    public func resetToZero() {
        for index in 0..<stateCount {
            real[index] = 0
            imag[index] = 0
        }
        real[0] = 1
    }

    public func copy() -> CPUStateVector {
        let clone: CPUStateVector
        do {
            clone = try CPUStateVector(qubitCount: qubitCount)
        } catch {
            preconditionFailure("CPUStateVector.copy failed for valid qubitCount \(qubitCount)")
        }
        clone.real = real
        clone.imag = imag
        return clone
    }

    public func probabilities() -> [QFloat] {
        (0..<stateCount).map { index in
            QFloat(real[index] * real[index] + imag[index] * imag[index])
        }
    }

    func setAmplitudes(real newReal: [Double], imag newImag: [Double]) {
        precondition(newReal.count == stateCount && newImag.count == stateCount)
        real = newReal
        imag = newImag
    }

    func initialize(qubits: [Int], amplitudes: [ComplexAmplitude]) throws {
        let embedded = QubitIndexing.embeddedAmplitudes(
            qubits: qubits,
            amplitudes: amplitudes,
            qubitCount: qubitCount
        )
        try UnitaryValidation.validateUnitNorm(embedded)
        for index in 0..<stateCount {
            real[index] = Double(embedded[index].real)
            imag[index] = Double(embedded[index].imaginary)
        }
    }
}
