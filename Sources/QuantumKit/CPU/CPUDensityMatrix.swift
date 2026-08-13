import Foundation

/// Host-side density matrix used by ``CPUDensityMatrixEngine``.
///
/// Thread-safety: one ``CPUDensityMatrix`` must not be mutated concurrently. Distinct instances may
/// be evolved in parallel on separate engines; do not share one matrix across threads.
public final class CPUDensityMatrix: @unchecked Sendable {
    public static let maxQubitCount = 8

    public let qubitCount: Int
    public let stateCount: Int
    public let elementCount: Int
    /// Row-major ρ, real/imag parts as Double.
    public private(set) var real: [Double]
    public private(set) var imag: [Double]

    public init(qubitCount: Int) throws {
        guard qubitCount > 0 else { throw CPUEngineError.invalidQubitCount(qubitCount) }
        guard qubitCount <= Self.maxQubitCount else {
            throw CPUEngineError.qubitCountExceedsLimit(max: Self.maxQubitCount, requested: qubitCount)
        }
        self.qubitCount = qubitCount
        self.stateCount = 1 << qubitCount
        self.elementCount = stateCount * stateCount
        self.real = Array(repeating: 0, count: elementCount)
        self.imag = Array(repeating: 0, count: elementCount)
        self.real[0] = 1
    }

    public func resetToZero() {
        for index in 0..<elementCount {
            real[index] = 0
            imag[index] = 0
        }
        real[0] = 1
    }

    public func copy() -> CPUDensityMatrix {
        let clone: CPUDensityMatrix
        do {
            clone = try CPUDensityMatrix(qubitCount: qubitCount)
        } catch {
            preconditionFailure("CPUDensityMatrix.copy failed for valid qubitCount \(qubitCount)")
        }
        clone.real = real
        clone.imag = imag
        return clone
    }

    public func probabilities() -> [QFloat] {
        (0..<stateCount).map { index in
            QFloat(max(0, real[index * stateCount + index]))
        }
    }

    /// Diagonal populations in Double for Float64 / analytic comparisons.
    public func probabilitiesDouble() -> [Double] {
        (0..<stateCount).map { index in
            max(0, real[index * stateCount + index])
        }
    }

    public func snapshot() -> CPUDensityMatrixSnapshot {
        CPUDensityMatrixSnapshot(qubitCount: qubitCount, real: real, imag: imag)
    }

    public func restore(from snapshot: CPUDensityMatrixSnapshot) throws {
        guard snapshot.qubitCount == qubitCount else {
            throw CheckpointError.qubitCountMismatch(expected: qubitCount, actual: snapshot.qubitCount)
        }
        guard snapshot.real.count == elementCount, snapshot.imag.count == elementCount else {
            throw CheckpointError.elementCountMismatch(expected: elementCount, actual: snapshot.real.count)
        }
        setMatrix(real: snapshot.real, imag: snapshot.imag)
    }

    func setMatrix(real newReal: [Double], imag newImag: [Double]) {
        precondition(newReal.count == elementCount && newImag.count == elementCount)
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
        // ρ = |ψ⟩⟨ψ|
        for row in 0..<stateCount {
            for column in 0..<stateCount {
                let index = row * stateCount + column
                let aRe = Double(embedded[row].real)
                let aIm = Double(embedded[row].imaginary)
                let bRe = Double(embedded[column].real)
                let bIm = Double(embedded[column].imaginary)
                // a * conj(b)
                real[index] = aRe * bRe + aIm * bIm
                imag[index] = aIm * bRe - aRe * bIm
            }
        }
    }

    func trace() -> Double {
        var sum = 0.0
        for index in 0..<stateCount {
            sum += real[index * stateCount + index]
        }
        return sum
    }
}
