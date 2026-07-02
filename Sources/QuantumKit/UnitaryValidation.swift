import Foundation

enum UnitaryValidation {

    static let normTolerance = 1e-5
    static let unitarityTolerance = 1e-4

    static func validateUnitNorm(_ amplitudes: [ComplexAmplitude]) throws {
        var squaredNorm = 0.0
        for amplitude in amplitudes {
            squaredNorm += Double(amplitude.real * amplitude.real + amplitude.imaginary * amplitude.imaginary)
        }
        guard abs(squaredNorm - 1.0) <= normTolerance else {
            throw StateVectorInitializationError.nonUnitNorm(squaredNorm: squaredNorm)
        }
    }

    static func validateUnitary(matrix: [ComplexAmplitude], dimension: Int) throws {
        let expectedCount = dimension * dimension
        guard matrix.count == expectedCount else {
            throw QuantumCircuitError.invalidAlgorithmParameter(
                reason: "custom unitary matrix size \(matrix.count) does not match \(dimension)×\(dimension)"
            )
        }

        for column in 0..<dimension {
            for row in 0..<dimension {
                var realPart = 0.0
                var imagPart = 0.0
                for k in 0..<dimension {
                    let uKR = Double(matrix[k * dimension + row].real)
                    let uKI = Double(matrix[k * dimension + row].imaginary)
                    let uKC = Double(matrix[k * dimension + column].real)
                    let uKD = Double(matrix[k * dimension + column].imaginary)
                    // (U†U)[row,col] = Σ_k conj(U[k,row]) · U[k,col]
                    realPart += (uKR * uKC) + (uKI * uKD)
                    imagPart += (uKR * uKD) - (uKI * uKC)
                }
                let expectedReal = row == column ? 1.0 : 0.0
                let expectedImag = 0.0
                if abs(realPart - expectedReal) > unitarityTolerance
                    || abs(imagPart - expectedImag) > unitarityTolerance {
                    throw QuantumCircuitError.invalidAlgorithmParameter(
                        reason: "matrix is not unitary"
                    )
                }
            }
        }
    }

    static func adjoint(matrix: [ComplexAmplitude], dimension: Int) -> [ComplexAmplitude] {
        var dagger = [ComplexAmplitude](repeating: ComplexAmplitude(real: 0, imaginary: 0), count: matrix.count)
        for row in 0..<dimension {
            for column in 0..<dimension {
                let element = matrix[row * dimension + column]
                dagger[column * dimension + row] = ComplexAmplitude(
                    real: element.real,
                    imaginary: -element.imaginary
                )
            }
        }
        return dagger
    }
}

enum QubitIndexing {

    static func subspaceGlobalIndex(
        outerIndex: Int,
        subIndex: Int,
        targetQubits: [Int],
        qubitCount: Int
    ) -> Int {
        let targetMask = targetQubits.reduce(0) { $0 | (1 << $1) }
        var global = 0

        var passivePosition = 0
        for qubit in 0..<qubitCount where (targetMask & (1 << qubit)) == 0 {
            if (outerIndex >> passivePosition) & 1 != 0 {
                global |= 1 << qubit
            }
            passivePosition += 1
        }

        for (localBit, qubit) in targetQubits.enumerated() {
            if (subIndex >> localBit) & 1 != 0 {
                global |= 1 << qubit
            }
        }
        return global
    }

    static func partialOutcomeIndex(stateIndex: Int, qubits: [Int]) -> Int {
        var outcome = 0
        for (position, qubit) in qubits.enumerated() {
            let bit = (stateIndex >> qubit) & 1
            outcome |= bit << position
        }
        return outcome
    }

    static func embeddedAmplitudes(
        qubits: [Int],
        amplitudes: [ComplexAmplitude],
        qubitCount: Int
    ) -> [ComplexAmplitude] {
        let subspaceDimension = 1 << qubits.count
        let stateCount = 1 << qubitCount
        var embedded = [ComplexAmplitude](
            repeating: ComplexAmplitude(real: 0, imaginary: 0),
            count: stateCount
        )
        let outerCount = 1 << (qubitCount - qubits.count)
        for outer in 0..<outerCount {
            for sub in 0..<subspaceDimension {
                let global = subspaceGlobalIndex(
                    outerIndex: outer,
                    subIndex: sub,
                    targetQubits: qubits,
                    qubitCount: qubitCount
                )
                embedded[global] = amplitudes[sub]
            }
        }
        return embedded
    }
}
