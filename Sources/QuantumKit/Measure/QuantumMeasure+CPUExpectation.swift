import Foundation

extension QuantumMeasurement {

    /// Exact ⟨ψ|P|ψ⟩ on a CPU statevector (Double amplitudes).
    public static func expectation(
        state: CPUStateVector,
        paulis: [Int: Pauli]
    ) throws -> QFloat {
        var flipMask = 0
        var yMask = 0
        var zMask = 0
        var yCount = 0

        for (qubit, pauli) in paulis {
            guard qubit >= 0, qubit < state.qubitCount else {
                throw QuantumMeasurementError.qubitIndexOutOfBounds(
                    index: qubit,
                    qubitCount: state.qubitCount
                )
            }
            let bit = 1 << qubit
            switch pauli {
            case .i:
                continue
            case .x:
                flipMask |= bit
            case .y:
                flipMask |= bit
                yMask |= bit
                yCount += 1
            case .z:
                zMask |= bit
            }
        }

        let phaseBaseReal: Double
        let phaseBaseImag: Double
        switch yCount & 3 {
        case 0: (phaseBaseReal, phaseBaseImag) = (1, 0)
        case 1: (phaseBaseReal, phaseBaseImag) = (0, 1)
        case 2: (phaseBaseReal, phaseBaseImag) = (-1, 0)
        default: (phaseBaseReal, phaseBaseImag) = (0, -1)
        }

        let signMask = yMask | zMask
        var expectation = 0.0
        for index in 0..<state.stateCount {
            let partner = index ^ flipMask
            let aRe = state.real[index]
            let aIm = state.imag[index]
            let bRe = state.real[partner]
            let bIm = state.imag[partner]

            var phaseReal = phaseBaseReal
            var phaseImag = phaseBaseImag
            if signMask != 0, ((index & signMask).nonzeroBitCount & 1) == 1 {
                phaseReal = -phaseReal
                phaseImag = -phaseImag
            }

            // Re[ conj(a_partner) * phase * a_index ]
            let prodRe = bRe * (phaseReal * aRe - phaseImag * aIm)
                + bIm * (phaseReal * aIm + phaseImag * aRe)
            expectation += prodRe
        }
        return QFloat(expectation)
    }

    /// Exact Tr(ρP) on a CPU density matrix.
    public static func expectation(
        density: CPUDensityMatrix,
        paulis: [Int: Pauli]
    ) throws -> QFloat {
        var flipMask = 0
        var yMask = 0
        var zMask = 0
        var yCount = 0

        for (qubit, pauli) in paulis {
            guard qubit >= 0, qubit < density.qubitCount else {
                throw QuantumMeasurementError.qubitIndexOutOfBounds(
                    index: qubit,
                    qubitCount: density.qubitCount
                )
            }
            let bit = 1 << qubit
            switch pauli {
            case .i:
                continue
            case .x:
                flipMask |= bit
            case .y:
                flipMask |= bit
                yMask |= bit
                yCount += 1
            case .z:
                zMask |= bit
            }
        }

        let phaseBaseReal: Double
        let phaseBaseImag: Double
        switch yCount & 3 {
        case 0: (phaseBaseReal, phaseBaseImag) = (1, 0)
        case 1: (phaseBaseReal, phaseBaseImag) = (0, 1)
        case 2: (phaseBaseReal, phaseBaseImag) = (-1, 0)
        default: (phaseBaseReal, phaseBaseImag) = (0, -1)
        }

        let signMask = yMask | zMask
        let stateCount = density.stateCount
        var expectation = 0.0
        for column in 0..<stateCount {
            let row = column ^ flipMask
            let rhoReal = density.real[row * stateCount + column]
            let rhoImag = density.imag[row * stateCount + column]
            if rhoReal == 0 && rhoImag == 0 { continue }

            var phaseReal = phaseBaseReal
            var phaseImag = phaseBaseImag
            if signMask != 0, ((column & signMask).nonzeroBitCount & 1) == 1 {
                phaseReal = -phaseReal
                phaseImag = -phaseImag
            }
            expectation += (rhoReal * phaseReal) - (rhoImag * phaseImag)
        }
        return QFloat(expectation)
    }
}

extension Hamiltonian {
    /// Exact ⟨ψ|H|ψ⟩ on a CPU statevector.
    public func expectation(state: CPUStateVector) throws -> QFloat {
        var total: QFloat = 0
        for term in terms {
            total += term.coefficient * (try QuantumMeasurement.expectation(state: state, paulis: term.paulis))
        }
        return total
    }

    /// Exact Tr(ρH) on a CPU density matrix.
    public func expectation(density: CPUDensityMatrix) throws -> QFloat {
        var total: QFloat = 0
        for term in terms {
            total += term.coefficient * (try QuantumMeasurement.expectation(density: density, paulis: term.paulis))
        }
        return total
    }
}
