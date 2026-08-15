import Foundation

extension QuantumMeasurement {

    /// Exact ⟨P⟩ = Tr(ρP) for a Pauli tensor product on a density matrix.
    public static func expectationPauli(
        density: DensityMatrix,
        paulis: [Int: Pauli],
        engine: DensityMatrixEngine
    ) throws -> QFloat {
        var flipMask = 0
        var yMask = 0
        var zMask = 0
        var yCount = 0

        for (qubit, pauli) in paulis {
            guard qubit >= 0, qubit < density.qubitCount else {
                throw QuantumMeasurementError.qubitIndexOutOfBounds(index: qubit, qubitCount: density.qubitCount)
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
        try engine.drainPipeline()

        let real = density.metalRealBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imag = density.metalImagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let stateCount = density.stateCount

        var expectation = 0.0
        for column in 0..<stateCount {
            let row = column ^ flipMask
            let rhoReal = Double(real[row * stateCount + column])
            let rhoImag = Double(imag[row * stateCount + column])
            if rhoReal == 0 && rhoImag == 0 { continue }

            var phaseReal = phaseBaseReal
            var phaseImag = phaseBaseImag
            if signMask != 0 {
                let negative = ((column & signMask).nonzeroBitCount & 1) == 1
                if negative {
                    phaseReal = -phaseReal
                    phaseImag = -phaseImag
                }
            }

            expectation += (rhoReal * phaseReal) - (rhoImag * phaseImag)
        }

        return QFloat(expectation)
    }

    /// ⟨Z_{qubits...}⟩ on a density matrix.
    public static func expectationPauliZ(
        density: DensityMatrix,
        engine: DensityMatrixEngine,
        qubits: [Int]
    ) throws -> QFloat {
        try validateQubits(qubits, qubitCount: density.qubitCount)
        let paulis = Dictionary(uniqueKeysWithValues: qubits.map { ($0, Pauli.z) })
        return try expectationPauli(density: density, paulis: paulis, engine: engine)
    }
}
