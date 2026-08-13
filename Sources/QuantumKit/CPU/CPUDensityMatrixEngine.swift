import Foundation

/// Host-side density-matrix engine mirroring ``DensityMatrixEngine`` CPTP semantics without Metal.
public final class CPUDensityMatrixEngine: @unchecked Sendable {
    public let renormalizationInterval: Int

    public init(renormalizationInterval: Int = 50) {
        self.renormalizationInterval = max(renormalizationInterval, 0)
    }

    public func execute(
        _ circuit: QuantumCircuit,
        on density: CPUDensityMatrix,
        noise: NoiseModel? = nil
    ) throws -> CircuitExecutionResult {
        var rng: QuantumRNG = .hardware
        return try executeRNG(circuit, on: density, rng: &rng, noise: noise)
    }

    public func executeRNG(
        _ circuit: QuantumCircuit,
        on density: CPUDensityMatrix,
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil
    ) throws -> CircuitExecutionResult {
        guard circuit.qubitCount == density.qubitCount else {
            throw CPUEngineError.qubitCountMismatch(circuit: circuit.qubitCount, state: density.qubitCount)
        }
        try circuit.requireFullyBound()

        var measurementOutcomes: [[Int]] = []
        var appliedGateCount = 0
        var classicalMemory = ClassicalMemory(
            registerWidths: circuit.classicalRegisters.map(\.bitCount)
        )

        func executeRuntimeGate(_ gate: Gate, at gateIndex: Int) throws {
            switch gate {
            case .measure(let spec):
                let bits = try applyMeasurement(
                    qubits: spec.qubits,
                    on: density,
                    rng: &rng,
                    noise: noise
                )
                measurementOutcomes.append(bits)
                let outcome = bits.enumerated().reduce(0) { partial, entry in
                    partial | (entry.element << entry.offset)
                }
                try classicalMemory.writeOutcome(
                    outcome,
                    measuredQubitCount: spec.qubits.count,
                    register: spec.classicalRegister,
                    bitOffset: spec.classicalBitOffset
                )
                if let noise, noise.hasLocalizedGateNoise {
                    try applyPointNoise(after: gate, at: gateIndex, on: density, noise: noise)
                }

            case .reset(let qubit):
                applyReset(qubit: qubit, on: density)
                if let noise {
                    try applyPointNoise(after: gate, at: gateIndex, on: density, noise: noise)
                }

            case .barrier, .delay:
                if let noise {
                    try applyPointNoise(after: gate, at: gateIndex, on: density, noise: noise)
                }

            case .id:
                break

            case .c_if(let classicalRegister, let expectedValue, let conditionedGate):
                if classicalMemory.value(ofRegister: classicalRegister) == expectedValue {
                    try executeRuntimeGate(conditionedGate, at: gateIndex)
                }

            case .initialize(let qubits, let amplitudes):
                try density.initialize(qubits: qubits, amplitudes: amplitudes)
                if let noise {
                    try applyPointNoise(after: gate, at: gateIndex, on: density, noise: noise)
                }

            default:
                try applyUnitaryGate(gate, on: density)
                if let noise {
                    try applyNoiseChannels(after: gate, at: gateIndex, on: density, noise: noise)
                }
            }

            appliedGateCount += 1
            if renormalizationInterval > 0, appliedGateCount % renormalizationInterval == 0 {
                renormalizeTrace(of: density)
            }
        }

        for (gateIndex, gate) in circuit.gates.enumerated() {
            try executeRuntimeGate(gate, at: gateIndex)
        }
        renormalizeTrace(of: density)

        return CircuitExecutionResult(
            measurementOutcomes: measurementOutcomes,
            classicalMemory: classicalMemory
        )
    }

    // MARK: - Unitary

    func applyUnitaryGate(_ gate: Gate, on density: CPUDensityMatrix) throws {
        let pieces = try QuantumEngine.expandForExecution(gate)
        for piece in pieces {
            try applyExpandedUnitary(piece, on: density)
        }
    }

    private func applyExpandedUnitary(_ gate: Gate, on density: CPUDensityMatrix) throws {
        if case .customUnitary(let matrix, let qubits) = gate, qubits.count > 1 {
            try applyCustomUnitary(matrix: matrix, qubits: qubits, on: density)
            return
        }
        if case .id = gate { return }
        if case .barrier = gate { return }
        if case .delay = gate { return }

        let unitary = try CircuitUnitary.matrix(for: gate, qubitCount: density.qubitCount)
        applyFullUnitary(unitary, on: density)
    }

    private func applyFullUnitary(_ unitary: UnitaryMatrix, on density: CPUDensityMatrix) {
        let dim = density.stateCount
        // temp = U ρ
        var tempRe = Array(repeating: 0.0, count: dim * dim)
        var tempIm = Array(repeating: 0.0, count: dim * dim)
        for row in 0..<dim {
            for column in 0..<dim {
                var sumRe = 0.0
                var sumIm = 0.0
                for k in 0..<dim {
                    let u = unitary[row, k]
                    let rRe = density.real[k * dim + column]
                    let rIm = density.imag[k * dim + column]
                    sumRe += u.re * rRe - u.im * rIm
                    sumIm += u.re * rIm + u.im * rRe
                }
                tempRe[row * dim + column] = sumRe
                tempIm[row * dim + column] = sumIm
            }
        }
        // ρ' = temp U†  (U†[column,k] = conj(U[k,column]))
        var outRe = Array(repeating: 0.0, count: dim * dim)
        var outIm = Array(repeating: 0.0, count: dim * dim)
        for row in 0..<dim {
            for column in 0..<dim {
                var sumRe = 0.0
                var sumIm = 0.0
                for k in 0..<dim {
                    let u = unitary[column, k] // then conjugate
                    let uRe = u.re
                    let uIm = -u.im
                    let tRe = tempRe[row * dim + k]
                    let tIm = tempIm[row * dim + k]
                    sumRe += tRe * uRe - tIm * uIm
                    sumIm += tRe * uIm + tIm * uRe
                }
                outRe[row * dim + column] = sumRe
                outIm[row * dim + column] = sumIm
            }
        }
        density.setMatrix(real: outRe, imag: outIm)
    }

    private func applyCustomUnitary(
        matrix: [ComplexAmplitude],
        qubits: [Int],
        on density: CPUDensityMatrix
    ) throws {
        // Build full embedded unitary then apply.
        let dim = density.stateCount
        let subDim = 1 << qubits.count
        guard matrix.count == subDim * subDim else {
            throw CPUEngineError.unsupportedOnCPU(reason: "customUnitary size mismatch")
        }
        let targetMask = qubits.reduce(0) { $0 | (1 << $1) }
        let passiveMask = ((1 << density.qubitCount) - 1) & ~targetMask
        var elements = Array(repeating: UnitaryComplex.zero, count: dim * dim)
        for row in 0..<dim {
            for column in 0..<dim {
                guard (row & passiveMask) == (column & passiveMask) else { continue }
                let subRow = QubitIndexing.partialOutcomeIndex(stateIndex: row, qubits: qubits)
                let subColumn = QubitIndexing.partialOutcomeIndex(stateIndex: column, qubits: qubits)
                let entry = matrix[subRow * subDim + subColumn]
                elements[row * dim + column] = UnitaryComplex(
                    re: Double(entry.real),
                    im: Double(entry.imaginary)
                )
            }
        }
        applyFullUnitary(UnitaryMatrix(dimension: dim, elements: elements), on: density)
    }

    // MARK: - Kraus

    /// Applies Σ_k K ρ K† for single-qubit Kraus operators given as length-4 row-major matrices.
    func applyKraus1Q(
        on density: CPUDensityMatrix,
        qubit: Int,
        kraus: [[(re: Double, im: Double)]]
    ) {
        let dim = density.stateCount
        var outRe = Array(repeating: 0.0, count: dim * dim)
        var outIm = Array(repeating: 0.0, count: dim * dim)
        let bit = 1 << qubit

        for op in kraus {
            // Apply K on left: temp = K ρ, then out += temp K†
            var tempRe = Array(repeating: 0.0, count: dim * dim)
            var tempIm = Array(repeating: 0.0, count: dim * dim)

            for row in 0..<dim {
                for column in 0..<dim {
                    let rowBit = (row & bit) != 0 ? 1 : 0
                    var sumRe = 0.0
                    var sumIm = 0.0
                    for b in 0...1 {
                        let src = (row & ~bit) | (b << qubit)
                        let k = op[rowBit * 2 + b]
                        let rRe = density.real[src * dim + column]
                        let rIm = density.imag[src * dim + column]
                        sumRe += k.re * rRe - k.im * rIm
                        sumIm += k.re * rIm + k.im * rRe
                    }
                    tempRe[row * dim + column] = sumRe
                    tempIm[row * dim + column] = sumIm
                }
            }

            for row in 0..<dim {
                for column in 0..<dim {
                    let colBit = (column & bit) != 0 ? 1 : 0
                    var sumRe = 0.0
                    var sumIm = 0.0
                    for b in 0...1 {
                        let srcCol = (column & ~bit) | (b << qubit)
                        // K†[b, colBit] = conj(K[colBit, b])
                        let k = op[colBit * 2 + b]
                        let kRe = k.re
                        let kIm = -k.im
                        let tRe = tempRe[row * dim + srcCol]
                        let tIm = tempIm[row * dim + srcCol]
                        sumRe += tRe * kRe - tIm * kIm
                        sumIm += tRe * kIm + tIm * kRe
                    }
                    outRe[row * dim + column] += sumRe
                    outIm[row * dim + column] += sumIm
                }
            }
        }
        density.setMatrix(real: outRe, imag: outIm)
    }

    // MARK: - Noise channels

    private func applyNoiseChannels(
        after gate: Gate,
        at gateIndex: Int,
        on density: CPUDensityMatrix,
        noise: NoiseModel
    ) throws {
        var seen = Set<Int>()
        let affected = gate.affectedQubits.filter { seen.insert($0).inserted }

        if noise.appliesDepolarizing {
            try applyDepolarizing(on: density, qubits: affected, probability: noise.depolarizingProbability)
        }
        for qubit in affected {
            if noise.appliesAmplitudeDamping {
                let gamma = noise.effectiveAmplitudeDampingProbability
                applyAmplitudeDamping(on: density, qubit: qubit, gamma: Double(gamma))
            }
            if noise.appliesPhaseDamping {
                let lambda = noise.effectivePhaseDampingProbability
                applyPhaseDamping(on: density, qubit: qubit, lambda: Double(lambda))
            }
        }
        if noise.hasLocalizedGateNoise {
            try applyLocalized(after: gate, at: gateIndex, on: density, noise: noise, affected: affected)
        }
    }

    private func applyPointNoise(
        after gate: Gate,
        at gateIndex: Int,
        on density: CPUDensityMatrix,
        noise: NoiseModel
    ) throws {
        if case .reset(let qubit) = gate, noise.resetErrorProbability > 0 {
            applyPauliFlip(on: density, qubit: qubit, axis: 1, probability: Double(noise.resetErrorProbability))
        }
        if case .initialize(let qubits, _) = gate, noise.preparationErrorProbability > 0 {
            for qubit in qubits {
                applyPauliFlip(on: density, qubit: qubit, axis: 1, probability: Double(noise.preparationErrorProbability))
            }
        }
        if case .delay(let duration, let qubit) = gate, noise.thermalRelaxationOnDelay {
            applyThermal(on: density, qubit: qubit, duration: Double(duration), t1: Double(noise.t1), t2: Double(noise.t2))
        }
        if noise.hasLocalizedGateNoise {
            var seen = Set<Int>()
            let affected = gate.affectedQubits.filter { seen.insert($0).inserted }
            try applyLocalized(after: gate, at: gateIndex, on: density, noise: noise, affected: affected)
        }
    }

    private func applyLocalized(
        after gate: Gate,
        at gateIndex: Int,
        on density: CPUDensityMatrix,
        noise: NoiseModel,
        affected: [Int]
    ) throws {
        for rule in noise.matchingLocalizedRules(for: gate, affectedQubits: affected, gateIndex: gateIndex) {
            let qubits = rule.target.applicationQubits(gate: gate, affectedQubits: affected)
            guard !qubits.isEmpty else { continue }
            try applyLocalizedChannel(rule.channel, after: gate, on: density, qubits: qubits)
        }
    }

    private func applyLocalizedChannel(
        _ channel: QuantumChannel,
        after gate: Gate,
        on density: CPUDensityMatrix,
        qubits: [Int]
    ) throws {
        switch channel {
        case .depolarizing(let probability):
            try applyDepolarizing(on: density, qubits: qubits, probability: probability)
        case .amplitudeDamping(let gamma):
            for qubit in qubits { applyAmplitudeDamping(on: density, qubit: qubit, gamma: Double(gamma)) }
        case .phaseDamping(let lambda):
            for qubit in qubits { applyPhaseDamping(on: density, qubit: qubit, lambda: Double(lambda)) }
        case .pauliXFlip(let p):
            for qubit in qubits { applyPauliFlip(on: density, qubit: qubit, axis: 1, probability: Double(p)) }
        case .pauliYFlip(let p):
            for qubit in qubits { applyPauliFlip(on: density, qubit: qubit, axis: 2, probability: Double(p)) }
        case .pauliZFlip(let p):
            for qubit in qubits { applyPauliFlip(on: density, qubit: qubit, axis: 3, probability: Double(p)) }
        case .coherentOverRotation(let axis, let angle):
            guard abs(angle) > 0 else { return }
            for qubit in qubits {
                try applyUnitaryGate(rotationGate(axis: axis, angle: angle, target: qubit), on: density)
            }
        case .coherentUnitaryError(let axis, let angle, let probability):
            guard probability > 0, abs(angle) > 0 else { return }
            for qubit in qubits {
                applyCoherentMixture(on: density, qubit: qubit, axis: axis, angle: Double(angle), probability: Double(probability))
            }
        case .idleThermalRelaxation(let t1, let t2):
            guard case .delay(let duration, _) = gate else { return }
            for qubit in qubits {
                applyThermal(on: density, qubit: qubit, duration: Double(duration), t1: Double(t1), t2: Double(t2))
            }
        case .correlatedPauli(let axis, let probability):
            guard qubits.count == 2, probability > 0 else { return }
            applyCorrelatedPauli(
                on: density,
                qubitA: qubits[0],
                qubitB: qubits[1],
                axis: axis,
                probability: Double(probability)
            )
        case .correlatedZZ(let angle):
            guard qubits.count == 2, abs(angle) > 0 else { return }
            try applyUnitaryGate(.rzz(theta: QFloatExpr(angle), q1: qubits[0], q2: qubits[1]), on: density)
        case .kraus1Q(let operators):
            guard !operators.isEmpty else { return }
            let kraus = operators.map { op in
                op.map { (re: Double($0.real), im: Double($0.imaginary)) }
            }
            for qubit in qubits {
                applyKraus1Q(on: density, qubit: qubit, kraus: kraus)
            }
        case .pauliChannel(let px, let py, let pz):
            let pI = max(0, 1 - px - py - pz)
            guard px + py + pz > 0 else { return }
            let kI = sqrt(Double(pI))
            let kx = sqrt(Double(max(0, px)))
            let ky = sqrt(Double(max(0, py)))
            let kz = sqrt(Double(max(0, pz)))
            let ops: [[(re: Double, im: Double)]] = [
                [(kI, 0), (0, 0), (0, 0), (kI, 0)],
                [(0, 0), (kx, 0), (kx, 0), (0, 0)],
                [(0, 0), (0, -ky), (0, ky), (0, 0)],
                [(kz, 0), (0, 0), (0, 0), (-kz, 0)],
            ]
            for qubit in qubits {
                applyKraus1Q(on: density, qubit: qubit, kraus: ops)
            }
        }
    }

    private func applyDepolarizing(on density: CPUDensityMatrix, qubits: [Int], probability: QFloat) throws {
        guard probability > 0, !qubits.isEmpty else { return }
        switch qubits.count {
        case 1:
            // (1-p)ρ + (p/3)(XρX+YρY+ZρZ)
            applyPauliMixture1Q(on: density, qubit: qubits[0], probability: Double(probability))
        case 2:
            applyCorrelatedTwoQubitDepolarizing(
                on: density,
                qubitA: qubits[0],
                qubitB: qubits[1],
                probability: Double(probability)
            )
        default:
            for qubit in qubits {
                applyPauliMixture1Q(on: density, qubit: qubit, probability: Double(probability))
            }
        }
    }

    private func applyPauliMixture1Q(on density: CPUDensityMatrix, qubit: Int, probability p: Double) {
        let keep = sqrt(max(0, 1 - p))
        let jump = sqrt(max(0, p / 3))
        applyKraus1Q(on: density, qubit: qubit, kraus: [
            [(keep, 0), (0, 0), (0, 0), (keep, 0)],
            [(0, 0), (jump, 0), (jump, 0), (0, 0)],
            [(0, 0), (0, -jump), (0, jump), (0, 0)],
            [(jump, 0), (0, 0), (0, 0), (-jump, 0)],
        ])
    }

    private func applyCorrelatedTwoQubitDepolarizing(
        on density: CPUDensityMatrix,
        qubitA: Int,
        qubitB: Int,
        probability p: Double
    ) {
        // Exact channel via 16 Pauli branches is expensive on CPU; apply as
        // (1-p)ρ + (p/15) Σ_{P≠I} PρP using successive conjugations accumulated.
        let original = density.copy()
        var accRe = original.real.map { $0 * (1 - p) }
        var accIm = original.imag.map { $0 * (1 - p) }
        let weight = p / 15.0
        for code in 1...15 {
            let axisA = code / 4
            let axisB = code % 4
            density.setMatrix(real: original.real, imag: original.imag)
            if let gateA = pauliGate(axis: axisA, on: qubitA) {
                try? applyUnitaryGate(gateA, on: density)
            }
            if let gateB = pauliGate(axis: axisB, on: qubitB) {
                try? applyUnitaryGate(gateB, on: density)
            }
            for index in 0..<density.elementCount {
                accRe[index] += weight * density.real[index]
                accIm[index] += weight * density.imag[index]
            }
        }
        density.setMatrix(real: accRe, imag: accIm)
    }

    private func applyAmplitudeDamping(on density: CPUDensityMatrix, qubit: Int, gamma: Double) {
        guard gamma > 0 else { return }
        let keep = sqrt(max(0, 1 - gamma))
        let relax = sqrt(max(0, gamma))
        applyKraus1Q(on: density, qubit: qubit, kraus: [
            [(1, 0), (0, 0), (0, 0), (keep, 0)],
            [(0, 0), (relax, 0), (0, 0), (0, 0)],
        ])
    }

    private func applyPhaseDamping(on density: CPUDensityMatrix, qubit: Int, lambda: Double) {
        guard lambda > 0 else { return }
        let keep = sqrt(max(0, 1 - lambda))
        let dephase = sqrt(max(0, lambda))
        applyKraus1Q(on: density, qubit: qubit, kraus: [
            [(1, 0), (0, 0), (0, 0), (keep, 0)],
            [(0, 0), (0, 0), (0, 0), (dephase, 0)],
        ])
    }

    private func applyPauliFlip(on density: CPUDensityMatrix, qubit: Int, axis: Int, probability p: Double) {
        guard p > 0 else { return }
        let kI = sqrt(max(0, 1 - p))
        let kP = sqrt(max(0, p))
        let ops: [[(re: Double, im: Double)]]
        switch axis {
        case 1:
            ops = [
                [(kI, 0), (0, 0), (0, 0), (kI, 0)],
                [(0, 0), (kP, 0), (kP, 0), (0, 0)],
            ]
        case 2:
            ops = [
                [(kI, 0), (0, 0), (0, 0), (kI, 0)],
                [(0, 0), (0, -kP), (0, kP), (0, 0)],
            ]
        default:
            ops = [
                [(kI, 0), (0, 0), (0, 0), (kI, 0)],
                [(kP, 0), (0, 0), (0, 0), (-kP, 0)],
            ]
        }
        applyKraus1Q(on: density, qubit: qubit, kraus: ops)
    }

    private func applyCoherentMixture(
        on density: CPUDensityMatrix,
        qubit: Int,
        axis: CoherentRotationAxis,
        angle: Double,
        probability p: Double
    ) {
        let original = density.copy()
        let kI = sqrt(max(0, 1 - p))
        let kU = sqrt(max(0, p))
        // (1-p)ρ branch
        var accRe = original.real.map { $0 * (1 - p) }
        var accIm = original.imag.map { $0 * (1 - p) }
        // p UρU†
        density.setMatrix(real: original.real, imag: original.imag)
        try? applyUnitaryGate(rotationGate(axis: axis, angle: QFloat(angle), target: qubit), on: density)
        for index in 0..<density.elementCount {
            accRe[index] += p * density.real[index]
            accIm[index] += p * density.imag[index]
        }
        _ = kI
        _ = kU
        density.setMatrix(real: accRe, imag: accIm)
    }

    private func applyCorrelatedPauli(
        on density: CPUDensityMatrix,
        qubitA: Int,
        qubitB: Int,
        axis: CoherentRotationAxis,
        probability p: Double
    ) {
        let original = density.copy()
        var accRe = original.real.map { $0 * (1 - p) }
        var accIm = original.imag.map { $0 * (1 - p) }
        density.setMatrix(real: original.real, imag: original.imag)
        let gateA: Gate
        let gateB: Gate
        switch axis {
        case .x: gateA = .x(target: qubitA); gateB = .x(target: qubitB)
        case .y: gateA = .y(target: qubitA); gateB = .y(target: qubitB)
        case .z: gateA = .z(target: qubitA); gateB = .z(target: qubitB)
        }
        try? applyUnitaryGate(gateA, on: density)
        try? applyUnitaryGate(gateB, on: density)
        for index in 0..<density.elementCount {
            accRe[index] += p * density.real[index]
            accIm[index] += p * density.imag[index]
        }
        density.setMatrix(real: accRe, imag: accIm)
    }

    private func applyThermal(
        on density: CPUDensityMatrix,
        qubit: Int,
        duration: Double,
        t1: Double,
        t2: Double
    ) {
        if t1 > 0, duration > 0 {
            let gamma = 1 - exp(-duration / t1)
            applyAmplitudeDamping(on: density, qubit: qubit, gamma: gamma)
        }
        if t2 > 0, duration > 0 {
            let inverseT2 = 1.0 / t2
            let inversePure = t1 > 0 ? inverseT2 - 1.0 / (2.0 * t1) : inverseT2
            if inversePure > 0 {
                let lambda = 1 - exp(-2.0 * duration * inversePure)
                applyPhaseDamping(on: density, qubit: qubit, lambda: lambda)
            }
        }
    }

    // MARK: - Measure / reset

    private func applyMeasurement(
        qubits: [Int],
        on density: CPUDensityMatrix,
        rng: inout QuantumRNG,
        noise: NoiseModel?
    ) throws -> [Int] {
        guard !qubits.isEmpty else {
            throw QuantumMeasurementError.emptyQubitSelection
        }
        let mode = noise?.measurementMode ?? .projective
        let preDephasing = noise?.measurementDephasingProbability ?? 0
        var bits: [Int] = []
        bits.reserveCapacity(qubits.count)

        for qubit in qubits {
            if preDephasing > 0 {
                applyPhaseDamping(on: density, qubit: qubit, lambda: Double(preDephasing))
            }
            let p0 = diagonalPopulation(of: density, qubit: qubit, bit: 0)
            let outcome = rng.nextUnitDouble() < p0 ? 0 : 1

            switch mode {
            case .projective:
                let probability = outcome == 0 ? p0 : 1 - p0
                guard probability > 0 else {
                    throw CPUEngineError.zeroProbabilityMeasurement(qubit: qubit)
                }
                let scale = 1.0 / sqrt(probability)
                if outcome == 0 {
                    applyKraus1Q(on: density, qubit: qubit, kraus: [[(scale, 0), (0, 0), (0, 0), (0, 0)]])
                } else {
                    applyKraus1Q(on: density, qubit: qubit, kraus: [[(0, 0), (0, 0), (0, 0), (scale, 0)]])
                }
            case .dephasingOnly:
                applyPhaseDamping(on: density, qubit: qubit, lambda: 1)
            }
            bits.append(outcome)
        }

        if let noise {
            return noise.flipReadoutBits(bits, rng: &rng)
        }
        return bits
    }

    private func applyReset(qubit: Int, on density: CPUDensityMatrix) {
        // Kraus: |0⟩⟨0|, |0⟩⟨1|
        applyKraus1Q(on: density, qubit: qubit, kraus: [
            [(1, 0), (0, 0), (0, 0), (0, 0)],
            [(0, 0), (1, 0), (0, 0), (0, 0)],
        ])
    }

    private func diagonalPopulation(of density: CPUDensityMatrix, qubit: Int, bit: Int) -> Double {
        let mask = 1 << qubit
        var sum = 0.0
        for index in 0..<density.stateCount {
            let bitValue = (index & mask) != 0 ? 1 : 0
            guard bitValue == bit else { continue }
            sum += density.real[index * density.stateCount + index]
        }
        return max(0, sum)
    }

    private func renormalizeTrace(of density: CPUDensityMatrix) {
        let tr = density.trace()
        guard tr > 0 else { return }
        let inv = 1.0 / tr
        var outRe = density.real
        var outIm = density.imag
        for index in 0..<density.elementCount {
            outRe[index] *= inv
            outIm[index] *= inv
        }
        density.setMatrix(real: outRe, imag: outIm)
    }

    private func rotationGate(axis: CoherentRotationAxis, angle: QFloat, target: Int) -> Gate {
        let theta = QFloatExpr(angle)
        switch axis {
        case .x: return .rx(theta: theta, target: target)
        case .y: return .ry(theta: theta, target: target)
        case .z: return .rz(theta: theta, target: target)
        }
    }

    private func pauliGate(axis: Int, on qubit: Int) -> Gate? {
        switch axis {
        case 1: return .x(target: qubit)
        case 2: return .y(target: qubit)
        case 3: return .z(target: qubit)
        default: return nil
        }
    }
}
