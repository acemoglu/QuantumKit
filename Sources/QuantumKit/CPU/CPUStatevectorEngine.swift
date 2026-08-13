import Foundation

/// Host-side statevector engine mirroring ``QuantumEngine/executeRNG`` semantics without Metal.
///
/// Thread-safety: the engine itself is safe to share across threads. Do **not** mutate the same
/// ``CPUStateVector`` from concurrent `execute` / `executeRNG` calls. Concurrent runs on distinct
/// states are supported.
public final class CPUStatevectorEngine: @unchecked Sendable {
    public let renormalizationInterval: Int

    public init(renormalizationInterval: Int = 50) {
        self.renormalizationInterval = max(renormalizationInterval, 0)
    }

    public func execute(
        _ circuit: QuantumCircuit,
        on state: CPUStateVector,
        noise: NoiseModel? = nil
    ) throws -> CircuitExecutionResult {
        var rng: QuantumRNG = .hardware
        return try executeRNG(circuit, on: state, rng: &rng, noise: noise)
    }

    public func executeRNG(
        _ circuit: QuantumCircuit,
        on state: CPUStateVector,
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil,
        runState: CircuitRunState = .full,
        cancellationCheck: (() throws -> Void)? = nil
    ) throws -> CircuitExecutionResult {
        guard circuit.qubitCount == state.qubitCount else {
            throw CPUEngineError.qubitCountMismatch(circuit: circuit.qubitCount, state: state.qubitCount)
        }
        try circuit.requireFullyBound()
        let instructionRange = try CircuitRunStateValidation.resolvedRange(
            gateCount: circuit.gates.count,
            runState: runState
        )

        if let noise, noise.hasLocalizedGateNoise {
            throw CPUEngineError.localizedNoiseRequiresDensityMatrixBackend
        }
        if let noise, noise.measurementMode != .projective {
            throw CPUEngineError.nonProjectiveMeasurementRequiresDensityMatrixBackend
        }

        let noiseEnabled = noise?.hasGateNoise == true
        var measurementOutcomes = runState.measurementOutcomes
        var appliedGateCount = runState.appliedGateCount
        var classicalMemory = runState.classicalMemory
            ?? ClassicalMemory(registerWidths: circuit.classicalRegisters.map(\.bitCount))

        func applyNoise(after gate: Gate) throws {
            guard noiseEnabled, let noise else { return }
            if noise.appliesDepolarizing {
                try applyDepolarizing(after: gate, on: state, probability: noise.depolarizingProbability, rng: &rng)
            }
            if noise.appliesAmplitudeDamping {
                try applyAmplitudeDamping(
                    after: gate,
                    on: state,
                    gamma: noise.effectiveAmplitudeDampingProbability,
                    rng: &rng
                )
            }
            if noise.appliesPhaseDamping {
                try applyPhaseDamping(
                    after: gate,
                    on: state,
                    lambda: noise.effectivePhaseDampingProbability,
                    rng: &rng
                )
            }
        }

        func executeRuntimeGate(_ gate: Gate) throws {
            switch gate {
            case .measure(let spec):
                if let noise, noise.measurementDephasingProbability > 0 {
                    try applyPhaseDamping(
                        after: gate,
                        on: state,
                        lambda: noise.measurementDephasingProbability,
                        rng: &rng
                    )
                }
                let outcome = try measureCollapse(
                    on: state,
                    qubits: spec.qubits,
                    rng: &rng,
                    noise: noise
                )
                measurementOutcomes.append(bits(from: outcome, count: spec.qubits.count))
                try classicalMemory.writeOutcome(
                    outcome,
                    measuredQubitCount: spec.qubits.count,
                    register: spec.classicalRegister,
                    bitOffset: spec.classicalBitOffset
                )

            case .reset(let qubit):
                try resetQubit(on: state, qubit: qubit, rng: &rng)
                if let noise, noise.resetErrorProbability > 0,
                   rng.nextUnitFloat() < noise.resetErrorProbability {
                    try applyUnitaryGate(.x(target: qubit), on: state)
                }

            case .barrier:
                break

            case .delay(let duration, _):
                if let noise, noise.thermalRelaxationOnDelay {
                    let gamma = noise.amplitudeDampingProbability(forDuration: duration)
                    if gamma > 0 {
                        try applyAmplitudeDamping(after: gate, on: state, gamma: gamma, rng: &rng)
                    }
                    let lambda = noise.phaseDampingProbability(forDuration: duration)
                    if lambda > 0 {
                        try applyPhaseDamping(after: gate, on: state, lambda: lambda, rng: &rng)
                    }
                }

            case .id:
                break

            case .c_if(let classicalRegister, let expectedValue, let conditionedGate):
                if classicalMemory.value(ofRegister: classicalRegister) == expectedValue {
                    try executeRuntimeGate(conditionedGate)
                }

            case .initialize(let qubits, let amplitudes):
                try state.initialize(qubits: qubits, amplitudes: amplitudes)
                if let noise, noise.preparationErrorProbability > 0 {
                    for qubit in qubits where rng.nextUnitFloat() < noise.preparationErrorProbability {
                        try applyUnitaryGate(.x(target: qubit), on: state)
                    }
                }

            default:
                try applyUnitaryGate(gate, on: state)
                if noiseEnabled {
                    try applyNoise(after: gate)
                }
            }

            appliedGateCount += 1
            if renormalizationInterval > 0, appliedGateCount % renormalizationInterval == 0 {
                try normalize(state)
            }
        }

        for index in instructionRange {
            try cancellationCheck?()
            try executeRuntimeGate(circuit.gates[index])
        }
        try normalize(state)

        return CircuitExecutionResult(
            measurementOutcomes: measurementOutcomes,
            classicalMemory: classicalMemory,
            appliedGateCount: appliedGateCount
        )
    }

// MARK: - Unitaries

    func applyUnitaryGate(_ gate: Gate, on state: CPUStateVector) throws {
        let pieces = try QuantumEngine.expandForExecution(gate)
        for piece in pieces {
            try applyExpandedUnitary(piece, on: state)
        }
    }

    private func applyExpandedUnitary(_ gate: Gate, on state: CPUStateVector) throws {
        if case .customUnitary(let matrix, let qubits) = gate, qubits.count > 1 {
            try applyCustomUnitary(matrix: matrix, qubits: qubits, on: state)
            return
        }
        if case .id = gate { return }
        if case .barrier = gate { return }
        if case .delay = gate { return }

        let unitary = try CircuitUnitary.matrix(for: gate, qubitCount: state.qubitCount)
        applyFullUnitary(unitary, on: state)
    }

    private func applyFullUnitary(_ unitary: UnitaryMatrix, on state: CPUStateVector) {
        let dim = state.stateCount
        var outReal = Array(repeating: 0.0, count: dim)
        var outImag = Array(repeating: 0.0, count: dim)
        for row in 0..<dim {
            var sumRe = 0.0
            var sumIm = 0.0
            for column in 0..<dim {
                let u = unitary[row, column]
                let inRe = state.real[column]
                let inIm = state.imag[column]
                sumRe += u.re * inRe - u.im * inIm
                sumIm += u.re * inIm + u.im * inRe
            }
            outReal[row] = sumRe
            outImag[row] = sumIm
        }
        state.setAmplitudes(real: outReal, imag: outImag)
    }

    private func applyCustomUnitary(
        matrix: [ComplexAmplitude],
        qubits: [Int],
        on state: CPUStateVector
    ) throws {
        let dimension = 1 << qubits.count
        guard matrix.count == dimension * dimension else {
            throw CPUEngineError.unsupportedOnCPU(
                reason: "customUnitary matrix size mismatch for \(qubits.count) qubits"
            )
        }
        let outerCount = 1 << (state.qubitCount - qubits.count)
        var outReal = state.real
        var outImag = state.imag

        for outer in 0..<outerCount {
            var sourceReal = [Double](repeating: 0, count: dimension)
            var sourceImag = [Double](repeating: 0, count: dimension)
            var globals = [Int](repeating: 0, count: dimension)
            for sub in 0..<dimension {
                let global = QubitIndexing.subspaceGlobalIndex(
                    outerIndex: outer,
                    subIndex: sub,
                    targetQubits: qubits,
                    qubitCount: state.qubitCount
                )
                globals[sub] = global
                sourceReal[sub] = state.real[global]
                sourceImag[sub] = state.imag[global]
            }
            for row in 0..<dimension {
                var sumRe = 0.0
                var sumIm = 0.0
                for column in 0..<dimension {
                    let element = matrix[row * dimension + column]
                    let inRe = sourceReal[column]
                    let inIm = sourceImag[column]
                    sumRe += Double(element.real) * inRe - Double(element.imaginary) * inIm
                    sumIm += Double(element.real) * inIm + Double(element.imaginary) * inRe
                }
                outReal[globals[row]] = sumRe
                outImag[globals[row]] = sumIm
            }
        }
        state.setAmplitudes(real: outReal, imag: outImag)
    }

    // MARK: - Measure / reset

    func measureCollapse(
        on state: CPUStateVector,
        qubits: [Int],
        rng: inout QuantumRNG,
        noise: NoiseModel? = nil
    ) throws -> Int {
        guard !qubits.isEmpty else {
            throw QuantumMeasurementError.emptyQubitSelection
        }

        var cumulative = [Double](repeating: 0, count: 1 << qubits.count)
        for index in 0..<state.stateCount {
            let amp2 = state.real[index] * state.real[index] + state.imag[index] * state.imag[index]
            let outcome = QubitIndexing.partialOutcomeIndex(stateIndex: index, qubits: qubits)
            cumulative[outcome] += amp2
        }
        // Turn into CDF
        var cdf = cumulative
        for index in 1..<cdf.count {
            cdf[index] += cdf[index - 1]
        }
        let total = cdf.last ?? 0
        guard total > 0 else { throw CPUEngineError.zeroStateNorm }

        let roll = rng.nextUnitDouble() * total
        var outcome = cdf.count - 1
        for index in 0..<cdf.count where roll < cdf[index] {
            outcome = index
            break
        }

        // Collapse
        var outReal = Array(repeating: 0.0, count: state.stateCount)
        var outImag = Array(repeating: 0.0, count: state.stateCount)
        var kept = 0.0
        for index in 0..<state.stateCount {
            let partial = QubitIndexing.partialOutcomeIndex(stateIndex: index, qubits: qubits)
            guard partial == outcome else { continue }
            outReal[index] = state.real[index]
            outImag[index] = state.imag[index]
            kept += state.real[index] * state.real[index] + state.imag[index] * state.imag[index]
        }
        guard kept > 0 else { throw CPUEngineError.zeroStateNorm }
        let inv = 1.0 / sqrt(kept)
        for index in 0..<state.stateCount {
            outReal[index] *= inv
            outImag[index] *= inv
        }
        state.setAmplitudes(real: outReal, imag: outImag)

        if let noise {
            return noise.flipReadoutOutcome(outcome, measuredQubitCount: qubits.count, rng: &rng)
        }
        return outcome
    }

    func resetQubit(on state: CPUStateVector, qubit: Int, rng: inout QuantumRNG) throws {
        let outcome = try measureCollapse(on: state, qubits: [qubit], rng: &rng, noise: nil)
        if outcome & 1 == 1 {
            try applyUnitaryGate(.x(target: qubit), on: state)
        }
    }

    // MARK: - Noise unraveling

    private func applyDepolarizing(
        after gate: Gate,
        on state: CPUStateVector,
        probability: QFloat,
        rng: inout QuantumRNG
    ) throws {
        guard probability > 0 else { return }
        var seen = Set<Int>()
        let qubits = gate.affectedQubits.filter { seen.insert($0).inserted }
        switch qubits.count {
        case 0:
            return
        case 1:
            guard rng.nextUnitFloat() < probability else { return }
            try applyUnitaryGate(randomSingleQubitPauli(on: qubits[0], rng: &rng), on: state)
        case 2:
            guard rng.nextUnitFloat() < probability else { return }
            for pauli in randomTwoQubitPauli(on: qubits[0], and: qubits[1], rng: &rng) {
                try applyUnitaryGate(pauli, on: state)
            }
        default:
            for qubit in qubits {
                guard rng.nextUnitFloat() < probability else { continue }
                try applyUnitaryGate(randomSingleQubitPauli(on: qubit, rng: &rng), on: state)
            }
        }
    }

    private func applyAmplitudeDamping(
        after gate: Gate,
        on state: CPUStateVector,
        gamma: QFloat,
        rng: inout QuantumRNG
    ) throws {
        guard gamma > 0 else { return }
        for qubit in Set(gate.affectedQubits) {
            let p1 = qubitOnePopulation(on: state, qubit: qubit)
            let jumpProbability = Double(gamma) * p1
            guard jumpProbability > 0 else { continue }
            if rng.nextUnitFloat() < QFloat(jumpProbability) {
                applyAmplitudeDampingJump(on: state, qubit: qubit)
            } else {
                applyAmplitudeDampingNoJump(on: state, qubit: qubit, factor: Double((1 - gamma).squareRoot()))
            }
            try normalize(state)
        }
    }

    private func applyPhaseDamping(
        after gate: Gate,
        on state: CPUStateVector,
        lambda: QFloat,
        rng: inout QuantumRNG
    ) throws {
        guard lambda > 0 else { return }
        for qubit in Set(gate.affectedQubits) {
            let p1 = qubitOnePopulation(on: state, qubit: qubit)
            let jumpProbability = Double(lambda) * p1
            guard jumpProbability > 0 else { continue }
            if rng.nextUnitFloat() < QFloat(jumpProbability) {
                applyPhaseDampingJump(on: state, qubit: qubit)
            } else {
                applyPhaseDampingNoJump(on: state, qubit: qubit, factor: Double((1 - lambda).squareRoot()))
            }
            try normalize(state)
        }
    }

    private func qubitOnePopulation(on state: CPUStateVector, qubit: Int) -> Double {
        let bit = 1 << qubit
        var total = 0.0
        for index in 0..<state.stateCount where (index & bit) != 0 {
            total += state.real[index] * state.real[index] + state.imag[index] * state.imag[index]
        }
        return total
    }

    private func applyAmplitudeDampingJump(on state: CPUStateVector, qubit: Int) {
        let bit = 1 << qubit
        var outReal = Array(repeating: 0.0, count: state.stateCount)
        var outImag = Array(repeating: 0.0, count: state.stateCount)
        for index in 0..<state.stateCount where (index & bit) != 0 {
            let partner = index ^ bit
            outReal[partner] = state.real[index]
            outImag[partner] = state.imag[index]
        }
        state.setAmplitudes(real: outReal, imag: outImag)
    }

    private func applyAmplitudeDampingNoJump(on state: CPUStateVector, qubit: Int, factor: Double) {
        let bit = 1 << qubit
        var outReal = state.real
        var outImag = state.imag
        for index in 0..<state.stateCount where (index & bit) != 0 {
            outReal[index] *= factor
            outImag[index] *= factor
        }
        state.setAmplitudes(real: outReal, imag: outImag)
    }

    private func applyPhaseDampingJump(on state: CPUStateVector, qubit: Int) {
        let bit = 1 << qubit
        var outReal = Array(repeating: 0.0, count: state.stateCount)
        var outImag = Array(repeating: 0.0, count: state.stateCount)
        for index in 0..<state.stateCount where (index & bit) != 0 {
            outReal[index] = state.real[index]
            outImag[index] = state.imag[index]
        }
        state.setAmplitudes(real: outReal, imag: outImag)
    }

    private func applyPhaseDampingNoJump(on state: CPUStateVector, qubit: Int, factor: Double) {
        applyAmplitudeDampingNoJump(on: state, qubit: qubit, factor: factor)
    }

    private func randomSingleQubitPauli(on qubit: Int, rng: inout QuantumRNG) -> Gate {
        let roll = rng.nextUnitFloat()
        if roll < 1.0 / 3.0 { return .x(target: qubit) }
        if roll < 2.0 / 3.0 { return .y(target: qubit) }
        return .z(target: qubit)
    }

    private func randomTwoQubitPauli(on qubitA: Int, and qubitB: Int, rng: inout QuantumRNG) -> [Gate] {
        let index = min(Int(rng.nextUnitFloat() * 15), 14) + 1
        let axisA = index / 4
        let axisB = index % 4
        var paulis: [Gate] = []
        if let a = pauliGate(axis: axisA, on: qubitA) { paulis.append(a) }
        if let b = pauliGate(axis: axisB, on: qubitB) { paulis.append(b) }
        return paulis
    }

    private func pauliGate(axis: Int, on qubit: Int) -> Gate? {
        switch axis {
        case 1: return .x(target: qubit)
        case 2: return .y(target: qubit)
        case 3: return .z(target: qubit)
        default: return nil
        }
    }

    private func normalize(_ state: CPUStateVector) throws {
        var total = 0.0
        for index in 0..<state.stateCount {
            total += state.real[index] * state.real[index] + state.imag[index] * state.imag[index]
        }
        guard total > 0 else { throw CPUEngineError.zeroStateNorm }
        let inv = 1.0 / sqrt(total)
        var outReal = state.real
        var outImag = state.imag
        for index in 0..<state.stateCount {
            outReal[index] *= inv
            outImag[index] *= inv
        }
        state.setAmplitudes(real: outReal, imag: outImag)
    }

    private func bits(from outcome: Int, count: Int) -> [Int] {
        (0..<count).map { (outcome >> $0) & 1 }
    }
}
