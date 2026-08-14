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

    /// After ``CircuitExecutionCancellationError``, `state` is undefined and must not be reused;
    /// allocate a fresh ``CPUStateVector`` for any later run.
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
        // Includes nested c_if bodies so renorm cadence matches a continuous run.
        var renormTick = runState.unitaryRenormCount ?? runState.appliedGateCount
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

        func applyCircuitUnitary(_ gate: Gate) throws {
            let pieces = try QuantumEngine.expandForExecution(gate)
            for piece in pieces {
                try applyExpandedUnitary(piece, on: state)
                renormTick += 1
                if renormalizationInterval > 0, renormTick % renormalizationInterval == 0 {
                    try normalize(state)
                }
            }
        }

        func executeRuntimeGate(_ gate: Gate, countsTowardApplied: Bool = true) throws {
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
                    try applyCircuitUnitary(.x(target: qubit))
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
                    try executeRuntimeGate(conditionedGate, countsTowardApplied: false)
                }

            case .initialize(let qubits, let amplitudes):
                try state.initialize(qubits: qubits, amplitudes: amplitudes)
                if let noise, noise.preparationErrorProbability > 0 {
                    for qubit in qubits where rng.nextUnitFloat() < noise.preparationErrorProbability {
                        try applyCircuitUnitary(.x(target: qubit))
                    }
                }

            default:
                try applyCircuitUnitary(gate)
                if noiseEnabled {
                    try applyNoise(after: gate)
                }
            }

            if countsTowardApplied {
                appliedGateCount += 1
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
            appliedGateCount: appliedGateCount,
            unitaryRenormCount: renormTick
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
        switch gate {
        case .customUnitary(let matrix, let qubits):
            if qubits.count == 1, let target = qubits.first {
                try apply1QMatrix(matrix, target: target, on: state)
            } else {
                try applyCustomUnitary(matrix: matrix, qubits: qubits, on: state)
            }

        case .unitary1(let matrix, let target):
            try apply1QMatrix(matrix, target: target, on: state)

        case .id, .barrier, .delay:
            return

        case .h(let target):
            let v = 1.0 / sqrt(2.0)
            apply1Q(u00: (v, 0), u01: (v, 0), u10: (v, 0), u11: (-v, 0), target: target, on: state)

        case .x(let target):
            applyPauliX(target: target, on: state)

        case .y(let target):
            apply1Q(u00: (0, 0), u01: (0, -1), u10: (0, 1), u11: (0, 0), target: target, on: state)

        case .z(let target):
            applyPhaseOnBit(target: target, re: -1, im: 0, on: state)

        case .s(let target):
            applyPhaseOnBit(target: target, re: 0, im: 1, on: state)

        case .sdg(let target):
            applyPhaseOnBit(target: target, re: 0, im: -1, on: state)

        case .t(let target):
            let v = 1.0 / sqrt(2.0)
            applyPhaseOnBit(target: target, re: v, im: v, on: state)

        case .tdg(let target):
            let v = 1.0 / sqrt(2.0)
            applyPhaseOnBit(target: target, re: v, im: -v, on: state)

        case .sx(let target):
            apply1Q(
                u00: (0.5, 0.5), u01: (0.5, -0.5),
                u10: (0.5, -0.5), u11: (0.5, 0.5),
                target: target, on: state
            )

        case .sxdg(let target):
            apply1Q(
                u00: (0.5, -0.5), u01: (0.5, 0.5),
                u10: (0.5, 0.5), u11: (0.5, -0.5),
                target: target, on: state
            )

        case .p(let theta, let target):
            let angle = Double(try theta.requireLiteral())
            applyPhaseOnBit(target: target, re: cos(angle), im: sin(angle), on: state)

        case .rx(let theta, let target):
            try applyRotation(.x, theta: theta, target: target, on: state)

        case .ry(let theta, let target):
            try applyRotation(.y, theta: theta, target: target, on: state)

        case .rz(let theta, let target):
            try applyRotation(.z, theta: theta, target: target, on: state)

        case .u(let theta, let phi, let lambda, let target):
            let thetaV = Double(try theta.requireLiteral())
            let phiV = Double(try phi.requireLiteral())
            let lambdaV = Double(try lambda.requireLiteral())
            let half = thetaV / 2.0
            let c = cos(half)
            let s = sin(half)
            apply1Q(
                u00: (c, 0),
                u01: (-s * cos(lambdaV), -s * sin(lambdaV)),
                u10: (s * cos(phiV), s * sin(phiV)),
                u11: (c * cos(phiV + lambdaV), c * sin(phiV + lambdaV)),
                target: target,
                on: state
            )

        case .cx(let control, let target):
            applyControlledX(controlMask: 1 << control, target: target, on: state)

        case .cz(let control, let target):
            applyPhaseOnMask(mask: (1 << control) | (1 << target), re: -1, im: 0, on: state)

        case .swap(let q1, let q2):
            applySwap(q1: q1, q2: q2, on: state)

        case .ccx(let control1, let control2, let target):
            applyControlledX(controlMask: (1 << control1) | (1 << control2), target: target, on: state)

        case .mcx(let controls, let target):
            let mask = controls.reduce(0) { $0 | (1 << $1) }
            applyControlledX(controlMask: mask, target: target, on: state)

        case .mcz(let controls, let target):
            let mask = controls.reduce(0) { $0 | (1 << $1) } | (1 << target)
            applyPhaseOnMask(mask: mask, re: -1, im: 0, on: state)

        case .crx(let theta, let control, let target):
            try applyControlledRotation(.x, theta: theta, controlMask: 1 << control, target: target, on: state)

        case .cry(let theta, let control, let target):
            try applyControlledRotation(.y, theta: theta, controlMask: 1 << control, target: target, on: state)

        case .crz(let theta, let control, let target):
            try applyControlledRotation(.z, theta: theta, controlMask: 1 << control, target: target, on: state)

        case .cp(let theta, let control, let target):
            let angle = Double(try theta.requireLiteral())
            applyPhaseOnMask(
                mask: (1 << control) | (1 << target),
                re: cos(angle),
                im: sin(angle),
                on: state
            )

        default:
            throw CPUEngineError.unsupportedOnCPU(
                reason: "CPU statevector apply has no subspace kernel for \(gate)"
            )
        }
    }

    private enum CPURotationAxis {
        case x, y, z
    }

    private func applyRotation(
        _ axis: CPURotationAxis,
        theta: QFloatExpr,
        target: Int,
        on state: CPUStateVector
    ) throws {
        try applyControlledRotation(axis, theta: theta, controlMask: 0, target: target, on: state)
    }

    private func applyControlledRotation(
        _ axis: CPURotationAxis,
        theta: QFloatExpr,
        controlMask: Int,
        target: Int,
        on state: CPUStateVector
    ) throws {
        let half = Double(try theta.requireLiteral()) / 2.0
        let c = cos(half)
        let s = sin(half)
        switch axis {
        case .x:
            apply1Q(
                u00: (c, 0), u01: (0, -s),
                u10: (0, -s), u11: (c, 0),
                target: target, on: state, controlMask: controlMask
            )
        case .y:
            apply1Q(
                u00: (c, 0), u01: (-s, 0),
                u10: (s, 0), u11: (c, 0),
                target: target, on: state, controlMask: controlMask
            )
        case .z:
            apply1Q(
                u00: (cos(half), -sin(half)), u01: (0, 0),
                u10: (0, 0), u11: (cos(half), sin(half)),
                target: target, on: state, controlMask: controlMask
            )
        }
    }

    private func apply1QMatrix(_ matrix: [ComplexAmplitude], target: Int, on state: CPUStateVector) throws {
        guard matrix.count == 4 else {
            throw CPUEngineError.unsupportedOnCPU(reason: "1-qubit matrix must contain 4 elements")
        }
        apply1Q(
            u00: (Double(matrix[0].real), Double(matrix[0].imaginary)),
            u01: (Double(matrix[1].real), Double(matrix[1].imaginary)),
            u10: (Double(matrix[2].real), Double(matrix[2].imaginary)),
            u11: (Double(matrix[3].real), Double(matrix[3].imaginary)),
            target: target,
            on: state
        )
    }

    private func apply1Q(
        u00: (Double, Double),
        u01: (Double, Double),
        u10: (Double, Double),
        u11: (Double, Double),
        target: Int,
        on state: CPUStateVector,
        controlMask: Int = 0
    ) {
        let bit = 1 << target
        var outReal = state.real
        var outImag = state.imag
        for index in 0..<state.stateCount {
            let partner = index ^ bit
            if index > partner { continue }
            if controlMask != 0, (index & controlMask) != controlMask { continue }

            let aRe = state.real[index]
            let aIm = state.imag[index]
            let bRe = state.real[partner]
            let bIm = state.imag[partner]

            outReal[index] = u00.0 * aRe - u00.1 * aIm + u01.0 * bRe - u01.1 * bIm
            outImag[index] = u00.0 * aIm + u00.1 * aRe + u01.0 * bIm + u01.1 * bRe
            outReal[partner] = u10.0 * aRe - u10.1 * aIm + u11.0 * bRe - u11.1 * bIm
            outImag[partner] = u10.0 * aIm + u10.1 * aRe + u11.0 * bIm + u11.1 * bRe
        }
        state.setAmplitudes(real: outReal, imag: outImag)
    }

    private func applyPauliX(target: Int, on state: CPUStateVector) {
        applyControlledX(controlMask: 0, target: target, on: state)
    }

    private func applyControlledX(controlMask: Int, target: Int, on state: CPUStateVector) {
        let bit = 1 << target
        var outReal = state.real
        var outImag = state.imag
        for index in 0..<state.stateCount {
            let partner = index ^ bit
            if index > partner { continue }
            if controlMask != 0, (index & controlMask) != controlMask { continue }
            outReal[index] = state.real[partner]
            outImag[index] = state.imag[partner]
            outReal[partner] = state.real[index]
            outImag[partner] = state.imag[index]
        }
        state.setAmplitudes(real: outReal, imag: outImag)
    }

    private func applyPhaseOnBit(target: Int, re: Double, im: Double, on state: CPUStateVector) {
        applyPhaseOnMask(mask: 1 << target, re: re, im: im, on: state)
    }

    private func applyPhaseOnMask(mask: Int, re: Double, im: Double, on state: CPUStateVector) {
        var outReal = state.real
        var outImag = state.imag
        for index in 0..<state.stateCount where (index & mask) == mask {
            let inRe = state.real[index]
            let inIm = state.imag[index]
            outReal[index] = re * inRe - im * inIm
            outImag[index] = re * inIm + im * inRe
        }
        state.setAmplitudes(real: outReal, imag: outImag)
    }

    private func applySwap(q1: Int, q2: Int, on state: CPUStateVector) {
        if q1 == q2 { return }
        let bitA = 1 << q1
        let bitB = 1 << q2
        var outReal = state.real
        var outImag = state.imag
        for index in 0..<state.stateCount {
            let a = (index & bitA) != 0
            let b = (index & bitB) != 0
            guard a != b else { continue }
            let partner = index ^ bitA ^ bitB
            if index > partner { continue }
            outReal[index] = state.real[partner]
            outImag[index] = state.imag[partner]
            outReal[partner] = state.real[index]
            outImag[partner] = state.imag[index]
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
