import Foundation

public enum EstimatorError: Error, Equatable {
    case unsupportedBackend
    case invalidShotCount(Int)
    case invalidPrecision(QFloat)
    /// Trajectory expectations are ensemble estimates; set ``EstimatorOptions/shots`` or ``QuantumRunOptions/shots``.
    case shotsRequiredForTrajectory
}

/// Options for ``Estimator`` beyond the shared ``QuantumRunOptions``.
///
/// When `shots` or `precision` is set, the estimator uses Pauli measurement sampling
/// instead of exact analytic expectations.
public struct EstimatorOptions: Sendable, Equatable {
    /// Explicit shot count for Pauli sampling. Wins over ``precision`` when both are set.
    public var shots: Int?
    /// Target absolute standard-error scale; translated to `shots ≈ 1/precision²` when
    /// `shots` is `nil`.
    public var precision: QFloat?

    public init(shots: Int? = nil, precision: QFloat? = nil) {
        self.shots = shots
        self.precision = precision
    }

    /// Exact analytic path (default).
    public static let exact = EstimatorOptions()

    /// Resolves the shot count to use, or `nil` for the exact estimator.
    public func resolvedShots() throws -> Int? {
        if let shots {
            guard shots > 0 else { throw EstimatorError.invalidShotCount(shots) }
            return shots
        }
        if let precision {
            guard precision > 0 else { throw EstimatorError.invalidPrecision(precision) }
            let estimate = ceil(1.0 / (Double(precision) * Double(precision)))
            return max(1, Int(estimate))
        }
        return nil
    }
}

/// Result of an ``Estimator`` run: the Hamiltonian expectation value plus execution metadata.
public struct EstimatorResult: Sendable, Equatable {
    public let value: QFloat
    public let metadata: QuantumResultMetadata
    /// Shot count when sampling was used; `nil` for exact expectations.
    public let shots: Int?
    /// Estimated standard error of the mean for shot-based runs (`≈ √(var/shots)`).
    public let standardError: QFloat?

    public init(
        value: QFloat,
        metadata: QuantumResultMetadata,
        shots: Int? = nil,
        standardError: QFloat? = nil
    ) {
        self.value = value
        self.metadata = metadata
        self.shots = shots
        self.standardError = standardError
    }
}

/// High-level primitive for evaluating ⟨ψ|H|ψ⟩ (or Tr(ρH)) after circuit evolution.
///
/// Exact path: GPU Pauli kernels on ``StatevectorBackend`` / ``DensityMatrixBackend``, and
/// host Double paths on ``CPUStatevectorBackend`` / ``CPUDensityMatrixBackend``.
/// Shot path (``EstimatorOptions/shots`` / ``precision``): Pauli-basis sampling on
/// statevector, density-matrix (prepared-ρ batching), CPU, and trajectory backends.
/// Trajectory requires shots (no exact mixed-state path).
public struct Estimator: Sendable {
    public init() {}

    public func run(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        backend: any QuantumBackend,
        options: QuantumRunOptions = QuantumRunOptions(),
        estimatorOptions: EstimatorOptions = .exact
    ) throws -> EstimatorResult {
        let started = DispatchTime.now()
        let resolvedShots = try estimatorOptions.resolvedShots() ?? options.shots

        if let shots = resolvedShots {
            return try estimateShots(
                circuit: circuit,
                hamiltonian: hamiltonian,
                backend: backend,
                options: options,
                shots: shots,
                started: started
            )
        }

        let value: QFloat
        let method: QuantumSimulationMethod
        let deviceName: String?

        if let statevectorBackend = backend as? StatevectorBackend {
            method = .statevector
            deviceName = MetalRuntime.deviceName
            value = try estimateExact(
                circuit: circuit,
                hamiltonian: hamiltonian,
                engine: statevectorBackend.engine,
                options: options
            )
        } else if let densityBackend = backend as? DensityMatrixBackend {
            method = .densityMatrix
            deviceName = MetalRuntime.deviceName
            value = try estimateExact(
                circuit: circuit,
                hamiltonian: hamiltonian,
                engine: densityBackend.engine,
                options: options
            )
        } else if let cpuSV = backend as? CPUStatevectorBackend {
            method = .statevector
            deviceName = "CPU"
            value = try estimateExactCPU(
                circuit: circuit,
                hamiltonian: hamiltonian,
                engine: cpuSV.engine,
                options: options
            )
        } else if let cpuDM = backend as? CPUDensityMatrixBackend {
            method = .densityMatrix
            deviceName = "CPU"
            value = try estimateExactCPU(
                circuit: circuit,
                hamiltonian: hamiltonian,
                engine: cpuDM.engine,
                options: options
            )
        } else if backend is TrajectoryBackend {
            throw EstimatorError.shotsRequiredForTrajectory
        } else {
            throw EstimatorError.unsupportedBackend
        }

        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = QuantumResultMetadata(
            method: method,
            seed: options.seed,
            deviceName: deviceName,
            wallClockNanoseconds: elapsed,
            qubitCount: circuit.qubitCount,
            gateCount: circuit.gates.count,
            noiseSnapshot: options.noise
        )

        return EstimatorResult(value: value, metadata: metadata)
    }

    private func estimateExact(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        engine: QuantumEngine,
        options: QuantumRunOptions
    ) throws -> QFloat {
        let state = try StateVector(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: options.noise)
        return try hamiltonian.expectation(state: state, engine: engine)
    }

    private func estimateExact(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        engine: DensityMatrixEngine,
        options: QuantumRunOptions
    ) throws -> QFloat {
        let density = try DensityMatrix(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: options.noise)
        return try hamiltonian.expectation(density: density, engine: engine)
    }

    private func estimateExactCPU(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        engine: CPUStatevectorEngine,
        options: QuantumRunOptions
    ) throws -> QFloat {
        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try engine.executeRNG(circuit, on: state, rng: &rng, noise: options.noise)
        return try hamiltonian.expectation(state: state)
    }

    private func estimateExactCPU(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        engine: CPUDensityMatrixEngine,
        options: QuantumRunOptions
    ) throws -> QFloat {
        let density = try CPUDensityMatrix(qubitCount: circuit.qubitCount)
        var rng = makePrimitiveRNG(seed: options.seed)
        _ = try engine.executeRNG(circuit, on: density, rng: &rng, noise: options.noise)
        return try hamiltonian.expectation(density: density)
    }

    private func estimateShots(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        backend: any QuantumBackend,
        options: QuantumRunOptions,
        shots: Int,
        started: DispatchTime
    ) throws -> EstimatorResult {
        guard shots > 0 else { throw EstimatorError.invalidShotCount(shots) }

        var rng = makePrimitiveRNG(seed: options.seed)
        var total: QFloat = 0
        var varianceAccumulator: QFloat = 0
        let method: QuantumSimulationMethod
        let deviceName: String?

        if let statevectorBackend = backend as? StatevectorBackend {
            method = .statevector
            deviceName = MetalRuntime.deviceName
            for term in hamiltonian.terms {
                let (mean, secondMoment) = try PauliShotEstimator.estimateTerm(
                    circuit: circuit,
                    term: term,
                    engine: statevectorBackend.engine,
                    shots: shots,
                    rng: &rng,
                    noise: options.noise,
                    sampleOptions: options.sampleOptions
                )
                total += term.coefficient * mean
                let termVar = max(0, secondMoment - mean * mean)
                varianceAccumulator += term.coefficient * term.coefficient * termVar
            }
        } else if let densityBackend = backend as? DensityMatrixBackend {
            method = .densityMatrix
            deviceName = MetalRuntime.deviceName
            for term in hamiltonian.terms {
                let (mean, secondMoment) = try PauliShotEstimator.estimateTerm(
                    circuit: circuit,
                    term: term,
                    engine: densityBackend.engine,
                    shots: shots,
                    rng: &rng,
                    noise: options.noise
                )
                total += term.coefficient * mean
                let termVar = max(0, secondMoment - mean * mean)
                varianceAccumulator += term.coefficient * term.coefficient * termVar
            }
        } else if let cpuDensity = backend as? CPUDensityMatrixBackend {
            method = .densityMatrix
            deviceName = "CPU"
            for term in hamiltonian.terms {
                let (mean, secondMoment) = try PauliShotEstimator.estimateTerm(
                    circuit: circuit,
                    term: term,
                    engine: cpuDensity.engine,
                    shots: shots,
                    rng: &rng,
                    noise: options.noise
                )
                total += term.coefficient * mean
                let termVar = max(0, secondMoment - mean * mean)
                varianceAccumulator += term.coefficient * term.coefficient * termVar
            }
        } else if let trajectory = backend as? TrajectoryBackend {
            method = .trajectory
            deviceName = trajectory.deviceName
            // Trajectory shot estimates reuse the wrapped backend via run + Z parity on counts.
            for (termIndex, term) in hamiltonian.terms.enumerated() {
                let (mean, secondMoment) = try PauliShotEstimator.estimateTermTrajectory(
                    circuit: circuit,
                    term: term,
                    backend: trajectory,
                    shots: shots,
                    seedBase: options.seed,
                    termIndex: termIndex,
                    rng: &rng,
                    noise: options.noise,
                    sampleOptions: options.sampleOptions
                )
                total += term.coefficient * mean
                let termVar = max(0, secondMoment - mean * mean)
                varianceAccumulator += term.coefficient * term.coefficient * termVar
            }
        } else if let cpuSV = backend as? CPUStatevectorBackend {
            method = .statevector
            deviceName = "CPU"
            for term in hamiltonian.terms {
                let (mean, secondMoment) = try PauliShotEstimator.estimateTermCPU(
                    circuit: circuit,
                    term: term,
                    engine: cpuSV.engine,
                    shots: shots,
                    rng: &rng,
                    noise: options.noise
                )
                total += term.coefficient * mean
                let termVar = max(0, secondMoment - mean * mean)
                varianceAccumulator += term.coefficient * term.coefficient * termVar
            }
        } else {
            throw EstimatorError.unsupportedBackend
        }

        let standardError = sqrt(varianceAccumulator / QFloat(shots))
        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = QuantumResultMetadata(
            method: method,
            seed: options.seed,
            deviceName: deviceName,
            wallClockNanoseconds: elapsed,
            qubitCount: circuit.qubitCount,
            gateCount: circuit.gates.count,
            noiseSnapshot: options.noise
        )

        return EstimatorResult(
            value: total,
            metadata: metadata,
            shots: shots,
            standardError: standardError
        )
    }
}

/// Pauli-string shot estimation via basis-change + computational-basis sampling.
enum PauliShotEstimator {
    static func estimateTerm(
        circuit: QuantumCircuit,
        term: PauliTerm,
        engine: QuantumEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        sampleOptions: SampleCountOptions
    ) throws -> (mean: QFloat, secondMoment: QFloat) {
        let support = term.paulis.keys.filter { term.paulis[$0] != .i }.sorted()
        if support.isEmpty {
            return (1, 1)
        }

        let measureCircuit = try basisChangedCircuit(circuit: circuit, term: term, support: support)
        let counts = try QuantumMeasurement.runSampleCountsRNG(
            circuit: measureCircuit,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise,
            options: sampleOptions
        )
        return moments(from: counts, qubits: support, shots: shots)
    }

    static func estimateTerm(
        circuit: QuantumCircuit,
        term: PauliTerm,
        engine: DensityMatrixEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?
    ) throws -> (mean: QFloat, secondMoment: QFloat) {
        let support = term.paulis.keys.filter { term.paulis[$0] != .i }.sorted()
        if support.isEmpty {
            return (1, 1)
        }

        let measureCircuit = try basisChangedCircuit(circuit: circuit, term: term, support: support)
        let counts = try DensityMatrixShotSampler.runSampleCountsRNG(
            circuit: measureCircuit,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise
        )
        return moments(from: counts, qubits: support, shots: shots)
    }

    static func estimateTerm(
        circuit: QuantumCircuit,
        term: PauliTerm,
        engine: CPUDensityMatrixEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?
    ) throws -> (mean: QFloat, secondMoment: QFloat) {
        let support = term.paulis.keys.filter { term.paulis[$0] != .i }.sorted()
        if support.isEmpty {
            return (1, 1)
        }

        let measureCircuit = try basisChangedCircuit(circuit: circuit, term: term, support: support)
        let counts = try DensityMatrixShotSampler.runSampleCountsRNG(
            circuit: measureCircuit,
            engine: engine,
            shots: shots,
            rng: &rng,
            noise: noise
        )
        return moments(from: counts, qubits: support, shots: shots)
    }

    static func estimateTermTrajectory(
        circuit: QuantumCircuit,
        term: PauliTerm,
        backend: TrajectoryBackend,
        shots: Int,
        seedBase: UInt64?,
        termIndex: Int = 0,
        rng: inout QuantumRNG,
        noise: NoiseModel?,
        sampleOptions: SampleCountOptions
    ) throws -> (mean: QFloat, secondMoment: QFloat) {
        let support = term.paulis.keys.filter { term.paulis[$0] != .i }.sorted()
        if support.isEmpty {
            return (1, 1)
        }

        let measureCircuit = try basisChangedCircuit(circuit: circuit, term: term, support: support)
        // Advance per Hamiltonian term so multi-term estimates stay independent.
        let seed = (seedBase ?? UInt64(rng.nextUnitDouble() * Double(UInt64.max))) &+ UInt64(termIndex)
        let result = try backend.run(
            circuit: measureCircuit,
            options: QuantumRunOptions(
                noise: noise,
                seed: seed,
                shots: shots,
                sampleOptions: sampleOptions
            )
        )
        let counts = result.shotCounts ?? ShotCounts(shots: shots, counts: [:])
        return moments(from: counts, qubits: support, shots: shots)
    }

    static func estimateTermCPU(
        circuit: QuantumCircuit,
        term: PauliTerm,
        engine: CPUStatevectorEngine,
        shots: Int,
        rng: inout QuantumRNG,
        noise: NoiseModel?
    ) throws -> (mean: QFloat, secondMoment: QFloat) {
        let support = term.paulis.keys.filter { term.paulis[$0] != .i }.sorted()
        if support.isEmpty {
            return (1, 1)
        }

        let measureCircuit = try basisChangedCircuit(circuit: circuit, term: term, support: support)
        var histogram: [Int: Int] = [:]
        for _ in 0..<shots {
            let state = try CPUStateVector(qubitCount: measureCircuit.qubitCount)
            _ = try engine.executeRNG(measureCircuit, on: state, rng: &rng, noise: noise)
            let outcome = try engine.measureCollapse(
                on: state,
                qubits: Array(0..<measureCircuit.qubitCount),
                rng: &rng,
                noise: noise
            )
            histogram[outcome, default: 0] += 1
        }
        return moments(
            from: ShotCounts(shots: shots, counts: histogram),
            qubits: support,
            shots: shots
        )
    }

    private static func basisChangedCircuit(
        circuit: QuantumCircuit,
        term: PauliTerm,
        support: [Int]
    ) throws -> QuantumCircuit {
        var measureCircuit = try QuantumCircuit(
            qubitCount: circuit.qubitCount,
            classicalRegisters: circuit.classicalRegisters
        )
        for gate in circuit.gates {
            try measureCircuit.apply(gate)
        }

        for qubit in support {
            switch term.paulis[qubit] {
            case .x:
                try measureCircuit.h(qubit)
            case .y:
                try measureCircuit.sdg(qubit)
                try measureCircuit.h(qubit)
            case .z, .i, .none:
                break
            }
        }
        return measureCircuit
    }

    private static func moments(
        from counts: ShotCounts,
        qubits: [Int],
        shots: Int
    ) -> (mean: QFloat, secondMoment: QFloat) {
        var sum: QFloat = 0
        for (outcome, count) in counts.counts {
            sum += parityEigenvalue(outcome: outcome, qubits: qubits) * QFloat(count)
        }
        let mean = sum / QFloat(shots)
        return (mean, 1)
    }

    private static func parityEigenvalue(outcome: Int, qubits: [Int]) -> QFloat {
        var parity = 0
        for qubit in qubits {
            parity ^= (outcome >> qubit) & 1
        }
        return parity == 0 ? 1 : -1
    }
}
