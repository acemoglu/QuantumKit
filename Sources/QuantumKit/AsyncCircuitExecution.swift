import Foundation

/// Maps Swift concurrency cancellation onto ``CircuitExecutionCancellationError``.
enum CircuitCancellation {
    static func check() throws {
        do {
            try Task.checkCancellation()
        } catch is CancellationError {
            throw CircuitExecutionCancellationError.cancelled
        }
    }

    static func mapCancellation<T>(_ body: () throws -> T) throws -> T {
        do {
            return try body()
        } catch is CancellationError {
            throw CircuitExecutionCancellationError.cancelled
        } catch let error as CircuitExecutionCancellationError {
            throw error
        }
    }
}

extension CPUStatevectorEngine {
    /// Async wrapper around ``executeRNG`` with cooperative cancellation between gates.
    public func executeRNGAsync(
        _ circuit: QuantumCircuit,
        on state: CPUStateVector,
        seed: UInt64? = nil,
        noise: NoiseModel? = nil,
        runState: CircuitRunState = .full
    ) async throws -> CircuitExecutionResult {
        var rng: QuantumRNG = seed.map { .seeded($0) } ?? .hardware
        return try CircuitCancellation.mapCancellation {
            try self.executeRNG(
                circuit,
                on: state,
                rng: &rng,
                noise: noise,
                runState: runState,
                cancellationCheck: { try CircuitCancellation.check() }
            )
        }
    }
}

extension CPUDensityMatrixEngine {
    public func executeRNGAsync(
        _ circuit: QuantumCircuit,
        on density: CPUDensityMatrix,
        seed: UInt64? = nil,
        noise: NoiseModel? = nil,
        runState: CircuitRunState = .full
    ) async throws -> CircuitExecutionResult {
        var rng: QuantumRNG = seed.map { .seeded($0) } ?? .hardware
        return try CircuitCancellation.mapCancellation {
            try self.executeRNG(
                circuit,
                on: density,
                rng: &rng,
                noise: noise,
                runState: runState,
                cancellationCheck: { try CircuitCancellation.check() }
            )
        }
    }
}

extension QuantumEngine {
    public func executeRNGAsync(
        _ circuit: QuantumCircuit,
        on state: StateVector,
        seed: UInt64? = nil,
        noise: NoiseModel? = nil,
        runState: CircuitRunState = .full
    ) async throws -> CircuitExecutionResult {
        var rng: QuantumRNG = seed.map { .seeded($0) } ?? .hardware
        return try CircuitCancellation.mapCancellation {
            try self.executeRNG(
                circuit,
                on: state,
                rng: &rng,
                noise: noise,
                runState: runState,
                cancellationCheck: { try CircuitCancellation.check() }
            )
        }
    }
}

extension DensityMatrixEngine {
    public func executeRNGAsync(
        _ circuit: QuantumCircuit,
        on density: DensityMatrix,
        seed: UInt64? = nil,
        noise: NoiseModel? = nil,
        runState: CircuitRunState = .full
    ) async throws -> CircuitExecutionResult {
        var rng: QuantumRNG = seed.map { .seeded($0) } ?? .hardware
        return try CircuitCancellation.mapCancellation {
            try self.executeRNG(
                circuit,
                on: density,
                rng: &rng,
                noise: noise,
                runState: runState,
                cancellationCheck: { try CircuitCancellation.check() }
            )
        }
    }
}

extension QuantumBackend {
    /// Async ``run`` with cooperative cancellation between gates / shot batches.
    ///
    /// On cancel, throws ``CircuitExecutionCancellationError/cancelled``. Engines drain Metal
    /// pipelines (or finish the current CPU gate) before returning so buffer pools are not leaked.
    public func runAsync(
        circuit: QuantumCircuit,
        options: QuantumRunOptions = QuantumRunOptions()
    ) async throws -> QuantumResult {
        if let cpu = self as? CPUStatevectorBackend {
            return try await cpu.runAsync(circuit: circuit, options: options)
        }
        if let cpu = self as? CPUDensityMatrixBackend {
            return try await cpu.runAsync(circuit: circuit, options: options)
        }
        if let metal = self as? StatevectorBackend {
            return try await metal.runAsync(circuit: circuit, options: options)
        }
        if let metal = self as? DensityMatrixBackend {
            return try await metal.runAsync(circuit: circuit, options: options)
        }
        if let trajectory = self as? TrajectoryBackend {
            return try await trajectory.runAsync(circuit: circuit, options: options)
        }
        return try CircuitCancellation.mapCancellation {
            try CircuitCancellation.check()
            return try self.run(circuit: circuit, options: options)
        }
    }
}

extension CPUStatevectorBackend {
    public func runAsync(
        circuit: QuantumCircuit,
        options: QuantumRunOptions = QuantumRunOptions()
    ) async throws -> QuantumResult {
        try CircuitCancellation.mapCancellation {
            try self.runCancellable(circuit: circuit, options: options)
        }
    }

    func runCancellable(
        circuit: QuantumCircuit,
        options: QuantumRunOptions
    ) throws -> QuantumResult {
        try requireQubitCount(circuit.qubitCount)
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        var rng = makeCPURNG(seed: options.seed)

        if let shots = options.shots {
            guard shots > 0 else { throw QuantumMeasurementError.invalidShotCount(shots) }
            var histogram: [Int: Int] = [:]
            for _ in 0..<shots {
                try CircuitCancellation.check()
                let state = try CPUStateVector(qubitCount: circuit.qubitCount)
                _ = try engine.executeRNG(
                    circuit,
                    on: state,
                    rng: &rng,
                    noise: options.noise,
                    cancellationCheck: { try CircuitCancellation.check() }
                )
                let outcome = try engine.measureCollapse(
                    on: state,
                    qubits: Array(0..<circuit.qubitCount),
                    rng: &rng,
                    noise: options.noise
                )
                histogram[outcome, default: 0] += 1
            }
            return QuantumResult(
                metadata: makeCPUMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .statevector
                ),
                shotCounts: ShotCounts(shots: shots, counts: histogram)
            )
        }

        let state = try CPUStateVector(qubitCount: circuit.qubitCount)
        let execution = try engine.executeRNG(
            circuit,
            on: state,
            rng: &rng,
            noise: options.noise,
            cancellationCheck: { try CircuitCancellation.check() }
        )
        return QuantumResult(
            metadata: makeCPUMetadata(
                circuit: circuit,
                options: options,
                started: started,
                method: .statevector
            ),
            execution: execution
        )
    }
}

extension CPUDensityMatrixBackend {
    public func runAsync(
        circuit: QuantumCircuit,
        options: QuantumRunOptions = QuantumRunOptions()
    ) async throws -> QuantumResult {
        try CircuitCancellation.mapCancellation {
            try self.runCancellable(circuit: circuit, options: options)
        }
    }

    func runCancellable(
        circuit: QuantumCircuit,
        options: QuantumRunOptions
    ) throws -> QuantumResult {
        try requireQubitCount(circuit.qubitCount)
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        var rng = makeCPURNG(seed: options.seed)

        if let shots = options.shots {
            let counts = try DensityMatrixShotSampler.runSampleCountsRNG(
                circuit: circuit,
                engine: engine,
                shots: shots,
                rng: &rng,
                noise: options.noise,
                cancellationCheck: { try CircuitCancellation.check() }
            )
            return QuantumResult(
                metadata: makeCPUMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .densityMatrix
                ),
                shotCounts: counts
            )
        }

        let density = try CPUDensityMatrix(qubitCount: circuit.qubitCount)
        let execution = try engine.executeRNG(
            circuit,
            on: density,
            rng: &rng,
            noise: options.noise,
            cancellationCheck: { try CircuitCancellation.check() }
        )
        return QuantumResult(
            metadata: makeCPUMetadata(
                circuit: circuit,
                options: options,
                started: started,
                method: .densityMatrix
            ),
            execution: execution
        )
    }
}

extension StatevectorBackend {
    public func runAsync(
        circuit: QuantumCircuit,
        options: QuantumRunOptions = QuantumRunOptions()
    ) async throws -> QuantumResult {
        try CircuitCancellation.mapCancellation {
            try self.runCancellable(circuit: circuit, options: options)
        }
    }

    func runCancellable(
        circuit: QuantumCircuit,
        options: QuantumRunOptions
    ) throws -> QuantumResult {
        let started = DispatchTime.now()
        try circuit.requireFullyBound()

        if let shots = options.shots {
            var rng = makeRNG(seed: options.seed)
            let counts = try QuantumMeasurement.runSampleCountsRNG(
                circuit: circuit,
                engine: engine,
                shots: shots,
                rng: &rng,
                noise: options.noise,
                options: options.sampleOptions,
                cancellationCheck: { try CircuitCancellation.check() }
            )
            return QuantumResult(
                metadata: makeMetadata(circuit: circuit, options: options, started: started),
                shotCounts: counts
            )
        }

        let state = try StateVector(qubitCount: circuit.qubitCount)
        var rng = makeRNG(seed: options.seed)
        let execution = try engine.executeRNG(
            circuit,
            on: state,
            rng: &rng,
            noise: options.noise,
            cancellationCheck: { try CircuitCancellation.check() }
        )
        return QuantumResult(
            metadata: makeMetadata(circuit: circuit, options: options, started: started),
            execution: execution
        )
    }
}

extension DensityMatrixBackend {
    public func runAsync(
        circuit: QuantumCircuit,
        options: QuantumRunOptions = QuantumRunOptions()
    ) async throws -> QuantumResult {
        try CircuitCancellation.mapCancellation {
            try self.runCancellable(circuit: circuit, options: options)
        }
    }

    func runCancellable(
        circuit: QuantumCircuit,
        options: QuantumRunOptions
    ) throws -> QuantumResult {
        let started = DispatchTime.now()
        try circuit.requireFullyBound()
        var rng = makeRNG(seed: options.seed)

        if let shots = options.shots {
            let counts = try DensityMatrixShotSampler.runSampleCountsRNG(
                circuit: circuit,
                engine: engine,
                shots: shots,
                rng: &rng,
                noise: options.noise,
                cancellationCheck: { try CircuitCancellation.check() }
            )
            return QuantumResult(
                metadata: makeMetadata(
                    circuit: circuit,
                    options: options,
                    started: started,
                    method: .densityMatrix
                ),
                shotCounts: counts
            )
        }

        let density = try DensityMatrix(qubitCount: circuit.qubitCount)
        let execution = try engine.executeRNG(
            circuit,
            on: density,
            rng: &rng,
            noise: options.noise,
            cancellationCheck: { try CircuitCancellation.check() }
        )
        return QuantumResult(
            metadata: makeMetadata(
                circuit: circuit,
                options: options,
                started: started,
                method: .densityMatrix
            ),
            execution: execution
        )
    }
}
