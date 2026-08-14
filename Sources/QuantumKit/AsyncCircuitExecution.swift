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
}
