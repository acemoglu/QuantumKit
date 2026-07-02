import Foundation

public enum EstimatorError: Error, Equatable {
    case unsupportedBackend
}

/// Result of an ``Estimator`` run: the Hamiltonian expectation value plus execution metadata.
public struct EstimatorResult: Sendable, Equatable {
    public let value: QFloat
    public let metadata: QuantumResultMetadata

    public init(value: QFloat, metadata: QuantumResultMetadata) {
        self.value = value
        self.metadata = metadata
    }
}

/// High-level primitive for evaluating ⟨ψ|H|ψ⟩ (or Tr(ρH)) after circuit evolution.
///
/// Routes to exact GPU Pauli kernels on ``StatevectorBackend`` and exact Tr(ρH) on
/// ``DensityMatrixBackend``.
public struct Estimator: Sendable {
    public init() {}

    public func run(
        circuit: QuantumCircuit,
        hamiltonian: Hamiltonian,
        backend: any QuantumBackend,
        options: QuantumRunOptions = QuantumRunOptions()
    ) throws -> EstimatorResult {
        let started = DispatchTime.now()

        let value: QFloat
        let method: QuantumSimulationMethod

        if let statevectorBackend = backend as? StatevectorBackend {
            method = .statevector
            value = try estimate(
                circuit: circuit,
                hamiltonian: hamiltonian,
                engine: statevectorBackend.engine,
                options: options
            )
        } else if let densityBackend = backend as? DensityMatrixBackend {
            method = .densityMatrix
            value = try estimate(
                circuit: circuit,
                hamiltonian: hamiltonian,
                engine: densityBackend.engine,
                options: options
            )
        } else {
            throw EstimatorError.unsupportedBackend
        }

        let elapsed = DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds
        let metadata = QuantumResultMetadata(
            method: method,
            seed: options.seed,
            deviceName: MetalRuntime.deviceName,
            wallClockNanoseconds: elapsed,
            qubitCount: circuit.qubitCount,
            gateCount: circuit.gates.count,
            noiseSnapshot: options.noise
        )

        return EstimatorResult(value: value, metadata: metadata)
    }

    private func estimate(
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

    private func estimate(
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
}
