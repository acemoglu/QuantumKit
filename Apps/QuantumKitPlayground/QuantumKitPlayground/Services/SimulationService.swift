import Foundation
import QuantumKit

enum SimulationServiceError: LocalizedError {
    case backendUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .backendUnavailable(let detail):
            return "Could not create simulation backend: \(detail)"
        }
    }
}

/// Parse / run helpers. Safe to call from a nonisolated / detached task.
struct SimulationService: Sendable {
    nonisolated func parse(source: String) throws -> QuantumCircuit {
        try QuantumCircuit(openQASM: source)
    }

    nonisolated func run(circuit: QuantumCircuit, settings: PlaygroundSettings) throws -> PlaygroundRunOutput {
        var policy = SimulationPolicy.default
        policy.devicePreference = settings.devicePreference
        let backend = try makeBackend(for: circuit, policy: policy)
        let options = QuantumRunOptions(
            seed: settings.effectiveSeed,
            shots: settings.shots
        )

        let started = DispatchTime.now()
        let result = try backend.run(circuit: circuit, options: options)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1_000_000
        let histogram = result.bitstringCounts ?? [:]

        return PlaygroundRunOutput(
            circuit: circuit,
            result: result,
            asciiDiagram: circuit.asciiDiagram(),
            wallClockMilliseconds: elapsed,
            histogram: histogram
        )
    }

    private nonisolated func makeBackend(
        for circuit: QuantumCircuit,
        policy: SimulationPolicy
    ) throws -> any QuantumBackend {
        do {
            return try QuantumBackendFactory.makeRecommended(
                circuit: circuit,
                policy: policy,
                renormalizationInterval: settings.renormalizationInterval
            )
        } catch {
            throw SimulationServiceError.backendUnavailable(
                (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            )
        }
    }
}
