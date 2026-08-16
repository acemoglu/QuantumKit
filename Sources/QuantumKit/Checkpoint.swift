import Foundation

public enum CheckpointError: Error, Equatable {
    case qubitCountMismatch(expected: Int, actual: Int)
    case elementCountMismatch(expected: Int, actual: Int)
    case instructionIndexOutOfBounds(index: Int, gateCount: Int)
    case instructionRangeInvalid(from: Int, to: Int, gateCount: Int)
}

/// Cooperative cancellation while executing a circuit or shot batch.
///
/// After this error, the quantum state passed into the cancelled `executeRNG` / `run` is
/// undefined and **must not** be reused. Allocate a fresh ``StateVector`` / ``DensityMatrix``
/// / CPU state for any later run. Metal engines drain the command queue before throwing so
/// buffer pools are not leaked.
public enum CircuitExecutionCancellationError: Error, Equatable, Sendable {
    case cancelled
}

/// Slice / resume controls for ``executeRNG`` on Metal and CPU engines.
///
/// Quantum amplitudes are restored separately via snapshot APIs; this value carries the classical
/// cursor (instruction index, classical memory, prior mid-circuit outcomes, and renorm counters).
///
/// RNG state is intentionally not checkpointed — callers that need bit-exact stochastic resume must
/// keep the same ``QuantumRNG`` instance across slices.
public struct CircuitRunState: Sendable, Equatable {
    /// Inclusive start index into ``QuantumCircuit/gates``.
    public var fromInstruction: Int
    /// Exclusive end index; `nil` runs through the end of the circuit.
    public var toInstruction: Int?
    /// When non-`nil`, replaces the fresh zeroed classical memory for this invocation.
    public var classicalMemory: ClassicalMemory?
    /// Mid-circuit outcomes already collected before ``fromInstruction``.
    public var measurementOutcomes: [[Int]]
    /// Number of **top-level circuit instructions** already processed (portable resume cursor).
    /// Does not include nested ``Gate/c_if`` / ``Gate/while_c`` bodies.
    public var appliedGateCount: Int
    /// Unitary-piece counter driving renormalization cadence on Metal and CPU (expanded gates
    /// may contribute several pieces). Measure / reset / barrier / delay / `id` / the `c_if`
    /// / `while_c` wrapper do not increment this. Nested `c_if` / `while_c` bodies that apply
    /// unitaries do.
    /// When `nil` on resume, engines seed from ``appliedGateCount`` for backward compatibility.
    public var unitaryRenormCount: Int?
    /// When non-`nil`, seeds ``CPUStateVector/cumulativeGlobalPhaseRadians`` /
    /// ``StateVector/cumulativeGlobalPhaseRadians`` at the start of the slice.
    public var cumulativeGlobalPhaseRadians: Double?

    public init(
        fromInstruction: Int = 0,
        toInstruction: Int? = nil,
        classicalMemory: ClassicalMemory? = nil,
        measurementOutcomes: [[Int]] = [],
        appliedGateCount: Int = 0,
        unitaryRenormCount: Int? = nil,
        cumulativeGlobalPhaseRadians: Double? = nil
    ) {
        self.fromInstruction = fromInstruction
        self.toInstruction = toInstruction
        self.classicalMemory = classicalMemory
        self.measurementOutcomes = measurementOutcomes
        self.appliedGateCount = appliedGateCount
        self.unitaryRenormCount = unitaryRenormCount
        self.cumulativeGlobalPhaseRadians = cumulativeGlobalPhaseRadians
    }

    /// Full-circuit execution from a fresh classical state.
    public static let full = CircuitRunState()

    /// Resume classical cursor after a ``CircuitCheckpoint`` (restore quantum state separately).
    public static func resume(from checkpoint: CircuitCheckpoint) -> CircuitRunState {
        CircuitRunState(
            fromInstruction: checkpoint.nextInstructionIndex,
            toInstruction: nil,
            classicalMemory: checkpoint.classicalMemory,
            measurementOutcomes: checkpoint.measurementOutcomes,
            appliedGateCount: checkpoint.appliedGateCount,
            unitaryRenormCount: checkpoint.unitaryRenormCount,
            cumulativeGlobalPhaseRadians: checkpoint.cumulativeGlobalPhaseRadians
        )
    }
}

/// Classical + instruction cursor captured between partial circuit runs.
public struct CircuitCheckpoint: Sendable, Equatable, Codable {
    /// Next gate index to execute (exclusive end of the completed slice).
    public let nextInstructionIndex: Int
    public let classicalMemory: ClassicalMemory
    public let measurementOutcomes: [[Int]]
    public let appliedGateCount: Int
    /// Unitary-piece renorm counter at checkpoint time.
    public let unitaryRenormCount: Int?
    /// SV cumulative global phase at checkpoint time (see ``GlobalPhaseTracking``).
    public let cumulativeGlobalPhaseRadians: Double?

    public init(
        nextInstructionIndex: Int,
        classicalMemory: ClassicalMemory,
        measurementOutcomes: [[Int]],
        appliedGateCount: Int,
        unitaryRenormCount: Int? = nil,
        cumulativeGlobalPhaseRadians: Double? = nil
    ) {
        self.nextInstructionIndex = nextInstructionIndex
        self.classicalMemory = classicalMemory
        self.measurementOutcomes = measurementOutcomes
        self.appliedGateCount = appliedGateCount
        self.unitaryRenormCount = unitaryRenormCount
        self.cumulativeGlobalPhaseRadians = cumulativeGlobalPhaseRadians
    }

    public static func make(
        nextInstructionIndex: Int,
        from result: CircuitExecutionResult
    ) -> CircuitCheckpoint {
        CircuitCheckpoint(
            nextInstructionIndex: nextInstructionIndex,
            classicalMemory: result.classicalMemory,
            measurementOutcomes: result.measurementOutcomes,
            appliedGateCount: result.appliedGateCount,
            unitaryRenormCount: result.unitaryRenormCount,
            cumulativeGlobalPhaseRadians: result.cumulativeGlobalPhaseRadians
        )
    }
}

/// Host-side copy of a Metal or CPU statevector (computational-basis amplitudes).
public struct StateVectorSnapshot: Sendable, Equatable, Codable {
    public let qubitCount: Int
    /// Real parts stored as Float32 to match Metal ``QFloat`` buffers without widening mutation APIs.
    public let real: [Float]
    public let imag: [Float]
    /// Cumulative global phase \(\Phi\) in radians (see ``GlobalPhaseTracking``).
    public let cumulativeGlobalPhaseRadians: Double

    public init(
        qubitCount: Int,
        real: [Float],
        imag: [Float],
        cumulativeGlobalPhaseRadians: Double = 0
    ) {
        self.qubitCount = qubitCount
        self.real = real
        self.imag = imag
        self.cumulativeGlobalPhaseRadians = cumulativeGlobalPhaseRadians
    }

    /// Pre-B16 snapshots omit \(\Phi\); missing key decodes as `0`.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        qubitCount = try container.decode(Int.self, forKey: .qubitCount)
        real = try container.decode([Float].self, forKey: .real)
        imag = try container.decode([Float].self, forKey: .imag)
        cumulativeGlobalPhaseRadians =
            try container.decodeIfPresent(Double.self, forKey: .cumulativeGlobalPhaseRadians) ?? 0
    }
}

/// Host-side copy of a density matrix in row-major order.
public struct DensityMatrixSnapshot: Sendable, Equatable, Codable {
    public let qubitCount: Int
    public let real: [Float]
    public let imag: [Float]

    public init(qubitCount: Int, real: [Float], imag: [Float]) {
        self.qubitCount = qubitCount
        self.real = real
        self.imag = imag
    }
}

/// Double-precision statevector snapshot for ``CPUStateVector``.
public struct CPUStateVectorSnapshot: Sendable, Equatable, Codable {
    public let qubitCount: Int
    public let real: [Double]
    public let imag: [Double]
    /// Cumulative global phase \(\Phi\) in radians (see ``GlobalPhaseTracking``).
    public let cumulativeGlobalPhaseRadians: Double

    public init(
        qubitCount: Int,
        real: [Double],
        imag: [Double],
        cumulativeGlobalPhaseRadians: Double = 0
    ) {
        self.qubitCount = qubitCount
        self.real = real
        self.imag = imag
        self.cumulativeGlobalPhaseRadians = cumulativeGlobalPhaseRadians
    }

    /// Pre-B16 snapshots omit \(\Phi\); missing key decodes as `0`.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        qubitCount = try container.decode(Int.self, forKey: .qubitCount)
        real = try container.decode([Double].self, forKey: .real)
        imag = try container.decode([Double].self, forKey: .imag)
        cumulativeGlobalPhaseRadians =
            try container.decodeIfPresent(Double.self, forKey: .cumulativeGlobalPhaseRadians) ?? 0
    }
}

/// Double-precision density-matrix snapshot for ``CPUDensityMatrix``.
public struct CPUDensityMatrixSnapshot: Sendable, Equatable, Codable {
    public let qubitCount: Int
    public let real: [Double]
    public let imag: [Double]

    public init(qubitCount: Int, real: [Double], imag: [Double]) {
        self.qubitCount = qubitCount
        self.real = real
        self.imag = imag
    }
}

enum CircuitRunStateValidation {
    static func resolvedRange(
        gateCount: Int,
        runState: CircuitRunState
    ) throws -> Range<Int> {
        let from = runState.fromInstruction
        let to = runState.toInstruction ?? gateCount
        guard from >= 0, from <= gateCount else {
            throw CheckpointError.instructionIndexOutOfBounds(index: from, gateCount: gateCount)
        }
        guard to >= from, to <= gateCount else {
            throw CheckpointError.instructionRangeInvalid(from: from, to: to, gateCount: gateCount)
        }
        return from..<to
    }
}
