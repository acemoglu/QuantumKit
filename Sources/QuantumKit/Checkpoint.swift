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
    /// Does not include nested ``Gate/c_if`` bodies.
    public var appliedGateCount: Int
    /// Unitary-piece counter driving renormalization cadence on Metal and CPU (expanded gates
    /// may contribute several pieces). Measure / reset / barrier / delay / `id` / the `c_if`
    /// wrapper do not increment this. Nested `c_if` bodies that apply unitaries do.
    /// When `nil` on resume, engines seed from ``appliedGateCount`` for backward compatibility.
    public var unitaryRenormCount: Int?

    public init(
        fromInstruction: Int = 0,
        toInstruction: Int? = nil,
        classicalMemory: ClassicalMemory? = nil,
        measurementOutcomes: [[Int]] = [],
        appliedGateCount: Int = 0,
        unitaryRenormCount: Int? = nil
    ) {
        self.fromInstruction = fromInstruction
        self.toInstruction = toInstruction
        self.classicalMemory = classicalMemory
        self.measurementOutcomes = measurementOutcomes
        self.appliedGateCount = appliedGateCount
        self.unitaryRenormCount = unitaryRenormCount
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
            unitaryRenormCount: checkpoint.unitaryRenormCount
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

    public init(
        nextInstructionIndex: Int,
        classicalMemory: ClassicalMemory,
        measurementOutcomes: [[Int]],
        appliedGateCount: Int,
        unitaryRenormCount: Int? = nil
    ) {
        self.nextInstructionIndex = nextInstructionIndex
        self.classicalMemory = classicalMemory
        self.measurementOutcomes = measurementOutcomes
        self.appliedGateCount = appliedGateCount
        self.unitaryRenormCount = unitaryRenormCount
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
            unitaryRenormCount: result.unitaryRenormCount
        )
    }
}

/// Host-side copy of a Metal or CPU statevector (computational-basis amplitudes).
public struct StateVectorSnapshot: Sendable, Equatable, Codable {
    public let qubitCount: Int
    /// Real parts stored as Float32 to match Metal ``QFloat`` buffers without widening mutation APIs.
    public let real: [Float]
    public let imag: [Float]

    public init(qubitCount: Int, real: [Float], imag: [Float]) {
        self.qubitCount = qubitCount
        self.real = real
        self.imag = imag
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

    public init(qubitCount: Int, real: [Double], imag: [Double]) {
        self.qubitCount = qubitCount
        self.real = real
        self.imag = imag
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
