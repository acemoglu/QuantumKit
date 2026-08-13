import Foundation

public enum CheckpointError: Error, Equatable {
    case qubitCountMismatch(expected: Int, actual: Int)
    case elementCountMismatch(expected: Int, actual: Int)
    case instructionIndexOutOfBounds(index: Int, gateCount: Int)
    case instructionRangeInvalid(from: Int, to: Int, gateCount: Int)
}

/// Cooperative cancellation while executing a circuit or shot batch.
public enum CircuitExecutionCancellationError: Error, Equatable, Sendable {
    case cancelled
}

/// Slice / resume controls for ``executeRNG`` on Metal and CPU engines.
///
/// Quantum amplitudes are restored separately via snapshot APIs; this value carries the classical
/// cursor (instruction index, classical memory, prior mid-circuit outcomes, and renorm counter).
public struct CircuitRunState: Sendable, Equatable {
    /// Inclusive start index into ``QuantumCircuit/gates``.
    public var fromInstruction: Int
    /// Exclusive end index; `nil` runs through the end of the circuit.
    public var toInstruction: Int?
    /// When non-`nil`, replaces the fresh zeroed classical memory for this invocation.
    public var classicalMemory: ClassicalMemory?
    /// Mid-circuit outcomes already collected before ``fromInstruction``.
    public var measurementOutcomes: [[Int]]
    /// Renormalization counter carried across resume (engine-specific gate/unitary tally).
    public var appliedGateCount: Int

    public init(
        fromInstruction: Int = 0,
        toInstruction: Int? = nil,
        classicalMemory: ClassicalMemory? = nil,
        measurementOutcomes: [[Int]] = [],
        appliedGateCount: Int = 0
    ) {
        self.fromInstruction = fromInstruction
        self.toInstruction = toInstruction
        self.classicalMemory = classicalMemory
        self.measurementOutcomes = measurementOutcomes
        self.appliedGateCount = appliedGateCount
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
            appliedGateCount: checkpoint.appliedGateCount
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

    public init(
        nextInstructionIndex: Int,
        classicalMemory: ClassicalMemory,
        measurementOutcomes: [[Int]],
        appliedGateCount: Int
    ) {
        self.nextInstructionIndex = nextInstructionIndex
        self.classicalMemory = classicalMemory
        self.measurementOutcomes = measurementOutcomes
        self.appliedGateCount = appliedGateCount
    }

    public static func make(
        nextInstructionIndex: Int,
        from result: CircuitExecutionResult
    ) -> CircuitCheckpoint {
        CircuitCheckpoint(
            nextInstructionIndex: nextInstructionIndex,
            classicalMemory: result.classicalMemory,
            measurementOutcomes: result.measurementOutcomes,
            appliedGateCount: result.appliedGateCount
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
