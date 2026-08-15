import Foundation
import Metal

public enum DensityMatrixError: Error {
    case invalidQubitCount(Int)
    case qubitCountExceedsLimit(max: Int, requested: Int)
    case matrixElementCountOverflow(qubitCount: Int)
    case bufferAllocationFailed(requiredBytes: Int)
}

/// GPU-resident density matrix for exact open-system simulation.
///
/// Stores a `2^n x 2^n` complex matrix in row-major order using two shared Metal buffers
/// (real + imaginary components). Designed for low-qubit, high-fidelity noisy simulation.
///
/// Prefer ``init(qubitCount:)`` (resolves ``MetalRuntime``) so callers never touch Metal.
/// Read state via engine probabilities / expectations or ``snapshotHostMatrix()`` — raw buffers
/// are package-`internal` `metal*` (not Swift `private`). Engine pairing uses
/// package-`internal` ``init(qubitCount:on:)``.
///
/// Thread-safety: a single ``DensityMatrix`` must not be mutated concurrently. Distinct matrices
/// may be evolved in parallel with the same or separate ``DensityMatrixEngine`` instances.
/// Prefer one engine per concurrent worker when sharing the runtime device.
public final class DensityMatrix {
    public static let maxQubitCount = 14

    public let qubitCount: Int
    public let stateCount: Int
    public let elementCount: Int

    /// Owning Metal device for buffer allocation (package-internal; prefer device-free construction).
    let metalDevice: MTLDevice

    /// Package-internal real-matrix storage (engine / measure only).
    let metalRealBuffer: MTLBuffer
    /// Package-internal imaginary-matrix storage (engine / measure only).
    let metalImagBuffer: MTLBuffer

    /// Creates a GPU density matrix via ``MetalRuntime/sharedDevice()`` (no caller Metal imports).
    public convenience init(qubitCount: Int) throws {
        try self.init(qubitCount: qubitCount, on: MetalRuntime.sharedDevice())
    }

    /// Package-internal designated initializer; always honors `device` for buffer correctness.
    init(qubitCount: Int, on device: MTLDevice) throws {
        guard qubitCount > 0 else {
            throw DensityMatrixError.invalidQubitCount(qubitCount)
        }
        guard qubitCount <= Self.maxQubitCount else {
            throw DensityMatrixError.qubitCountExceedsLimit(max: Self.maxQubitCount, requested: qubitCount)
        }

        let stateCount = 1 << qubitCount
        guard stateCount > 0 else {
            throw DensityMatrixError.matrixElementCountOverflow(qubitCount: qubitCount)
        }

        let elementCount = stateCount * stateCount
        let bytes = elementCount * MemoryLayout<QFloat>.stride
        guard let realBuffer = device.makeBuffer(length: bytes, options: .storageModeShared),
              let imagBuffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw DensityMatrixError.bufferAllocationFailed(requiredBytes: bytes * 2)
        }

        self.qubitCount = qubitCount
        self.stateCount = stateCount
        self.elementCount = elementCount
        self.metalDevice = device
        self.metalRealBuffer = realBuffer
        self.metalImagBuffer = imagBuffer

        resetToGroundState()
    }

    /// Resets to `|0...0><0...0|` without reallocating buffers.
    public func resetToGroundState() {
        let real = metalRealBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imag = metalImagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        real.update(repeating: 0, count: elementCount)
        imag.update(repeating: 0, count: elementCount)
        real[0] = 1.0
    }

    /// Host copy of ρ after GPU work has drained. Prefer ``DensityMatrixEngine/snapshot(_:)``.
    public func snapshotHostMatrix() -> DensityMatrixSnapshot {
        let realPointer = metalRealBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = metalImagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        var real = [Float](repeating: 0, count: elementCount)
        var imag = [Float](repeating: 0, count: elementCount)
        for index in 0..<elementCount {
            real[index] = realPointer[index]
            imag[index] = imagPointer[index]
        }
        return DensityMatrixSnapshot(qubitCount: qubitCount, real: real, imag: imag)
    }

    /// Restores ρ from a host snapshot into the existing shared buffers.
    public func restoreHostMatrix(from snapshot: DensityMatrixSnapshot) throws {
        guard snapshot.qubitCount == qubitCount else {
            throw CheckpointError.qubitCountMismatch(expected: qubitCount, actual: snapshot.qubitCount)
        }
        guard snapshot.real.count == elementCount, snapshot.imag.count == elementCount else {
            throw CheckpointError.elementCountMismatch(expected: elementCount, actual: snapshot.real.count)
        }
        let realPointer = metalRealBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = metalImagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        for index in 0..<elementCount {
            realPointer[index] = snapshot.real[index]
            imagPointer[index] = snapshot.imag[index]
        }
    }
}
