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
/// Explicit ``MTLDevice`` allocation and ``device`` remain available but are **deprecated** (H6b).
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

    /// Deprecated accessor for the owning Metal device. Prefer device-free construction.
    @available(*, deprecated, message: "Prefer device-free init(qubitCount:). Reading MTLDevice is deprecated; removal planned for a future major (H6c).")
    public var device: MTLDevice { metalDevice }

    public let realBuffer: MTLBuffer
    public let imagBuffer: MTLBuffer

    /// Creates a GPU density matrix via ``MetalRuntime/sharedDevice()`` (no caller Metal imports).
    public convenience init(qubitCount: Int) throws {
        try self.init(qubitCount: qubitCount, on: MetalRuntime.sharedDevice())
    }

    /// Deprecated advanced path: allocate on an explicit ``MTLDevice``. Prefer ``init(qubitCount:)``.
    ///
    /// Still allocates on the **passed** device (not ignored). Scheduled for removal in a future major (H6c).
    @available(*, deprecated, message: "Prefer init(qubitCount:). Explicit MTLDevice allocation is deprecated; removal planned for a future major (H6c).")
    public convenience init(qubitCount: Int, device: MTLDevice) throws {
        try self.init(qubitCount: qubitCount, on: device)
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
        self.realBuffer = realBuffer
        self.imagBuffer = imagBuffer

        resetToGroundState()
    }

    /// Resets to `|0...0><0...0|` without reallocating buffers.
    public func resetToGroundState() {
        let real = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imag = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        real.update(repeating: 0, count: elementCount)
        imag.update(repeating: 0, count: elementCount)
        real[0] = 1.0
    }

    /// Host copy of ρ after GPU work has drained. Prefer ``DensityMatrixEngine/snapshot(_:)``.
    public func snapshotHostMatrix() -> DensityMatrixSnapshot {
        let realPointer = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)
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
        let realPointer = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        for index in 0..<elementCount {
            realPointer[index] = snapshot.real[index]
            imagPointer[index] = snapshot.imag[index]
        }
    }
}
