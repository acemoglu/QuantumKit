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
public final class DensityMatrix {
    public static let maxQubitCount = 14

    public let qubitCount: Int
    public let stateCount: Int
    public let elementCount: Int

    public let device: MTLDevice
    public let realBuffer: MTLBuffer
    public let imagBuffer: MTLBuffer

    /// Creates a GPU density matrix on the default system Metal device.
    public convenience init(qubitCount: Int) throws {
        try self.init(qubitCount: qubitCount, device: MetalRuntime.sharedDevice())
    }

    public init(qubitCount: Int, device: MTLDevice) throws {
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
        self.device = device
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
}
