//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Metal

public enum StateVectorError: Error {
    case invalidQubitCount(Int)
    case qubitCountExceedsLimit(max: Int, requested: Int)
    case stateCountOverflow(qubitCount: Int)
    case bufferAllocationFailed(requiredBytes: Int)
}

/// A GPU-resident quantum state vector.
///
/// `StateVector` is a **reference type**: its amplitudes live in shared Metal buffers, so two
/// variables that refer to the same instance read and write the same GPU memory (mutating one
/// via ``resetToZero()`` or a kernel mutates the other). Create a fresh instance per independent
/// state rather than copying.
///
/// Prefer ``init(qubitCount:)`` (resolves ``MetalRuntime``) so callers never touch Metal.
/// Read state via ``QuantumMeasurement/amplitudes(state:)``, ``QuantumMeasurement/probabilities(state:engine:)``,
/// or ``snapshotHostAmplitudes()`` — raw buffers are package-`internal` `metal*` (H7 soft; not Swift `private`).
/// Explicit ``MTLDevice`` allocation remains available but is **deprecated** (H6b); removal is H6c.
///
/// - Important: A single `StateVector` is **not** safe to mutate from multiple threads at once.
///   Distinct `StateVector` instances may be operated on concurrently (see ``QuantumEngine``).
public final class StateVector {

    public static let maxQubitCount = 28

    public let qubitCount: Int
    public let stateCount: Int

    /// Package-internal real-amplitude storage (engine / measure only).
    let metalRealBuffer: MTLBuffer
    /// Package-internal imaginary-amplitude storage (engine / measure only).
    let metalImagBuffer: MTLBuffer

    /// Deprecated raw buffer accessor. Prefer amplitudes / probabilities APIs; storage is package-`internal`.
    ///
    /// Scheduled for removal in a future major (H7b).
    @available(*, deprecated, message: "Use amplitudes/probabilities APIs; buffers are package-internal. Removal planned for a future major (H7b).")
    public var realBuffer: MTLBuffer { metalRealBuffer }

    /// Deprecated raw buffer accessor. Prefer amplitudes / probabilities APIs; storage is package-`internal`.
    ///
    /// Scheduled for removal in a future major (H7b).
    @available(*, deprecated, message: "Use amplitudes/probabilities APIs; buffers are package-internal. Removal planned for a future major (H7b).")
    public var imagBuffer: MTLBuffer { metalImagBuffer }

    /// Creates a GPU state vector via ``MetalRuntime/sharedDevice()`` (no caller Metal imports).
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
            throw StateVectorError.invalidQubitCount(qubitCount)
        }

        guard qubitCount <= Self.maxQubitCount else {
            throw StateVectorError.qubitCountExceedsLimit(max: Self.maxQubitCount, requested: qubitCount)
        }

        let computedStateCount = 1 << qubitCount
        guard computedStateCount > 0 else {
            throw StateVectorError.stateCountOverflow(qubitCount: qubitCount)
        }

        let bufferSize = computedStateCount * MemoryLayout<QFloat>.stride

        guard let realBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let imagBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            throw StateVectorError.bufferAllocationFailed(requiredBytes: bufferSize * 2)
        }

        self.qubitCount = qubitCount
        self.stateCount = computedStateCount
        self.metalRealBuffer = realBuffer
        self.metalImagBuffer = imagBuffer

        let realPointer = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        realPointer.update(repeating: 0.0, count: stateCount)
        imagPointer.update(repeating: 0.0, count: stateCount)

        realPointer[0] = 1.0
    }

    /// Resets amplitudes to |0…0⟩ without reallocating GPU buffers.
    public func resetToZero() {
        let realPointer = metalRealBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = metalImagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        realPointer.update(repeating: 0, count: stateCount)
        imagPointer.update(repeating: 0, count: stateCount)
        realPointer[0] = 1.0
    }

    /// Copies amplitudes from shared Metal buffers into a host snapshot.
    ///
    /// Call only after GPU work on this state has completed (e.g. via ``QuantumEngine/drainPipeline()``
    /// or after ``QuantumEngine/executeRNG`` returns). Prefer ``QuantumEngine/snapshot(_:)``.
    public func snapshotHostAmplitudes() -> StateVectorSnapshot {
        let realPointer = metalRealBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = metalImagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        var real = [Float](repeating: 0, count: stateCount)
        var imag = [Float](repeating: 0, count: stateCount)
        for index in 0..<stateCount {
            real[index] = realPointer[index]
            imag[index] = imagPointer[index]
        }
        return StateVectorSnapshot(qubitCount: qubitCount, real: real, imag: imag)
    }

    /// Writes a prior ``snapshotHostAmplitudes()`` back into the shared buffers (host-side only).
    public func restoreHostAmplitudes(from snapshot: StateVectorSnapshot) throws {
        guard snapshot.qubitCount == qubitCount else {
            throw CheckpointError.qubitCountMismatch(expected: qubitCount, actual: snapshot.qubitCount)
        }
        guard snapshot.real.count == stateCount, snapshot.imag.count == stateCount else {
            throw CheckpointError.elementCountMismatch(expected: stateCount, actual: snapshot.real.count)
        }
        let realPointer = metalRealBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = metalImagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        for index in 0..<stateCount {
            realPointer[index] = snapshot.real[index]
            imagPointer[index] = snapshot.imag[index]
        }
    }
}
