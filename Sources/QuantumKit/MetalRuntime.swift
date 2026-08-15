import Foundation
import Metal

/// Shared Metal device access for QuantumKit.
///
/// Prefer device-free convenience initializers (``StateVector/init(qubitCount:)``,
/// ``DensityMatrix/init(qubitCount:)``, ``StateVectorBatch/init(qubitCount:capacity:)``,
/// ``QuantumEngine/init(renormalizationInterval:)``, backends via ``QuantumBackendFactory``)
/// so callers never touch Metal.
///
/// ## Advanced interop — ``sharedDevice()``
///
/// ``sharedDevice()`` is the **only** public ``MTLDevice`` entry point. Keep it for rare cases
/// where application Metal code must share the same GPU as QuantumKit (custom command queues,
/// external buffer interop, diagnostics). Prefer **not** to call it from ordinary simulation
/// paths — use device-free inits instead.
///
/// ## Public `MTLDevice` inventory (H6c)
/// - ``MetalRuntime/sharedDevice()`` — **kept** (advanced / discouraged)
/// - ``MetalRuntime/deviceName`` — **kept** (diagnostics; wraps ``sharedDevice()``)
///
/// Removed in H6c (previously soft-deprecated public device surfaces):
/// - `StateVector.init(qubitCount:device:)`
/// - `DensityMatrix.init(qubitCount:device:)`
/// - `StateVectorBatch.init(qubitCount:device:capacity:)`
/// - `DensityMatrix.device`
///
/// Kept package-`internal` for engine / batch pairing:
/// - `StateVector.init(qubitCount:on:)`
/// - `DensityMatrix.init(qubitCount:on:)` / `metalDevice`
/// - `StateVectorBatch.init(qubitCount:on:capacity:)`
/// - engine `device` properties (`QuantumEngine`, `DensityMatrixEngine`)
///
/// Removed in H6b (previously deprecated **ignore-device** shims — `_ = device`; not honored):
/// - `QuantumMeasurement.runSampleCounts(...device:...)`
/// - `QuantumMeasurement.runSampleCountsRNG(...device:...)`
/// - `QuantumEngine.executeTrajectorySampleCounts(...device:...)`
/// - `QuantumEngine.executeTrajectorySampleCountsRNG(...device:...)`
///
/// ## Public `MTLBuffer` inventory (H7b)
/// Amplitude storage is package-`internal` `metalRealBuffer` / `metalImagBuffer`
/// (same-module / `@testable` visible — not Swift `private`). Normal clients use
/// amplitudes / probabilities / snapshot APIs — no public `MTLBuffer`.
///
/// Removed in H7b (previously soft-deprecated public shims):
/// - `StateVector.realBuffer` / `StateVector.imagBuffer`
/// - `DensityMatrix.realBuffer` / `DensityMatrix.imagBuffer`
/// - `QuantumEngine.executeProbabilityKernel(on:outputBuffer:)`
///
/// Kept package-`internal`:
/// - `metalRealBuffer` / `metalImagBuffer` on ``StateVector`` / ``DensityMatrix``
/// - `QuantumEngine.executeProbabilityKernel(on:into:)` for engines / measure
public enum MetalRuntime: Sendable {

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cachedDevice: MTLDevice?

    /// Returns the default system Metal device, creating and caching it on first use.
    ///
    /// - Important: Prefer device-free inits on state / engine / backend types. Call this only
    ///   when you must share an explicit ``MTLDevice`` with other Metal code (advanced interop).
    ///   Ordinary clients should never need this API.
    public static func sharedDevice() throws -> MTLDevice {
        lock.lock()
        defer { lock.unlock() }

        if let cachedDevice {
            return cachedDevice
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw QuantumEngineError.deviceNotFound
        }

        cachedDevice = device
        return device
    }

    /// Human-readable GPU name for result metadata and diagnostics.
    public static var deviceName: String? {
        (try? sharedDevice())?.name
    }
}
