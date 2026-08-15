import Foundation
import Metal

/// Shared Metal device access for QuantumKit.
///
/// Prefer device-free convenience initializers (``StateVector/init(qubitCount:)``,
/// ``DensityMatrix/init(qubitCount:)``, ``StateVectorBatch/init(qubitCount:capacity:)``,
/// ``QuantumEngine/init(renormalizationInterval:)``, backends via ``QuantumBackendFactory``)
/// so callers never touch Metal.
///
/// ``sharedDevice()`` is retained for advanced interop (sharing an ``MTLDevice`` with other
/// Metal code). Prefer not to call it from application code.
///
/// ## Public `MTLDevice` inventory (H6b)
/// - ``MetalRuntime/sharedDevice()`` — **kept** (advanced / discouraged)
/// - ``StateVector/init(qubitCount:device:)`` — **deprecated** (honors passed device; H6c removal)
/// - ``DensityMatrix/init(qubitCount:device:)`` — **deprecated** (honors passed device; H6c removal)
/// - ``DensityMatrix/device`` — **deprecated** accessor (H6c removal)
/// - ``StateVectorBatch/init(qubitCount:device:capacity:)`` — **deprecated** (honors passed device; H6c removal)
///
/// Removed in H6b (previously deprecated **ignore-device** shims — `_ = device`; not honored):
/// - `QuantumMeasurement.runSampleCounts(...device:...)`
/// - `QuantumMeasurement.runSampleCountsRNG(...device:...)`
/// - `QuantumEngine.executeTrajectorySampleCounts(...device:...)`
/// - `QuantumEngine.executeTrajectorySampleCountsRNG(...device:...)`
///
/// In-repo test callers that used the measure `device:` overloads (and other explicit-device
/// construction) were migrated in H6b to the device-free APIs. Do not treat “no remaining
/// labeled call sites” as proof that external clients never used those overloads.
///
/// H6c (future major): remove the deprecated public `device:` inits / ``DensityMatrix/device``.
/// H7 (out of scope): encapsulate raw `MTLBuffer` surfaces.
public enum MetalRuntime: Sendable {

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cachedDevice: MTLDevice?

    /// Returns the default system Metal device, creating and caching it on first use.
    ///
    /// - Important: Prefer device-free inits on state / engine / backend types. Call this only
    ///   when you must share an explicit ``MTLDevice`` with other Metal code (advanced).
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
