import Foundation
import Metal

/// Shared Metal device access for QuantumKit.
///
/// Callers should prefer convenience initializers on ``StateVector`` and ``DensityMatrix``
/// rather than passing an ``MTLDevice`` directly.
public enum MetalRuntime: Sendable {

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cachedDevice: MTLDevice?

    /// Returns the default system Metal device, creating and caching it on first use.
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
