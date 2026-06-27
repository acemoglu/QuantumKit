import XCTest
import Metal
@testable import QuantumKit

/// Shared base for the QuantumKit test suite.
///
/// The individual test groups live in dedicated `extension QuantumKitTests`
/// files (gates, noise, measurement, Shor, algorithms, ...). This file keeps
/// the class declaration and helpers shared across all of them.
final class QuantumKitTests: XCTestCase {

    func makeDevice() -> MTLDevice? {
        MTLCreateSystemDefaultDevice()
    }
}
