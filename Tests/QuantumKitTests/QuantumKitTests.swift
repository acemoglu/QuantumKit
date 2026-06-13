import XCTest
import Metal
@testable import QuantumKit

final class QuantumKitTests: XCTestCase {

    func testBellStateEntanglement() throws {
        let engine = try QuantumEngine()

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }
        let state = StateVector(qubitCount: 2, device: device)

        var circuit = QuantumCircuit(qubitCount: 2)
        circuit.h(0)
        circuit.cx(0, 1)

        try engine.execute(circuit, on: state)

        let result = try QuantumMeasurement.measure(state: state, engine: engine)

        print("🚀 QUANTUM COLLAPSE RESULT: \(result)")

        let isZeroZero = (result[0] == 0 && result[1] == 0)
        let isOneOne = (result[0] == 1 && result[1] == 1)

        XCTAssertTrue(isZeroZero || isOneOne, "Entanglement broken! Collapsed into an impossible state: \(result)")
    }
}
