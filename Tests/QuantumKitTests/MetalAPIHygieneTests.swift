import XCTest
@testable import QuantumKit
import Metal

/// H6c: recommended construction is device-free. Explicit public `device:` inits /
/// ``DensityMatrix/device`` are gone; package-`internal` `init(…on:)` remains for engine pairing.
///
/// H7b: amplitude storage is package-`internal` (`metalRealBuffer` / `metalImagBuffer`,
/// not Swift `private`); public `realBuffer` / `imagBuffer` / `outputBuffer:` shims are gone.
/// Normal use needs no `MTLBuffer` / `MTLDevice`.
extension QuantumKitTests {

    /// Construct and inspect SV/DM via public device-free + probabilities APIs only
    /// (no `MTLBuffer` / `MTLDevice` in the client path).
    func testDeviceFreeStateInspectionWithoutMTLBuffer() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        let dmEngine = try DensityMatrixEngine()
        let state = try StateVector(qubitCount: 1)
        let density = try DensityMatrix(qubitCount: 1)

        XCTAssertEqual(state.qubitCount, 1)
        XCTAssertEqual(state.stateCount, 2)
        XCTAssertEqual(density.qubitCount, 1)
        XCTAssertEqual(density.elementCount, 4)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try engine.execute(circuit, on: state)
        try dmEngine.execute(circuit, on: density)

        let amplitudes = QuantumMeasurement.amplitudes(state: state)
        let invSqrt2 = QFloat(1.0 / 2.0.squareRoot())
        XCTAssertEqual(amplitudes.count, 2)
        XCTAssertEqual(amplitudes[0].real, invSqrt2, accuracy: 1e-5)
        XCTAssertEqual(amplitudes[1].real, invSqrt2, accuracy: 1e-5)

        let probs = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probs[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(probs[1], 0.5, accuracy: 1e-5)

        let dmProbs = dmEngine.probabilities(of: density)
        XCTAssertEqual(dmProbs[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(dmProbs[1], 0.5, accuracy: 1e-5)

        let snapshot = state.snapshotHostAmplitudes()
        XCTAssertEqual(snapshot.real.count, 2)
        let dmSnapshot = density.snapshotHostMatrix()
        XCTAssertEqual(dmSnapshot.real.count, 4)
    }

    func testDeviceFreeStateVectorAndDensityMatrixConstruction() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let shared = try MetalRuntime.sharedDevice()
        let state = try StateVector(qubitCount: 2)
        let density = try DensityMatrix(qubitCount: 2)

        XCTAssertEqual(state.qubitCount, 2)
        XCTAssertEqual(state.stateCount, 4)
        XCTAssertEqual(density.qubitCount, 2)
        XCTAssertEqual(density.elementCount, 16)
        // Package-internal storage tracks MetalRuntime shared device.
        XCTAssertTrue(state.metalRealBuffer.device === shared)
        XCTAssertTrue(state.metalImagBuffer.device === shared)
        XCTAssertTrue(density.metalRealBuffer.device === shared)
        XCTAssertTrue(density.metalImagBuffer.device === shared)
        XCTAssertTrue(density.metalDevice === shared)
    }

    func testDeviceFreeEnginesBackendsAndBatchConstruction() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let shared = try MetalRuntime.sharedDevice()
        let engine = try QuantumEngine()
        let dmEngine = try DensityMatrixEngine()
        let batch = try StateVectorBatch(qubitCount: 2, capacity: 4)
        let svBackend = try StatevectorBackend()
        let dmBackend = try DensityMatrixBackend()
        let factorySV = try QuantumBackendFactory.makeStatevector()
        let factoryDM = try QuantumBackendFactory.makeDensityMatrix()

        XCTAssertEqual(engine.renormalizationInterval, 50)
        XCTAssertEqual(dmEngine.renormalizationInterval, 50)
        XCTAssertTrue(engine.device === shared)
        XCTAssertTrue(dmEngine.device === shared)
        XCTAssertEqual(batch.capacity, 4)
        XCTAssertEqual(batch.states.count, 4)
        XCTAssertTrue(batch.states[0].metalRealBuffer.device === shared)
        XCTAssertEqual(svBackend.method, .statevector)
        XCTAssertEqual(dmBackend.method, .densityMatrix)
        XCTAssertEqual(factorySV.method, .statevector)
        XCTAssertEqual(factoryDM.method, .densityMatrix)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let state = try StateVector(qubitCount: 1)
        XCTAssertTrue(state.metalRealBuffer.device === engine.device)
        try engine.execute(circuit, on: state)
        let probs = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probs[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(probs[1], 0.5, accuracy: 1e-5)

        let density = try DensityMatrix(qubitCount: 1)
        XCTAssertTrue(density.metalRealBuffer.device === dmEngine.device)
        try dmEngine.execute(circuit, on: density)
        let dmProbs = dmEngine.probabilities(of: density)
        XCTAssertEqual(dmProbs[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(dmProbs[1], 0.5, accuracy: 1e-5)

        let result = try svBackend.run(circuit: circuit)
        XCTAssertEqual(result.metadata.method, .statevector)
    }

    /// Internal `on:` path used by batching / DM shot sampling must track `engine.device`.
    func testInternalOnDeviceInitsPairWithEngineDevice() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        let dmEngine = try DensityMatrixEngine()

        let state = try StateVector(qubitCount: 1, on: engine.device)
        let batch = try StateVectorBatch(qubitCount: 1, on: engine.device, capacity: 2)
        let density = try DensityMatrix(qubitCount: 1, on: dmEngine.device)

        XCTAssertTrue(state.metalRealBuffer.device === engine.device)
        XCTAssertTrue(state.metalImagBuffer.device === engine.device)
        XCTAssertTrue(batch.states[0].metalRealBuffer.device === engine.device)
        XCTAssertTrue(batch.states[1].metalRealBuffer.device === engine.device)
        XCTAssertTrue(density.metalRealBuffer.device === dmEngine.device)
        XCTAssertTrue(density.metalImagBuffer.device === dmEngine.device)
        XCTAssertTrue(density.metalDevice === dmEngine.device)
    }
}
