import XCTest
@testable import QuantumKit
import Metal

/// H6b: recommended construction is device-free; deprecated explicit-device APIs remain callable
/// and must allocate on the **passed** device (not silently redirected to ``MetalRuntime``).
///
/// Typical Apple Silicon hosts expose a **single** ``MTLDevice``, so these tests cannot prove
/// distinct-device honor at runtime. The honor guarantee is the code path
/// `init(…device:)` → package-internal `init(…on:)` → `device.makeBuffer` (not a synthetic
/// second-device fixture). When `MTLCopyAllDevices()` yields a second device, the deprecated
/// path additionally asserts buffers are not redirected to ``MetalRuntime/sharedDevice()``.
extension QuantumKitTests {

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
        XCTAssertTrue(state.realBuffer.device === shared)
        XCTAssertTrue(state.imagBuffer.device === shared)
        XCTAssertTrue(density.realBuffer.device === shared)
        XCTAssertTrue(density.imagBuffer.device === shared)
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
        XCTAssertTrue(batch.states[0].realBuffer.device === shared)
        XCTAssertEqual(svBackend.method, .statevector)
        XCTAssertEqual(dmBackend.method, .densityMatrix)
        XCTAssertEqual(factorySV.method, .statevector)
        XCTAssertEqual(factoryDM.method, .densityMatrix)

        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        let state = try StateVector(qubitCount: 1)
        XCTAssertTrue(state.realBuffer.device === engine.device)
        try engine.execute(circuit, on: state)
        let probs = try QuantumMeasurement.probabilities(state: state, engine: engine)
        XCTAssertEqual(probs[0], 0.5, accuracy: 1e-5)
        XCTAssertEqual(probs[1], 0.5, accuracy: 1e-5)

        let density = try DensityMatrix(qubitCount: 1)
        XCTAssertTrue(density.realBuffer.device === dmEngine.device)
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

        XCTAssertTrue(state.realBuffer.device === engine.device)
        XCTAssertTrue(state.imagBuffer.device === engine.device)
        XCTAssertTrue(batch.states[0].realBuffer.device === engine.device)
        XCTAssertTrue(batch.states[1].realBuffer.device === engine.device)
        XCTAssertTrue(density.realBuffer.device === dmEngine.device)
        XCTAssertTrue(density.imagBuffer.device === dmEngine.device)
    }

    @available(*, deprecated)
    func testDeprecatedExplicitDeviceInitsStillHonorPassedDevice() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let shared = try MetalRuntime.sharedDevice()
        // Single-GPU hosts: this only proves honor vs the shared/system device (=== smoke).
        // Distinct-device non-redirect is opportunistic below; the binding guarantee is the
        // `device:` → `on:` → `makeBuffer` path in StateVector / DensityMatrix / StateVectorBatch.
        try assertDeprecatedInitsAllocateOn(shared)

        // Opportunistic only — most Macs have one MTLDevice; do not invent a second device.
        let distinct = MTLCopyAllDevices().first { $0.registryID != shared.registryID }
        if let distinct {
            try assertDeprecatedInitsAllocateOn(distinct)
            let state = try StateVector(qubitCount: 1, device: distinct)
            XCTAssertFalse(state.realBuffer.device === shared)
        }
    }

    @available(*, deprecated)
    private func assertDeprecatedInitsAllocateOn(_ device: MTLDevice) throws {
        let state = try StateVector(qubitCount: 1, device: device)
        let density = try DensityMatrix(qubitCount: 1, device: device)
        let batch = try StateVectorBatch(qubitCount: 1, device: device, capacity: 2)

        XCTAssertEqual(state.qubitCount, 1)
        XCTAssertTrue(state.realBuffer.device === device)
        XCTAssertTrue(state.imagBuffer.device === device)
        XCTAssertEqual(density.qubitCount, 1)
        XCTAssertTrue(density.device === device)
        XCTAssertTrue(density.realBuffer.device === device)
        XCTAssertTrue(density.imagBuffer.device === device)
        XCTAssertTrue(density.device === density.realBuffer.device)
        XCTAssertEqual(batch.capacity, 2)
        XCTAssertTrue(batch.states[0].realBuffer.device === device)
        XCTAssertTrue(batch.states[1].realBuffer.device === device)
    }
}
