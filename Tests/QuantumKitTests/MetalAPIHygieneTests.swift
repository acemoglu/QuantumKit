import XCTest
@testable import QuantumKit
import Metal

/// H6b: recommended construction is device-free; deprecated explicit-device APIs remain callable
/// and must allocate on the **passed** device (not silently redirected to ``MetalRuntime``).
///
/// H7 soft: amplitude storage is package-`internal` (`metalRealBuffer` / `metalImagBuffer`,
/// not Swift `private`); public `realBuffer` / `imagBuffer` are soft-deprecated wrappers.
/// Normal use needs no `MTLBuffer`.
///
/// Typical Apple Silicon hosts expose a **single** ``MTLDevice``, so these tests cannot prove
/// distinct-device honor at runtime. The honor guarantee is the code path
/// `init(…device:)` → package-internal `init(…on:)` → `device.makeBuffer` (not a synthetic
/// second-device fixture). When `MTLCopyAllDevices()` yields a second device, the deprecated
/// path additionally asserts buffers are not redirected to ``MetalRuntime/sharedDevice()``.
extension QuantumKitTests {

    /// H7: construct and inspect SV/DM via public device-free + probabilities APIs only
    /// (no `MTLBuffer` / `realBuffer` / `imagBuffer` in the client path).
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
        // Package-internal storage (not the deprecated public buffer wrappers).
        XCTAssertTrue(state.metalRealBuffer.device === shared)
        XCTAssertTrue(state.metalImagBuffer.device === shared)
        XCTAssertTrue(density.metalRealBuffer.device === shared)
        XCTAssertTrue(density.metalImagBuffer.device === shared)
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
    }

    /// Soft-deprecated public buffer wrappers still return the same storage (H7 → H7b).
    @available(*, deprecated)
    func testDeprecatedPublicBufferAccessorsStillExposeInternalStorage() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let state = try StateVector(qubitCount: 1)
        let density = try DensityMatrix(qubitCount: 1)
        XCTAssertTrue(state.realBuffer === state.metalRealBuffer)
        XCTAssertTrue(state.imagBuffer === state.metalImagBuffer)
        XCTAssertTrue(density.realBuffer === density.metalRealBuffer)
        XCTAssertTrue(density.imagBuffer === density.metalImagBuffer)
    }

    /// Soft-deprecated `outputBuffer:` shim still encodes Born probs (H7 → H7b).
    @available(*, deprecated)
    func testDeprecatedExecuteProbabilityKernelOutputBufferShim() throws {
        guard MetalRuntime.isAvailable else {
            throw XCTSkip("Metal device unavailable")
        }

        let engine = try QuantumEngine()
        let state = try StateVector(qubitCount: 1)
        var circuit = try QuantumCircuit(qubitCount: 1)
        try circuit.h(0)
        try engine.execute(circuit, on: state)

        let byteCount = state.stateCount * MemoryLayout<QFloat>.stride
        guard let deprecatedBuffer = engine.device.makeBuffer(length: byteCount, options: .storageModeShared),
              let intoBuffer = engine.device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            XCTFail("Failed to allocate probability output buffers")
            return
        }

        try engine.executeProbabilityKernel(on: state, outputBuffer: deprecatedBuffer)
        try engine.executeProbabilityKernel(on: state, into: intoBuffer)

        let deprecated = deprecatedBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let into = intoBuffer.contents().assumingMemoryBound(to: QFloat.self)
        var deprecatedSum: Double = 0
        for index in 0..<state.stateCount {
            let value = deprecated[index]
            XCTAssertTrue(value.isFinite)
            XCTAssertGreaterThanOrEqual(value, 0)
            deprecatedSum += Double(value)
            XCTAssertEqual(value, into[index], accuracy: 1e-5)
        }
        XCTAssertEqual(deprecatedSum, 1.0, accuracy: 1e-5)
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
            XCTAssertFalse(state.metalRealBuffer.device === shared)
        }
    }

    @available(*, deprecated)
    private func assertDeprecatedInitsAllocateOn(_ device: MTLDevice) throws {
        let state = try StateVector(qubitCount: 1, device: device)
        let density = try DensityMatrix(qubitCount: 1, device: device)
        let batch = try StateVectorBatch(qubitCount: 1, device: device, capacity: 2)

        XCTAssertEqual(state.qubitCount, 1)
        XCTAssertTrue(state.metalRealBuffer.device === device)
        XCTAssertTrue(state.metalImagBuffer.device === device)
        XCTAssertEqual(density.qubitCount, 1)
        XCTAssertTrue(density.device === device)
        XCTAssertTrue(density.metalRealBuffer.device === device)
        XCTAssertTrue(density.metalImagBuffer.device === device)
        XCTAssertTrue(density.device === density.metalRealBuffer.device)
        XCTAssertEqual(batch.capacity, 2)
        XCTAssertTrue(batch.states[0].metalRealBuffer.device === device)
        XCTAssertTrue(batch.states[1].metalRealBuffer.device === device)
    }
}
