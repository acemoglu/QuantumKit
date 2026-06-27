import Foundation
import Metal

public enum DensityMatrixEngineError: Error {
    case deviceNotFound
    case commandQueueCreationFailed
    case commandBufferCreationFailed
    case commandBufferExecutionFailed(underlying: Error?)
    case functionNotFound(String)
    case libraryNotFound
    case qubitCountMismatch(circuit: Int, matrix: Int)
    case unsupportedGate(Gate)
    case nonUnitaryGateUnsupported(Gate)
}

public final class DensityMatrixEngine: @unchecked Sendable {
    struct Pipelines: @unchecked Sendable {
        let leftMultiplySingleQubit: MTLComputePipelineState
        let rightMultiplySingleQubitDagger: MTLComputePipelineState
        let applyKrausSingleQubit: MTLComputePipelineState

        init(device: MTLDevice, library: MTLLibrary) throws {
            guard let leftFunc = library.makeFunction(name: "dm_left_multiply_single_qubit") else {
                throw DensityMatrixEngineError.functionNotFound("dm_left_multiply_single_qubit")
            }
            self.leftMultiplySingleQubit = try device.makeComputePipelineState(function: leftFunc)

            guard let rightFunc = library.makeFunction(name: "dm_right_multiply_single_qubit_dagger") else {
                throw DensityMatrixEngineError.functionNotFound("dm_right_multiply_single_qubit_dagger")
            }
            self.rightMultiplySingleQubitDagger = try device.makeComputePipelineState(function: rightFunc)

            guard let krausFunc = library.makeFunction(name: "dm_apply_kraus_single_qubit") else {
                throw DensityMatrixEngineError.functionNotFound("dm_apply_kraus_single_qubit")
            }
            self.applyKrausSingleQubit = try device.makeComputePipelineState(function: krausFunc)
        }
    }

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let bufferPool: BufferPool
    let pipelines: Pipelines

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw DensityMatrixEngineError.deviceNotFound
        }
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            throw DensityMatrixEngineError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        self.bufferPool = BufferPool(device: device)
        self.pipelines = try Pipelines(device: device, library: try Self.loadLibrary(device: device))
    }

    public func execute(_ circuit: QuantumCircuit, on density: DensityMatrix, noise: NoiseModel? = nil) throws {
        guard circuit.qubitCount == density.qubitCount else {
            throw DensityMatrixEngineError.qubitCountMismatch(circuit: circuit.qubitCount, matrix: density.qubitCount)
        }

        let scratchBytes = density.elementCount * MemoryLayout<QFloat>.stride
        let scratchReal = try bufferPool.acquire(length: scratchBytes)
        let scratchImag = try bufferPool.acquire(length: scratchBytes)
        defer {
            bufferPool.release(scratchReal)
            bufferPool.release(scratchImag)
        }

        for gate in circuit.gates {
            switch gate {
            case .measure, .reset:
                throw DensityMatrixEngineError.nonUnitaryGateUnsupported(gate)
            default:
                try applyUnitaryGate(gate, on: density, scratchReal: scratchReal, scratchImag: scratchImag)
                if let noise {
                    try applyNoiseChannels(after: gate, on: density, noise: noise, scratchReal: scratchReal, scratchImag: scratchImag)
                }
            }
        }
    }

    public func probabilities(of density: DensityMatrix) -> [QFloat] {
        let real = density.realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        return (0..<density.stateCount).map { basis in
            real[basis * density.stateCount + basis]
        }
    }

    private func applyUnitaryGate(
        _ gate: Gate,
        on density: DensityMatrix,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        let encoded = try encodeSingleQubitUnitary(gate)
        let matrixBuffer = try makeSharedBuffer(
            values: encoded.matrix,
            byteCount: encoded.matrix.count * MemoryLayout<SIMD2<QFloat>>.stride
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DensityMatrixEngineError.commandBufferCreationFailed
        }

        var stateCount = UInt32(density.stateCount)
        var target = UInt32(encoded.target)
        var controlMask = encoded.controlMask

        // Pass A: temp = U * rho
        encoder.setComputePipelineState(pipelines.leftMultiplySingleQubit)
        encoder.setBuffer(density.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(density.imagBuffer, offset: 0, index: 1)
        encoder.setBuffer(scratchReal, offset: 0, index: 2)
        encoder.setBuffer(scratchImag, offset: 0, index: 3)
        encoder.setBytes(&stateCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&target, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&controlMask, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 7)
        dispatchMatrixElements(
            encoder: encoder,
            elementCount: density.elementCount,
            pipeline: pipelines.leftMultiplySingleQubit
        )

        // Pass B: rho = temp * U†
        encoder.setComputePipelineState(pipelines.rightMultiplySingleQubitDagger)
        encoder.setBuffer(scratchReal, offset: 0, index: 0)
        encoder.setBuffer(scratchImag, offset: 0, index: 1)
        encoder.setBuffer(density.realBuffer, offset: 0, index: 2)
        encoder.setBuffer(density.imagBuffer, offset: 0, index: 3)
        encoder.setBytes(&stateCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&target, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&controlMask, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 7)
        dispatchMatrixElements(
            encoder: encoder,
            elementCount: density.elementCount,
            pipeline: pipelines.rightMultiplySingleQubitDagger
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DensityMatrixEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    private func applyNoiseChannels(
        after gate: Gate,
        on density: DensityMatrix,
        noise: NoiseModel,
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        let affected = Set(gate.affectedQubits)
        for qubit in affected {
            if noise.appliesDepolarizing {
                let p = noise.depolarizingProbability
                let k0 = sqrt(max(0, 1 - p))
                let k1 = sqrt(p / 3)
                try applyKrausChannel(
                    on: density,
                    targetQubit: qubit,
                    kraus: [
                        [complex(k0, 0), complex(0, 0), complex(0, 0), complex(k0, 0)],
                        [complex(0, 0), complex(k1, 0), complex(k1, 0), complex(0, 0)],
                        [complex(0, 0), complex(0, -k1), complex(0, k1), complex(0, 0)],
                        [complex(k1, 0), complex(0, 0), complex(0, 0), complex(-k1, 0)],
                    ],
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

            if noise.appliesAmplitudeDamping {
                let gamma = noise.effectiveAmplitudeDampingProbability
                let keep = sqrt(max(0, 1 - gamma))
                let relax = sqrt(max(0, gamma))
                try applyKrausChannel(
                    on: density,
                    targetQubit: qubit,
                    kraus: [
                        [complex(1, 0), complex(0, 0), complex(0, 0), complex(keep, 0)],
                        [complex(0, 0), complex(relax, 0), complex(0, 0), complex(0, 0)],
                    ],
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }

            if noise.appliesPhaseDamping {
                let lambda = noise.phaseDampingProbability
                let keep = sqrt(max(0, 1 - lambda))
                let dephase = sqrt(max(0, lambda))
                try applyKrausChannel(
                    on: density,
                    targetQubit: qubit,
                    kraus: [
                        [complex(1, 0), complex(0, 0), complex(0, 0), complex(keep, 0)],
                        [complex(0, 0), complex(0, 0), complex(0, 0), complex(dephase, 0)],
                    ],
                    scratchReal: scratchReal,
                    scratchImag: scratchImag
                )
            }
        }
    }

    private func applyKrausChannel(
        on density: DensityMatrix,
        targetQubit: Int,
        kraus: [[SIMD2<QFloat>]],
        scratchReal: MTLBuffer,
        scratchImag: MTLBuffer
    ) throws {
        let flat = kraus.flatMap { $0 }
        let krausBuffer = try makeSharedBuffer(
            values: flat,
            byteCount: flat.count * MemoryLayout<SIMD2<QFloat>>.stride
        )

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DensityMatrixEngineError.commandBufferCreationFailed
        }

        var stateCount = UInt32(density.stateCount)
        var target = UInt32(targetQubit)
        var krausCount = UInt32(kraus.count)

        encoder.setComputePipelineState(pipelines.applyKrausSingleQubit)
        encoder.setBuffer(density.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(density.imagBuffer, offset: 0, index: 1)
        encoder.setBuffer(scratchReal, offset: 0, index: 2)
        encoder.setBuffer(scratchImag, offset: 0, index: 3)
        encoder.setBytes(&stateCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&target, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&krausCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(krausBuffer, offset: 0, index: 7)
        dispatchMatrixElements(
            encoder: encoder,
            elementCount: density.elementCount,
            pipeline: pipelines.applyKrausSingleQubit
        )
        encoder.endEncoding()

        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw DensityMatrixEngineError.commandBufferCreationFailed
        }
        blit.copy(from: scratchReal, sourceOffset: 0, to: density.realBuffer, destinationOffset: 0, size: density.realBuffer.length)
        blit.copy(from: scratchImag, sourceOffset: 0, to: density.imagBuffer, destinationOffset: 0, size: density.imagBuffer.length)
        blit.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DensityMatrixEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    private func dispatchMatrixElements(
        encoder: MTLComputeCommandEncoder,
        elementCount: Int,
        pipeline: MTLComputePipelineState
    ) {
        let threads = MTLSize(width: max(elementCount, 1), height: 1, depth: 1)
        let width = min(max(1, pipeline.threadExecutionWidth), pipeline.maxTotalThreadsPerThreadgroup)
        let group = MTLSize(width: width, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: group)
    }

    private struct EncodedSingleQubitUnitary {
        let target: Int
        let controlMask: UInt32
        let matrix: [SIMD2<QFloat>] // row-major [u00, u01, u10, u11]
    }

    private func encodeSingleQubitUnitary(_ gate: Gate) throws -> EncodedSingleQubitUnitary {
        switch gate {
        case .h(let target):
            let v = QFloat(0.5).squareRoot()
            return .init(target: target, controlMask: 0, matrix: [complex(v, 0), complex(v, 0), complex(v, 0), complex(-v, 0)])
        case .x(let target):
            return .init(target: target, controlMask: 0, matrix: [complex(0, 0), complex(1, 0), complex(1, 0), complex(0, 0)])
        case .y(let target):
            return .init(target: target, controlMask: 0, matrix: [complex(0, 0), complex(0, -1), complex(0, 1), complex(0, 0)])
        case .z(let target):
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(-1, 0)])
        case .s(let target):
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 1)])
        case .t(let target):
            let v = QFloat(0.5).squareRoot()
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(v, v)])
        case .sdg(let target):
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(0, -1)])
        case .tdg(let target):
            let v = QFloat(0.5).squareRoot()
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(v, -v)])
        case .sx(let target):
            return .init(
                target: target,
                controlMask: 0,
                matrix: [
                    complex(0.5, 0.5), complex(0.5, -0.5),
                    complex(0.5, -0.5), complex(0.5, 0.5),
                ]
            )
        case .sxdg(let target):
            return .init(
                target: target,
                controlMask: 0,
                matrix: [
                    complex(0.5, -0.5), complex(0.5, 0.5),
                    complex(0.5, 0.5), complex(0.5, -0.5),
                ]
            )
        case .p(let theta, let target):
            let c = cos(theta)
            let s = sin(theta)
            return .init(target: target, controlMask: 0, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(c, s)])
        case .u(let theta, let phi, let lambda, let target):
            let c = cos(theta * 0.5)
            let s = sin(theta * 0.5)
            let eLambda = complex(cos(lambda), sin(lambda))
            let ePhi = complex(cos(phi), sin(phi))
            let ePhiLambda = complex(cos(phi + lambda), sin(phi + lambda))
            return .init(
                target: target,
                controlMask: 0,
                matrix: [
                    complex(c, 0),
                    complexMul(complex(-s, 0), eLambda),
                    complexMul(complex(s, 0), ePhi),
                    complexMul(complex(c, 0), ePhiLambda),
                ]
            )
        case .rx(let theta, let target):
            let c = cos(theta * 0.5)
            let s = sin(theta * 0.5)
            return .init(target: target, controlMask: 0, matrix: [complex(c, 0), complex(0, -s), complex(0, -s), complex(c, 0)])
        case .ry(let theta, let target):
            let c = cos(theta * 0.5)
            let s = sin(theta * 0.5)
            return .init(target: target, controlMask: 0, matrix: [complex(c, 0), complex(-s, 0), complex(s, 0), complex(c, 0)])
        case .rz(let theta, let target):
            let half = theta * 0.5
            return .init(
                target: target,
                controlMask: 0,
                matrix: [
                    complex(cos(half), -sin(half)),
                    complex(0, 0),
                    complex(0, 0),
                    complex(cos(half), sin(half)),
                ]
            )
        case .cx(let control, let target):
            return .init(target: target, controlMask: bitMask(of: control), matrix: [complex(0, 0), complex(1, 0), complex(1, 0), complex(0, 0)])
        case .cz(let control, let target):
            return .init(target: target, controlMask: bitMask(of: control), matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(-1, 0)])
        case .ccx(let control1, let control2, let target):
            return .init(target: target, controlMask: bitMask(of: control1) | bitMask(of: control2), matrix: [complex(0, 0), complex(1, 0), complex(1, 0), complex(0, 0)])
        case .mcx(let controls, let target):
            return .init(target: target, controlMask: controls.reduce(0) { $0 | bitMask(of: $1) }, matrix: [complex(0, 0), complex(1, 0), complex(1, 0), complex(0, 0)])
        case .mcz(let controls, let target):
            return .init(target: target, controlMask: controls.reduce(0) { $0 | bitMask(of: $1) }, matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(-1, 0)])
        case .crx(let theta, let control, let target):
            let c = cos(theta * 0.5)
            let s = sin(theta * 0.5)
            return .init(target: target, controlMask: bitMask(of: control), matrix: [complex(c, 0), complex(0, -s), complex(0, -s), complex(c, 0)])
        case .cry(let theta, let control, let target):
            let c = cos(theta * 0.5)
            let s = sin(theta * 0.5)
            return .init(target: target, controlMask: bitMask(of: control), matrix: [complex(c, 0), complex(-s, 0), complex(s, 0), complex(c, 0)])
        case .crz(let theta, let control, let target):
            let half = theta * 0.5
            return .init(
                target: target,
                controlMask: bitMask(of: control),
                matrix: [
                    complex(cos(half), -sin(half)),
                    complex(0, 0),
                    complex(0, 0),
                    complex(cos(half), sin(half)),
                ]
            )
        case .cp(let theta, let control, let target):
            return .init(
                target: target,
                controlMask: bitMask(of: control),
                matrix: [complex(1, 0), complex(0, 0), complex(0, 0), complex(cos(theta), sin(theta))]
            )
        case .swap, .measure, .reset:
            throw DensityMatrixEngineError.unsupportedGate(gate)
        }
    }

    private func complex(_ re: QFloat, _ im: QFloat) -> SIMD2<QFloat> {
        SIMD2<QFloat>(re, im)
    }

    private func complexMul(_ a: SIMD2<QFloat>, _ b: SIMD2<QFloat>) -> SIMD2<QFloat> {
        complex((a.x * b.x) - (a.y * b.y), (a.x * b.y) + (a.y * b.x))
    }

    private func bitMask(of qubit: Int) -> UInt32 {
        UInt32(1) << UInt32(qubit)
    }

    private func makeSharedBuffer<T>(values: [T], byteCount: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw DensityMatrixError.bufferAllocationFailed(requiredBytes: byteCount)
        }
        guard byteCount > 0 else { return buffer }
        values.withUnsafeBytes { source in
            buffer.contents().copyMemory(from: source.baseAddress!, byteCount: byteCount)
        }
        return buffer
    }

    static func loadLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let library = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return library
        }

        let bundle = Bundle.module
        let candidateURLs = [
            bundle.url(forResource: "DensityMatrixKernels", withExtension: "metalsrc", subdirectory: "Metal"),
            bundle.url(forResource: "DensityMatrixKernels", withExtension: "metalsrc"),
        ]
        guard let sourceURL = candidateURLs.compactMap({ $0 }).first else {
            throw DensityMatrixEngineError.libraryNotFound
        }
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        return try device.makeLibrary(source: source, options: nil)
    }
}
