//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation
import Metal

public enum QuantumEngineError: Error {

    case deviceNotFound
    case libraryNotFound
    case functionNotFound(String)
    case pipelineStateCreationFailed
    case commandBufferCreationFailed
    case commandQueueCreationFailed
    case qubitCountMismatch(circuit: Int, state: Int)

}

public struct Pipelines {

    let hadamard: MTLComputePipelineState
    let pauliX: MTLComputePipelineState
    let pauliY: MTLComputePipelineState
    let pauliZ: MTLComputePipelineState
    let cnot: MTLComputePipelineState
    let ccx: MTLComputePipelineState
    let rotZ: MTLComputePipelineState

    let probabilities: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let hFunc = library.makeFunction(name: "hadamard_gate") else { throw QuantumEngineError.functionNotFound("hadamard_gate") }
        self.hadamard = try device.makeComputePipelineState(function: hFunc)

        guard let xFunc = library.makeFunction(name: "pauli_x_gate") else { throw QuantumEngineError.functionNotFound("pauli_x_gate") }
        self.pauliX = try device.makeComputePipelineState(function: xFunc)

        guard let yFunc = library.makeFunction(name: "pauli_y_gate") else { throw QuantumEngineError.functionNotFound("pauli_y_gate") }
        self.pauliY = try device.makeComputePipelineState(function: yFunc)

        guard let zFunc = library.makeFunction(name: "pauli_z_gate") else { throw QuantumEngineError.functionNotFound("pauli_z_gate") }
        self.pauliZ = try device.makeComputePipelineState(function: zFunc)

        guard let cxFunc = library.makeFunction(name: "cnot_gate") else { throw QuantumEngineError.functionNotFound("cnot_gate") }
        self.cnot = try device.makeComputePipelineState(function: cxFunc)

        guard let ccxFunc = library.makeFunction(name: "ccx_gate") else { throw QuantumEngineError.functionNotFound("ccx_gate") }
        self.ccx = try device.makeComputePipelineState(function: ccxFunc)

        guard let rzFunc = library.makeFunction(name: "rz_gate") else { throw QuantumEngineError.functionNotFound("rz_gate") }
        self.rotZ = try device.makeComputePipelineState(function: rzFunc)

        guard let probFunc = library.makeFunction(name: "compute_probabilities") else { throw QuantumEngineError.functionNotFound("compute_probabilities") }
        self.probabilities = try device.makeComputePipelineState(function: probFunc)

    }
}

public class QuantumEngine {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelines: Pipelines

    private static func loadMetalLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let library = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return library
        }

        if let library = device.makeDefaultLibrary() {
            return library
        }

        let bundle = Bundle.module
        let candidateURLs = [
            bundle.url(forResource: "Gates", withExtension: "metal", subdirectory: "Metal"),
            bundle.url(forResource: "Gates", withExtension: "metal"),
        ]

        guard let metalURL = candidateURLs.compactMap({ $0 }).first else {
            throw QuantumEngineError.libraryNotFound
        }

        let source = try String(contentsOf: metalURL, encoding: .utf8)
        return try device.makeLibrary(source: source, options: nil)
    }

    public init() throws {
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else { throw QuantumEngineError.deviceNotFound }
        self.device = defaultDevice

        guard let queue = device.makeCommandQueue() else { throw QuantumEngineError.commandQueueCreationFailed }
        self.commandQueue = queue

        let library = try Self.loadMetalLibrary(device: device)
        self.pipelines = try Pipelines(device: device, library: library)
    }

    private func dispatchPairwiseGate(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        state: StateVector,
        configure: (MTLComputeCommandEncoder) -> Void
    ) {
        let pairCount = state.stateCount / 2

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        configure(encoder)

        let threadsPerGrid = MTLSize(width: pairCount, height: 1, depth: 1)
        let executionWidth = pipeline.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: min(executionWidth, pairCount), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    private func dispatchFullStateKernel(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        state: StateVector,
        configure: (MTLComputeCommandEncoder) -> Void
    ) {
        let threadCount = max(state.stateCount, 1)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        configure(encoder)

        let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
        let executionWidth = pipeline.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: min(executionWidth, threadCount), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    public func execute(_ circuit: QuantumCircuit, on state: StateVector) throws {
        guard circuit.qubitCount == state.qubitCount else {
            throw QuantumEngineError.qubitCountMismatch(circuit: circuit.qubitCount, state: state.qubitCount)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        for gate in circuit.gates {
            switch gate {
            case .h(let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.hadamard, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                }

            case .x(let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.pauliX, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                }

            case .y(let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.pauliY, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                }

            case .z(let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.pauliZ, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                }

            case .cx(let control, let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.cnot, state: state) { encoder in
                    var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                    encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                }

            case .ccx(let control1, let control2, let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.ccx, state: state) { encoder in
                    var qubits = SIMD3<UInt32>(x: UInt32(control1), y: UInt32(control2), z: UInt32(target))
                    encoder.setBytes(&qubits, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
                }

            case .rz(let theta, let target):
                dispatchPairwiseGate(encoder: computeEncoder, pipeline: pipelines.rotZ, state: state) { encoder in
                    var targetQubit = UInt32(target)
                    encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                    var thetaValue = Float(theta)
                    encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
                }

            case .rx:
                print("⚠️ Warning: Rx gate has not been connected to Metal yet")
            }
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    public func executeProbabilityKernel(on state: StateVector, outputBuffer: MTLBuffer) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        dispatchFullStateKernel(encoder: computeEncoder, pipeline: pipelines.probabilities, state: state) { encoder in
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
