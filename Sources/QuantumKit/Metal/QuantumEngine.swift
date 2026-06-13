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
    
}

public struct Pipelines {

    let hadamard: MTLComputePipelineState
    let pauliX: MTLComputePipelineState
    let pauliY: MTLComputePipelineState
    let pauliZ: MTLComputePipelineState
    let cnot: MTLComputePipelineState
    
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
        
        guard let probFunc = library.makeFunction(name: "compute_probabilities") else { throw QuantumEngineError.functionNotFound("compute_probabilities") }
        self.probabilities = try device.makeComputePipelineState(function: probFunc)

    }
}

public class QuantumEngine {
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelines: Pipelines
    
    private static func loadMetalLibrary(device: MTLDevice) throws -> MTLLibrary {
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
    
    
    
    // MARK: - Execute Layer
    
    public func execute(_ circuit: QuantumCircuit, on state: StateVector) throws {

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }
        
        for gate in circuit.gates {
            switch gate {
            case .h(let target):
                // Set the Engine on Hadamard mode
                computeEncoder.setComputePipelineState(pipelines.hadamard)
                
                // Show the memory address
                computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
                
                // Send the targeted qubit info
                var targetQubit = target
                computeEncoder.setBytes(&targetQubit, length: MemoryLayout<Int>.stride, index: 2)
                
                // Thread
                let threadsPerGrid = MTLSize(width: state.stateCount / 2, height: 1, depth: 1)
                let w = pipelines.hadamard.threadExecutionWidth
                let threadsPerThreadgroup = MTLSize(width: min(w, state.stateCount / 2), height: 1, depth: 1)
                
                computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                
                
            case .x(let target):
                computeEncoder.setComputePipelineState(pipelines.pauliX)
                computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
                
                var targetQubit = UInt32(target)
                computeEncoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                
                let threadsPerGrid = MTLSize(width: state.stateCount / 2, height: 1, depth: 1)
                let w = pipelines.pauliX.threadExecutionWidth
                let threadsPerThreadgroup = MTLSize(width: min(w, state.stateCount / 2), height: 1, depth: 1)
                computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                
            case .y(let target):
                computeEncoder.setComputePipelineState(pipelines.pauliY)
                computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
                
                var targetQubit = UInt32(target)
                computeEncoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                
                let threadsPerGrid = MTLSize(width: state.stateCount / 2, height: 1, depth: 1)
                let w = pipelines.pauliY.threadExecutionWidth
                let threadsPerThreadgroup = MTLSize(width: min(w, state.stateCount / 2), height: 1, depth: 1)
                computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                
                
            case .z(let target):
                computeEncoder.setComputePipelineState(pipelines.pauliZ)
                computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
                
                var targetQubit = UInt32(target)
                computeEncoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                
                let threadsPerGrid = MTLSize(width: state.stateCount / 2, height: 1, depth: 1)
                let w = pipelines.pauliZ.threadExecutionWidth
                let threadsPerThreadgroup = MTLSize(width: min(w, state.stateCount / 2), height: 1, depth: 1)
                computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                
                
            case .cx(let control, let target):
                computeEncoder.setComputePipelineState(pipelines.cnot)
                computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
                
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                computeEncoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                
                let threadsPerGrid = MTLSize(width: state.stateCount / 2, height: 1, depth: 1)
                let w = pipelines.cnot.threadExecutionWidth
                let threadsPerThreadgroup = MTLSize(width: min(w, state.stateCount / 2), height: 1, depth: 1)
                computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                
            case .rx(let theta, let target):
                print("⚠️ Warning: Rx gate has not been connected to Metal yet")
                
            default:
                print("⚠️ Warning: This gate has not been connected to Metal yet")
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
        
        computeEncoder.setComputePipelineState(pipelines.probabilities)
        computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
        
        let threadsPerGrid = MTLSize(width: state.stateCount, height: 1, depth: 1)
        let w = pipelines.probabilities.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: min(w, state.stateCount), height: 1, depth: 1)
        
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted() 
    }
    
}


