//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Metal

public enum QuantumEngineError: Error {
    
    case deviceNotFound
    case libraryNotFound
    case functionNotFound(String)
    case pipelineStateCreationFailed
    case commandBufferCreationFailed
    case commandQueueCreationFailed
}

public class QuantumEngine {
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    private var hadamardPipelineState: MTLComputePipelineState
    
    public init() throws {
        
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            throw QuantumEngineError.deviceNotFound
        }
        
        self.device = defaultDevice
        
        guard let queue = device.makeCommandQueue() else {
            throw QuantumEngineError.commandQueueCreationFailed
        }
        
        self.commandQueue = queue
        
        guard let library = device.makeDefaultLibrary() else {
            throw QuantumEngineError.libraryNotFound
        }
                
        guard let hFunction = library.makeFunction(name: "hadamard_gate") else {
            throw QuantumEngineError.functionNotFound("hadamard_gate")
        }
        
        self.hadamardPipelineState = try device.makeComputePipelineState(function: hFunction)
    }
    
    
    private func loadPipelines() throws {

        guard let library = device.makeDefaultLibrary() else {
            throw QuantumEngineError.libraryNotFound
        }
        
        guard let hFunction = library.makeFunction(name: "hadamard_gate") else {
            throw QuantumEngineError.functionNotFound("hadamard_gate")
        }
        
        self.hadamardPipelineState = try device.makeComputePipelineState(function: hFunction)
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
                computeEncoder.setComputePipelineState(hadamardPipelineState)
                
                // Show the memory address
                computeEncoder.setBuffer(state.realBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
                
                // Send the targeted qubit info
                var targetQubit = target
                computeEncoder.setBytes(&targetQubit, length: MemoryLayout<Int>.stride, index: 2)
                
                // Thread
                let threadsPerGrid = MTLSize(width: state.stateCount / 2, height: 1, depth: 1)
                let w = hadamardPipelineState.threadExecutionWidth
                let threadsPerThreadgroup = MTLSize(width: min(w, state.stateCount / 2), height: 1, depth: 1)
                
                computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                
            default:
                print("⚠️ Warning: This gate has not been connected yet")
            }
        }
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    
}


