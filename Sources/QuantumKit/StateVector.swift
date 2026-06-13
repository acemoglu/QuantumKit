//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Metal

public struct StateVector {
    
    public let qubitCount: Int
    public let stateCount: Int
    
    public let realBuffer: MTLBuffer
    public let imagBuffer: MTLBuffer
    
    public init(qubitCount: Int, device: MTLDevice) {
        
        guard qubitCount > 0 else {
            fatalError("qubitCount must be greater than 0")
        }
        
        self.qubitCount = qubitCount
        self.stateCount = 1 << qubitCount // Bit shift
        
        let bufferSize = stateCount * MemoryLayout<QFloat>.stride
        
        guard let rBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let iBuf = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            fatalError("Metal buffer unfailed! RAM capacity could pass the limit!")
        }
        
        self.realBuffer = rBuf
        self.imagBuffer = iBuf
        
        
        let realPointer = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)
        
        
        realPointer.assign(repeating: 0.0, count: stateCount)
        imagPointer.assign(repeating: 0.0, count: stateCount)
        
        realPointer[0] = 1.0
    }
}
