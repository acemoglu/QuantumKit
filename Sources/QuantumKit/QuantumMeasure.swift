//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation
import Metal

public struct QuantumMeasurement {
    
    public static func measure(state: StateVector, engine: QuantumEngine) throws -> [Int] {
        
        let stateCount = state.stateCount
        let bufferSize = stateCount * MemoryLayout<QFloat>.stride
        
        guard let probBuffer = state.realBuffer.device.makeBuffer(length: bufferSize, options: .storageModeShared)
        else {
            fatalError("Probability Buffer alligment failed.")
        }
        
        try engine.executeProbabilityKernel(on: state, outputBuffer: probBuffer)
                
        let probPointer = probBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let diceRoll = TRNGCollapse.generateHardwareFloat()
        
        
        var cumulative: QFloat = 0.0
        var collapsedIndex: Int = 0
        
        for i in 0..<stateCount {
            cumulative += probPointer[i]
            if diceRoll <= cumulative {
                collapsedIndex = i
                break
            }
        }
        
        return toBitArray(value: collapsedIndex, qubitCount: state.qubitCount)
    }

    private static func toBitArray(value: Int, qubitCount: Int) -> [Int] {
        var bits = [Int](repeating: 0, count: qubitCount)
        for i in 0..<qubitCount {
            bits[qubitCount - 1 - i] = (value & (1 << i)) != 0 ? 1 : 0
        }
        
        return bits
    }
    
}
