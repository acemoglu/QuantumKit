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
        let diceRoll = TRNGCollapse.generateHardwareFloat()
        let collapsedIndex = try engine.executeMeasurementCollapse(on: state, diceRoll: diceRoll)
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
