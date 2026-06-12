//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//


public struct StateVector {
    
    public var qubitCount: Int
    
    public var realParts: [QFloat]
    public var imagParts: [QFloat]
    
    public var stateCount:Int {
        return 1 << qubitCount
    }
    
    
    public init(qubitCount: Int) {
        
        guard qubitCount > 0 else {
            fatalError("qubiqCount must bigger than 0")
        }
        
        self.qubitCount = qubitCount
        let count = 1 << qubitCount
        
        self.realParts = Array(repeating: 0.0, count: count)
        self.imagParts = Array(repeating: 0.0, count: count)
        
        //system initial possibility
        self.realParts[0] = 1.0
    }
    
}
