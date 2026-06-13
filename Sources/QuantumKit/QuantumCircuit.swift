//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//



public struct QuantumCircuit {
    
    public let qubitCount: Int
    public private(set) var gates: [Gate] = []
    
    
    public init(qubitCount:Int) {
        
        guard qubitCount > 0 else {
            fatalError("qubitCount must be bigger than 0")
        }
        
        self.qubitCount = qubitCount
        
    }
    
    
    public mutating func apply(_ gate: Gate) {
        gates.append(gate)
    }
    
    // MARK: - Method Chaining API
    
    @discardableResult
    public mutating func h(_ target: Int) -> QuantumCircuit {
        self.apply(.h(target: target))
        return self
    }
    
    @discardableResult
    public mutating func cx(_ control:Int, _ target:Int) -> QuantumCircuit {
        self.apply(.cx(control:control,target:target))
        return self
    }
    
    @discardableResult
    public mutating func rx(theta:QFloat, _ target:Int) -> QuantumCircuit {
        self.apply(.rx(theta:theta,target:target))
        return self
    }
    
    @discardableResult
    public mutating func x(_ target:Int) -> QuantumCircuit {
        self.apply(.x(target:target))
        return self
    }
    
    @discardableResult
    public mutating func y(_ target:Int) -> QuantumCircuit {
        self.apply(.y(target:target))
        return self
    }
    
    @discardableResult
    public mutating func z(_ target:Int) -> QuantumCircuit {
        self.apply(.z(target:target))
        return self
    }
    
}

