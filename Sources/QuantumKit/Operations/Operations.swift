//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation

public enum Gate {
    
    /// Hadamard gate
    case h(target:Int)
    
    /// Pauli X, Y, Z  gates:
    case x(target:Int)
    case y(target:Int)
    case z(target:Int)
    
    /// Entangles two qubits. Controlled-NOT gate
    case cx(control:Int, target:Int)
    
    /// Parametrized rotation around the X-axis
    case rx(theta:QFloat, target:Int)
        
}
