//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import XCTest
import Metal

final class QuantumKitTests: XCTestCase {
    
    func testBellStateEntanglement() throws {
        //  Boot up the Engine and Memory
        let engine = try QuantumEngine()
        
        // Locate the device and create a 2-qubit (4 possible states) universe
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("Apple Silicon GPU not found!")
            return
        }
        let state = StateVector(qubitCount: 2, device: device)
        
        // Build the Circuit (Abstract syntax tree)
        var circuit = QuantumCircuit(qubitCount: 2)
        
        // Bell State (Entanglement) formulation: Hadamard first, then CNOT
        circuit.h(0)
        circuit.cx(0, 1)
        
        // Awaken the GPU and compute
        try engine.execute(circuit, on: state)
        
        // Collapse the system using Hardware TRNG!
        let result = try QuantumMeasurement.measure(state: state, engine: engine)
        
        
        print("🚀 QUANTUM COLLAPSE RESULT: \(result)")
        
        // The Core Test (Mathematical Proof)
        // Since the two qubits are entangled, the result MUST be EITHER [0, 0] OR [1, 1].
        // It can NEVER collapse into [0, 1] or [1, 0]!
        
        let isZeroZero = (result[0] == 0 && result[1] == 0)
        let isOneOne = (result[0] == 1 && result[1] == 1)
        
        XCTAssertTrue(isZeroZero || isOneOne, "Entanglement broken! Collapsed into an impossible state: \(result)")
    }
}
