//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation

extension QuantumCircuit {

    public mutating func applyCPHASE(theta: Double, control: Int, target: Int) throws {
        let halfTheta = QFloat(theta / 2.0)
        try cx(control, target)
        try rz(theta: -halfTheta, target)
        try cx(control, target)
        try rz(theta: halfTheta, control)
        try rz(theta: halfTheta, target)
    }

    public mutating func applyQFT() throws {
        let n = qubitCount
        for j in 0..<n {
            try h(j)
            for k in (j + 1)..<n {
                let theta = Double.pi / Double(1 << (k - j))
                try applyCPHASE(theta: theta, control: k, target: j)
            }
        }
    }

    public mutating func applySwap(q1: Int, q2: Int) throws {
        try cx(q1, q2)
        try cx(q2, q1)
        try cx(q1, q2)
    }

    public mutating func applyBellState(control: Int = 0, target: Int = 1) throws {
        try h(control)
        try cx(control, target)
    }

    public mutating func applyModularExponentiation(
        a: Int,
        modulus N: Int,
        controlRegister: ClosedRange<Int>,
        targetRegister: ClosedRange<Int>
    ) throws {
        guard N > 1 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Modulus must be greater than 1")
        }

        guard a >= 0 else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Base must be non-negative")
        }

        guard !controlRegister.isEmpty, !targetRegister.isEmpty else {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Control and target registers must not be empty")
        }

        for index in controlRegister {
            try validateRegisterIndex(index)
        }

        for index in targetRegister {
            try validateRegisterIndex(index)
        }

        if controlRegister.overlaps(targetRegister) {
            throw QuantumCircuitError.invalidAlgorithmParameter(reason: "Control and target registers must not overlap")
        }

        let runningBase = a % N
        var currentMultiplier = runningBase

        for controlQubit in controlRegister {
            // TODO: Implement controlled modular multiplication:
            //       |c>|y> -> |c>|(y * currentMultiplier) mod N> when c = |1>
            
            try applyControlledModularMultiply(
                multiplier: currentMultiplier,
                modulus: N,
                control: controlQubit,
                targetRegister: targetRegister
            )

            currentMultiplier = (currentMultiplier * currentMultiplier) % N
        }
    }

    private mutating func applyControlledModularMultiply(
        multiplier: Int,
        modulus: Int,
        control: Int,
        targetRegister: ClosedRange<Int>
    ) throws {
        // TODO: Decompose into quantum adders/multipliers using ccx and swap primitives.
        _ = multiplier
        _ = modulus
        _ = control
        _ = targetRegister
    }

    private func validateRegisterIndex(_ index: Int) throws {
        guard index >= 0, index < qubitCount else {
            throw QuantumCircuitError.qubitIndexOutOfBounds(index: index, qubitCount: qubitCount)
        }
    }
}
