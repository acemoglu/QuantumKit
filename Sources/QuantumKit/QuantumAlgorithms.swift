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

    public mutating func applyBellState(control: Int = 0, target: Int = 1) throws {
        try h(control)
        try cx(control, target)
    }
}
