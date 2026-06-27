//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation
import Security

public struct TRNGCollapse {
    
    /// A hardware-entropy `Float32` uniformly distributed in the half-open range `[0, 1)`.
    public static func generateHardwareFloat() -> Float32 {

        var randomBytes = [UInt8](repeating: 0, count: 4)
        let status = SecRandomCopyBytes(kSecRandomDefault, randomBytes.count, &randomBytes)

        if status == errSecSuccess {
            let randomUInt32 = randomBytes.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
            return QuantumRNG.unitFloat(from: randomUInt32)
        } else {
            var fallbackRNG = SystemRandomNumberGenerator()
            let randomUInt32 = fallbackRNG.next() as UInt32
            return QuantumRNG.unitFloat(from: randomUInt32)
        }
    }

    /// A hardware-entropy `Double` uniformly distributed in the half-open range `[0, 1)`.
    public static func generateHardwareDouble() -> Double {

        var randomBytes = [UInt8](repeating: 0, count: 8)
        let status = SecRandomCopyBytes(kSecRandomDefault, randomBytes.count, &randomBytes)

        if status == errSecSuccess {
            let randomUInt64 = randomBytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            return QuantumRNG.unitDouble(from: randomUInt64)
        } else {
            var fallbackRNG = SystemRandomNumberGenerator()
            let randomUInt64 = fallbackRNG.next() as UInt64
            return QuantumRNG.unitDouble(from: randomUInt64)
        }
    }
}
