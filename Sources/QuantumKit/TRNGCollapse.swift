//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Foundation
import Security

public struct TRNGCollapse {
    
    public static func generateHardwareFloat() -> Float32 {
        
        var randomBytes = [UInt8](repeating: 0, count: 4)
        let status = SecRandomCopyBytes(kSecRandomDefault, randomBytes.count, &randomBytes)
        
        if status == errSecSuccess {
            let randomUInt32 = randomBytes.withUnsafeBytes { $0.load(as: UInt32.self)}
            return Float32(randomUInt32) / Float32(UInt32.max)
            
        } else {
            print("⚠️ TRNG Warning: Hardware entropy is busy, temporarily switching to classic generator.")
            var fallbackRNG = SystemRandomNumberGenerator()
            let randomUInt32 = fallbackRNG.next(upperBound: UInt32.max)
            return Float32(randomUInt32) / Float32(UInt32.max)
        }
    }
}
