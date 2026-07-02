import Foundation

func makePrimitiveRNG(seed: UInt64?) -> QuantumRNG {
    if let seed {
        return .seeded(seed)
    }
    return .hardware
}
