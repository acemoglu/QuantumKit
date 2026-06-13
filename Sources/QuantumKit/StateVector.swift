//
//  File.swift
//  QuantumKit
//
//  Created by Bugra Acemoglu on 13.06.2026.
//

import Metal

public enum StateVectorError: Error {
    case invalidQubitCount(Int)
    case qubitCountExceedsLimit(max: Int, requested: Int)
    case stateCountOverflow(qubitCount: Int)
    case bufferAllocationFailed(requiredBytes: Int)
}

public struct StateVector {

    public static let maxQubitCount = 28

    public let qubitCount: Int
    public let stateCount: Int

    public let realBuffer: MTLBuffer
    public let imagBuffer: MTLBuffer

    public init(qubitCount: Int, device: MTLDevice) throws {
        guard qubitCount > 0 else {
            throw StateVectorError.invalidQubitCount(qubitCount)
        }

        guard qubitCount <= Self.maxQubitCount else {
            throw StateVectorError.qubitCountExceedsLimit(max: Self.maxQubitCount, requested: qubitCount)
        }

        let computedStateCount = 1 << qubitCount
        guard computedStateCount > 0 else {
            throw StateVectorError.stateCountOverflow(qubitCount: qubitCount)
        }

        let bufferSize = computedStateCount * MemoryLayout<QFloat>.stride

        guard let realBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let imagBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            throw StateVectorError.bufferAllocationFailed(requiredBytes: bufferSize * 2)
        }

        self.qubitCount = qubitCount
        self.stateCount = computedStateCount
        self.realBuffer = realBuffer
        self.imagBuffer = imagBuffer

        let realPointer = realBuffer.contents().assumingMemoryBound(to: QFloat.self)
        let imagPointer = imagBuffer.contents().assumingMemoryBound(to: QFloat.self)

        realPointer.update(repeating: 0.0, count: stateCount)
        imagPointer.update(repeating: 0.0, count: stateCount)

        realPointer[0] = 1.0
    }
}
