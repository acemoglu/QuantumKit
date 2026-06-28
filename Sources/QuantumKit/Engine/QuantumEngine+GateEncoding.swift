import Foundation
import Metal

extension QuantumEngine {
    static let pairCountBufferIndex = 15
    struct RenormalizationScratch {
        let hiA: MTLBuffer
        let loA: MTLBuffer
        let hiB: MTLBuffer
        let loB: MTLBuffer
    }

    /// Uses a SIMD-aligned width so each threadgroup maps cleanly onto SIMDs while still pushing
    /// occupancy near the hardware maximum.
    private func optimalThreadgroupWidth(
        pipeline: MTLComputePipelineState,
        workItems: Int
    ) -> Int {
        let simdWidth = max(1, pipeline.threadExecutionWidth)
        let maxWidth = pipeline.maxTotalThreadsPerThreadgroup
        let clamped = min(maxWidth, max(workItems, simdWidth))
        let aligned = (clamped / simdWidth) * simdWidth
        return max(simdWidth, min(workItems, aligned == 0 ? simdWidth : aligned))
    }

    func dispatchPairwiseGate(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        state: StateVector,
        configure: (MTLComputeCommandEncoder) -> Void
    ) {
        let pairCount = state.stateCount / 2
        guard pairCount > 0 else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        configure(encoder)
        var pairCountValue = UInt32(pairCount)
        encoder.setBytes(&pairCountValue, length: MemoryLayout<UInt32>.stride, index: Self.pairCountBufferIndex)

        let threadsPerGrid = MTLSize(width: pairCount, height: 1, depth: 1)
        let threadgroupWidth = optimalThreadgroupWidth(pipeline: pipeline, workItems: pairCount)
        // One complex pair (r0, i0, r1, i1) per lane in threadgroup memory.
        encoder.setThreadgroupMemoryLength(
            MemoryLayout<SIMD4<Float>>.stride * threadgroupWidth,
            index: 0
        )
        let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    func shouldRenormalize(afterAppliedGateCount gateCount: Int) -> Bool {
        renormalizationInterval > 0 && gateCount % renormalizationInterval == 0
    }

    func renormBlockCount(for elementCount: Int) -> Int {
        max((elementCount + Self.renormalizationBlockSize - 1) / Self.renormalizationBlockSize, 1)
    }

    func makeRenormalizationScratch(stateCount: Int) throws -> RenormalizationScratch {
        let maxPartialCount = renormBlockCount(for: stateCount)
        let bytes = maxPartialCount * MemoryLayout<QFloat>.stride
        return RenormalizationScratch(
            hiA: try bufferPool.acquire(length: bytes),
            loA: try bufferPool.acquire(length: bytes),
            hiB: try bufferPool.acquire(length: bytes),
            loB: try bufferPool.acquire(length: bytes)
        )
    }

    func releaseRenormalizationScratch(_ scratch: RenormalizationScratch) {
        bufferPool.release(scratch.hiA)
        bufferPool.release(scratch.loA)
        bufferPool.release(scratch.hiB)
        bufferPool.release(scratch.loB)
    }

    func encodeStateRenormalization(
        encoder: MTLComputeCommandEncoder,
        state: StateVector,
        scratch: RenormalizationScratch
    ) throws {
        let blockSize = Self.renormalizationBlockSize
        let threadgroupSize = MTLSize(width: blockSize, height: 1, depth: 1)

        var stateElementCount = UInt32(state.stateCount)
        var partialCount = renormBlockCount(for: state.stateCount)

        encoder.setComputePipelineState(pipelines.renormStateNormPartial)
        encoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        encoder.setBytes(&stateElementCount, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBuffer(scratch.hiA, offset: 0, index: 3)
        encoder.setBuffer(scratch.loA, offset: 0, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: partialCount, height: 1, depth: 1),
            threadsPerThreadgroup: threadgroupSize
        )

        var readHi = scratch.hiA
        var readLo = scratch.loA
        var writeHi = scratch.hiB
        var writeLo = scratch.loB

        while partialCount > 1 {
            var partialElementCount = UInt32(partialCount)
            let nextCount = renormBlockCount(for: partialCount)

            encoder.setComputePipelineState(pipelines.renormCompensatedPartial)
            encoder.setBuffer(readHi, offset: 0, index: 0)
            encoder.setBuffer(readLo, offset: 0, index: 1)
            encoder.setBytes(&partialElementCount, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBuffer(writeHi, offset: 0, index: 3)
            encoder.setBuffer(writeLo, offset: 0, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: nextCount, height: 1, depth: 1),
                threadsPerThreadgroup: threadgroupSize
            )

            swap(&readHi, &writeHi)
            swap(&readLo, &writeLo)
            partialCount = nextCount
        }

        dispatchFullStateKernel(encoder: encoder, pipeline: pipelines.renormScale, state: state) { encoder in
            var elementCount = UInt32(state.stateCount)
            encoder.setBuffer(readHi, offset: 0, index: 2)
            encoder.setBuffer(readLo, offset: 0, index: 3)
            encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.stride, index: 4)
        }
    }

    func dispatchFullStateKernel(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        state: StateVector,
        configure: (MTLComputeCommandEncoder) -> Void
    ) {
        let threadCount = max(state.stateCount, 1)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        configure(encoder)

        let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
        // See `dispatchPairwiseGate`: these kernels carry no threadgroup state, so we size the group
        // at the pipeline maximum for occupancy instead of a single execution width.
        let threadgroupWidth = min(pipeline.maxTotalThreadsPerThreadgroup, threadCount)
        let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    func executeUnitaryGate(_ gate: Gate, on state: StateVector) throws {
        var gateCounter = 0
        try executeUnitaryGate(gate, on: state, gateCounter: &gateCounter)
    }

    func executeUnitaryGate(_ gate: Gate, on state: StateVector, gateCounter: inout Int) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }
        var scratch: RenormalizationScratch?
        defer {
            if let scratch {
                releaseRenormalizationScratch(scratch)
            }
        }

        encodeUnitaryGate(gate, encoder: computeEncoder, state: state)
        gateCounter += 1
        if shouldRenormalize(afterAppliedGateCount: gateCounter) {
            let localScratch = try makeRenormalizationScratch(stateCount: state.stateCount)
            scratch = localScratch
            try encodeStateRenormalization(encoder: computeEncoder, state: state, scratch: localScratch)
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    func flushUnitaryGates(_ gates: [Gate], on state: StateVector) throws {
        var gateCounter = 0
        try flushUnitaryGates(gates, on: state, gateCounter: &gateCounter)
    }

    func flushUnitaryGates(_ gates: [Gate], on state: StateVector, gateCounter: inout Int) throws {
        guard !gates.isEmpty else { return }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }
        var scratch: RenormalizationScratch?
        defer {
            if let scratch {
                releaseRenormalizationScratch(scratch)
            }
        }

        for gate in gates {
            encodeUnitaryGate(gate, encoder: computeEncoder, state: state)
            gateCounter += 1
            if shouldRenormalize(afterAppliedGateCount: gateCounter) {
                if scratch == nil {
                    scratch = try makeRenormalizationScratch(stateCount: state.stateCount)
                }
                if let scratch {
                    try encodeStateRenormalization(encoder: computeEncoder, state: state, scratch: scratch)
                }
            }
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    /// Packs a list of control qubit indices into a bitmask (bit `q` set ⟺ qubit `q` is a control).
    static func controlMask(_ controls: [Int]) -> UInt32 {
        var mask: UInt32 = 0
        for control in controls {
            mask |= UInt32(1) << UInt32(control)
        }
        return mask
    }

    func encodeUnitaryGate(
        _ gate: Gate,
        encoder: MTLComputeCommandEncoder,
        state: StateVector
    ) {
        // Reduce unbounded rotation/phase angles into [-π, π] (in Double) before they are cast to
        // Float32 for the GPU, preventing Float32 trig loss of significance on huge inputs.
        let gate = gate.angleWrapped
        switch gate {
        case .h(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.hadamard, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .x(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.pauliX, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .y(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.pauliY, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .z(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.pauliZ, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .s(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.phaseS, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .t(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.phaseT, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .sdg(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.phaseSDagger, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .tdg(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.phaseTDagger, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .sx(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.sqrtX, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .sxdg(let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.sqrtXDagger, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .p(let theta, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.phase, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .u(let theta, let phi, let lambda, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.universal, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                var angles = SIMD3<Float>(x: Float(theta), y: Float(phi), z: Float(lambda))
                encoder.setBytes(&angles, length: MemoryLayout<SIMD3<Float>>.stride, index: 3)
            }

        case .cx(let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cnot, state: state) { encoder in
                var packed = SIMD2<UInt32>(x: UInt32(1) << UInt32(control), y: UInt32(target))
                encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .cz(let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cz, state: state) { encoder in
                var packed = SIMD2<UInt32>(x: UInt32(1) << UInt32(control), y: UInt32(target))
                encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .swap(let q1, let q2):
            dispatchFullStateKernel(encoder: encoder, pipeline: pipelines.swapGate, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(q1), y: UInt32(q2))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .ccx(let control1, let control2, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.ccx, state: state) { encoder in
                let mask = (UInt32(1) << UInt32(control1)) | (UInt32(1) << UInt32(control2))
                var packed = SIMD2<UInt32>(x: mask, y: UInt32(target))
                encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .crx(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cRotX, state: state) { encoder in
                var packed = SIMD2<UInt32>(x: UInt32(1) << UInt32(control), y: UInt32(target))
                encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .cry(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cRotY, state: state) { encoder in
                var packed = SIMD2<UInt32>(x: UInt32(1) << UInt32(control), y: UInt32(target))
                encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .crz(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cRotZ, state: state) { encoder in
                var packed = SIMD2<UInt32>(x: UInt32(1) << UInt32(control), y: UInt32(target))
                encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .cp(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cPhase, state: state) { encoder in
                var packed = SIMD2<UInt32>(x: UInt32(1) << UInt32(control), y: UInt32(target))
                encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .mcx(let controls, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.mcx, state: state) { encoder in
                var packed = SIMD2<UInt32>(x: Self.controlMask(controls), y: UInt32(target))
                encoder.setBytes(&packed, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .mcz(let controls, let target):
            dispatchFullStateKernel(encoder: encoder, pipeline: pipelines.mcz, state: state) { encoder in
                var fullMask = Self.controlMask(controls) | (UInt32(1) << UInt32(target))
                encoder.setBytes(&fullMask, length: MemoryLayout<UInt32>.stride, index: 2)
            }

        case .rz(let theta, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.rotZ, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .rx(let theta, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.rotX, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .ry(let theta, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.rotY, state: state) { encoder in
                var targetQubit = UInt32(target)
                encoder.setBytes(&targetQubit, length: MemoryLayout<UInt32>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .measure, .reset:
            break
        }
    }
}
