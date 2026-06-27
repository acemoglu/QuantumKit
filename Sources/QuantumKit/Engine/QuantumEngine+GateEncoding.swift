import Foundation
import Metal

extension QuantumEngine {

    func dispatchPairwiseGate(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        state: StateVector,
        configure: (MTLComputeCommandEncoder) -> Void
    ) {
        let pairCount = state.stateCount / 2

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(state.realBuffer, offset: 0, index: 0)
        encoder.setBuffer(state.imagBuffer, offset: 0, index: 1)
        configure(encoder)

        let threadsPerGrid = MTLSize(width: pairCount, height: 1, depth: 1)
        // Fill the threadgroup up to the pipeline's hardware maximum (≈1024) rather than a single
        // SIMD width (≈32); these gate kernels use no threadgroup memory and index purely off
        // thread_position_in_grid, so a wider group only raises occupancy.
        let threadgroupWidth = min(pipeline.maxTotalThreadsPerThreadgroup, max(pairCount, 1))
        let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
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
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        encodeUnitaryGate(gate, encoder: computeEncoder, state: state)

        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
        }
    }

    func flushUnitaryGates(_ gates: [Gate], on state: StateVector) throws {
        guard !gates.isEmpty else { return }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QuantumEngineError.commandBufferCreationFailed
        }

        for gate in gates {
            encodeUnitaryGate(gate, encoder: computeEncoder, state: state)
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
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .cz(let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cz, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .swap(let q1, let q2):
            dispatchFullStateKernel(encoder: encoder, pipeline: pipelines.swapGate, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(q1), y: UInt32(q2))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            }

        case .ccx(let control1, let control2, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.ccx, state: state) { encoder in
                var qubits = SIMD3<UInt32>(x: UInt32(control1), y: UInt32(control2), z: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
            }

        case .crx(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cRotX, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .cry(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cRotY, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .crz(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cRotZ, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                var thetaValue = Float(theta)
                encoder.setBytes(&thetaValue, length: MemoryLayout<Float>.stride, index: 3)
            }

        case .cp(let theta, let control, let target):
            dispatchPairwiseGate(encoder: encoder, pipeline: pipelines.cPhase, state: state) { encoder in
                var qubits = SIMD2<UInt32>(x: UInt32(control), y: UInt32(target))
                encoder.setBytes(&qubits, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
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
