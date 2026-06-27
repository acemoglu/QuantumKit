import Foundation
import Metal

extension QuantumEngine {

    func makeSharedBuffer(length: Int) throws -> MTLBuffer {
        guard length > 0,
              let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw QuantumEngineError.bufferAllocationFailed(requiredBytes: max(length, 0))
        }
        return buffer
    }

    /// Allocates the per-level block-sum scratch buffers (hi and lo halves) for a compensated scan
    /// over `stateCount` elements. When `pooled` is true the buffers come from ``bufferPool`` and the
    /// caller must hand them back with ``releasePrefixSumAuxBuffers(_:)``.
    func makePrefixSumAuxBuffers(
        stateCount: Int,
        pooled: Bool = false
    ) throws -> (hi: [MTLBuffer], lo: [MTLBuffer]) {
        var hi: [MTLBuffer] = []
        var lo: [MTLBuffer] = []

        var currentCount = stateCount
        while currentCount > 1 {
            let blockCount = max((currentCount + Self.scanBlockSize - 1) / Self.scanBlockSize, 1)
            let byteCount = blockCount * MemoryLayout<QFloat>.stride
            hi.append(pooled ? try bufferPool.acquire(length: byteCount) : try makeSharedBuffer(length: byteCount))
            lo.append(pooled ? try bufferPool.acquire(length: byteCount) : try makeSharedBuffer(length: byteCount))
            currentCount = blockCount
        }

        return (hi, lo)
    }

    /// Returns the buffers from a pooled ``makePrefixSumAuxBuffers(stateCount:pooled:)`` call.
    func releasePrefixSumAuxBuffers(_ aux: (hi: [MTLBuffer], lo: [MTLBuffer])) {
        for buffer in aux.hi { bufferPool.release(buffer) }
        for buffer in aux.lo { bufferPool.release(buffer) }
    }

    func dispatchPrefixSumPhase(
        encoder: MTLComputeCommandEncoder,
        dataHi: MTLBuffer,
        dataLo: MTLBuffer,
        blockHi: MTLBuffer,
        blockLo: MTLBuffer,
        elementCount: Int,
        phase: UInt32,
        readInputLo: UInt32
    ) {
        let blockSize = Self.scanBlockSize
        let threadCount = max(elementCount, 1)
        var countValue = UInt32(elementCount)
        var phaseValue = phase
        var readInputLoValue = readInputLo

        encoder.setComputePipelineState(pipelines.prefixSum)
        encoder.setBuffer(dataHi, offset: 0, index: 0)
        encoder.setBuffer(dataLo, offset: 0, index: 1)
        encoder.setBuffer(blockHi, offset: 0, index: 2)
        encoder.setBuffer(blockLo, offset: 0, index: 3)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&phaseValue, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&readInputLoValue, length: MemoryLayout<UInt32>.stride, index: 6)

        let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: min(blockSize, threadCount), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    /// Compensated (double-single) inclusive prefix sum over `(hiBuffer, loBuffer)`. The leaf level
    /// (`level == 0`) feeds raw float probabilities (no `lo` term yet); recursive block-sum passes
    /// carry the compensation through.
    func encodeInclusivePrefixSum(
        encoder: MTLComputeCommandEncoder,
        hiBuffer: MTLBuffer,
        loBuffer: MTLBuffer,
        elementCount: Int,
        auxiliaryHi: [MTLBuffer],
        auxiliaryLo: [MTLBuffer],
        level: Int = 0
    ) throws {
        guard elementCount > 1 else { return }
        guard level < auxiliaryHi.count else {
            throw QuantumEngineError.prefixSumBufferLevelMissing(level: level)
        }

        let blockSize = Self.scanBlockSize
        let numBlocks = (elementCount + blockSize - 1) / blockSize
        let blockHi = auxiliaryHi[level]
        let blockLo = auxiliaryLo[level]
        let readInputLo: UInt32 = level == 0 ? 0 : 1

        dispatchPrefixSumPhase(
            encoder: encoder,
            dataHi: hiBuffer,
            dataLo: loBuffer,
            blockHi: blockHi,
            blockLo: blockLo,
            elementCount: elementCount,
            phase: 0,
            readInputLo: readInputLo
        )

        if numBlocks > 1 {
            try encodeInclusivePrefixSum(
                encoder: encoder,
                hiBuffer: blockHi,
                loBuffer: blockLo,
                elementCount: numBlocks,
                auxiliaryHi: auxiliaryHi,
                auxiliaryLo: auxiliaryLo,
                level: level + 1
            )

            dispatchPrefixSumPhase(
                encoder: encoder,
                dataHi: hiBuffer,
                dataLo: loBuffer,
                blockHi: blockHi,
                blockLo: blockLo,
                elementCount: elementCount,
                phase: 2,
                readInputLo: readInputLo
            )
        }
    }

    // MARK: - Benchmark baseline (naive Float32 scan)

    func dispatchPrefixSumPhaseNaive(
        encoder: MTLComputeCommandEncoder,
        dataBuffer: MTLBuffer,
        blockSumsBuffer: MTLBuffer,
        elementCount: Int,
        phase: UInt32
    ) {
        let blockSize = Self.scanBlockSize
        let threadCount = max(elementCount, 1)
        var countValue = UInt32(elementCount)
        var phaseValue = phase

        encoder.setComputePipelineState(pipelines.prefixSumNaive)
        encoder.setBuffer(dataBuffer, offset: 0, index: 0)
        encoder.setBuffer(blockSumsBuffer, offset: 0, index: 1)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&phaseValue, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadsPerGrid = MTLSize(width: threadCount, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: min(blockSize, threadCount), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    func encodeInclusivePrefixSumNaive(
        encoder: MTLComputeCommandEncoder,
        buffer: MTLBuffer,
        elementCount: Int,
        auxiliaryBuffers: [MTLBuffer],
        level: Int = 0
    ) throws {
        guard elementCount > 1 else { return }
        guard level < auxiliaryBuffers.count else {
            throw QuantumEngineError.prefixSumBufferLevelMissing(level: level)
        }

        let blockSize = Self.scanBlockSize
        let numBlocks = (elementCount + blockSize - 1) / blockSize
        let blockSumsBuffer = auxiliaryBuffers[level]

        dispatchPrefixSumPhaseNaive(
            encoder: encoder,
            dataBuffer: buffer,
            blockSumsBuffer: blockSumsBuffer,
            elementCount: elementCount,
            phase: 0
        )

        if numBlocks > 1 {
            try encodeInclusivePrefixSumNaive(
                encoder: encoder,
                buffer: blockSumsBuffer,
                elementCount: numBlocks,
                auxiliaryBuffers: auxiliaryBuffers,
                level: level + 1
            )

            dispatchPrefixSumPhaseNaive(
                encoder: encoder,
                dataBuffer: buffer,
                blockSumsBuffer: blockSumsBuffer,
                elementCount: elementCount,
                phase: 2
            )
        }
    }

    /// Timing comparison of the CDF prefix-sum stage only: the uncompensated Float32 baseline versus
    /// the shipped compensated (double-single) scan, averaged over `iterations` GPU runs.
    public struct ScanBenchmarkResult: Sendable {
        public let stateCount: Int
        public let iterations: Int
        public let naiveMillisecondsAverage: Double
        public let compensatedMillisecondsAverage: Double

        /// Extra wall-clock cost of the compensated scan, as a percentage of the naive scan.
        public var overheadPercent: Double {
            guard naiveMillisecondsAverage > 0 else { return 0 }
            return (compensatedMillisecondsAverage - naiveMillisecondsAverage) / naiveMillisecondsAverage * 100
        }
    }

    /// Benchmarks only the CDF/scan stage for a `qubitCount`-wide probability array.
    ///
    /// Both variants run the identical hierarchical scan structure over the same uniform input; only
    /// the per-element work (and the extra `lo` compensation buffer traffic) differs. Buffer refills
    /// are excluded from the timing, so the measurement reflects the GPU scan alone.
    public func benchmarkPrefixSumScan(qubitCount: Int, iterations: Int) throws -> ScanBenchmarkResult {
        precondition(qubitCount > 0 && iterations > 0)
        let stateCount = 1 << qubitCount
        let bytes = stateCount * MemoryLayout<QFloat>.stride

        let source = try makeSharedBuffer(length: bytes)
        let sourcePointer = source.contents().assumingMemoryBound(to: QFloat.self)
        let uniform = QFloat(1.0 / Double(stateCount))
        for index in 0..<stateCount { sourcePointer[index] = uniform }

        let naiveData = try makeSharedBuffer(length: bytes)
        var naiveAux: [MTLBuffer] = []
        var remaining = stateCount
        while remaining > 1 {
            let blockCount = max((remaining + Self.scanBlockSize - 1) / Self.scanBlockSize, 1)
            naiveAux.append(try makeSharedBuffer(length: blockCount * MemoryLayout<QFloat>.stride))
            remaining = blockCount
        }

        let compensatedHi = try makeSharedBuffer(length: bytes)
        let compensatedLo = try makeSharedBuffer(length: bytes)
        let compensatedAux = try makePrefixSumAuxBuffers(stateCount: stateCount)

        func refill(_ buffer: MTLBuffer) {
            buffer.contents().copyMemory(from: source.contents(), byteCount: bytes)
        }

        func runNaiveOnce() throws {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw QuantumEngineError.commandBufferCreationFailed
            }
            try encodeInclusivePrefixSumNaive(
                encoder: encoder,
                buffer: naiveData,
                elementCount: stateCount,
                auxiliaryBuffers: naiveAux
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
            }
        }

        func runCompensatedOnce() throws {
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw QuantumEngineError.commandBufferCreationFailed
            }
            try encodeInclusivePrefixSum(
                encoder: encoder,
                hiBuffer: compensatedHi,
                loBuffer: compensatedLo,
                elementCount: stateCount,
                auxiliaryHi: compensatedAux.hi,
                auxiliaryLo: compensatedAux.lo
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw QuantumEngineError.commandBufferExecutionFailed(underlying: error)
            }
        }

        // Warm-up (pipeline/caches) — untimed.
        refill(naiveData); try runNaiveOnce()
        refill(compensatedHi); try runCompensatedOnce()

        var naiveTotal = 0.0
        for _ in 0..<iterations {
            refill(naiveData)
            let start = CFAbsoluteTimeGetCurrent()
            try runNaiveOnce()
            naiveTotal += CFAbsoluteTimeGetCurrent() - start
        }

        var compensatedTotal = 0.0
        for _ in 0..<iterations {
            refill(compensatedHi)
            let start = CFAbsoluteTimeGetCurrent()
            try runCompensatedOnce()
            compensatedTotal += CFAbsoluteTimeGetCurrent() - start
        }

        return ScanBenchmarkResult(
            stateCount: stateCount,
            iterations: iterations,
            naiveMillisecondsAverage: naiveTotal / Double(iterations) * 1000,
            compensatedMillisecondsAverage: compensatedTotal / Double(iterations) * 1000
        )
    }
}
