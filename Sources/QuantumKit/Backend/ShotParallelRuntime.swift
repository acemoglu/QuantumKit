import Foundation

/// Host-side helpers for independent-shot parallelism. DispatchQueue workers do **not** inherit
/// ``TaskLocal``; the owner captures the recorder and cancellation task, then installs them
/// explicitly. Workers may only ``SimulationProfileRecorder/timeGate`` (via engine `executeRNG`).
enum ShotParallelRuntime {

    /// Binds the owner's recorder on a GCD worker and raises nesting so the worker is never
    /// a phase / ``finishProfile`` owner.
    static func installWorkerRecorder<T>(
        recorder: SimulationProfileRecorder?,
        gateRecordingSuppressed: Bool,
        _ body: () throws -> T
    ) rethrows -> T {
        try SimulationProfiling.$recorder.withValue(recorder) {
            try SimulationProfiling.$recorderNestingDepth.withValue(1) {
                try SimulationProfiling.$gateRecordingSuppressed.withValue(
                    gateRecordingSuppressed,
                    operation: body
                )
            }
        }
    }

    /// Cooperative cancel that still sees the **owner** Swift Task after hopping to GCD
    /// (`Task.checkCancellation()` on a queue worker would miss it).
    static func makeCancellationPing(
        _ cancellationCheck: (() throws -> Void)?
    ) -> CancellationPing {
        CancellationPing(cancellationCheck: cancellationCheck)
    }

    static func concurrentMap(
        count: Int,
        _ body: @escaping @Sendable (Int) throws -> Int
    ) throws -> [Int] {
        guard count > 0 else { return [] }
        var outcomes = [Int](repeating: 0, count: count)
        let errors = ConcurrentFirstError()
        outcomes.withUnsafeMutableBufferPointer { buffer in
            let slots = OutcomeSlots(buffer)
            DispatchQueue.concurrentPerform(iterations: count) { index in
                if errors.hasValue { return }
                do {
                    slots.store(try body(index), at: index)
                } catch {
                    errors.store(error)
                }
            }
        }
        if let error = errors.take() {
            throw error
        }
        return outcomes
    }
}

final class CancellationPing: @unchecked Sendable {
    private let ownerCancelled: OwnerTaskCancellation
    private let check: (() throws -> Void)?

    init(cancellationCheck: (() throws -> Void)?) {
        self.ownerCancelled = OwnerTaskCancellation()
        self.check = cancellationCheck
    }

    func call() throws {
        if ownerCancelled.isCancelled() {
            throw CircuitExecutionCancellationError.cancelled
        }
        try check?()
    }
}

final class ConcurrentFirstError: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Error?

    var hasValue: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stored != nil
    }

    func store(_ error: Error) {
        lock.lock()
        if stored == nil {
            stored = error
        }
        lock.unlock()
    }

    func take() -> Error? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

final class OwnerTaskCancellation: @unchecked Sendable {
    private let isOwnerCancelled: () -> Bool

    init() {
        let owner = withUnsafeCurrentTask { $0 }
        self.isOwnerCancelled = { owner?.isCancelled ?? false }
    }

    func isCancelled() -> Bool { isOwnerCancelled() }
}

final class OutcomeSlots: @unchecked Sendable {
    private let pointer: UnsafeMutablePointer<Int>

    init(_ buffer: UnsafeMutableBufferPointer<Int>) {
        self.pointer = buffer.baseAddress!
    }

    func store(_ value: Int, at index: Int) {
        pointer[index] = value
    }
}
