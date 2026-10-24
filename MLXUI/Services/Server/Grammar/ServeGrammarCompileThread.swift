import Foundation

/// The one thread every grammar compile runs on (WW4-1-3, ruling AV).
///
/// xgrammar copies the parsed schema recursively (picojson), so a deeply nested schema is a *stack* overflow, not a slow
/// compile: the test host died with SIGBUS at 101 nested containers. How deep is safe depends on the stack the thread was
/// given, and a Swift concurrency pool thread, the main thread and a `Thread` each have a different one — so the depth limit
/// in `ServeSchemaCheck.Bounds.maxJSONDepth` is only meaningful on a thread whose stack we set. This is that thread: 16 MB,
/// fed by a serial queue, never the main thread and never a cooperative-pool thread.
///
/// A job already running cannot be interrupted (xgrammar has no cancel); a job still queued when its caller is cancelled is
/// skipped.
nonisolated final class ServeGrammarCompileThread: @unchecked Sendable {
    static let stackBytes = 16 * 1024 * 1024
    static let shared = ServeGrammarCompileThread()

    private let condition = NSCondition()
    private var queue: [() -> Void] = []
    private var thread: Thread?

    /// Runs `work` on the compile thread. Throws `CancellationError` if the caller was cancelled before the job started.
    func run<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        let state = CancelState()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<T, Error>) in
                enqueue {
                    if state.isCancelled {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    do { continuation.resume(returning: try work()) } catch { continuation.resume(throwing: error) }
                }
            }
        } onCancel: {
            state.cancel()
        }
    }

    /// For tests: run synchronously-queued work and report which thread and stack size it ran with.
    var threadName: String { "ServeGrammarCompileThread" }

    private func enqueue(_ job: @escaping () -> Void) {
        condition.lock()
        defer { condition.unlock() }
        if thread == nil {
            let t = Thread { [unowned self] in self.loop() }
            t.name = threadName
            t.stackSize = Self.stackBytes
            t.qualityOfService = .userInitiated
            thread = t
            t.start()
        }
        queue.append(job)
        condition.signal()
    }

    private func loop() {
        while true {
            condition.lock()
            while queue.isEmpty { condition.wait() }
            let job = queue.removeFirst()
            condition.unlock()
            job()
        }
    }

    private final class CancelState: @unchecked Sendable {
        private let lock = NSLock()
        private var cancelled = false
        var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
        func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    }
}
