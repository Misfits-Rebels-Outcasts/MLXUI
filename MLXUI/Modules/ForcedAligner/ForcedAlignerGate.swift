import Foundation

/// Where a queued alignment says that it is waiting (`true`) and that its turn has come (`false`). Like
/// `AlignTranscript` it travels with the run: the Run screen sets it around its run, a flow row sets none. A call that
/// never had to wait reports nothing.
nonisolated enum AlignWaiting {
    @TaskLocal static var report: (@Sendable (Bool) -> Void)?
}

/// One alignment at a time per key (the model folder), first come first served. The engine loads the model inside every
/// call and a flow row and a Run sheet in one language share one memory booking, so two calls at once would hold two
/// resident copies under it.
///
/// A turn is a flag held across the body's suspensions (not this actor's isolation), so a second call cannot start its
/// load while the first is suspended in its own. The body runs off the actor, which stays free to queue and to drop
/// waiters while a synchronous forward pass runs. The shared audio thread (`MLXAudioThread`) is not involved.
///
/// Owner ruling 20 (Boo, 2026-11-27): "A run waiting for the engine responds to Stop, to Done and to a flow being
/// cancelled, and never starts after it has been cancelled. Test it: start run A, queue run B, cancel B, then A finishes
/// and B never runs. The Run screen shows 'Waiting for the last alignment to finish…' while it waits."
actor ForcedAlignerGate {
    nonisolated private struct Waiter {
        let id: UUID
        let turn: CheckedContinuation<Void, any Error>
    }

    private var taken: Set<String> = []
    private var queues: [String: [Waiter]] = [:]

    /// Runs `body` when `key` is free, and frees it however `body` ends. A call cancelled while it waits, or just as
    /// its turn comes, throws `CancellationError` and never runs `body`.
    func run<T: Sendable>(_ key: String, body: @concurrent @Sendable () async throws -> T) async throws -> T {
        try await acquire(key)
        defer { release(key) }
        try Task.checkCancellation()   // cancelled between the grant and here
        return try await body()
    }

    /// How many calls are waiting for `key`.
    func queuedCount(_ key: String) -> Int { queues[key]?.count ?? 0 }

    private func acquire(_ key: String) async throws {
        try Task.checkCancellation()
        if taken.insert(key).inserted { return }
        let id = UUID()
        let report = AlignWaiting.report
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (turn: CheckedContinuation<Void, any Error>) in
                // Already cancelled: `onCancel` may have run before there was a waiter to drop.
                if Task.isCancelled {
                    turn.resume(throwing: CancellationError())
                    return
                }
                queues[key, default: []].append(Waiter(id: id, turn: turn))
                report?(true)
            }
        } onCancel: {
            Task { await self.leave(key, id) }
        }
        report?(false)
    }

    /// A cancelled waiter leaves the queue. Nothing to do if its turn came first (`run` checks for that).
    private func leave(_ key: String, _ id: UUID) {
        guard let index = queues[key]?.firstIndex(where: { $0.id == id }),
              let waiter = queues[key]?.remove(at: index) else { return }
        waiter.turn.resume(throwing: CancellationError())
    }

    /// Passes the turn to the longest waiter, or frees the key.
    private func release(_ key: String) {
        if let queue = queues[key], !queue.isEmpty {
            queues[key]?.removeFirst().turn.resume()
        } else {
            queues[key] = nil
            taken.remove(key)
        }
    }
}
