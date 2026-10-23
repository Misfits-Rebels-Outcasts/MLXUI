import Foundation

/// One dedicated OS thread for the audio engines that call mlx-audio-swift's `generate` (WW2-3-6, journal 2026-557).
///
/// **Why it exists.** Under mlx-swift 0.32 a subscript with an `MLXArray` index that follows another index
/// (`input[0..., 0..., xLow]`) records an op on a *thread-bound* stream, and such an op can only be evaluated by the thread that built it. mlx-audio-swift's
/// `interpolate1d` (Kokoro's `SineGenerator`) does exactly that, and `generate` returns its result *lazy*, so
/// awaiting it from the main actor and reading it there raises "There is no Stream(gpu, N) in current thread" — and the
/// next read of the buffer-less array segfaults (Part B, B10). The fix is to build and read on one thread: the engine's
/// whole body runs on this executor, entered from anywhere through `run`.
///
/// **Remove when upstream fixes it.** Delete this file and its two call sites (`KokoroEngine.synthesize`,
/// `MLXAudioTTSEngine.synthesize`) once mlx-audio-swift evaluates before returning or stops using the mixed subscript, or
/// mlx-swift stops recording it on a thread-bound stream. The issue links go in `RSI/notes/ww2/upstream-issues.md` when
/// posted; `MLXThreadStreamTests` fails when the underlying behaviour changes, which is the cue to revisit.
///
/// macOS 15+ only (`TaskExecutor`): on macOS 14 `run` just runs the body, and the hazard stays for the main-actor Run UI.
nonisolated enum MLXAudioThread {
    /// Run `operation` — load, `generate`, evaluate, read — on the audio thread. Safe to call from any actor.
    static func run<T: Sendable>(_ operation: @escaping @Sendable () async throws -> T) async throws -> T {
        if #available(macOS 15.0, *) {
            return try await PinnedThread.shared.run(operation)
        }
        return try await operation()
    }
}

@available(macOS 15.0, *)
private nonisolated final class PinnedThread: TaskExecutor, @unchecked Sendable {
    static let shared = PinnedThread(name: "net.connectcode.mlxui.mlx-audio")

    private let condition = NSCondition()
    private var jobs: [UnownedJob] = []

    private init(name: String) {
        let thread = Thread { [unowned self] in
            while true {
                condition.lock()
                while jobs.isEmpty { condition.wait() }
                let job = jobs.removeFirst()
                condition.unlock()
                job.runSynchronously(on: asUnownedTaskExecutor())
            }
        }
        thread.name = name
        thread.start()
    }

    func enqueue(_ job: consuming ExecutorJob) {
        let unowned = UnownedJob(job)
        condition.lock()
        jobs.append(unowned)
        condition.signal()
        condition.unlock()
    }

    func run<T: Sendable>(_ operation: @escaping @Sendable () async throws -> T) async throws -> T {
        try await withTaskExecutorPreference(self) { try await Self.perform(operation) }
    }

    /// `@concurrent`: a nonisolated async function honours the task's executor preference; a closure inheriting the
    /// caller's actor (the main actor, for the Run UI) would run there regardless.
    @concurrent private static func perform<T: Sendable>(
        _ operation: @Sendable () async throws -> T
    ) async throws -> T {
        try await operation()
    }
}
