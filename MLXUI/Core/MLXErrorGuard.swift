import Foundation
import MLX
import os

/// MLX-ERR-1 — an MLX error must never kill the app.
///
/// **The bug.** mlx-swift's error path ends in `fatalError` whenever no handler is installed:
/// `ErrorHandler.dispatch` (`mlx-swift/Source/MLX/ErrorHandler.swift:345`) is reached from
/// `errorHandlerTrampoline` ← `_mlx_error` ← `mlx_eval`, and with an empty handler stack it calls
/// `fatalError(message)`. The trampoline is not opt-in: `MLXArray.swift:18` calls `initError()`,
/// so **creating any `MLXArray` installs it** — which every MLX path in this app does. There is
/// no non-deprecated global installer in mlx-swift, and `MLXUI/` installs none of the scoped
/// ones, so in this app *any* MLX error on *any* path is a `fatalError`: the process dies with
/// Xcode's "hit program assert". That is what 13 `MLXUI-*.ips` reports on this machine show
/// (journal `2026-464` read every one); 8 of them are an MLX error inside the Local Server's
/// prompt-cache prefill, and the rest are teardown/exception paths this type cannot help with.
///
/// **Two layers, because they cover different code.**
///
/// 1. ``guarded(_:)`` wraps the outer boundary of work we own — `RealExecutor.execute` (every
///    flow row, hence every engine) and `MLXChatBackend.run` (the Local Server) — using
///    mlx-swift's **scoped, non-deprecated** `withError`. An MLX error becomes a thrown
///    `MLXError` carrying MLX's own message, so the row or the request reports it like any other
///    failure instead of killing the process.
/// 2. ``installFallback()`` covers everything *not* wrapped — today the per-model Run UIs'
///    engines. mlx-swift offers no non-deprecated process-wide install (see the note on
///    `installFallback`), so this deliberately uses the deprecated `setErrorHandler`. Without it
///    an MLX error in a Run UI still kills the app; with it the message is recorded by
///    ``lastFallbackMessage()`` and logged as a fault.
///
/// **Known limitation, deliberate, and not a substitute for layer 1.** The fallback does not make
/// the *result* correct: mlx-swift's own docs warn that "using the `MLXArray` produced after an
/// error will likely result in additional errors", so an engine that reported an error may go on
/// to produce a bad array. Layer 2 trades "the app dies with no explanation" for "the app logs a
/// fault and keeps running", which is the right trade for a shipping app but means the fallback
/// must stay **visible** rather than silent — hence the `os.Logger` fault and the recorded
/// message. Coverage should be extended to the Run-UI engines so those paths get the honest
/// thrown error too, not just the log.
nonisolated enum MLXErrorGuard {

    // MARK: - Scoped guard (the one to prefer)

    /// Run `body` with mlx-swift's scoped error handler installed, converting an MLX error into a
    /// thrown ``MLXError`` at block exit instead of `fatalError`.
    ///
    /// The handler is **task-local** in mlx-swift, so it follows the `await` chain and any `MainActor`
    /// hop inside `body`; MLX work spawned in a *detached* task would not inherit it. Nothing in
    /// `FlowKit/` or `Modules/` detaches, so the flow and server boundaries are covered
    /// completely.
    static func guarded<R>(_ body: () async throws -> R) async throws -> R {
        try await withError { try await body() }
    }

    /// Synchronous form, for callers that are not `async`.
    static func guarded<R>(_ body: () throws -> R) throws -> R {
        try withError { try body() }
    }

    /// The sentence to show a person for an error raised under ``guarded(_:)``.
    ///
    /// mlx-swift's own `MLXError.errorDescription` is `"MLX Error: <message>"` — accurate but not
    /// this product's voice (one plain sentence, and what to try). Kept short on purpose: the
    /// message is MLX's own diagnostic text, so it is often the most useful part.
    static func sentence(for error: Error) -> String {
        if let mlx = error as? MLXError, case .caught(let message) = mlx {
            // "Try again", not "Try the row again": this sentence is used by flow rows, by every
            // per-model Run UI and (MLX-ERR-3) by the chat, and two of those three have no row.
            return "the model's MLX runtime reported an error — \(message). Try again; if it repeats, re-install the model."
        }
        return (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    }

    /// Whether `error` is one of ours from mlx-swift. Lets a caller in a module that must stay
    /// free of `import MLX` (e.g. `FlowKit/RealExecutor`) still recognise the case and re-voice it.
    static func isMLXError(_ error: Error) -> Bool {
        error is MLXError
    }

    // MARK: - The Run-UI entry point (MLX-ERR-2)

    /// Wrap a **Run UI's** engine call.
    ///
    /// This is `guarded(_:)` plus one rule that only matters for a view model: an MLX error that
    /// escaped through the process-wide fallback *during this call* also fails the run.
    ///
    /// That second rule is what makes the fallback safe to ship. Layer 2 keeps the app alive by
    /// recording and logging instead of dying — which, on its own, would let a Run UI paint a
    /// **successful** result produced by an engine that had already errored (mlx-swift's own docs
    /// warn the array from a failed op is suspect). A view model that shows "done" there is worse
    /// than one that crashed: it is the CFM-R16-1 shape, "a wrong output nobody notices". So a
    /// wrapped Run UI may never report success for a run in which the fallback fired.
    ///
    /// Uses a **generation counter**, not the message text: the same MLX error happening twice in a
    /// row would be invisible to a text comparison.
    static func guardedRun<R>(_ body: () async throws -> R) async throws -> R {
        let start = recorder.generation
        let value = try await guarded(body)
        if let escaped = recorder.firstMessage(after: start) {
            throw MLXError.caught(escaped)
        }
        return value
    }

    /// Synchronous form of ``guardedRun(_:)``.
    static func guardedRun<R>(_ body: () throws -> R) throws -> R {
        let start = recorder.generation
        let value = try guarded(body)
        if let escaped = recorder.firstMessage(after: start) {
            throw MLXError.caught(escaped)
        }
        return value
    }

    /// Consume a stream that MLX work feeds, **stopping at the first MLX error**.
    ///
    /// MLX-ERR-3 (owner ruling R-ME3-1, 2026-10-12: *"stop at the error … The partial reply stays,
    /// and the turn ends with the error sentence visible"*). `guardedRun` cannot do this: it
    /// converts at **block exit**, so a mid-stream error would surface only when the producer
    /// finished and the consumer would keep going until then. This checks **per element**, before
    /// the element is handed on, so the loop stops at the error and everything already consumed
    /// stays where the caller put it.
    ///
    /// - Parameter onEach: called for each element **before** it is passed on. Return `false` to
    ///   stop consuming deliberately (e.g. the user switched conversations).
    ///
    /// **`makeStream` is a closure because it has to be, not for style.** mlx-swift's handler is
    /// `@TaskLocal`, and both producers in this app (`AgentSession.rawStreamResponse` and
    /// mlx-swift-lm's `ChatSession.streamMap`) do their work in a plain `Task {}` — which inherits
    /// task-locals from the task that *creates* it. Create the stream inside this scope and the
    /// handler reaches the generation; pass in an already-created stream and it does not. Measured
    /// by MLX-ERR-3's point-2 probe (journal `2026-469`):
    ///
    /// | stream created | error caught by | elements delivered | fallback |
    /// |---|---|---|---|
    /// | **inside** this scope | the scoped `ErrorBox` → thrown | 2 of 5 (**stopped at the error**) | 0 |
    /// | outside, passed in | nothing — the fallback recorded it | 5 of 5 (drained) | 1 |
    ///
    /// The outside case fails **silently**: the error goes to the process-wide fallback,
    /// `check()` never throws, and the loop runs to completion — precisely the bug this helper
    /// exists to fix, while looking correct at the call site. `aStreamMadeOutsideTheScopeDefeatsTheGuard`
    /// in `MLXErrorGuardTests` pins that so the closure parameter cannot be "simplified" away.
    static func guardedStream<S: AsyncSequence>(
        _ makeStream: () -> S,
        onEach: (S.Element) throws -> Bool
    ) async throws {
        try await withError { box in
            for try await element in makeStream() {
                try box.check()
                let keepGoing = try onEach(element)
                if !keepGoing { return }
            }
        }
    }

    /// How many messages the fallback has recorded. Exposed for tests and for the generation check
    /// above; a plain count is enough because it only ever moves forward.
    static func fallbackGeneration() -> Int {
        recorder.generation    }

    // MARK: - Process-wide fallback

    /// Install the process-wide fallback, once. Safe to call more than once.
    ///
    /// **This deliberately uses a deprecated API, and that is the point.** mlx-swift's
    /// deprecation notice points at `withErrorHandler`/`withError`, but those install a
    /// *task-local* handler: they cannot cover a path we have not wrapped, which is exactly the
    /// gap this fills. `setErrorHandler` is sync with mlx-swift's `ErrorHandler.globalHandler`
    /// slot, which `dispatch` consults *after* the task-local stack — so a `guarded` body keeps
    /// its precise thrown error and everything else falls back to the recorder.
    ///
    /// It must go through mlx-swift's slot rather than calling the C `mlx_set_error_handler`
    /// directly: mlx-swift installs its own trampoline lazily (on the first `MLXArray`), which
    /// would silently overwrite a handler we installed underneath it.
    static func installFallback() {
        _ = fallbackInstalled
    }

    /// The last message the fallback recorded, or nil if no unguarded MLX error has happened.
    /// Exists so the fallback cannot fail *silently* — a test asserts it, and a diagnostic or a
    /// future banner can read it.
    static func lastFallbackMessage() -> String? {
        recorder.message
    }

    /// Clear the recorded message (tests).
    static func resetFallbackMessage() {
        recorder.message = nil
    }

    // MARK: - Internals

    private static let recorder = MLXErrorRecorder()

    private static let fallbackInstalled: Void = {
        // `setErrorHandler` is deprecated in favour of the scoped API; see `installFallback`'s
        // doc comment for why the global slot is the only correct hook for this layer.
        MLXErrorGuard.installDeprecatedGlobalHandler(mlxGlobalErrorHandler)
    }()

    /// Isolates the one deprecated call so the deprecation is acknowledged in a single, named
    /// place instead of at a bare call site.
    @available(*, deprecated, message: "MLX-ERR-1: call MLXErrorGuard.installFallback() instead.")
    private static func installDeprecatedGlobalHandler(
        _ handler: (@convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void)?
    ) {
        setErrorHandler(handler)
    }

    /// Thread-safe holder for the last unguarded message. A class so the `@convention(c)`
    /// trampoline below can reach it without capturing anything.
    private final class MLXErrorRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private let log = Logger(subsystem: "net.connectcode.mlxui", category: "mlx-error")
        private var stored: String?
        /// Monotonic count of recorded messages — lets a caller ask "did one happen during this
        /// call?" even when the text repeats. Only ever incremented.
        private var count = 0

        var message: String? {
            get { lock.withLock { stored } }
            set { lock.withLock { stored = newValue } }
        }

        var generation: Int { lock.withLock { count } }

        /// The first message recorded after `generation`, or nil if none was.
        func firstMessage(after generation: Int) -> String? {
            lock.withLock { count > generation ? stored : nil }
        }

        func record(_ message: String) {
            let isFirst = lock.withLock { () -> Bool in
                count += 1
                guard stored == nil else { return false }
                stored = message
                return true
            }
            // `.fault`, not `.error`: the fallback firing means an MLX error escaped every
            // guarded boundary, so the result of that work is suspect (see the type's "Known
            // limitation"). This has to be loud rather than silent, and the first one names the
            // missing wrap explicitly so the log says what to do about it.
            if isFirst {
                log.fault("an MLX error escaped every guarded boundary — wrap this path in MLXErrorGuard.guarded: \(message, privacy: .public)")
            } else {
                log.fault("unguarded MLX error: \(message, privacy: .public)")
            }
        }
    }
}

/// The `@convention(c)` trampoline mlx-swift calls for an error outside every `guarded` block. A
/// top-level function so it captures nothing; it forwards to the recorder.
private let mlxGlobalErrorHandler:
    @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = { message, _ in
        let text = message.map { String(cString: $0) } ?? ""
        MLXErrorGuard.recordFallback(text)
    }

extension MLXErrorGuard {
    /// Bridge for the C trampoline (which cannot see `private` members).
    fileprivate static func recordFallback(_ message: String) {
        recorder.record(message)
    }
}
