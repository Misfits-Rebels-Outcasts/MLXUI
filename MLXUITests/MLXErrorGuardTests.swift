import Testing
import Foundation
import MLX
@testable import MLXUI

/// MLX-ERR-1 (journal `2026-466`, backlog BG-8) — an MLX error must never kill the app.
///
/// mlx-swift's `ErrorHandler.dispatch` calls `fatalError` when no handler is installed, and
/// `MLXArray`'s own init installs the trampoline, so *any* MLX error on an unguarded path was a
/// process kill. The first test below is the regression test for that: **before the fix it does
/// not fail — it aborts the whole test process**, which is why the bug went unnoticed.
///
/// Serialized: MLX's default stream is not parallel-safe (the same reason the other MLX suites
/// are), and the fallback's recorded message is process-wide.
@Suite(.serialized)
struct MLXErrorGuardTests {

    // MARK: - The guard turns the error into a throw

    /// Inside `guarded`, the same expression throws `MLXError` carrying MLX's message — instead
    /// of calling `fatalError` and taking the process with it.
    @Test func guardedTurnsAnMLXErrorIntoAThrow() async throws {
        var caught: Error?
        do {
            _ = try await MLXErrorGuard.guarded {
                triggerAnMLXError()
                return 0
            }
            Issue.record("expected the MLX error to be thrown")
        } catch {
            caught = error
        }
        let error = try #require(caught)
        #expect(MLXErrorGuard.isMLXError(error), "expected an MLXError, got \(error)")
        // MLX's own message travels with it — that is the part worth showing a person.
        #expect(!MLXErrorGuard.sentence(for: error).isEmpty)
    }

    /// The sync overload behaves the same way (both are thin wrappers over `withError`).
    @Test func guardedSyncTurnsAnMLXErrorIntoAThrow() throws {
        #expect(throws: MLXError.self) {
            try MLXErrorGuard.guarded { triggerAnMLXError() }
        }
    }

    /// The product voice: `MLXError`'s own `errorDescription` is `"MLX Error: …"`, which is not
    /// this app's voice. `sentence(for:)` must produce the honest sentence *and* keep MLX's
    /// diagnostic text, since that is usually the actionable part.
    @Test func theSentenceIsTheProductsVoiceAndKeepsTheDetail() {
        let mlx = MLXError.caught("Shapes [2,5] and [3,5] cannot be broadcast")
        let sentence = MLXErrorGuard.sentence(for: mlx)
        #expect(sentence.contains("Shapes [2,5] and [3,5] cannot be broadcast"))
        #expect(sentence.contains("MLX"))
        #expect(!sentence.hasPrefix("MLX Error:"))
        // A non-MLX error passes through as its own description, not swallowed.
        struct Other: LocalizedError { var errorDescription: String? { "some other failure" } }
        #expect(MLXErrorGuard.sentence(for: Other()) == "some other failure")
    }

    // MARK: - The fallback catches what the guard does not

    /// **The layer-2 contract**: an MLX error raised *outside* any `guarded` block must not kill
    /// the process. It is recorded (and logged as a fault) so it cannot pass silently — the
    /// fallback trades a crash for a visible fault, never for a silent wrong answer.
    @Test @MainActor func theFallbackRecordsAnUnguardedErrorInsteadOfKillingTheApp() {
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)

        triggerAnMLXError()   // deliberately unguarded — this used to be fatal

        let recorded = MLXErrorGuard.lastFallbackMessage()
        #expect(recorded != nil, "the fallback must record the message, never swallow it")
        #expect(recorded?.isEmpty == false)
        // Still alive and usable, which is the whole point.
        #expect(MLXArray(0 ..< 3, [3]).shape == [3])
        MLXErrorGuard.resetFallbackMessage()
    }

    /// `installFallback()` is idempotent (it is called from `MLXUIApp.init()` and tests both).
    @Test @MainActor func installingTheFallbackTwiceKeepsItWorking() {
        MLXErrorGuard.installFallback()
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        triggerAnMLXError()
        #expect(MLXErrorGuard.lastFallbackMessage() != nil)
        MLXErrorGuard.resetFallbackMessage()
    }

    // MARK: - The wiring: a failing MLX op inside a flow row

    /// Proves `RealExecutor.execute` is actually guarded — the fallback alone would keep the
    /// process alive but let the row return a bogus asset. A row whose stage leaves an MLX error
    /// behind must **fail with a sentence**, not succeed quietly and not kill the app.
    @Test func aFailingMLXOpInAFlowRowFailsTheRowWithASentence() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxerr-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let ws = FlowWorkspace(root: root)

        // The row has to resolve to an installed catalog entry for `execute` to reach
        // `makeModelStage` at all — otherwise it fails on the model, not on the MLX op, and the
        // test would pass for the wrong reason.
        let entry = makeEntry(id: "test--mlxerr-llm", family: "TestFamily",
                              displayName: "MLXErr LLM", modelType: .llm,
                              hfModelId: "test/mlxerr-llm")

        let executor = RealExecutor(
            workspace: ws, flowID: "mlxerr", blobDirectory: root.appendingPathComponent(".blobs"),
            makeModelStage: { _, _ in
                // Triggers the MLX error and then returns **normally**, so the error is the only
                // failure and `withError`'s block-exit check gets to see it.
                TextStageThatRaisesAnMLXError()
            },
            installedModelIDs: [entry.id], catalog: [entry])

        let row = Row(task: "Summarize", model: entry.displayName, settings: "in 3 bullets")
        let input = Asset(items: [Item(kind: .text, value: "a memo", path: nil, sourceText: nil)])
        do {
            _ = try await executor.execute(path: "2", row: row, inputs: [input],
                                           transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("expected the row to fail with the MLX sentence")
        } catch {
            let message = String(describing: error)
            #expect(message.contains("MLX"), "the row's sentence should name the MLX runtime: \(message)")
            #expect(message.contains("2"), "the row's sentence should name row 2: \(message)")
        }
    }

    /// The semantics worth knowing: mlx-swift's `withError` converts the error at **block exit**,
    /// so if the body throws something of its own after the MLX error, *that* error propagates and
    /// the MLX sentence never surfaces. The app still does not die — which is the guard's job —
    /// but the case is not reported as an MLX error, and a caller relying on the MLX sentence
    /// would miss this shape. Pinned so the next reader does not assume the sentence always wins.
    @Test func aThrowAfterTheMLXErrorPropagatesInsteadOfTheMLXSentence() async throws {
        var caught: Error?
        do {
            _ = try await MLXErrorGuard.guarded { () -> Int in
                triggerAnMLXError()
                throw FlowError.unsupportedTask(row: "2", task: "Test")
            }
        } catch {
            caught = error
        }
        let error = try #require(caught)
        #expect(!MLXErrorGuard.isMLXError(error),
                "the body's own throw must win — if this starts failing, `withError`'s contract changed")
    }

    // MARK: - MLX-ERR-3: guardedStream, and the ordering that makes it work

    /// A producer shaped exactly like the two real ones (`AgentSession.rawStreamResponse`,
    /// mlx-swift-lm's `ChatSession.streamMap`): `makeStream()` plus a plain `Task {}` that yields
    /// some elements, trips an MLX error, then keeps going.
    private func streamThatErrorsAfter(_ chunks: Int, thenKeepsGoing late: Int)
        -> AsyncThrowingStream<String, Error> {
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let task = Task {
            for i in 0 ..< chunks {
                continuation.yield("chunk-\(i)")
                try? await Task.sleep(for: .milliseconds(2))
            }
            triggerAnMLXError()
            for i in 0 ..< late {              // must never reach the caller
                continuation.yield("late-\(i)")
                try? await Task.sleep(for: .milliseconds(2))
            }
            continuation.finish()
        }
        _ = task
        return stream
    }

    /// **R-ME3-1**: the loop stops AT the error, and what was consumed before it stays.
    @Test func guardedStreamStopsAtTheErrorAndKeepsWhatCameBefore() async throws {
        MLXErrorGuard.installFallback()
        let before = MLXErrorGuard.fallbackGeneration()
        var caught: Error?
        var delivered: [String] = []
        do {
            try await MLXErrorGuard.guardedStream({ self.streamThatErrorsAfter(2, thenKeepsGoing: 3) }) { chunk in
                delivered.append(chunk)
                return true
            }
            Issue.record("expected the MLX error to end the stream")
        } catch {
            caught = error
        }
        let error = try #require(caught)
        #expect(MLXErrorGuard.isMLXError(error), "expected an MLXError, got \(error)")
        #expect(delivered == ["chunk-0", "chunk-1"],
                "the two chunks before the error must survive and nothing after it may arrive")
        #expect(MLXErrorGuard.fallbackGeneration() == before,
                "the scoped handler should have caught it — nothing may reach the fallback")
    }

    /// `onEach` returning `false` stops deliberate consumption (the conversation-switch case),
    /// without inventing an error.
    @Test func guardedStreamHonoursAnEarlyStopFromOnEach() async throws {
        var delivered: [String] = []
        try await MLXErrorGuard.guardedStream({ self.streamThatErrorsAfter(4, thenKeepsGoing: 0) }) { chunk in
            delivered.append(chunk)
            return delivered.count < 2      // stop after two
        }
        #expect(delivered == ["chunk-0", "chunk-1"])
    }

    /// **The trap the point-2 probe found, pinned.** A stream created *outside* the scope defeats
    /// the guard completely: the producer's `Task {}` does not inherit mlx-swift's `@TaskLocal`
    /// handler, so the error goes to the process-wide fallback, `check()` never throws, and the
    /// loop drains. That is why `guardedStream` takes a closure — this test exists so nobody
    /// "simplifies" it into `guardedStream(someStream) { … }` and silently reintroduces the bug.
    @Test func aStreamMadeOutsideTheScopeDefeatsTheGuard() async throws {
        MLXErrorGuard.installFallback()
        let before = MLXErrorGuard.fallbackGeneration()
        let preMade = streamThatErrorsAfter(2, thenKeepsGoing: 3)   // created BEFORE any scope

        var caught: Error?
        var delivered: [String] = []
        do {
            try await withError { box in
                for try await chunk in preMade {
                    try box.check()
                    delivered.append(chunk)
                }
            }
        } catch { caught = error }

        #expect(caught == nil, "the scoped handler cannot see a producer created outside its scope")
        #expect(delivered.count == 5, "so the loop drains every element, including the late ones")
        #expect(MLXErrorGuard.fallbackGeneration() > before,
                "and the error can only be seen by the process-wide fallback")
    }

    // MARK: - MLX-ERR-2: the Run UIs

    /// The rule that makes the process-wide fallback safe to ship: if an MLX error escaped through
    /// the **fallback** during a guarded call, the call must fail even though its body returned
    /// normally. Otherwise a Run UI would paint a successful result produced by an engine that had
    /// already errored — the "wrong output nobody notices" shape.
    ///
    /// The error has to happen somewhere the scoped handler does **not** reach, or the block-exit
    /// check would catch it instead and this would pass for the wrong reason. A **detached** task
    /// is the faithful case: detached tasks do not inherit task-local values, so mlx-swift's
    /// `@TaskLocal` handler is absent there and the message goes to the fallback — exactly what a
    /// stray engine thread does in the app.
    @Test func anMLXErrorEscapingThroughTheFallbackStillFailsTheCall() async throws {
        var caught: Error?
        do {
            _ = try await MLXErrorGuard.guardedRun { () -> Int in
                await Task.detached { triggerAnMLXError() }.value
                return 42          // the body "succeeds"
            }
            Issue.record("expected the escaped MLX error to fail the call")
        } catch {
            caught = error
        }
        let error = try #require(caught)
        #expect(MLXErrorGuard.isMLXError(error), "expected an MLXError, got \(error)")
        #expect(!MLXErrorGuard.sentence(for: error).isEmpty)
    }

    /// A generation counter, not a text comparison: the same MLX error happening twice must still
    /// be visible as "something happened during this call".
    @Test func theFallbackGenerationAdvancesOnEveryMessage() {
        MLXErrorGuard.installFallback()
        let before = MLXErrorGuard.fallbackGeneration()
        triggerAnMLXError()
        triggerAnMLXError()
        #expect(MLXErrorGuard.fallbackGeneration() >= before + 2)
    }

    /// **The requirement's end-to-end test**: one Run UI's view model, driven with an injected MLX
    /// error, must put the sentence in its own error bar and must not report a result.
    ///
    /// `MusicGenRunModel.generate(prompt:stage:)` is the cleanest seam of the fourteen — the stage
    /// is a parameter — and it is the same shape as Flux/SDXL/TTS/ASR/VLM/OCR, so this covers the
    /// pattern rather than one screen. `runOneShot` is what the other modules' tests would reuse.
    @Test @MainActor func aRunUIWithAnInjectedMLXErrorShowsTheSentenceAndNoResult() async throws {
        let model = MusicGenRunModel()
        model.generate(prompt: "calm ambient background music", stage: AudioStageThatRaisesAnMLXError())

        let settled = await waitUntil { !model.isRunning }
        #expect(settled, "the run should settle rather than stay in `isRunning`")

        let shown = try #require(model.errorText, "the Run UI must show an error, not a result")
        #expect(shown.contains("MLX"), "expected the MLX sentence, got: \(shown)")
        #expect(shown.contains("runtime"), "expected the guard's sentence, got: \(shown)")
        #expect(model.didGenerate == false,
                "a run in which MLX reported an error must never report success")
    }

    /// Poll the main actor until `condition` holds. The Run models start an unstructured `Task` and
    /// have no completion hook, so a test has to wait for the state they write; sleeping yields the
    /// main actor so that task can run.
    @MainActor
    private func waitUntil(timeout: Duration = .seconds(20),
                           _ condition: () -> Bool) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while ContinuousClock.now < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }
}

/// A deterministic MLX error with no weights, no model and no GPU-heavy path: mlx-swift's own
/// documented broadcast failure. `[2, 5] + [3, 5]` are not broadcast-compatible. A free function
/// so `@Sendable` closures (the stage factory) can reference it.
private func triggerAnMLXError() {
    let a = MLXArray(0 ..< 10, [2, 5])
    let b = MLXArray(0 ..< 15, [3, 5])
    _ = a + b
}

/// A `.text → .text` stage that raises an MLX error and then returns its input normally, so the
/// guard's wiring can be exercised without a model, weights, or a GPU-heavy path. Passing the
/// input through is deliberate: the row must fail because of the *MLX error*, not because the
/// stage refused anything.
private nonisolated struct TextStageThatRaisesAnMLXError: PipelineStage {
    let id = "mlxerr.stage"
    let name = "MLX error stage"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        triggerAnMLXError()
        progress(1.0)
        return input
    }
}

/// A `.text → .audio` stage that raises an MLX error and then returns a (silent) buffer normally.
/// The Run UI under test must fail because of the **MLX error**, not because the stage refused
/// anything — so it returns a shape the view model would otherwise accept.
private nonisolated struct AudioStageThatRaisesAnMLXError: PipelineStage {
    let id = "mlxerr.audio"
    let name = "MLX error audio stage"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .audio }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        triggerAnMLXError()
        progress(1.0)
        return .audio(AudioBuffer(samples: [0, 0], sampleRate: 32_000))
    }
}
