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
