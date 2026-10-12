import Testing
import Foundation
import MLX
@testable import MLXUI

/// MLX-ERR-3 (owner rulings R-ME3-1/R-ME3-2, 2026-10-12) — an MLX error mid-conversation must show
/// in the chat **and stop the turn**, not be logged and ignored.
///
/// Before this change the chat path was the one generation path MLXErrorGuard did not cover, so an
/// MLX error during a turn went to the process-wide fallback: the app survived, the error was
/// logged, and the assistant kept "responding".
///
/// The seam is the one-method `ChatStreamingSession` protocol (R-ME3-2) rather than a closure,
/// precisely so this test can assert the turn **stopped early** — the visible difference R-ME3-1
/// creates. `streamingSessionForTesting` replaces the session *before* the weights load, because
/// nothing in this suite can build a `ModelContainer`.
@MainActor
@Suite(.serialized)
struct ModelRunnerStreamErrorTests {

    private let entry = makeEntry(id: "model-a", displayName: "Test Model")

    /// A producer shaped like the real ones (`AgentSession.rawStreamResponse`,
    /// mlx-swift-lm `ChatSession.streamMap`): `makeStream()` plus a plain `Task {}` — which is what
    /// lets it inherit the guard's `@TaskLocal` handler.
    private final class FakeStreamingSession: ChatStreamingSession, @unchecked Sendable {
        let chunksBeforeError: Int
        let chunksAfterError: Int
        /// `nil` = a clean stream that never reports an MLX error (the control case).
        let errorAfter: Int?

        init(chunksBeforeError: Int, errorAfter: Int?, chunksAfterError: Int) {
            self.chunksBeforeError = chunksBeforeError
            self.errorAfter = errorAfter
            self.chunksAfterError = chunksAfterError
        }

        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
            let before = chunksBeforeError
            let after = chunksAfterError
            let tripError = errorAfter != nil
            let task = Task {
                for i in 0 ..< before {
                    continuation.yield("part-\(i)")
                    try? await Task.sleep(for: .milliseconds(2))
                }
                if tripError {
                    // The MLX error, mid-stream, exactly where an engine would raise one.
                    let a = MLXArray(0 ..< 10, [2, 5])
                    let b = MLXArray(0 ..< 15, [3, 5])
                    _ = a + b
                }
                for i in 0 ..< after {
                    continuation.yield("late-\(i)")
                    try? await Task.sleep(for: .milliseconds(2))
                }
                continuation.finish()
            }
            _ = task
            return stream
        }
    }

    private func makeRunner(session: FakeStreamingSession) throws -> (ModelRunner, URL) {
        let store = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-err-3-store-\(UUID().uuidString)")
        let modelDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-err-3-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let runner = ModelRunner()
        runner.prepare(for: entry, in: store)          // temp conversation store, not the real one
        runner.modelDirectoryForTesting = { _ in modelDir }
        runner.streamingSessionForTesting = { session }
        return (runner, store)
    }

    private func assistantText(_ runner: ModelRunner) -> String {
        runner.transcript.items.compactMap { item -> String? in
            guard case .message(let message) = item, message.role == .assistant else { return nil }
            return message.text
        }.joined()
    }

    private func waitUntil(_ timeout: Duration = .seconds(20), _ condition: () -> Bool) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while ContinuousClock.now < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    /// **The end-to-end test.** The turn stops at the error: what the assistant produced before it
    /// is kept, nothing after it arrives, and the sentence is visible — which is what the owner
    /// ruled ("the partial reply stays, and the turn ends with the error sentence visible").
    @Test func anMLXErrorMidStreamStopsTheTurnAndShowsTheSentence() async throws {
        let session = FakeStreamingSession(chunksBeforeError: 2, errorAfter: 2, chunksAfterError: 3)
        let (runner, store) = try makeRunner(session: session)
        defer { try? FileManager.default.removeItem(at: store) }

        await runner.send("hello", for: entry)
        let settled = await waitUntil { !runner.isRunning }
        #expect(settled, "the turn should settle rather than stay running")

        let shown = assistantText(runner)
        #expect(shown == "part-0part-1",
                "the partial reply must stay and nothing after the error may arrive; got '\(shown)'")
        #expect(!shown.contains("late-"),
                "chunks produced after the MLX error must never reach the transcript")

        let error = try #require(runner.errorMessage, "the turn must end with a visible error")
        #expect(error.contains("MLX"), "expected the guard's MLX sentence, got: \(error)")
        #expect(error.contains("runtime"), "expected the guard's voice, got: \(error)")
        #expect(runner.isRunning == false)

    }

    /// The control: a stream that ends normally still delivers every chunk, so the test above is
    /// measuring the error and not the fake.
    @Test func aCleanStreamStillDeliversEveryChunk() async throws {
        let session = FakeStreamingSession(chunksBeforeError: 3, errorAfter: nil, chunksAfterError: 0)
        let (runner, store) = try makeRunner(session: session)
        defer { try? FileManager.default.removeItem(at: store) }

        await runner.send("hello", for: entry)
        _ = await waitUntil { !runner.isRunning }

        #expect(assistantText(runner) == "part-0part-1part-2")
        #expect(runner.errorMessage == nil, "a clean turn must not report an error")
    }
}
