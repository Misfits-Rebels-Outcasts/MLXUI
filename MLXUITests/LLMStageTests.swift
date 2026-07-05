import Testing
import Foundation
@testable import MLXUI

/// Covers M3 `LLMStage`: pure prompt composition and stage behavior with a mock
/// generator (no model load). See RSI/evals/eval-plan.md G2.
struct LLMStageTests {

    // MARK: LLMPrompt.compose

    @Test func composePrependsSystemPrompt() {
        #expect(LLMPrompt.compose(systemPrompt: "Summarize.", userText: "Hello") == "Summarize.\n\nHello")
    }

    @Test func composeReturnsUserTextWhenNoSystemPrompt() {
        #expect(LLMPrompt.compose(systemPrompt: nil, userText: "Hello") == "Hello")
    }

    @Test func composeTreatsBlankSystemPromptAsAbsent() {
        #expect(LLMPrompt.compose(systemPrompt: "   \n ", userText: "Hello") == "Hello")
    }

    // MARK: stage contract

    @Test func stageAcceptsAndProducesText() {
        let stage = LLMStage(id: "m", name: "Mock", systemPrompt: nil) { $0 }
        #expect(stage.accepts == .text)
        #expect(stage.produces == .text)
    }

    // MARK: run

    @Test func runComposesPromptAndReturnsGeneratedText() async throws {
        // The mock echoes the prompt it receives, so we can assert composition reached it.
        let stage = LLMStage(id: "m", name: "Mock", systemPrompt: "Summarize.") { prompt in
            "OUT<<\(prompt)>>"
        }
        let out = try await stage.run(.text("hi")) { _ in }
        guard case let .text(s) = out else {
            Issue.record("expected .text output")
            return
        }
        #expect(s == "OUT<<Summarize.\n\nhi>>")
    }

    @Test func runReportsProgressToCompletion() async throws {
        let stage = LLMStage(id: "m", name: "Mock", systemPrompt: nil) { $0 }
        let box = ProgressBox()
        _ = try await stage.run(.text("hi")) { box.record($0) }
        #expect(box.last == 1.0)
    }

    @Test func runRejectsNonTextInput() async {
        let stage = LLMStage(id: "m", name: "Mock", systemPrompt: nil) { $0 }
        await #expect(throws: StageError.self) {
            _ = try await stage.run(.audio(AudioBuffer(samples: [], sampleRate: 16_000))) { _ in }
        }
    }
}

/// Collects progress values from the `@Sendable` callback without data races.
private final class ProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []
    func record(_ v: Double) { lock.lock(); values.append(v); lock.unlock() }
    var last: Double? { lock.lock(); defer { lock.unlock() }; return values.last }
}
