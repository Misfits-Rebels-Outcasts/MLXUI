import Testing
import Foundation
@testable import MLXUI

/// M-demo integration: the full [Template → LLM → Save] chain runs end-to-end on the
/// real `Pipeline`, with a mock generator so no model is needed. Proves auto-chaining,
/// stage threading, and the file sink. See RSI/evals/eval-plan.md G2.
struct SummarizeDemoTests {

    @Test func demoChainTemplatesGeneratesAndSavesToFile() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("demo-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: url) }

        // Mock LLM echoes the prompt it received, so we can assert the template reached it.
        let mockLLM = LLMStage(id: "mock", name: "Mock LLM", systemPrompt: nil) { prompt in
            "SUMMARY of <<\(prompt)>>"
        }
        let pipeline = SummarizeDemo.pipeline(saveTo: url, llm: mockLLM)

        let result = try await pipeline.run(.text("A long transcript.")) { _ in }

        // The pipeline returns the saved text (Save passes its input through).
        guard case let .text(output) = result else {
            Issue.record("expected .text result")
            return
        }
        let expectedPrompt = "Summarize the following in 3 bullet points:\n\nA long transcript."
        #expect(output == "SUMMARY of <<\(expectedPrompt)>>")

        // And it actually landed on disk.
        let onDisk = try String(contentsOf: url, encoding: .utf8)
        #expect(onDisk == output)
    }

    @Test func demoChainValidatesAsTextThroughout() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("x.txt")
        let mockLLM = LLMStage(id: "mock", name: "Mock LLM", systemPrompt: nil) { $0 }
        // All three stages are text→text, so the chain is statically valid.
        try SummarizeDemo.pipeline(saveTo: url, llm: mockLLM).validate()
    }

    @Test func demoEmitsThreeStagesOfProgress() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("demo-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: url) }

        let mockLLM = LLMStage(id: "mock", name: "Mock LLM", systemPrompt: nil) { $0 }
        let starts = StartCounter()
        _ = try await SummarizeDemo.pipeline(saveTo: url, llm: mockLLM)
            .run(.text("hello")) { event in
                if case .stageStarted = event { starts.bump() }
            }
        #expect(starts.count == 3)   // Template, LLM, Save
    }
}

/// Thread-safe counter for the `@Sendable` event callback.
private final class StartCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func bump() { lock.lock(); n += 1; lock.unlock() }
    var count: Int { lock.lock(); defer { lock.unlock() }; return n }
}
