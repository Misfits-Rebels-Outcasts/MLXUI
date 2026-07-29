import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG1 — the transcript reducer: folding streamed text chunks + `AgentToolEvent`s into an
/// ordered `[TranscriptItem]`. Pure (no model, no view), so the interleaving is tested directly.
struct ChatTranscriptTests {

    // helpers to read items without pattern-matching noise at every call site
    private func assistantText(_ item: TranscriptItem?) -> String? {
        if case .message(let m) = item, m.role == .assistant { return m.text }
        return nil
    }
    private func userText(_ item: TranscriptItem?) -> String? {
        if case .message(let m) = item, m.role == .user { return m.text }
        return nil
    }
    private func tool(_ item: TranscriptItem?) -> ToolActivity? {
        if case .tool(let t) = item { return t }
        return nil
    }

    @Test func chunksAccumulateIntoOneBubble() {
        var b = TranscriptBuilder()
        b.appendAssistantChunk("Hel")
        b.appendAssistantChunk("lo")
        b.appendAssistantChunk(" world")
        #expect(b.items.count == 1)
        #expect(assistantText(b.items.first) == "Hello world")
        #expect(b.lastAssistantText == "Hello world")
    }

    @Test func emptyChunkIsIgnored() {
        var b = TranscriptBuilder()
        b.appendAssistantChunk("")
        #expect(b.isEmpty)
    }

    @Test func userMessageClosesAssistantBubble() {
        var b = TranscriptBuilder()
        b.appendAssistantChunk("hi")
        b.addUserMessage("next question")
        b.appendAssistantChunk("answer")
        #expect(b.items.count == 3)
        #expect(assistantText(b.items[0]) == "hi")
        #expect(userText(b.items[1]) == "next question")
        #expect(assistantText(b.items[2]) == "answer")
    }

    @Test func toolStartedClosesBubbleAndSubsequentTextStartsNewBubble() {
        var b = TranscriptBuilder()
        b.appendAssistantChunk("let me check")
        b.apply(.started(name: "echo", arguments: ["text": .string("hi")]))
        b.appendAssistantChunk("the result is hi")
        #expect(b.items.count == 3)
        #expect(assistantText(b.items[0]) == "let me check")
        #expect(tool(b.items[1])?.name == "echo")
        #expect(tool(b.items[1])?.status == .running)
        #expect(assistantText(b.items[2]) == "the result is hi")
    }

    @Test func finishedUpdatesTheRunningCard() {
        var b = TranscriptBuilder()
        b.apply(.started(name: "echo", arguments: ["text": .string("hi")]))
        b.apply(.finished(name: "echo", result: "echo: hi"))
        #expect(b.items.count == 1)
        #expect(tool(b.items[0])?.status == .finished)
        #expect(tool(b.items[0])?.result == "echo: hi")
    }

    @Test func failedUpdatesTheRunningCard() {
        var b = TranscriptBuilder()
        b.apply(.started(name: "boom", arguments: [:]))
        b.apply(.failed(name: "boom", message: "kaboom"))
        #expect(b.items.count == 1)
        #expect(tool(b.items[0])?.status == .failed("kaboom"))
        #expect(tool(b.items[0])?.result == "kaboom")
    }

    @Test func terminalEventsWithoutStartAppendACard() {
        var b = TranscriptBuilder()
        b.apply(.denied(name: "gated"))
        b.apply(.unknownTool(name: "nope"))
        b.apply(.budgetExceeded(name: "echo"))
        #expect(b.items.count == 3)
        #expect(tool(b.items[0])?.status == .denied)
        #expect(tool(b.items[1])?.status == .unknownTool)
        #expect(tool(b.items[2])?.status == .budgetExceeded)
    }

    @Test func fullInterleavedTurn() {
        var b = TranscriptBuilder()
        b.addUserMessage("echo hi")
        b.appendAssistantChunk("sure, ")
        b.appendAssistantChunk("calling the tool")
        b.apply(.started(name: "echo", arguments: ["text": .string("hi")]))
        b.apply(.finished(name: "echo", result: "echo: hi"))
        b.appendAssistantChunk("it said hi")
        #expect(b.items.count == 4)
        #expect(userText(b.items[0]) == "echo hi")
        #expect(assistantText(b.items[1]) == "sure, calling the tool")
        #expect(tool(b.items[2])?.status == .finished)
        #expect(assistantText(b.items[3]) == "it said hi")
    }

    @Test func resetClearsEverything() {
        var b = TranscriptBuilder()
        b.addUserMessage("hi")
        b.appendAssistantChunk("hello")
        b.reset()
        #expect(b.isEmpty)
        #expect(b.lastAssistantText == nil)
    }
}
