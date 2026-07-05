import Testing
@testable import MLXUI

/// Covers M6 Tools_Text: `TemplatePrompt`/`TemplatePromptStage` and the `TextChunker`
/// sentence/chunk utilities. See RSI/evals/eval-plan.md G2.
struct ToolsTextTests {

    // MARK: TemplatePrompt.apply

    @Test func applySubstitutesPlaceholder() {
        #expect(TemplatePrompt.apply(template: "Summarize:\n{input}", input: "hello")
                == "Summarize:\nhello")
    }

    @Test func applyReplacesEveryPlaceholderOccurrence() {
        #expect(TemplatePrompt.apply(template: "{input} / {input}", input: "x") == "x / x")
    }

    @Test func applyAppendsWhenNoPlaceholder() {
        #expect(TemplatePrompt.apply(template: "Summarize the following:", input: "hello")
                == "Summarize the following:\n\nhello")
    }

    @Test func applyWithEmptyTemplateReturnsInput() {
        #expect(TemplatePrompt.apply(template: "   ", input: "hello") == "hello")
    }

    // MARK: TemplatePromptStage

    @Test func stageTemplatesUpstreamText() async throws {
        let stage = TemplatePromptStage(template: "TL;DR:\n{input}")
        let out = try await stage.run(.text("a long transcript")) { _ in }
        guard case let .text(s) = out else { Issue.record("expected text"); return }
        #expect(s == "TL;DR:\na long transcript")
    }

    @Test func stageRejectsNonText() async {
        let stage = TemplatePromptStage(template: "{input}")
        await #expect(throws: StageError.self) {
            _ = try await stage.run(.audio(AudioBuffer(samples: [], sampleRate: 16_000))) { _ in }
        }
    }

    @Test func stageDeclaresTextToText() {
        let stage = TemplatePromptStage(template: "{input}")
        #expect(stage.accepts == .text)
        #expect(stage.produces == .text)
    }

    // MARK: TextChunker

    @Test func sentencesSplitsOnSentenceBoundaries() {
        let s = TextChunker.sentences("Hello world. How are you? I am fine.")
        #expect(s == ["Hello world.", "How are you?", "I am fine."])
    }

    @Test func sentencesIgnoresBlankInput() {
        #expect(TextChunker.sentences("   \n ").isEmpty)
    }

    @Test func chunksGroupsSentencesWithinLimit() {
        // "Hello world." (12) + " " + "How are you?" (12) = 25 > 20 → two chunks.
        let chunks = TextChunker.chunks("Hello world. How are you?", maxCharacters: 20)
        #expect(chunks == ["Hello world.", "How are you?"])
    }

    @Test func chunksKeepsSentencesTogetherUnderLimit() {
        let chunks = TextChunker.chunks("Hello world. How are you?", maxCharacters: 100)
        #expect(chunks == ["Hello world. How are you?"])
    }
}
