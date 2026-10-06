import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

// S2-3 — the reasoning *controls* (decode), the family table, and the wire shapes.

private func decode(_ json: String) throws -> ServeRequest {
    try OpenAIChatDialect.decode(Data(json.utf8))
}

private func rejection(_ json: String) -> ServeError? {
    do { _ = try decode(json); return nil } catch { return error as? ServeError }
}

private let hello = #"{"model":"m","messages":[{"role":"user","content":"x"}]}"#

struct ServeReasoningDecodeTests {

    // MARK: - reasoning_effort

    /// The documented set. `none`/`minimal` mean off, the rest mean on (S2-3 decision 1).
    @Test func reasoningEffortMapsToOnOrOff() throws {
        func effort(_ value: String) throws -> Bool? {
            try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":"\#(value)"}"#).reasoning
        }
        #expect(try effort("none") == false)
        #expect(try effort("minimal") == false)
        #expect(try effort("low") == true)
        #expect(try effort("medium") == true)
        #expect(try effort("high") == true)
        // Both are on the reference page; neither is a value the Qwen templates understand, which is
        // exactly why the string is not forwarded to them.
        #expect(try effort("xhigh") == true)
        #expect(try effort("max") == true)
    }

    @Test func aRequestWithNoControlAsksForNothing() throws {
        #expect(try decode(hello).reasoning == nil)
        #expect(try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":null}"#).reasoning == nil)
    }

    @Test func anUnknownReasoningEffortIs400() {
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":"lots"}"#)?.status == 400)
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":3}"#)?.status == 400)
    }

    // MARK: - chat_template_kwargs

    @Test func chatTemplateKwargsEnableThinkingIsRead() throws {
        #expect(try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"enable_thinking":true}}"#).reasoning == true)
        #expect(try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"enable_thinking":false}}"#).reasoning == false)
        // Another template's key is not ours to interpret — ignored, not an error.
        #expect(try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"add_vision_id":true}}"#).reasoning == nil)
    }

    /// Precedence: the explicit `enable_thinking` wins over `reasoning_effort` (decision 1).
    @Test func anExplicitEnableThinkingWins() throws {
        let withBoth = #"{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":false}}"#
        #expect(try decode(withBoth).reasoning == false)

        let theOtherWay = #"{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":true}}"#
        #expect(try decode(theOtherWay).reasoning == true)

        // `reasoning_effort` alone still works when the kwargs object doesn't mention it.
        let effortOnly = #"{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":"high","chat_template_kwargs":{"other":1}}"#
        #expect(try decode(effortOnly).reasoning == true)
    }

    @Test func malformedChatTemplateKwargsIs400() {
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"chat_template_kwargs":"yes"}"#)?.status == 400)
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"enable_thinking":"yes"}}"#)?.status == 400)
        // Wrong type *and* a bad effort: the effort is checked first, and both are 400s.
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":"lots","chat_template_kwargs":3}"#)?.status == 400)
    }

    // MARK: - history

    /// An incoming assistant `reasoning_content` never becomes `content` (decision 6) — it is
    /// accepted and dropped.
    @Test func incomingReasoningContentIsDropped() throws {
        let request = try decode(#"""
        {"model":"m","messages":[
          {"role":"assistant","content":"The answer is 4.","reasoning_content":"I added them."},
          {"role":"user","content":"thanks"}]}
        """#)
        #expect(request.messages[0].text == "The answer is 4.")
        #expect(!String(describing: request.messages[0]).contains("I added them"))
    }
}

struct ReasoningTemplateTests {

    /// The decision itself, in the one place both the route and the backend ask (decisions 1–2):
    /// off unless asked, and only for a family that reads the switch.
    @Test func theThinkingDecision() {
        #expect(!ReasoningTemplateTable.thinkingOn(requested: nil, forModelType: "qwen3"))
        #expect(!ReasoningTemplateTable.thinkingOn(requested: false, forModelType: "qwen3"))
        #expect(ReasoningTemplateTable.thinkingOn(requested: true, forModelType: "qwen3"))
        #expect(ReasoningTemplateTable.thinkingOn(requested: true, forModelType: "qwen3_5"))
        // Asked for, but the family has no thinking to give: ignored, not refused.
        #expect(!ReasoningTemplateTable.thinkingOn(requested: true, forModelType: "llama"))
        #expect(!ReasoningTemplateTable.thinkingOn(requested: true, forModelType: "gpt_oss"))
        #expect(!ReasoningTemplateTable.thinkingOn(requested: true, forModelType: ""))
    }

    /// The family table: only the families whose template reads `enable_thinking` — five of the
    /// twelve catalog chat models — can think. Everything else **ignores** the control (decision 2).
    @Test func onlyTheThinkingFamiliesReadEnableThinking() {
        for modelType in ["qwen3_5", "qwen3.5"] {
            #expect(ReasoningTemplateTable.thinkShape(forModelType: modelType) == .promptOpened)
            #expect(ReasoningTemplateTable.readsEnableThinking(forModelType: modelType))
        }
        #expect(ReasoningTemplateTable.thinkShape(forModelType: "qwen3") == .modelEmitted)
        #expect(ReasoningTemplateTable.readsEnableThinking(forModelType: "qwen3"))

        // The catalog's other seven: qwen2, llama, mistral3 (Ministral 3 / Devstral), gemma2,
        // gemma3_text — and gpt-oss, whose template reads `reasoning_effort` for harmony channels
        // but never `enable_thinking` (no harmony work in S2-3).
        for modelType in ["qwen2", "llama", "mistral3", "gemma2", "gemma3_text", "gpt_oss", "phi3", ""] {
            #expect(ReasoningTemplateTable.thinkShape(forModelType: modelType) == .none,
                    "\(modelType) has no think block")
            #expect(!ReasoningTemplateTable.readsEnableThinking(forModelType: modelType))
        }
    }
}

struct ServeReasoningEncodeTests {

    private func object(_ data: Data) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func message(_ body: Data) throws -> [String: Any] {
        try #require(((try object(body))["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])
    }

    /// Non-streaming: `message.reasoning_content` beside `content`, and absent when there was none.
    @Test func nonStreamingBodyCarriesReasoningContent() throws {
        let body = OpenAIChatDialect.completionBody(
            id: "i", created: 1, model: "m", text: "The answer is 4.", reasoning: "I added them.",
            finish: .stop, usage: ServeUsage(promptTokens: 3, completionTokens: 4))
        let withReasoning = try message(body)
        #expect(withReasoning["reasoning_content"] as? String == "I added them.")
        #expect(withReasoning["content"] as? String == "The answer is 4.")

        let plain = OpenAIChatDialect.completionBody(id: "i", created: 1, model: "m", text: "hi",
                                                     finish: .stop, usage: ServeUsage(promptTokens: 1, completionTokens: 1))
        #expect(try message(plain)["reasoning_content"] == nil)          // no empty placeholder
    }

    /// A turn that thought and then called a tool: reasoning plus calls, `content` null.
    @Test func reasoningAndCallsCoexist() throws {
        let body = OpenAIChatDialect.completionBody(
            id: "i", created: 1, model: "m", text: "", reasoning: "I should check the weather.",
            toolCalls: [ServeToolCall(id: "call_1", name: "get_weather", argumentsJSON: "{}")],
            finish: .toolCalls, usage: ServeUsage(promptTokens: 3, completionTokens: 9))
        let message = try message(body)
        #expect(message["reasoning_content"] as? String == "I should check the weather.")
        #expect(message["content"] is NSNull)
        #expect((message["tool_calls"] as? [[String: Any]])?.count == 1)
    }

    /// Streaming: `delta.reasoning_content`, with no `content` key alongside it.
    @Test func streamingReasoningChunkHasOnlyReasoningContent() throws {
        let chunk = OpenAIChatDialect.reasoningChunk(id: "i", created: 1, model: "m",
                                                     text: "I added them.", includeUsage: false)
        let delta = try #require(((try object(chunk))["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])
        #expect(delta["reasoning_content"] as? String == "I added them.")
        #expect(delta["content"] == nil)
        #expect(delta["tool_calls"] == nil)
    }
}
