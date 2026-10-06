import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

// S2-2 — the wire side of tool calls: decoding a client's `tools` / `tool_choice` / tool history,
// and encoding `tool_calls` back out. Route behaviour lives in `ServeChatTests` (where the route
// helpers are) and the streaming framer in `ServeToolCallScannerTests`.

private func decode(_ json: String) throws -> ServeRequest {
    try OpenAIChatDialect.decode(Data(json.utf8))
}

private func rejection(_ json: String) -> ServeError? {
    do { _ = try decode(json); return nil } catch { return error as? ServeError }
}

private let weatherToolJSON = #"""
{"type":"function","function":{"name":"get_weather","description":"Get the weather",
 "parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}
"""#

/// The full round trip a harness replays: two calls, three results (one with an explicit `name`,
/// one with an unknown id, one with a text-part array), then a closing assistant turn.
private let toolConversation = #"""
{"model":"m","messages":[
  {"role":"user","content":"Read a.txt"},
  {"role":"assistant","content":null,"tool_calls":[
    {"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}},
    {"id":"call_2","type":"function","function":{"name":"list_dir","arguments":"{}"}}]},
  {"role":"tool","tool_call_id":"call_1","content":"hello"},
  {"role":"tool","tool_call_id":"call_2","name":"explicit","content":[{"type":"text","text":"x"},{"type":"text","text":"y"}]},
  {"role":"tool","tool_call_id":"unknown","content":"z"},
  {"role":"assistant","content":"Done.","tool_calls":[{"id":"c3","type":"function","function":{"name":"f"}}]}
]}
"""#

struct ServeToolCallsDecodeTests {

    // MARK: - tools

    @Test func toolsAreDecodedVerbatim() throws {
        let request = try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[\#(weatherToolJSON)]}"#)
        #expect(request.toolsRequested)
        #expect(request.tools.count == 1)
        let tool = try #require(request.tools.first)
        #expect(tool.name == "get_weather")
        // Kept verbatim: the template renders this JSON and the parser reads the schema from it, so
        // a key this app doesn't model must still reach both.
        let object = try #require(tool.object)
        let function = try #require(object["function"] as? [String: any Sendable])
        #expect(function["description"] as? String == "Get the weather")
        #expect(tool.rawJSON.contains("\"required\":[\"location\"]"))
    }

    @Test func noToolsMeansNothingToHonour() throws {
        let request = try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#)
        #expect(!request.toolsRequested && request.tools.isEmpty && request.toolChoice == .auto)
    }

    @Test func aNonFunctionToolTypeIs400() {
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[{"type":"code_interpreter"}]}"#)?.status == 400)
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{}}]}"#)?.status == 400)
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tools":"nope"}"#)?.status == 400)
    }

    // MARK: - tool_choice

    @Test func toolChoiceIsDecoded() throws {
        func choice(_ value: String) throws -> ServeToolChoice {
            try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tool_choice":\#(value)}"#).toolChoice
        }
        #expect(try choice(#""auto""#) == .auto)
        #expect(try choice(#""none""#) == .none)
        #expect(try choice(#""required""#) == .required)
        #expect(try choice(#"{"type":"function","function":{"name":"get_weather"}}"#) == .function("get_weather"))
        #expect(try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#).toolChoice == .auto)
    }

    @Test func aBadToolChoiceIs400() {
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tool_choice":"sometimes"}"#)?.status == 400)
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"function","function":{}}}"#)?.status == 400)
        #expect(rejection(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tool_choice":7}"#)?.status == 400)
    }

    /// What each choice renders (S2-2 requirement 7). `none` renders nothing — which is also what
    /// stops the backend scanning for calls.
    @Test func renderedToolsFollowTheChoice() throws {
        let tools = #"[\#(weatherToolJSON),{"type":"function","function":{"name":"other"}}]"#
        func request(_ choice: String) throws -> ServeRequest {
            try decode(#"{"model":"m","messages":[{"role":"user","content":"x"}],"tools":\#(tools),"tool_choice":\#(choice)}"#)
        }
        #expect(try request(#""auto""#).renderedTools.map(\.name) == ["get_weather", "other"])
        #expect(try request(#""required""#).renderedTools.map(\.name) == ["get_weather", "other"])
        #expect(try request(#""none""#).renderedTools.isEmpty)
        #expect(try request(#"{"type":"function","function":{"name":"other"}}"#).renderedTools.map(\.name) == ["other"])
        // A name the client didn't send is best effort: every tool stays.
        #expect(try request(#"{"type":"function","function":{"name":"missing"}}"#).renderedTools.count == 2)
    }

    // MARK: - history

    /// Tool traffic stays structured (S2-2 requirement 3) — the model that can use tools gets the
    /// client's own turns, not a flattened approximation.
    @Test func toolHistoryStaysStructured() throws {
        let request = try decode(toolConversation)
        #expect(request.messages.map(\.role) == [.user, .assistant, .tool, .tool, .tool, .assistant])
        // `require` rather than `expect`: the assertions below index into the list, and a mutation
        // that merges the tool turns away must fail the test cleanly, not crash the run.
        try #require(request.messages.count == 6)

        let calls = request.messages[1].toolCalls
        #expect(calls.map(\.name) == ["read_file", "list_dir"])
        #expect(calls.map(\.id) == ["call_1", "call_2"])
        #expect(calls[0].argumentsJSON == #"{"path":"a.txt"}"#)
        #expect(calls[1].argumentsJSON == "{}")                       // absent arguments → {}

        // Two consecutive tool results are two results, not one glued turn.
        #expect(request.messages[2].toolCallID == "call_1")
        #expect(request.messages[3].toolCallID == "call_2")
        #expect(request.messages[3].toolName == "explicit")
        #expect(request.messages[3].text == "xy")                     // text parts joined
        #expect(request.messages[4].toolCallID == "unknown")
        #expect(request.messages[5].text == "Done.")
        #expect(request.messages[5].toolCalls.map(\.name) == ["f"])
    }

    /// An assistant turn carrying calls is never merged into another assistant turn — the calls
    /// would be lost, since merging only joins text.
    @Test func anAssistantTurnWithCallsIsNotMerged() throws {
        let request = try decode(#"""
        {"model":"m","messages":[
          {"role":"assistant","content":"thinking"},
          {"role":"assistant","tool_calls":[{"id":"a","function":{"name":"f","arguments":"{}"}}]},
          {"role":"assistant","content":"more"}]}
        """#)
        try #require(request.messages.count == 3)
        #expect(request.messages[1].toolCalls.count == 1)
    }

    @Test func argumentsSentAsAnObjectBecomeAJSONString() throws {
        let request = try decode(#"{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"a","function":{"name":"f","arguments":{"b":2,"a":1}}}]}]}"#)
        // Sorted so the same call from two clients compares equal; the wire format carries a string.
        #expect(request.messages[0].toolCalls[0].argumentsJSON == #"{"a":1,"b":2}"#)
    }

    @Test func aMalformedToolCallIs400() {
        #expect(rejection(#"{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"a","function":{}}]}]}"#)?.status == 400)
        #expect(rejection(#"{"model":"m","messages":[{"role":"assistant","tool_calls":"nope"}]}"#)?.status == 400)
        #expect(rejection(#"{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"a","function":{"name":"f","arguments":7}}]}]}"#)?.status == 400)
    }
}

struct ServeToolCallsEncodeTests {

    private func object(_ data: Data) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// The non-streaming shape (S2-2 requirement 5):
    /// `message.tool_calls[] {id, type: "function", function{name, arguments}}` with `arguments` a
    /// JSON **string**, and `finish_reason: "tool_calls"`.
    @Test func nonStreamingBodyCarriesToolCalls() throws {
        let body = OpenAIChatDialect.completionBody(
            id: "chatcmpl-1", created: 7, model: "m", text: "",
            toolCalls: [ServeToolCall(id: "call_1", name: "get_weather", argumentsJSON: #"{"location":"Paris"}"#)],
            finish: .toolCalls, usage: ServeUsage(promptTokens: 3, completionTokens: 4))
        let root = try object(body)
        let choice = try #require((root["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "tool_calls")
        let message = try #require(choice["message"] as? [String: Any])
        #expect(message["role"] as? String == "assistant")
        #expect(message["content"] as? String == "")
        let calls = try #require(message["tool_calls"] as? [[String: Any]])
        #expect(calls.count == 1)
        #expect(calls[0]["id"] as? String == "call_1")
        #expect(calls[0]["type"] as? String == "function")
        let function = try #require(calls[0]["function"] as? [String: Any])
        #expect(function["name"] as? String == "get_weather")
        #expect(function["arguments"] as? String == #"{"location":"Paris"}"#)
    }

    /// A turn with no calls keeps S1's body exactly: no `tool_calls` key at all.
    @Test func aPlainBodyHasNoToolCallsKey() throws {
        let body = OpenAIChatDialect.completionBody(id: "i", created: 1, model: "m", text: "hi",
                                                    finish: .stop, usage: ServeUsage(promptTokens: 1, completionTokens: 1))
        let message = try #require(((try object(body))["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])
        #expect(message["tool_calls"] == nil)
    }

    /// The streaming fragment: `index`, `id`, `type` and `function.name` on the first fragment,
    /// `function.arguments` alongside it (the parsers hand back a finished call).
    @Test func streamingToolCallChunkCarriesIndexIdAndArguments() throws {
        let chunk = OpenAIChatDialect.toolCallChunk(
            id: "chatcmpl-1", created: 7, model: "m",
            call: ServeToolCallDelta(index: 1, id: "call_2", name: "get_weather", argumentsFragment: #"{"location":"Paris"}"#),
            includeUsage: false)
        let root = try object(chunk)
        #expect(root["object"] as? String == "chat.completion.chunk")
        let choice = try #require((root["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] == nil || choice["finish_reason"] is NSNull)
        let delta = try #require(choice["delta"] as? [String: Any])
        let calls = try #require(delta["tool_calls"] as? [[String: Any]])
        #expect(calls[0]["index"] as? Int == 1)
        #expect(calls[0]["id"] as? String == "call_2")
        #expect(calls[0]["type"] as? String == "function")
        let function = try #require(calls[0]["function"] as? [String: Any])
        #expect(function["name"] as? String == "get_weather")
        #expect(function["arguments"] as? String == #"{"location":"Paris"}"#)
    }

    @Test func generatedCallIDsLookLikeOpenAIs() {
        let id = OpenAIChatDialect.makeToolCallID()
        #expect(id.hasPrefix("call_"))
        #expect(id.count == "call_".count + 24)
    }
}
