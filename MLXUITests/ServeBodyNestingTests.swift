import Foundation
import Testing

@testable import MLXUI

// S-FIX-2 (ruling AZ) — the request guard's cap on raw JSON nesting, before any dialect parses the body.

struct ServeBodyNestingTests {
    private static let paths = ["/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens", "/v1/responses"]

    /// An otherwise ordinary body whose `x` field holds `levels - 1` further levels (the outer object is level 1).
    private static func body(levels: Int, path: String, object: Bool = true) -> Data {
        let inner = max(0, levels - 1)
        let deep = object ? String(repeating: #"{"a":"#, count: inner) + "0" + String(repeating: "}", count: inner)
            : String(repeating: "[", count: inner) + String(repeating: "]", count: inner)
        let text = path == "/v1/responses" ? #"{"model":"m","input":"hi","x":\#(deep)}"#
            : path == "/v1/messages" || path == "/v1/messages/count_tokens"
            ? #"{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"x":\#(deep)}"#
            : #"{"model":"m","messages":[{"role":"user","content":"hi"}],"x":\#(deep)}"#
        return Data(text.utf8)
    }

    private static func send(_ data: Data, to path: String, log: RequestLog = RequestLog()) async -> (status: Int, body: String) {
        let head = ServeRequestHead(method: "POST", path: path, host: "127.0.0.1:1212", origin: nil, transferEncoding: nil,
                                    contentLength: String(data.count), userAgent: "test/1.0")
        let environment = ServeEnvironment(servedModelIDs: { [] }, installed: { InstalledModelIndex(entries: []) },
                                           appleFoundationReadiness: { nil }, created: 0, requestLog: log)
        let reply = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: environment, body: { data })
        guard case .full(let response) = reply else { return (-1, "") }
        return (response.status, String(decoding: response.body, as: UTF8.self))
    }

    // MARK: the scan

    @Test func theScanCountsTheDeepestNestingAndStopsAsSoonAsItIsOverTheCap() {
        func nesting(_ text: String, limit: Int = 128) -> Int { RequestGuard.nesting(of: Data(text.utf8), stopAfter: limit) }
        #expect(nesting("{}") == 1)
        #expect(nesting(#"{"a":[{"b":[]}]}"#) == 4)
        #expect(nesting(String(repeating: "[", count: 10_000), limit: 128) == 129, "stops at the first level over the cap")
        #expect(nesting("[[]][[]]") == 2, "siblings don't add up")
    }

    @Test func bracketsInsideStringsAndEscapedQuotesDoNotCount() {
        func nesting(_ text: String) -> Int { RequestGuard.nesting(of: Data(text.utf8), stopAfter: 1000) }
        #expect(nesting(#"{"a":"[[[[[[[[[[ {{{{{{{{{{"}"#) == 1)
        #expect(nesting(#"{"a":"quote \" then [[[[[[[[[["}"#) == 1, "an escaped quote does not end the string")
        #expect(nesting(#"{"a":"backslash \\","b":[[[1]]]}"#) == 4, "an escaped backslash does: the string ended before `b`")
        #expect(nesting("{\"é😀\":[1]}") == 2, "multi-byte characters are neutral")
    }

    // MARK: the guard, on every endpoint

    @Test(arguments: paths) func exactlyTheCapPassesAndOneMoreIsRefused(path: String) async {
        let at = await Self.send(Self.body(levels: RequestGuard.maxNesting, path: path), to: path)
        #expect(!at.body.contains("nested more than"), "\(path): 128 levels must reach the route (status \(at.status))")
        let over = await Self.send(Self.body(levels: RequestGuard.maxNesting + 1, path: path), to: path)
        #expect(over.status == 400, "\(path)")
        #expect(over.body.contains("request JSON is nested more than 128 levels deep"), "\(path): \(over.body)")
        // Only the OpenAI shape carries a `code`; Anthropic's envelope is `{"type":"error","error":{"type","message"}}`.
        if !path.hasPrefix("/v1/messages") { #expect(over.body.contains("request_too_deeply_nested")) }
    }

    @Test(arguments: paths) func nestedArraysAreCountedToo(path: String) async {
        let over = await Self.send(Self.body(levels: 129, path: path, object: false), to: path)
        #expect(over.status == 400 && over.body.contains("nested more than 128 levels"), "\(path)")
    }

    @Test func eachDialectAnswersInItsOwnErrorShape() async {
        let openAI = await Self.send(Self.body(levels: 200, path: "/v1/chat/completions"), to: "/v1/chat/completions")
        #expect(openAI.body.hasPrefix(#"{"error":{"#) && !openAI.body.contains(#""type":"error""#), "\(openAI.body)")
        let responses = await Self.send(Self.body(levels: 200, path: "/v1/responses"), to: "/v1/responses")
        #expect(responses.body.hasPrefix(#"{"error":{"#), "\(responses.body)")
        for path in ["/v1/messages", "/v1/messages/count_tokens"] {
            let anthropic = await Self.send(Self.body(levels: 200, path: path), to: path)
            #expect(anthropic.body.contains(#""type":"error""#) && anthropic.body.contains("invalid_request_error"), "\(path): \(anthropic.body)")
        }
    }

    @Test func aHostileBodyDoesNotCrashTheHost() async {
        // The shape that killed the host at ~475 levels before the guard (see ServeDeepBodyReproTests).
        for path in Self.paths {
            let reply = await Task.detached { await Self.send(Self.body(levels: 50_000, path: path), to: path) }.value
            #expect(reply.status == 400, "\(path)")
        }
    }

    @Test func theRefusalIsLoggedAsMetadataOnly() async {
        let log = RequestLog()
        _ = await Self.send(Self.body(levels: 300, path: "/v1/chat/completions"), to: "/v1/chat/completions", log: log)
        let entry = log.entries.first
        #expect(log.entries.count == 1)
        #expect(entry?.status == 400 && entry?.model == nil && entry?.promptTokens == nil && entry?.completionTokens == nil)
        #expect(entry?.userAgent == "test/1.0")
    }

    // MARK: a real-sized request passes

    /// 60 tools shaped like Claude Code's (named, described, `input_schema` with nested properties / items / anyOf), ~180 KB.
    @Test func aClaudeCodeSizedRequestWithLargeToolSchemasPasses() async {
        func schema(depth: Int) -> String {
            depth == 0 ? #"{"type":"string","description":"leaf"}"#
                : #"{"type":"object","properties":{"item":{"type":"array","items":\#(schema(depth: depth - 1))},"mode":{"anyOf":[{"enum":["a","b"]},{"type":"null"}]},"note":{"type":"string","description":"free text, \#(String(repeating: "x", count: 200))"}},"required":["item"]}"#
        }
        let tools = (0 ..< 60).map { #"{"name":"tool_\#($0)","description":"\#(String(repeating: "does a thing ", count: 40))","input_schema":\#(schema(depth: 5))}"# }
        let data = Data(#"{"model":"m","max_tokens":64,"system":"You are a coding agent.","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[\#(tools.joined(separator: ","))]}"#.utf8)
        let deepest = RequestGuard.nesting(of: data, stopAfter: 1000)
        #expect(data.count > 150_000, "the fixture is realistically large (\(data.count) bytes)")
        #expect(deepest < RequestGuard.maxNesting, "its deepest nesting is \(deepest)")
        let reply = await Self.send(data, to: "/v1/messages")
        #expect(!reply.body.contains("nested more than"), "\(reply.body)")
    }
}
