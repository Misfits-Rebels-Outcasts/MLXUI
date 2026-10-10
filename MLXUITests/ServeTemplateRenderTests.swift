import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

// S3-5b part 1 — a chat-template render failure is a clear 400 (code `template_error`) before the head,
// never S2-6's fail-open and never a generic `generation_failed`.

private struct SomeJinjaFailure: Error {}
/// Matched by type *name*, like the real swift-jinja one (which this target can't import).
private struct TemplateException: Error { var message: String? }

struct ChatTemplateRenderFailureTests {
    @Test func classifiesRefusalsMissingTemplatesAndRenderFailures() throws {
        // a template-authored refusal keeps S2-8's sentence
        if case .refusal(let refusal) = ChatTemplateFailure.classify(TemplateException(message: "roles must alternate")) {
            #expect(refusal.code == "template_error" && refusal.message.contains("roles must alternate"))
        } else { Issue.record("a raise_exception refusal must stay a refusal") }
        // no chat template at all: "can't count", not a failure of the request
        guard case .missingTemplate = ChatTemplateFailure.classify(MLXLMCommon.TokenizerError.missingChatTemplate) else {
            Issue.record("a missing template is not a render failure"); return
        }
        // anything else the template threw while running (swift-jinja's conversion error, …)
        guard case .renderFailure = ChatTemplateFailure.classify(SomeJinjaFailure()) else {
            Issue.record("an unclassified render error must be a render failure"); return
        }
    }

    @Test func theSentenceNamesTheToolsInOnePlainSentence() {
        #expect(ChatTemplateFailure.renderFailure(toolNames: []).message == "This model's chat template couldn't render this request.")
        let one = ChatTemplateFailure.renderFailure(toolNames: ["create-pod"])
        #expect(one.status == 400 && one.code == "template_error")
        #expect(one.message == "This model's chat template couldn't render the tool 'create-pod': its schema contains something the template can't serialise.")
        let two = ChatTemplateFailure.renderFailure(toolNames: ["a", "b"])
        #expect(two.message == "This model's chat template couldn't render 2 of the tools ('a', 'b'): their schemas contain something the template can't serialise.")
        let many = ChatTemplateFailure.renderFailure(toolNames: ["a", "b", "c", "d", "e"])
        #expect(many.message.contains("5 of the tools ('a', 'b', 'c' and 2 more)"))
    }

    @Test func theCulpritSearchFindsTheToolsThatFailAlone() {
        let tools: [[String: any Sendable]] = ["ok", "bad1", "fine", "bad2"].map {
            ["type": "function", "function": ["name": $0] as [String: any Sendable]] as [String: any Sendable]
        }
        let names = ChatTemplateFailure.failingToolNames(in: tools) { tool in
            !(ChatTemplateFailure.toolName(tool) ?? "").hasPrefix("bad")
        }
        #expect(names == ["bad1", "bad2"])
        #expect(ChatTemplateFailure.failingToolNames(in: tools) { _ in true }.isEmpty)
        // a flat (Anthropic/Responses-style) spec still yields its name
        #expect(ChatTemplateFailure.toolName(["name": "flat"]) == "flat")
    }
}

/// A backend whose pre-count reports what `MLXChatBackend` now throws for a render failure.
private final class RenderFailingBackend: ServeBackend, @unchecked Sendable {
    let modelType: String? = "qwen3"
    var contextLimit: Int? = 40_960
    let error: ServeError?
    init(error: ServeError?) { self.error = error }
    func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> { AsyncThrowingStream { $0.finish() } }
    func countPromptTokens(_ request: ServeRequest) async throws -> Int? {
        if let error { throw error }
        return nil
    }
}

private let model = "mlx-community/Qwen3-8B-4bit"

private func send(_ json: String, path: String, backend: ServeBackend) async -> ServeResponse? {
    let env = ServeEnvironment(
        servedModelIDs: { [model] },
        installed: { InstalledModelIndex(entries: [
            .init(id: "mlx-community--Qwen3-8B-4bit", hfModelId: model, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
        appleFoundationReadiness: { nil }, created: 1, backendFor: { $0 == model ? backend : nil },
        queue: ServeQueue(), requestLog: RequestLog())
    let body = Data(json.utf8)
    let head = ServeRequestHead(method: "POST", path: path, host: "127.0.0.1:1212", origin: nil, transferEncoding: nil,
                                contentLength: String(body.count), userAgent: nil)
    if case .full(let response) = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: env,
                                                         body: { body }) { return response }
    return nil
}

private func object(_ data: Data) -> [String: Any]? { (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] }

struct RenderFailureRouteTests {
    private let failure = ChatTemplateFailure.renderFailure(toolNames: ["create-pod"])

    /// The same 400 before the head in every dialect, and for `count_tokens` (not a 501).
    @Test func everyDialectAndCountTokensAnswerTheClear400() async throws {
        let backend = RenderFailingBackend(error: failure)
        let chat = try #require(await send(#"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#,
                                           path: "/v1/chat/completions", backend: backend))
        #expect(chat.status == 400)
        let inner = try #require(object(chat.body)?["error"] as? [String: Any])
        #expect(inner["code"] as? String == "template_error" && (inner["message"] as? String)?.contains("'create-pod'") == true)

        let body = #"{"model":"\#(model)","max_tokens":9,"messages":[{"role":"user","content":"hi"}]}"#
        for streamFlag in ["", #","stream":true"#] {
            let messages = try #require(await send(String(body.dropLast()) + streamFlag + "}", path: "/v1/messages", backend: backend))
            #expect(messages.status == 400 && object(messages.body)?["type"] as? String == "error")
            #expect(((object(messages.body)?["error"] as? [String: Any])?["message"] as? String)?.contains("'create-pod'") == true)
        }
        let responses = try #require(await send(#"{"model":"\#(model)","input":"hi","stream":true}"#, path: "/v1/responses", backend: backend))
        #expect(responses.status == 400 && (object(responses.body)?["error"] as? [String: Any])?["code"] as? String == "template_error")

        let counted = try #require(await send(#"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#,
                                              path: "/v1/messages/count_tokens", backend: backend))
        #expect(counted.status == 400)                                          // was 501 while the render failure fell open
        #expect(((object(counted.body)?["error"] as? [String: Any])?["message"] as? String)?.contains("'create-pod'") == true)
    }

    /// 501 stays for "no tokenizer / no template": a nil count.
    @Test func noTokenizerIsStillA501OnCountTokens() async throws {
        let counted = try #require(await send(#"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#,
                                              path: "/v1/messages/count_tokens", backend: RenderFailingBackend(error: nil)))
        #expect(counted.status == 501)
    }
}
