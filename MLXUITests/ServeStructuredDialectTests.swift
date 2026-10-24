import Testing
import Foundation
@testable import MLXUI

// WW4-1-4 — `response_format` (Chat Completions) and `text.format` (Responses) on the wire: decoded into `ServeStructure`,
// answered with each dialect's own shapes, and refused with 400s that name the client's own field. No real model: a scripted
// backend stands in for generation, and `MLXChatBackend` over a directory holding only a `config.json` stands in for the checks
// that run before any weights or tokenizer load (allow-list, schema, tools + format). `/v1/messages` is not covered on purpose:
// it does not read a structured-output field (design note: a later item).

private let model = "mlx-community/Qwen3-4B-4bit"

private let taskSchema = #"{"type":"object","properties":{"title":{"type":"string"},"done":{"type":"boolean"}},"required":["title","done"]}"#
private let taskReply = #"{"title":"Tidy","done":false}"#

/// Records what reached it; `prepareStructure` accepts anything (the checks are tested against the real backend below).
private nonisolated final class ScriptedStructuredBackend: ServeBackend, @unchecked Sendable {
    private let lock = NSLock()
    private var _requests: [ServeRequest] = []
    let script: [ServeEvent]
    var modelType: String? = "qwen3"

    init(_ script: [ServeEvent]) { self.script = script }

    var requests: [ServeRequest] { lock.lock(); defer { lock.unlock() }; return _requests }

    func prepareStructure(_ request: ServeRequest) async throws -> ServeGrammarGrant? { nil }

    func generate(_ request: ServeRequest) -> AsyncThrowingStream<ServeEvent, Error> {
        lock.lock(); _requests.append(request); lock.unlock()
        let script = script
        return AsyncThrowingStream { continuation in
            for event in script { continuation.yield(event) }
            continuation.finish()
        }
    }
}

private func reply(_ text: String = taskReply, finish: ServeFinishReason = .stop) -> [ServeEvent] {
    [.textDelta(text), .usage(ServeUsage(promptTokens: 9, completionTokens: 7)), .finish(finish)]
}

private func environment(_ backend: ServeBackend?) -> ServeEnvironment {
    ServeEnvironment(
        servedModelIDs: { [model] },
        installed: { InstalledModelIndex(entries: [
            .init(id: "mlx-community--Qwen3-4B-4bit", hfModelId: model, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
        appleFoundationReadiness: { nil },
        created: 1_700_000_000,
        backendFor: { $0 == model ? backend : nil },
        queue: ServeQueue(), requestLog: RequestLog(), keepAliveSeconds: 1.0)
}

/// A directory with only a `config.json`: enough for `MLXChatBackend` to know its `model_type`, so the checks that run before
/// any weights or tokenizer load are the real ones.
private func configOnlyBackend(modelType: String) throws -> MLXChatBackend {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ww414-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try Data(#"{"model_type":"\#(modelType)"}"#.utf8).write(to: dir.appendingPathComponent("config.json"))
    return MLXChatBackend(directory: dir)
}

private struct Wire {
    var status: Int
    var text: String
    var json: [String: Any]? { (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] }
    var errorCode: String? { (json?["error"] as? [String: Any])?["code"] as? String }
    var errorMessage: String { ((json?["error"] as? [String: Any])?["message"] as? String) ?? "" }
    /// The `data:` payloads of an SSE body, parsed.
    var dataEvents: [[String: Any]] {
        text.components(separatedBy: "\n\n").compactMap { block in
            guard let line = block.components(separatedBy: "\n").first(where: { $0.hasPrefix("data: ") }) else { return nil }
            return (try? JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8))) as? [String: Any]
        }
    }
}

private func send(_ path: String, _ json: String, _ env: ServeEnvironment) async -> Wire {
    let body = Data(json.utf8)
    let head = ServeRequestHead(method: "POST", path: path, host: "127.0.0.1:1212", origin: nil, transferEncoding: nil,
                                contentLength: String(body.count), userAgent: "test/1.0")
    switch await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: env, body: { body }) {
    case .full(let response):
        return Wire(status: response.status, text: String(decoding: response.body, as: UTF8.self))
    case .stream(let status, _, let bytes):
        var all: [UInt8] = []
        for await chunk in bytes { all += chunk }
        return Wire(status: status, text: String(decoding: all, as: UTF8.self))
    }
}

private func chat(_ fields: String, _ env: ServeEnvironment) async -> Wire {
    await send("/v1/chat/completions", #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}],\#(fields)}"#, env)
}

private func responses(_ fields: String, _ env: ServeEnvironment) async -> Wire {
    await send("/v1/responses", #"{"model":"\#(model)","input":"hi",\#(fields)}"#, env)
}

private let chatSchemaFormat = #""response_format":{"type":"json_schema","json_schema":{"name":"task","schema":\#(taskSchema),"strict":true}}"#
private let responsesSchemaFormat = #""text":{"format":{"type":"json_schema","name":"task","schema":\#(taskSchema),"strict":true}}"#

// MARK: - Decode (pure)

struct ServeStructureDecodingTests {
    @Test func chatJsonSchemaCarriesNameSchemaAndStrict() throws {
        let structure = try ServeStructureDecoding.chatResponseFormat(
            ["type": "json_schema", "json_schema": ["name": "task", "strict": true, "schema": ["type": "object"]]])
        #expect(structure == .jsonSchema(name: "task", schema: #"{"type":"object"}"#, strict: true))
    }

    @Test func aMissingNameIsAccepted() throws {
        let nested = try ServeStructureDecoding.chatResponseFormat(["type": "json_schema", "json_schema": ["schema": ["type": "object"]]])
        #expect(nested == .jsonSchema(name: nil, schema: #"{"type":"object"}"#, strict: false))
        let flat = try ServeStructureDecoding.responsesText(["format": ["type": "json_schema", "schema": ["type": "object"]]])
        #expect(flat == .jsonSchema(name: nil, schema: #"{"type":"object"}"#, strict: false))
    }

    @Test func textAbsentAndNullAreNoOps() throws {
        #expect(try ServeStructureDecoding.chatResponseFormat(["type": "text"]) == .none)
        #expect(try ServeStructureDecoding.chatResponseFormat(nil) == .none)
        #expect(try ServeStructureDecoding.chatResponseFormat(NSNull()) == .none)
        #expect(try ServeStructureDecoding.responsesText(["format": ["type": "text"]]) == .none)
        #expect(try ServeStructureDecoding.responsesText(nil) == .none)
        #expect(try ServeStructureDecoding.responsesText(["format": NSNull()]) == .none)
        #expect(try ServeStructureDecoding.responsesText(["verbosity": "low"]) == .none, "the rest of `text` is still ignored")
        #expect(try ServeStructureDecoding.responsesText("not an object") == .none, "R38: a `text` that is not an object is ignored")
    }

    @Test func theSchemaIsHandedOnAsTheClientSentItWithSlashesIntact() throws {
        let structure = try ServeStructureDecoding.chatResponseFormat(
            ["type": "json_schema", "json_schema": ["schema": ["$ref": "#/$defs/a", "$defs": ["a": ["type": "string"]]]]])
        guard case .jsonSchema(_, let schema, _) = structure else { Issue.record("not a schema"); return }
        #expect(schema.contains("#/$defs/a"), "\(schema)")
    }
}

// MARK: - Chat Completions

struct ChatStructuredFormatTests {
    @Test func aSchemaReplyComesBackWithFinishReasonStop() async throws {
        let backend = ScriptedStructuredBackend(reply())
        let wire = await chat(chatSchemaFormat, environment(backend))
        #expect(wire.status == 200)
        let choice = try #require((wire.json?["choices"] as? [[String: Any]])?.first)
        #expect((choice["message"] as? [String: Any])?["content"] as? String == taskReply)
        #expect(choice["finish_reason"] as? String == "stop")
        let request = try #require(backend.requests.first)
        #expect(request.structure == .jsonSchema(name: "task", schema: #"{"properties":{"done":{"type":"boolean"},"title":{"type":"string"}},"required":["title","done"],"type":"object"}"#, strict: true))
        #expect(request.structureNames == .chatCompletions)
    }

    @Test func jsonObjectConstrainsToAnObject() async throws {
        let backend = ScriptedStructuredBackend(reply("{}"))
        let wire = await chat(#""response_format":{"type":"json_object"}"#, environment(backend))
        #expect(wire.status == 200)
        #expect(backend.requests.first?.structure == .jsonObject)
    }

    @Test func strictFalseIsStillEnforcedAndAMissingNameIsFine() async throws {
        let backend = ScriptedStructuredBackend(reply())
        let wire = await chat(#""response_format":{"type":"json_schema","json_schema":{"strict":false,"schema":\#(taskSchema)}}"#, environment(backend))
        #expect(wire.status == 200)
        let structure = try #require(backend.requests.first?.structure)
        #expect(structure.isConstrained, "strict: false is not best-effort: the grammar still constrains the reply")
        guard case .jsonSchema(let name, _, let strict) = structure else { Issue.record("not a schema"); return }
        #expect(name == nil && strict == false)
    }

    @Test func aLengthStopIsFinishReasonLength() async throws {
        let backend = ScriptedStructuredBackend(reply(#"{"title":"Ti"#, finish: .length))
        let wire = await chat(chatSchemaFormat, environment(backend))
        #expect(wire.status == 200)
        let choice = try #require((wire.json?["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "length")
    }

    @Test func aStreamedSchemaReplyEndsWithStopAndDone() async throws {
        let backend = ScriptedStructuredBackend(reply())
        let wire = await chat(chatSchemaFormat + #","stream":true"#, environment(backend))
        #expect(wire.status == 200)
        let chunks = wire.dataEvents
        let content = chunks.compactMap { (($0["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["content"] as? String }.joined()
        #expect(content == taskReply)
        let finishes = chunks.compactMap { ($0["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String }
        #expect(finishes == ["stop"])
        #expect(wire.text.hasSuffix("data: [DONE]\n\n"))
    }

    @Test func aStreamedLengthStopIsFinishReasonLength() async throws {
        let backend = ScriptedStructuredBackend(reply(#"{"title":"Ti"#, finish: .length))
        let wire = await chat(#""response_format":{"type":"json_object"},"stream":true"#, environment(backend))
        let finishes = wire.dataEvents.compactMap { ($0["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String }
        #expect(finishes == ["length"])
    }

    /// Ruling AO: `{"type":"text"}` and an absent field are the same request — also with tools on.
    @Test func textFormatAndAbsentAreIdenticalIncludingWithTools() async throws {
        let tools = #""tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]"#
        for fields in [tools, tools + #","response_format":{"type":"text"}"#, #""response_format":{"type":"text"}"#, #""x":1"#] {
            let backend = ScriptedStructuredBackend(reply("hello"))
            let wire = await chat(fields, environment(backend))
            #expect(wire.status == 200, "\(fields): \(wire.text)")
            let request = try #require(backend.requests.first)
            #expect(request.structure == .none, "\(fields)")
            #expect(request.tools.count == (fields.contains("tools") ? 1 : 0), "\(fields)")
        }
    }

    @Test func eachMalformedFormatIs400NamingItsPath() async throws {
        let cases: [(String, String)] = [
            (#""response_format":"json_object""#, "'response_format' must be an object"),
            (#""response_format":{}"#, "'response_format.type' is required"),
            (#""response_format":{"type":7}"#, "'response_format.type' is required"),
            (#""response_format":{"type":"xml"}"#, "'response_format.type' must be 'text', 'json_object' or 'json_schema'; got 'xml'"),
            (#""response_format":{"type":"json_schema"}"#, "'response_format.json_schema' is required"),
            (#""response_format":{"type":"json_schema","json_schema":{"name":"n"}}"#, "'response_format.json_schema.schema' is required"),
            (#""response_format":{"type":"json_schema","json_schema":{"schema":[1]}}"#, "'response_format.json_schema.schema' is required"),
            (#""response_format":{"type":"json_schema","json_schema":{"name":4,"schema":{}}}"#, "'response_format.json_schema.name' must be a string"),
            (#""response_format":{"type":"json_schema","json_schema":{"strict":"yes","schema":{}}}"#, "'response_format.json_schema.strict' must be true or false"),
        ]
        for (fields, message) in cases {
            let backend = ScriptedStructuredBackend(reply())
            let wire = await chat(fields, environment(backend))
            #expect(wire.status == 400 && wire.errorCode == "invalid_response_format", "\(fields) → \(wire.text)")
            #expect(wire.errorMessage.contains(message), "\(fields) → \(wire.errorMessage)")
            #expect(backend.requests.isEmpty, "nothing reached the backend")
        }
    }

    // The checks below run in the real `MLXChatBackend`, before any tokenizer or weights.

    @Test func aModelThatIsNotOnTheAllowListIsRefusedNamingIt() async throws {
        let wire = await chat(chatSchemaFormat, environment(try configOnlyBackend(modelType: "llama")))
        #expect(wire.status == 400 && wire.errorCode == "structured_output_unsupported", "\(wire.text)")
        #expect(wire.errorMessage.contains(model) && wire.errorMessage.contains("response_format"))
    }

    @Test func aBackendThatCannotConstrainIsRefusedNamingTheModel() async throws {
        // The protocol default — what Apple Foundation Models and every other unchecked backend gets.
        let wire = await chat(chatSchemaFormat, environment(FakeBackend([.text("x"), .finish(.stop)])))
        #expect(wire.status == 400 && wire.errorCode == "structured_output_unsupported", "\(wire.text)")
        #expect(wire.errorMessage.contains(model))
        let object = await chat(#""response_format":{"type":"json_object"}"#, environment(FakeBackend([.text("x"), .finish(.stop)])))
        #expect(object.status == 400 && object.errorCode == "structured_output_unsupported")
    }

    @Test func anUnusableSchemaIsRefusedWithTheRealParamPath() async throws {
        let schema = #"{"type":"object","properties":{"q":{"type":"integer","multipleOf":5}}}"#
        let wire = await chat(#""response_format":{"type":"json_schema","json_schema":{"name":"n","schema":\#(schema)}}"#,
                              environment(try configOnlyBackend(modelType: "qwen3")))
        #expect(wire.status == 400 && wire.errorCode == "invalid_json_schema", "\(wire.text)")
        #expect(wire.errorMessage.contains("response_format.json_schema.schema/properties/q/multipleOf"), "\(wire.errorMessage)")
    }

    @Test func toolsTogetherWithAJsonFormatAreRefusedNamingTheField() async throws {
        let tools = #""tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"#
        let wire = await chat(tools + chatSchemaFormat, environment(try configOnlyBackend(modelType: "qwen3")))
        #expect(wire.status == 400 && wire.errorCode == "unsupported_combination", "\(wire.text)")
        #expect(wire.errorMessage.contains("response_format"))
    }
}

// MARK: - Responses

struct ResponsesStructuredFormatTests {
    @Test func aSchemaReplyIsACompletedResponse() async throws {
        let backend = ScriptedStructuredBackend(reply())
        let wire = await responses(responsesSchemaFormat, environment(backend))
        #expect(wire.status == 200)
        #expect(wire.json?["status"] as? String == "completed")
        let message = try #require((wire.json?["output"] as? [[String: Any]])?.first)
        let text = ((message["content"] as? [[String: Any]])?.first)?["text"] as? String
        #expect(text == taskReply)
        let request = try #require(backend.requests.first)
        #expect(request.structure.isConstrained)
        #expect(request.structureNames == .responses)
        guard case .jsonSchema(let name, _, let strict) = request.structure else { Issue.record("not a schema"); return }
        #expect(name == "task" && strict)
    }

    @Test func jsonObjectConstrainsToAnObject() async throws {
        let backend = ScriptedStructuredBackend(reply("{}"))
        let wire = await responses(#""text":{"format":{"type":"json_object"}}"#, environment(backend))
        #expect(wire.status == 200)
        #expect(backend.requests.first?.structure == .jsonObject)
    }

    @Test func aMissingNameAndStrictFalseAreAccepted() async throws {
        let backend = ScriptedStructuredBackend(reply())
        let wire = await responses(#""text":{"verbosity":"low","format":{"type":"json_schema","strict":false,"schema":\#(taskSchema)}}"#,
                                   environment(backend))
        #expect(wire.status == 200)
        let structure = try #require(backend.requests.first?.structure)
        #expect(structure.isConstrained)
    }

    /// The Responses equivalent of `finish_reason: "length"` — what `OpenAIResponsesDialect` already emits for a length stop.
    @Test func aLengthStopIsIncompleteWithMaxOutputTokens() async throws {
        let backend = ScriptedStructuredBackend(reply(#"{"title":"Ti"#, finish: .length))
        let wire = await responses(responsesSchemaFormat, environment(backend))
        #expect(wire.status == 200)
        #expect(wire.json?["status"] as? String == "incomplete")
        #expect((wire.json?["incomplete_details"] as? [String: String])?["reason"] == "max_output_tokens")
    }

    @Test func aStreamedSchemaReplyEndsWithResponseCompleted() async throws {
        let backend = ScriptedStructuredBackend(reply())
        let wire = await responses(responsesSchemaFormat + #","stream":true"#, environment(backend))
        #expect(wire.status == 200)
        let types = wire.dataEvents.compactMap { $0["type"] as? String }
        #expect(types.last == "response.completed", "\(types)")
        let deltas = wire.dataEvents.filter { $0["type"] as? String == "response.output_text.delta" }.compactMap { $0["delta"] as? String }
        #expect(deltas.joined() == taskReply)
    }

    @Test func aStreamedLengthStopEndsWithResponseIncomplete() async throws {
        let backend = ScriptedStructuredBackend(reply(#"{"title":"Ti"#, finish: .length))
        let wire = await responses(#""text":{"format":{"type":"json_object"}},"stream":true"#, environment(backend))
        let last = try #require(wire.dataEvents.last)
        #expect(last["type"] as? String == "response.incomplete")
        let response = try #require(last["response"] as? [String: Any])
        #expect(response["status"] as? String == "incomplete")
        #expect((response["incomplete_details"] as? [String: String])?["reason"] == "max_output_tokens")
    }

    /// Ruling AO: `text.format` of type text — which Codex may send on every request — behaves exactly as today, tools or not.
    @Test func textFormatIsANoOpAlsoAlongsideTools() async throws {
        let tools = #""tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]"#
        for fields in [tools, tools + #","text":{"format":{"type":"text"}}"#, #""text":{"format":{"type":"text"}}"#,
                       #""text":{"verbosity":"low"}"#, #""x":1"#] {
            let backend = ScriptedStructuredBackend(reply("hello"))
            let wire = await responses(fields, environment(backend))
            #expect(wire.status == 200, "\(fields): \(wire.text)")
            let request = try #require(backend.requests.first)
            #expect(request.structure == .none, "\(fields)")
            #expect(request.tools.count == (fields.contains("tools") ? 1 : 0), "\(fields)")
        }
    }

    @Test func eachMalformedFormatIs400NamingItsPath() async throws {
        let cases: [(String, String)] = [
            (#""text":{"format":"json_object"}"#, "'text.format' must be an object"),
            (#""text":{"format":{}}"#, "'text.format.type' is required"),
            (#""text":{"format":{"type":"yaml"}}"#, "'text.format.type' must be 'text', 'json_object' or 'json_schema'; got 'yaml'"),
            (#""text":{"format":{"type":"json_schema","name":"n"}}"#, "'text.format.schema' is required"),
            (#""text":{"format":{"type":"json_schema","schema":"{}"}}"#, "'text.format.schema' is required"),
            (#""text":{"format":{"type":"json_schema","name":false,"schema":{}}}"#, "'text.format.name' must be a string"),
            (#""text":{"format":{"type":"json_schema","strict":1,"schema":{}}}"#, "'text.format.strict' must be true or false"),
        ]
        for (fields, message) in cases {
            let backend = ScriptedStructuredBackend(reply())
            let wire = await responses(fields, environment(backend))
            #expect(wire.status == 400 && wire.errorCode == "invalid_text_format", "\(fields) → \(wire.text)")
            #expect(wire.errorMessage.contains(message), "\(fields) → \(wire.errorMessage)")
            #expect(backend.requests.isEmpty)
        }
    }

    @Test func aModelThatIsNotOnTheAllowListIsRefusedNamingIt() async throws {
        let wire = await responses(responsesSchemaFormat, environment(try configOnlyBackend(modelType: "llama")))
        #expect(wire.status == 400 && wire.errorCode == "structured_output_unsupported", "\(wire.text)")
        #expect(wire.errorMessage.contains(model) && wire.errorMessage.contains("text.format"))
    }

    @Test func aBackendThatCannotConstrainIsRefusedNamingTheModel() async throws {
        let wire = await responses(responsesSchemaFormat, environment(FakeBackend([.text("x"), .finish(.stop)])))
        #expect(wire.status == 400 && wire.errorCode == "structured_output_unsupported", "\(wire.text)")
        #expect(wire.errorMessage.contains(model))
    }

    @Test func anUnusableSchemaIsRefusedWithTheRealParamPath() async throws {
        let schema = #"{"type":"object","properties":{"q":{"type":"integer","multipleOf":5}}}"#
        let wire = await responses(#""text":{"format":{"type":"json_schema","name":"n","schema":\#(schema)}}"#,
                                   environment(try configOnlyBackend(modelType: "qwen3")))
        #expect(wire.status == 400 && wire.errorCode == "invalid_json_schema", "\(wire.text)")
        #expect(wire.errorMessage.contains("text.format.schema/properties/q/multipleOf"), "\(wire.errorMessage)")
    }

    @Test func toolsTogetherWithAJsonFormatAreRefusedNamingTheField() async throws {
        let tools = #""tools":[{"type":"function","name":"f","parameters":{"type":"object"}}],"#
        let wire = await responses(tools + responsesSchemaFormat, environment(try configOnlyBackend(modelType: "qwen3")))
        #expect(wire.status == 400 && wire.errorCode == "unsupported_combination", "\(wire.text)")
        #expect(wire.errorMessage.contains("text.format"))
    }
}
