import Testing
import Foundation
@testable import MLXUI

// S3-3c — the `previous_response_id` store on `POST /v1/responses`.

private let model = "mlx-community/Qwen3-4B-4bit"

private func environment(_ backend: ServeBackend?, store: ResponseStore = ResponseStore()) -> ServeEnvironment {
    ServeEnvironment(
        servedModelIDs: { [model] },
        installed: { InstalledModelIndex(entries: [
            .init(id: "mlx-community--Qwen3-4B-4bit", hfModelId: model, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
        appleFoundationReadiness: { nil },
        created: 1_700_000_000,
        backendFor: { $0 == model ? backend : nil },
        queue: ServeQueue(), requestLog: RequestLog(), responseStore: store)
}

private func head(_ body: Data) -> ServeRequestHead {
    ServeRequestHead(method: "POST", path: "/v1/responses", host: "127.0.0.1:1212", origin: nil, transferEncoding: nil,
                     contentLength: String(body.count), userAgent: "codex/test")
}

private func send(_ fields: String, env: ServeEnvironment) async -> ServeResponse? {
    let body = Data(#"{"model":"\#(model)",\#(fields)}"#.utf8)
    if case .full(let response) = await ServeAPI.handle(head: head(body), guard: RequestGuard(port: 1212),
                                                         environment: env, body: { body }) { return response }
    return nil
}

/// A streamed request: the whole event text.
private func stream(_ fields: String, env: ServeEnvironment) async -> String {
    let body = Data(#"{"model":"\#(model)",\#(fields),"stream":true}"#.utf8)
    guard case .stream(_, _, let bytes) = await ServeAPI.handle(head: head(body), guard: RequestGuard(port: 1212),
                                                                 environment: env, body: { body }) else { return "" }
    var all: [UInt8] = []
    for await chunk in bytes { all += chunk }
    return String(decoding: all, as: UTF8.self)
}

private func object(_ data: Data) -> [String: Any]? { (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] }

private func responseID(_ response: ServeResponse?) -> String? { response.flatMap { object($0.body)?["id"] as? String } }

private func streamedID(_ text: String) -> String? {
    text.range(of: #""id":"resp_[0-9a-f]+""#, options: .regularExpression).map {
        String(text[$0]).replacingOccurrences(of: #""id":""#, with: "").replacingOccurrences(of: "\"", with: "")
    }
}

private func reply(_ text: String) -> MessagesBackend {
    MessagesBackend(events: [.textDelta(text), .usage(ServeUsage(promptTokens: 10, completionTokens: 2)), .finish(.stop)])
}

// MARK: - The store on its own

struct ResponseStoreTests {
    private final class Clock: @unchecked Sendable {
        var now = Date(timeIntervalSince1970: 1_000_000)
    }
    private func conversation(_ text: String = "hi") -> [ServeMessage] { [ServeMessage(role: .user, text: text)] }

    @Test func aSavedConversationComesBackAndAnUnknownIdDoesNot() {
        let store = ResponseStore()
        store.save(id: "resp_a", model: "m", conversation: conversation("one"))
        #expect(store.conversation(for: "resp_a") == conversation("one"))
        #expect(store.conversation(for: "resp_nope") == nil)
    }

    @Test func anEntryExpiresAfterTheTTL() {
        let clock = Clock()
        let store = ResponseStore(ttl: 60, now: { clock.now })
        store.save(id: "resp_a", model: "m", conversation: conversation())
        clock.now += 59
        #expect(store.conversation(for: "resp_a") != nil)
        clock.now += 2
        #expect(store.conversation(for: "resp_a") == nil)
        #expect(store.count == 0)
    }

    @Test func theEntryCapEvictsTheLeastRecentlyUsed() {
        let store = ResponseStore(maxEntries: 3)
        for id in ["a", "b", "c"] { store.save(id: id, model: "m", conversation: conversation(id)) }
        _ = store.conversation(for: "a")                                   // a is now the most recent; b the least
        store.save(id: "d", model: "m", conversation: conversation("d"))
        #expect(store.count == 3)
        #expect(store.conversation(for: "b") == nil)
        #expect(store.conversation(for: "a") != nil && store.conversation(for: "c") != nil && store.conversation(for: "d") != nil)
    }

    @Test func theByteCapEvictsTheLeastRecentlyUsedAndRefusesAnOversizedConversation() {
        let store = ResponseStore(maxBytes: 100)
        store.save(id: "a", model: "m", conversation: conversation(String(repeating: "x", count: 40)))
        store.save(id: "b", model: "m", conversation: conversation(String(repeating: "y", count: 40)))
        _ = store.conversation(for: "a")
        store.save(id: "c", model: "m", conversation: conversation(String(repeating: "z", count: 40)))     // 120 > 100
        #expect(store.conversation(for: "b") == nil)                                // the least recent went
        #expect(store.totalBytesHeld == 80)
        #expect(store.save(id: "huge", model: "m", conversation: conversation(String(repeating: "h", count: 101))) == false)
        #expect(store.conversation(for: "huge") == nil && store.conversation(for: "a") != nil)   // nothing else was flushed for it
    }

    @Test func evictingAModelRemovesOnlyItsResponses() {
        let store = ResponseStore()
        store.save(id: "a", model: "m1", conversation: conversation())
        store.save(id: "b", model: "m2", conversation: conversation())
        #expect(store.evict(model: "m1") == 1)
        #expect(store.conversation(for: "a") == nil && store.conversation(for: "b") != nil)
        store.removeAll()
        #expect(store.count == 0)
    }
}

// MARK: - Through the route

struct ResponsesPreviousResponseTests {

    /// The whole point: turn 2 sends only the new message and the id, and the model is shown turn 1.
    @Test func aTwoTurnRoundTripRemembersTheFirstTurn() async throws {
        let store = ResponseStore()
        let first = reply("Nice to meet you, Ann.")
        let turn1 = await send(#""input":"My name is Ann.""#, env: environment(first, store: store))
        let id = try #require(responseID(turn1))
        #expect(store.count == 1)

        let second = reply("Your name is Ann.")
        let turn2 = await send(#""input":"What is my name?","previous_response_id":"\#(id)""#, env: environment(second, store: store))
        #expect(turn2?.status == 200)
        #expect(second.generated.first?.messages == [
            ServeMessage(role: .user, text: "My name is Ann."),
            ServeMessage(role: .assistant, text: "Nice to meet you, Ann."),
            ServeMessage(role: .user, text: "What is my name?")])
        // …and turn 2 is itself stored, so a third turn sees all of it
        let id2 = try #require(responseID(turn2))
        let third = reply("ok")
        _ = await send(#""input":"Thanks","previous_response_id":"\#(id2)""#, env: environment(third, store: store))
        #expect(third.generated.first?.messages.map(\.text) == [
            "My name is Ann.", "Nice to meet you, Ann.", "What is my name?", "Your name is Ann.", "Thanks"])
    }

    @Test func aStreamedResponseIsStoredToo() async throws {
        let store = ResponseStore()
        let first = reply("Hello Ann")
        let text = await stream(#""input":"I am Ann""#, env: environment(first, store: store))
        let id = try #require(streamedID(text))
        #expect(text.contains("response.completed") && store.count == 1)

        let second = reply("Ann")
        _ = await send(#""input":"Who am I?","previous_response_id":"\#(id)""#, env: environment(second, store: store))
        #expect(second.generated.first?.messages == [
            ServeMessage(role: .user, text: "I am Ann"), ServeMessage(role: .assistant, text: "Hello Ann"),
            ServeMessage(role: .user, text: "Who am I?")])
        // a continuation can itself stream
        let third = reply("x")
        let streamed = await stream(#""input":"again","previous_response_id":"\#(id)""#, env: environment(third, store: store))
        #expect(streamed.contains("response.completed") && third.generated.first?.messages.count == 3)
    }

    @Test func storeFalseSavesNothing() async throws {
        let store = ResponseStore()
        let response = await send(#""input":"secret","store":false"#, env: environment(reply("ok"), store: store))
        let id = try #require(responseID(response))
        #expect(store.count == 0)
        let streamed = await stream(#""input":"secret","store":false"#, env: environment(reply("ok"), store: store))
        #expect(streamed.contains("response.completed") && store.count == 0)
        let again = await send(#""input":"more","previous_response_id":"\#(id)""#, env: environment(reply("ok"), store: store))
        #expect(again?.status == 400)
    }

    @Test func anUnknownOrExpiredIdIsTheNotFoundErrorNeverAFreshStart() async throws {
        let backend = reply("ok")
        let response = try #require(await send(#""input":"hi","previous_response_id":"resp_missing""#,
                                               env: environment(backend)))
        #expect(response.status == 400)
        let error = try #require(object(response.body)?["error"] as? [String: Any])
        #expect(error["message"] as? String == "Previous response with id 'resp_missing' not found.")
        #expect(error["code"] as? String == "previous_response_not_found" && error["type"] as? String == "invalid_request_error")
        #expect(backend.generated.isEmpty && backend.counted.isEmpty)               // nothing ran
        // an expired id is the same error
        final class Clock: @unchecked Sendable { var now = Date(timeIntervalSince1970: 5_000_000) }
        let clock = Clock()
        let store = ResponseStore(ttl: 30 * 60, now: { clock.now })
        let id = try #require(responseID(await send(#""input":"hi""#, env: environment(reply("ok"), store: store))))
        clock.now += 31 * 60
        let expired = await send(#""input":"hi","previous_response_id":"\#(id)""#, env: environment(reply("ok"), store: store))
        #expect(expired?.status == 400)
    }

    @Test func aFunctionCallContinuesWithOnlyItsOutput() async throws {
        let store = ResponseStore()
        let tools = #","tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"}}]"#
        let first = MessagesBackend(events: [
            .toolCallDelta(ServeToolCallDelta(index: 0, id: "call_9", name: "get_weather", argumentsFragment: #"{"city":"Oslo"}"#)),
            .usage(ServeUsage(promptTokens: 4, completionTokens: 6)), .finish(.toolCalls)])
        let id = try #require(responseID(await send(#""input":"weather in Oslo?""# + tools, env: environment(first, store: store))))

        let second = reply("3C and clear")
        let turn2 = await send(#""input":[{"type":"function_call_output","call_id":"call_9","output":"3C"}],"previous_response_id":"\#(id)""# + tools,
                               env: environment(second, store: store))
        #expect(turn2?.status == 200)
        #expect(second.generated.first?.messages == [
            ServeMessage(role: .user, text: "weather in Oslo?"),
            ServeMessage(role: .assistant, text: "", toolCalls: [ServeToolCall(id: "call_9", name: "get_weather", argumentsJSON: #"{"city":"Oslo"}"#)]),
            ServeMessage(role: .tool, text: "3C", toolCallID: "call_9")])
    }

    /// OpenAI: a continued request's `instructions` replace the earlier ones, which are not carried over; earlier *input*
    /// items (a Codex-style system/developer message among them) are.
    @Test func newInstructionsReplaceTheOldOnesAndAreNotStored() async throws {
        let store = ResponseStore()
        let id = try #require(responseID(await send(#""instructions":"Old rules.","input":"hi""#,
                                                    env: environment(reply("hello"), store: store))))
        let withNew = reply("ok")
        let turn2 = await send(#""instructions":"New rules.","input":"again","previous_response_id":"\#(id)""#,
                               env: environment(withNew, store: store))
        #expect(withNew.generated.first?.messages == [
            ServeMessage(role: .system, text: "New rules."), ServeMessage(role: .user, text: "hi"),
            ServeMessage(role: .assistant, text: "hello"), ServeMessage(role: .user, text: "again")])
        let without = reply("ok")
        _ = await send(#""input":"third","previous_response_id":"\#(try #require(responseID(turn2)))""#, env: environment(without, store: store))
        #expect(without.generated.first?.messages.contains { $0.text.contains("rules") } == false)
        // input-level system/developer messages ARE part of the conversation
        let storeB = ResponseStore()
        let idB = try #require(responseID(await send(
            #""input":[{"role":"developer","content":"Base."},{"role":"user","content":"hi"}]"#, env: environment(reply("hello"), store: storeB))))
        let continued = reply("ok")
        _ = await send(#""input":"more","previous_response_id":"\#(idB)""#, env: environment(continued, store: storeB))
        #expect(continued.generated.first?.messages.first == ServeMessage(role: .system, text: "Base."))
    }

    @Test func aConversationStoredForAnotherModelIsStillContinued() async throws {
        let store = ResponseStore()
        store.save(id: "resp_other", model: "some/other-model",
                   conversation: [ServeMessage(role: .user, text: "earlier"), ServeMessage(role: .assistant, text: "answer")])
        let backend = reply("ok")
        let response = await send(#""input":"next","previous_response_id":"resp_other""#, env: environment(backend, store: store))
        #expect(response?.status == 200)
        #expect(backend.generated.first?.messages.map(\.text) == ["earlier", "answer", "next"])
    }

    @Test func aFailedResponseSavesNothing() async throws {
        let store = ResponseStore()
        let failing = MessagesBackend(events: [.textDelta("par"), .error(ServeError(status: 500, message: "boom", code: "boom"))])
        let streamed = await stream(#""input":"hi""#, env: environment(failing, store: store))
        #expect(!streamed.contains("response.completed") && store.count == 0)
        _ = await send(#""input":"hi""#, env: environment(MessagesBackend(events: [.error(ServeError(status: 500, message: "boom", code: "boom"))]),
                                                       store: store))
        #expect(store.count == 0)
    }

    /// The prepended conversation is part of the prompt the pipeline counts and admits (S3-5a): the request the backend
    /// is asked to count carries it, and a prompt the machine cannot hold is refused with the stored turns in it.
    @Test func thePrependedConversationIsWhatAdmissionSees() async throws {
        let store = ResponseStore()
        let id = try #require(responseID(await send(#""input":"one""#, env: environment(reply("two"), store: store))))
        let backend = MessagesBackend(count: 39_492, contextLimit: 40_960)
        backend.memoryProfile = ServeMemoryAdmission.profile(
            configData: Data(#"{"model_type":"qwen3","num_hidden_layers":36,"num_key_value_heads":8,"head_dim":128,"max_position_embeddings":40960}"#.utf8),
            weightsBytes: 4_600_000_000, roomBytes: Int64(10.8 * 1_073_741_824))
        let response = try #require(await send(#""input":"three","previous_response_id":"\#(id)""#, env: environment(backend, store: store)))
        #expect(backend.counted.last?.messages.map(\.text) == ["one", "two", "three"])       // the count sees the whole conversation
        #expect(response.status == 400)
        #expect((object(response.body)?["error"] as? [String: Any])?["code"] as? String == "context_too_large_for_memory")
        #expect(backend.generated.isEmpty)
    }
}

// MARK: - Clearing

@MainActor
struct ResponseStoreClearingTests {
    @Test func stoppingTheServerClearsTheStore() async {
        let store = ResponseStore()
        store.save(id: "resp_a", model: model, conversation: [ServeMessage(role: .user, text: "hi")])
        let server = LocalServer(defaults: nil, listenPort: 0, environment: environment(nil, store: store))
        await server.stop()
        #expect(store.count == 0)
    }

    @Test func turningServeOffForAModelClearsItsResponsesButNotOthers() async {
        let store = ResponseStore()
        store.save(id: "resp_a", model: model, conversation: [ServeMessage(role: .user, text: "hi")])
        store.save(id: "resp_b", model: "mlx-community/other", conversation: [ServeMessage(role: .user, text: "hi")])
        let server = LocalServer(settings: ServeSettings().serving(model, true), defaults: nil, listenPort: 0,
                                 environment: environment(nil, store: store))
        let defaults = UserDefaults(suiteName: "resp-store-\(UUID().uuidString)")!
        let controller = LocalServerController(server: server, defaults: defaults,
                                               installed: { InstalledModelIndex(entries: []) },
                                               appleFoundation: { nil }, catalogLoaded: { true },
                                               evict: { _ in }, evictPromptCache: { _ in })
        await controller.setServed(model, false)
        #expect(store.conversation(for: "resp_a") == nil)
        #expect(store.conversation(for: "resp_b") != nil)
    }
}
