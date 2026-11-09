import Foundation
import Testing
@testable import MLXUI

/// MT2-3: the unified conversation list, the per-turn model, and "load once, refresh on change". Helpers `MT23*`.
@MainActor
@Suite(.serialized)
struct ChatUnifiedConversationsTests {

    private let modelA = makeEntry(id: "model-a", displayName: "Alpha")
    private let modelB = makeEntry(id: "model-b", displayName: "Beta")

    private final class MT23Counter: @unchecked Sendable {
        private let lock = NSLock(); private var n = 0
        func hit() { lock.withLock { n += 1 } }
        var count: Int { lock.withLock { n } }
    }

    private final class MT23OpenSession: ChatStreamingSession, @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: AsyncThrowingStream<String, Error>.Continuation?
        let chunks: [String]
        init(chunks: [String] = ["hi"]) { self.chunks = chunks }
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
            lock.withLock { self.continuation = continuation }
            for chunk in chunks { continuation.yield(chunk) }
            continuation.finish()
            return stream
        }
    }

    private func MT23Directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("mt23-\(UUID().uuidString)", isDirectory: true)
    }

    private func MT23Saved(_ dir: URL, model: String, title: String, at seconds: TimeInterval) -> Conversation {
        let c = Conversation(modelID: model, title: title, updatedAt: Date(timeIntervalSince1970: seconds),
                             items: [.message(ChatMessage(role: .user, text: title))])
        ConversationStore.save(c, in: dir)
        return c
    }

    private func MT23Wait(_ condition: () -> Bool) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while ContinuousClock.now < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    private func MT23Counting(_ dir: URL) -> (ChatController, MT23Counter) {
        let counter = MT23Counter()
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir,
                                        readAllConversations: { counter.hit(); return ConversationStore.allConversations(in: $0) })
        return (controller, counter)
    }

    // MARK: - files: old builds read new, new read old

    private let encoder: JSONEncoder = { let e = JSONEncoder(); e.dateEncodingStrategy = .iso8601; return e }()
    private let decoder: JSONDecoder = { let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601; return d }()

    @Test func aMainWrittenConversationAndAssistantMessageDecode() throws {
        let id = UUID()
        let json = """
        {"id":"\(id.uuidString)","modelID":"model-a","title":"old chat","createdAt":"2025-01-01T00:00:00Z",
         "updatedAt":"2025-06-01T00:00:00Z","items":[
          {"message":{"_0":{"id":"\(UUID().uuidString)","role":"user","text":"hi"}}},
          {"message":{"_0":{"id":"\(UUID().uuidString)","role":"assistant","text":"hello"}}}]}
        """
        let c = try decoder.decode(Conversation.self, from: Data(json.utf8))
        #expect(c.modelID == "model-a")
        #expect(c.attachedFolders.isEmpty)
        guard case .message(let reply) = c.items[1] else { Issue.record("not a message"); return }
        #expect(reply.role == .assistant)
        #expect(reply.modelID == nil)
    }

    @Test func modelIDRoundTripsWithAndWithout() throws {
        let stamped = ChatMessage(role: .assistant, text: "x", modelID: "model-b")
        let plain = ChatMessage(role: .assistant, text: "y")
        #expect(try decoder.decode(ChatMessage.self, from: encoder.encode(stamped)) == stamped)
        #expect(try decoder.decode(ChatMessage.self, from: encoder.encode(plain)) == plain)
        // the key is only written when set, so an unstamped message is byte-compatible with older builds
        let plainJSON = String(decoding: try encoder.encode(plain), as: UTF8.self)
        #expect(!plainJSON.contains("modelID"))
        #expect(String(decoding: try encoder.encode(stamped), as: UTF8.self).contains("\"modelID\""))
        // and Conversation.modelID is still a required, always-encoded key
        let c = Conversation(modelID: "model-a", items: [.message(stamped)])
        let data = try encoder.encode(c)
        #expect(String(decoding: data, as: UTF8.self).contains("\"modelID\":\"model-a\""))
        let back = try decoder.decode(Conversation.self, from: data)      // dates are second-precise on disk
        #expect(back.id == c.id && back.modelID == c.modelID && back.items == c.items)
    }

    @Test func aMessageWithUnknownExtraKeysStillDecodes() throws {
        // an older build reading a newer file is the mirror of this: keys are only ever added
        let json = #"{"id":"\#(UUID().uuidString)","role":"assistant","text":"t","modelID":"m","futureKey":1}"#
        #expect(try decoder.decode(ChatMessage.self, from: Data(json.utf8)).modelID == "m")
    }

    // MARK: - the turn stamps the model

    @Test func aTurnStampsTheAssistantBubbleAndTheConversationsLastModel() async throws {
        let dir = MT23Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt23-m-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.runner.modelDirectoryForTesting = { _ in modelDir }
        controller.runner.streamingSessionForTesting = { MT23OpenSession(chunks: ["one ", "two"]) }
        controller.refreshPageModel(installedChatModels: [modelA, modelB])
        controller.acquire(.page)
        let id = try #require(controller.model(for: .page)).id
        await controller.send("hello", from: .page)
        #expect(await MT23Wait { !controller.runner.isRunning && controller.runner.transcript.items.count == 2 })
        guard case .message(let reply) = controller.runner.transcript.items[1] else { Issue.record("no reply"); return }
        #expect(reply.text == "one two")
        #expect(reply.modelID == id)
        #expect(controller.conversations.first?.modelID == id)
        // saved: the file carries both stamps
        let onDisk = try #require(ConversationStore.allConversations(in: dir).first)
        #expect(onDisk.modelID == id)
        guard case .message(let saved) = onDisk.items[1] else { Issue.record("not saved"); return }
        #expect(saved.modelID == id)
    }

    @Test func extendingAnOpenBubbleKeepsItsStamp() {
        var builder = TranscriptBuilder()
        builder.appendAssistantChunk("a", modelID: "model-a")
        builder.appendAssistantChunk("b", modelID: "model-b")        // same bubble: not restamped
        guard case .message(let m) = builder.items[0] else { Issue.record("no bubble"); return }
        #expect(m.text == "ab")
        #expect(m.modelID == "model-a")
    }

    // MARK: - lists

    @Test func theBenchListStaysFilteredAndThePageListIsEveryChatNewestFirst() {
        let dir = MT23Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let a1 = MT23Saved(dir, model: "model-a", title: "a1", at: 1000)
        let b1 = MT23Saved(dir, model: "model-b", title: "b1", at: 3000)
        let a2 = MT23Saved(dir, model: "model-a", title: "a2", at: 2000)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.acquire(.bench, model: modelA)
        #expect(controller.conversations.map(\.id) == [a2.id, a1.id])
        controller.release(.bench)
        controller.acquire(.page)
        #expect(controller.conversations.map(\.id) == [b1.id, a2.id, a1.id])
    }

    @Test func theBadgeIsThePlainModelNameAndSaysWhenItIsNotInstalled() {
        let names = ["model-a": "Qwen3.5 9B"]
        let installed = ChatConversationBadge.make(modelID: "model-a", catalog: names, installedIDs: ["model-a"])
        #expect(installed.text == "Qwen3.5 9B")
        #expect(installed.installed)
        let missing = ChatConversationBadge.make(modelID: "model-a", catalog: names, installedIDs: [])
        #expect(missing.text == "Qwen3.5 9B · not installed")
        let unknown = ChatConversationBadge.make(modelID: "mlx-community--Gone-4bit", catalog: names, installedIDs: [])
        #expect(unknown.text == "Gone-4bit · not installed")
    }

    @Test func mergeKeepsLiveOverDiskAndBlankChats() {
        let live = Conversation(modelID: "model-a", title: "live", updatedAt: Date(timeIntervalSince1970: 500))
        var diskCopy = live; diskCopy.title = "stale"
        let other = Conversation(modelID: "model-b", title: "other", updatedAt: Date(timeIntervalSince1970: 900))
        let merged = ChatConversationList.merge(disk: [diskCopy, other], live: [live])
        #expect(merged.map(\.title) == ["other", "live"])
    }

    // MARK: - load once, refresh on change (owner ruling 2026-11-09)

    @Test func aSidebarClickAndAPageAppearReadNoFilesAfterTheFirstLoad() {
        let dir = MT23Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        _ = MT23Saved(dir, model: "model-a", title: "a", at: 1000)
        _ = MT23Saved(dir, model: "model-b", title: "b", at: 2000)
        let (controller, counter) = MT23Counting(dir)

        controller.refreshPageModel(installedChatModels: [modelA, modelB])      // first sidebar click
        #expect(counter.count == 1)
        controller.acquire(.page)                                               // page appears
        #expect(counter.count == 1)                                             // served from the cache

        for _ in 0..<5 {                                                        // click, appear, install change, again
            controller.refreshPageModel(installedChatModels: [modelA, modelB])
            controller.acquire(.page)
            controller.refreshPageModel(installedChatModels: [modelB])
        }
        #expect(counter.count == 1)
        #expect(controller.conversations.count == 2)
    }

    @Test func aChangeOnTheSheetRefreshesOnceOnTheNextPageAppear() {
        let dir = MT23Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        _ = MT23Saved(dir, model: "model-a", title: "a", at: 1000)
        let (controller, counter) = MT23Counting(dir)
        controller.refreshPageModel(installedChatModels: [modelA])
        controller.acquire(.page)
        #expect(counter.count == 1)

        controller.acquire(.bench, model: modelB)                               // the sheet opens and saves a chat
        let fresh = MT23Saved(dir, model: "model-b", title: "fresh", at: 5000)
        controller.release(.bench)
        controller.acquire(.page)
        #expect(counter.count == 2)                                             // one read
        #expect(controller.conversations.map(\.id).contains(fresh.id))
        controller.acquire(.page)
        controller.refreshPageModel(installedChatModels: [modelA, modelB])
        #expect(counter.count == 2)                                             // and no more
    }

    @Test func deletingAChatDropsTheCacheSoTheNextLoadIsFresh() {
        let dir = MT23Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let a = MT23Saved(dir, model: "model-a", title: "a", at: 1000)
        _ = MT23Saved(dir, model: "model-b", title: "b", at: 2000)
        let (controller, counter) = MT23Counting(dir)
        controller.acquire(.page)
        controller.delete(a.id, on: .page)
        #expect(!controller.conversations.contains { $0.id == a.id })
        controller.release(.page)
        controller.acquire(.bench, model: modelB)
        controller.release(.bench)
        controller.refreshPageModel(installedChatModels: [modelB])
        #expect(counter.count >= 2)                                             // re-read after the change, not stale
        #expect(!ConversationStore.allConversations(in: dir).contains { $0.id == a.id })
    }
}
