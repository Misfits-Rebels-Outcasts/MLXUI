import Foundation
import Testing
@testable import MLXUI

/// MT2-1: `ChatController` — the lease, the forwarding to the runner, and the drafts. Helpers are prefixed `MT21`.
/// The bench behaves as before: with one surface the state passes through unchanged.
@MainActor
@Suite(.serialized)
struct ChatControllerTests {

    private let entry = makeEntry(id: "model-a", displayName: "Model A")
    private let other = makeEntry(id: "model-b", displayName: "Model B")

    /// A stream that stays open until `finish()`, so a turn can be held in flight.
    private final class MT21OpenSession: ChatStreamingSession, @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: AsyncThrowingStream<String, Error>.Continuation?
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
            lock.withLock { self.continuation = continuation }
            continuation.yield("hi")
            return stream
        }
        func finish() { lock.withLock { continuation?.finish() } }
    }

    private func MT21Directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("mt21-\(UUID().uuidString)", isDirectory: true)
    }

    private func MT21Model() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mt21-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func MT21Wait(_ condition: () -> Bool) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while ContinuousClock.now < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    private func MT21Controller(_ directory: URL) -> ChatController {
        ChatController(runner: ModelRunner(), conversationDirectory: directory)
    }

    // MARK: - the bench passes state through unchanged

    @Test func acquiringTheBenchIsThePrepareCall() {
        let dir = MT21Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let saved = Conversation(modelID: "model-a", title: "old", items: [.message(ChatMessage(role: .user, text: "hello"))])
        ConversationStore.save(saved, in: dir)
        ConversationStore.save(Conversation(modelID: "model-b", title: "elsewhere",
                                            items: [.message(ChatMessage(role: .user, text: "x"))]), in: dir)

        let reference = ModelRunner()
        reference.prepare(for: entry, in: dir)

        let controller = MT21Controller(dir)
        #expect(controller.acquire(.bench, model: entry) == .granted)
        #expect(controller.holder == .bench)
        #expect(controller.conversations.map(\.id) == reference.conversations.map(\.id))
        #expect(controller.conversations.map(\.id) == [saved.id])
        #expect(controller.activeConversationID == reference.activeConversationID)
        #expect(controller.runner.transcript.items == reference.transcript.items)
        #expect(controller.model(for: .bench)?.id == "model-a")
    }

    @Test func selectAndNewConversationForwardOnlyForTheHolder() {
        let dir = MT21Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let first = Conversation(modelID: "model-a", title: "first", updatedAt: Date(timeIntervalSinceNow: -100),
                                 items: [.message(ChatMessage(role: .user, text: "1"))])
        let second = Conversation(modelID: "model-a", title: "second",
                                  items: [.message(ChatMessage(role: .user, text: "2"))])
        ConversationStore.save(first, in: dir); ConversationStore.save(second, in: dir)
        let controller = MT21Controller(dir)
        controller.acquire(.bench, model: entry)
        #expect(controller.activeConversationID == second.id)

        controller.select(first.id, on: .page)                 // not the holder: nothing happens
        #expect(controller.activeConversationID == second.id)
        controller.select(first.id, on: .bench)
        #expect(controller.activeConversationID == first.id)
        #expect(controller.runner.activeConversationID == first.id)

        let before = controller.conversations.count
        controller.newConversation(on: .bench)
        #expect(controller.conversations.count == before + 1)
        #expect(controller.activeConversationID != first.id)

        let victim = controller.activeConversationID
        controller.delete(second.id, on: .page)                // not the holder
        #expect(controller.conversations.contains { $0.id == second.id })
        controller.delete(second.id, on: .bench)
        #expect(!controller.conversations.contains { $0.id == second.id })
        #expect(controller.activeConversationID == victim)
    }

    // MARK: - generation

    @Test func sendFromANonHolderStartsNothing() async {
        let dir = MT21Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let controller = MT21Controller(dir)
        controller.acquire(.bench, model: entry)
        #expect(await controller.send("hello", from: .page) == true)
        #expect(!controller.runner.isRunning)
        #expect(controller.runner.transcript.items.isEmpty)
    }

    @Test func aRunningTurnBelongsToItsSurfaceAndOnlyThatSurfaceStopsIt() async throws {
        let dir = MT21Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let modelDir = try MT21Model(); defer { try? FileManager.default.removeItem(at: modelDir) }
        let session = MT21OpenSession()
        let controller = MT21Controller(dir)
        controller.runner.modelDirectoryForTesting = { _ in modelDir }
        controller.runner.streamingSessionForTesting = { session }
        controller.acquire(.bench, model: entry)

        await controller.send("hello", from: .bench)
        #expect(await MT21Wait { controller.runner.isRunning })
        #expect(controller.generationOwner == .bench)

        // The page may not take the runner while the bench generates; nothing is rebound.
        let active = controller.activeConversationID
        let items = controller.runner.transcript.items
        #expect(controller.acquire(.page) == .busy(holder: .bench))
        #expect(controller.holder == .bench)
        #expect(controller.activeConversationID == active)
        #expect(controller.runner.transcript.items == items)

        controller.stop(from: .page)                           // not the owner
        #expect(controller.runner.isRunning)
        controller.stop(from: .bench)
        #expect(!controller.runner.isRunning)
        #expect(controller.generationOwner == nil)
        session.finish()
    }

    @Test func aSecondSendWhileRunningIsStillANoOp() async throws {
        let dir = MT21Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let modelDir = try MT21Model(); defer { try? FileManager.default.removeItem(at: modelDir) }
        let session = MT21OpenSession()
        let controller = MT21Controller(dir)
        controller.runner.modelDirectoryForTesting = { _ in modelDir }
        controller.runner.streamingSessionForTesting = { session }
        controller.acquire(.bench, model: entry)
        await controller.send("one", from: .bench)
        #expect(await MT21Wait { controller.runner.isRunning })
        let count = controller.runner.transcript.items.count
        await controller.send("two", from: .bench)
        #expect(controller.runner.transcript.items.count == count)
        controller.stop(from: .bench)
        session.finish()
    }

    // MARK: - release and the page/bench swap

    @Test func aBlankChatWithFoldersSurvivesAPageAndBenchSwap() {
        let dir = MT21Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let controller = MT21Controller(dir)
        controller.acquire(.bench, model: entry)
        let blank = controller.activeConversationID
        controller.runner.attachFolder(path: "/tmp/mt21-folder")
        #expect(controller.runner.activeAttachedFolders == ["/tmp/mt21-folder"])

        #expect(controller.acquire(.page) == .granted)          // the bench's set is parked, not reloaded
        #expect(controller.holder == .page)
        #expect(controller.acquire(.bench, model: entry) == .granted)
        #expect(controller.activeConversationID == blank)
        #expect(controller.runner.activeAttachedFolders == ["/tmp/mt21-folder"])
    }

    @Test func releasingTheBenchClearsItsDraftAndTheHolder() {
        let dir = MT21Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let controller = MT21Controller(dir)
        controller.acquire(.bench, model: entry)
        controller.setDraft(ChatDraft(text: "half-typed"), for: .bench)
        controller.setDraft(ChatDraft(text: "page draft"), for: .page)
        controller.release(.bench)
        #expect(controller.holder == nil)
        #expect(controller.draft(for: .bench) == ChatDraft())
        #expect(controller.draft(for: .page).text == "page draft")
    }

    // MARK: - drafts

    @Test func draftsRoundTripPerSurface() {
        let controller = MT21Controller(MT21Directory())
        #expect(controller.draft(for: .bench) == ChatDraft())
        let file = URL(fileURLWithPath: "/tmp/a.txt")
        controller.setDraft(ChatDraft(text: "hello", files: [file]), for: .bench)
        #expect(controller.draft(for: .bench).text == "hello")
        #expect(controller.draft(for: .bench).files == [file])
        #expect(controller.draft(for: .page) == ChatDraft())
        controller.setDraft(ChatDraft(), for: .bench)
        #expect(controller.draft(for: .bench) == ChatDraft())
    }

    // MARK: - the store

    @Test func allConversationsListsEveryModelNewestFirstAndReadsAMainWrittenFile() throws {
        let dir = MT21Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        ConversationStore.save(Conversation(modelID: "model-a", title: "a", updatedAt: Date(timeIntervalSince1970: 1000),
                                            items: [.message(ChatMessage(role: .user, text: "a"))]), in: dir)
        ConversationStore.save(Conversation(modelID: "model-b", title: "b", updatedAt: Date(timeIntervalSince1970: 2_000_000_000),
                                            items: [.message(ChatMessage(role: .user, text: "b"))]), in: dir)
        // A file as `main` writes it: no attachedFolders, no workingFolder, no attachments.
        let id = UUID()
        let old = """
        {"id":"\(id.uuidString)","modelID":"model-c","title":"old","createdAt":"2025-01-01T00:00:00Z",
         "updatedAt":"2025-06-01T00:00:00Z","items":[{"message":{"_0":{"id":"\(UUID().uuidString)","role":"user","text":"hi"}}}]}
        """
        try Data(old.utf8).write(to: dir.appendingPathComponent("\(id.uuidString).json"))

        let all = ConversationStore.allConversations(in: dir)
        #expect(all.count == 3)
        #expect(all.contains { $0.id == id && $0.modelID == "model-c" && $0.attachedFolders.isEmpty })
        #expect(all.first?.title == "b")
    }

    @Test func prepareForModelIsTheScopedCall() {
        let dir = MT21Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        ConversationStore.save(Conversation(modelID: "model-a", title: "a",
                                            items: [.message(ChatMessage(role: .user, text: "a"))]), in: dir)
        ConversationStore.save(Conversation(modelID: "model-b", title: "b",
                                            items: [.message(ChatMessage(role: .user, text: "b"))]), in: dir)
        let byModel = ModelRunner(); byModel.prepare(for: entry, in: dir)
        let byScope = ModelRunner(); byScope.prepare(scope: .model("model-a"), in: dir)
        #expect(byModel.conversations == byScope.conversations)
        let all = ModelRunner(); all.prepare(scope: .all, in: dir)
        #expect(all.conversations.count == 2)
    }
}
