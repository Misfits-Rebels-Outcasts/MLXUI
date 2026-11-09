import Foundation
import Testing
@testable import MLXUI

/// MT2-5: the sheet's "Open in Chat" hand-over. Helpers are prefixed `MT25`. Serialized: the gate override is global.
@MainActor
@Suite(.serialized)
struct ChatOpenInChatTests {

    private let modelA = makeEntry(id: "model-a", displayName: "Alpha")
    private let modelB = makeEntry(id: "model-b", displayName: "Beta")

    private final class MT25Session: ChatStreamingSession, @unchecked Sendable {
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
            continuation.yield("streaming")
            return stream                                   // never finishes: a turn held in flight
        }
    }

    private func MT25Directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("mt25-\(UUID().uuidString)", isDirectory: true)
    }

    private func MT25Saved(_ dir: URL, model: String, title: String, at seconds: TimeInterval) -> Conversation {
        let c = Conversation(modelID: model, title: title, updatedAt: Date(timeIntervalSince1970: seconds),
                             items: [.message(ChatMessage(role: .user, text: title))])
        ConversationStore.save(c, in: dir)
        return c
    }

    private func MT25Wait(_ condition: () -> Bool) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while ContinuousClock.now < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    @Test func aBlankChatAndItsFoldersMoveToThePageWithoutAReload() {
        let dir = MT25Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let other = MT25Saved(dir, model: "model-b", title: "elsewhere", at: 1000)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.acquire(.bench, model: modelA)
        let blank = controller.activeConversationID
        controller.runner.attachFolder(path: "/tmp/mt25-folder")
        controller.setDraft(ChatDraft(text: "half typed"), for: .bench)
        #expect(controller.canOpenBenchInChat)

        #expect(controller.openBenchInChat(installedChatModels: [modelA, modelB]))
        #expect(controller.holder == .page)
        #expect(controller.activeConversationID == blank)                           // the same conversation
        #expect(controller.runner.activeAttachedFolders == ["/tmp/mt25-folder"])      // memory-only, so not reloaded
        #expect(controller.conversations.contains { $0.id == other.id })              // the page's list is every chat
        #expect(controller.conversations.contains { $0.id == blank })
        #expect(controller.draft(for: .page).text == "half typed")
        #expect(controller.draft(for: .bench) == ChatDraft())

        // the page binding that follows is a no-op, not a rebind
        #expect(controller.acquire(.page, model: modelA) == .granted)
        #expect(controller.activeConversationID == blank)
        #expect(controller.runner.activeAttachedFolders == ["/tmp/mt25-folder"])
    }

    @Test func aTurnStartedInTheSheetKeepsRunningOnThePage() async throws {
        let dir = MT25Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt25-m-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.runner.modelDirectoryForTesting = { _ in modelDir }
        controller.runner.streamingSessionForTesting = { MT25Session() }
        controller.acquire(.bench, model: modelA)
        await controller.send("hello", from: .bench)
        #expect(await MT25Wait { controller.runner.isRunning && controller.runner.transcript.items.count == 2 })
        #expect(controller.generationOwner == .bench)
        let active = controller.activeConversationID

        #expect(controller.openBenchInChat(installedChatModels: [modelA]))
        #expect(controller.runner.isRunning)
        #expect(controller.generationOwner == .page)                               // the lease passed with the turn
        #expect(controller.activeConversationID == active)

        // the sheet going away (explicitly): its stop and release leave the page's turn alone
        controller.stop(from: .bench)
        controller.release(.bench)
        #expect(controller.runner.isRunning)
        #expect(controller.holder == .page)
        #expect(controller.runner.transcript.items.count == 2)

        controller.stop(from: .page)
        #expect(!controller.runner.isRunning)
    }

    @Test func aClosedGateOffersNoButtonAndHandsNothingOver() {
        let dir = MT25Directory(); defer { ChatGate.overrideForTesting = nil }
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.acquire(.bench, model: modelA)
        ChatGate.overrideForTesting = false
        #expect(!controller.canOpenBenchInChat)
        #expect(!controller.openBenchInChat(installedChatModels: [modelA]))
        #expect(controller.holder == .bench)
        let app = AppState()
        #expect(!app.openBenchInChat())
        #expect(app.selectedSection == .home)
    }

    @Test func aWaitingSheetCannotOpenInChat() async throws {
        let dir = MT25Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt25-m-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.runner.modelDirectoryForTesting = { _ in modelDir }
        controller.runner.streamingSessionForTesting = { MT25Session() }
        controller.refreshPageModel(installedChatModels: [modelA])
        controller.acquire(.page)
        await controller.send("hello", from: .page)
        #expect(await MT25Wait { controller.runner.isRunning })
        #expect(controller.acquire(.bench, model: modelB) == .busy(holder: .page))
        #expect(!controller.canOpenBenchInChat)
        #expect(!controller.openBenchInChat(installedChatModels: [modelA, modelB]))
        controller.stopCurrentGeneration()
    }

    @Test func theAppSelectsTheChatItemAndThePageModelFollowsTheConversation() {
        let dir = MT25Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        _ = MT25Saved(dir, model: "model-b", title: "b chat", at: 1000)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.acquire(.bench, model: modelB)
        #expect(controller.openBenchInChat(installedChatModels: [modelA, modelB]))
        #expect(controller.model(for: .page)?.id == "model-b")                      // the conversation's model

        // not installed any more: the page keeps what it had
        let dir2 = MT25Directory(); defer { try? FileManager.default.removeItem(at: dir2) }
        _ = MT25Saved(dir2, model: "model-b", title: "b chat", at: 1000)
        let c2 = ChatController(runner: ModelRunner(), conversationDirectory: dir2)
        c2.refreshPageModel(installedChatModels: [modelA])
        c2.acquire(.bench, model: modelB)
        #expect(c2.openBenchInChat(installedChatModels: [modelA]))
        #expect(c2.model(for: .page)?.id == "model-a")

        // the AppState entry selects the Chat item
        let app = AppState()
        ChatGate.overrideForTesting = true; defer { ChatGate.overrideForTesting = nil }
        app.chat.acquire(.bench, model: modelA)
        #expect(app.openBenchInChat())
        #expect(app.selectedSection == .chat)
        #expect(app.chat.holder == .page)
    }

    @Test func theSheetsOwnListStaysFilteredUntilItIsHandedOver() {
        let dir = MT25Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let a = MT25Saved(dir, model: "model-a", title: "a", at: 1000)
        _ = MT25Saved(dir, model: "model-b", title: "b", at: 2000)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.acquire(.bench, model: modelA)
        #expect(controller.conversations.map(\.id) == [a.id])
        controller.openBenchInChat(installedChatModels: [modelA, modelB])
        #expect(controller.conversations.count == 2)
    }
}
