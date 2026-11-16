import Foundation
import Testing
@testable import MLXUI

/// MT2-2: `ChatGate`, the sidebar order, the page's model, and the lease between the page and the bench sheet
/// (controller level). Helpers are prefixed `MT22`. Serialized: the gate override is process-wide.
@MainActor
@Suite(.serialized)
struct ChatPageTests {

    private let modelA = makeEntry(id: "model-a", displayName: "Alpha")
    private let modelB = makeEntry(id: "model-b", displayName: "Beta")

    private final class MT22OpenSession: ChatStreamingSession, @unchecked Sendable {
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

    private func MT22Directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("mt22-\(UUID().uuidString)", isDirectory: true)
    }

    private func MT22ModelDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mt22-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func MT22Wait(_ condition: () -> Bool) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while ContinuousClock.now < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    private func MT22Saved(_ dir: URL, model: String, title: String, at seconds: TimeInterval) -> Conversation {
        let c = Conversation(modelID: model, title: title, updatedAt: Date(timeIntervalSince1970: seconds),
                             items: [.message(ChatMessage(role: .user, text: title))])
        ConversationStore.save(c, in: dir)
        return c
    }

    private func MT22Generating(_ dir: URL, _ session: MT22OpenSession, _ modelDir: URL) -> ChatController {
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.runner.modelDirectoryForTesting = { _ in modelDir }
        controller.runner.streamingSessionForTesting = { session }
        return controller
    }

    // MARK: - the gate (mirrors LocalServerOptionTests' table)

    @Test func theGateClosesEveryChatSurfaceAndKeepsWhatIsOnDisk() {
        let dir = MT22Directory(); defer { try? FileManager.default.removeItem(at: dir); ChatGate.overrideForTesting = nil }
        let saved = MT22Saved(dir, model: "model-a", title: "kept", at: 1000)

        for (available, expectRow) in [(true, true), (false, false)] {
            ChatGate.overrideForTesting = available
            #expect(ChatGate.isAvailable == available)
            // HC-1 ruling A: the sidebar has no Chat row at any gate state; Home shows the chat page only while open.
            #expect(SidebarItem.primaryItems(chatAvailable: ChatGate.isAvailable) == [.home])
            #expect((HomeSurface.resolve(chatAvailable: ChatGate.isAvailable) == .chat) == expectRow)
            let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
            let lease = controller.acquire(.page, model: modelA)
            #expect((lease == .granted) == available)
            #expect((lease == .unavailable) == !available)
            #expect(controller.holds(.page) == available)
            // persisted state survives the flag
            #expect(ConversationStore.allConversations(in: dir).map(\.id) == [saved.id])
        }
    }

    @Test func theBenchIsNotBehindTheChatGate() {
        let dir = MT22Directory(); defer { ChatGate.overrideForTesting = nil }
        ChatGate.overrideForTesting = false
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        #expect(controller.acquire(.bench, model: modelA) == .granted)
    }

    @Test func homeIsTheOnlyPrimaryItemAndShowsChatOnlyWhenTheGateSaysSo() {
        // HC-1 ruling A (Boo, 2026-11-16) supersedes MT ruling J ("Chat second after Home").
        #expect(SidebarItem.primaryItems(chatAvailable: true) == [.home])
        #expect(SidebarItem.primaryItems(chatAvailable: false) == [.home])
        #expect(HomeSurface.resolve(chatAvailable: true) == .chat)
        #expect(HomeSurface.resolve(chatAvailable: false) == .dashboard)
    }

    // MARK: - the page's model

    @Test func thePageModelRule() {
        let installed = [modelB, modelA]
        #expect(ChatPageModel.resolve(current: modelB, newestConversationModelIDs: ["model-a"], installed: installed)?.id == "model-b")
        #expect(ChatPageModel.resolve(current: nil, newestConversationModelIDs: ["gone", "model-b", "model-a"], installed: installed)?.id == "model-b")
        #expect(ChatPageModel.resolve(current: modelB, newestConversationModelIDs: [], installed: [modelA])?.id == "model-a")
        #expect(ChatPageModel.resolve(current: nil, newestConversationModelIDs: [], installed: installed)?.id == "model-a")   // by name
        #expect(ChatPageModel.resolve(current: modelA, newestConversationModelIDs: ["model-a"], installed: []) == nil)
    }

    @Test func refreshPicksTheNewestConversationsModelAndSelectingAChatFollowsItsModel() {
        let dir = MT22Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let old = MT22Saved(dir, model: "model-a", title: "old", at: 1000)
        let new = MT22Saved(dir, model: "model-b", title: "new", at: 2000)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.refreshPageModel(installedChatModels: [modelA, modelB])
        #expect(controller.model(for: .page)?.id == "model-b")
        controller.acquire(.page)
        #expect(controller.conversations.map(\.id) == [new.id, old.id])        // every model's chats
        controller.select(old.id, on: .page)
        #expect(controller.model(for: .page)?.id == "model-a")
        controller.newConversation(on: .page)
        #expect(controller.conversations.first { $0.id == controller.activeConversationID }?.modelID == "model-a")
    }

    @Test func noInstalledModelLeavesThePageWithoutOne() {
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: MT22Directory())
        controller.refreshPageModel(installedChatModels: [])
        #expect(controller.model(for: .page) == nil)
    }

    // MARK: - the lease between page and bench

    @Test func aPageTurnMakesTheBenchWaitAndItBindsWhenTheTurnEnds() async throws {
        let dir = MT22Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let modelDir = try MT22ModelDir(); defer { try? FileManager.default.removeItem(at: modelDir) }
        let session = MT22OpenSession()
        let controller = MT22Generating(dir, session, modelDir)
        controller.acquire(.page, model: modelA)
        await controller.send("hello", from: .page)
        #expect(await MT22Wait { controller.runner.isRunning })
        let items = controller.runner.transcript.items
        let active = controller.activeConversationID

        #expect(controller.acquire(.bench, model: modelB) == .busy(holder: .page))
        #expect(controller.waiting == [.bench])
        #expect(controller.holder == .page)                                     // nothing rebound
        #expect(controller.runner.transcript.items == items)
        #expect(controller.activeConversationID == active)
        #expect(!controller.bindWhenFree(.bench))                               // still generating

        controller.stop(from: .bench)                                           // not the owner
        #expect(controller.runner.isRunning)
        controller.stopCurrentGeneration()                                      // the explicit "Stop it"
        #expect(!controller.runner.isRunning)
        #expect(controller.bindWhenFree(.bench))
        #expect(controller.holder == .bench)
        #expect(controller.waiting.isEmpty)
        session.finish()
    }

    @Test func releasingAWaitingSurfaceClearsTheNotice() async throws {
        let dir = MT22Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let modelDir = try MT22ModelDir(); defer { try? FileManager.default.removeItem(at: modelDir) }
        let session = MT22OpenSession()
        let controller = MT22Generating(dir, session, modelDir)
        controller.acquire(.page, model: modelA)
        await controller.send("hello", from: .page)
        #expect(await MT22Wait { controller.runner.isRunning })
        controller.acquire(.bench, model: modelB)
        controller.release(.bench)
        #expect(controller.waiting.isEmpty)
        #expect(controller.runner.isRunning)                                    // the page's turn is untouched
        #expect(controller.generationOwner == .page)
        controller.stopCurrentGeneration()
        session.finish()
    }

    @Test func thePageKeepsItsConversationTranscriptAndDraftAcrossASheet() {
        let dir = MT22Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let a = MT22Saved(dir, model: "model-a", title: "page chat", at: 2000)
        _ = MT22Saved(dir, model: "model-b", title: "other", at: 1000)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.refreshPageModel(installedChatModels: [modelA, modelB])
        controller.acquire(.page)
        #expect(controller.activeConversationID == a.id)
        let items = controller.runner.transcript.items
        controller.setDraft(ChatDraft(text: "half a thought"), for: .page)
        controller.runner.attachFolder(path: "/tmp/mt22-folder")

        #expect(controller.acquire(.bench, model: modelB) == .granted)         // the sheet opens over the page
        #expect(controller.conversations.map(\.modelID) == ["model-b"])         // the sheet's list stays filtered
        controller.release(.bench)                                              // the sheet closes
        #expect(controller.pageShouldReacquire)
        #expect(controller.acquire(.page) == .granted)

        #expect(controller.activeConversationID == a.id)
        #expect(controller.runner.transcript.items == items)
        #expect(controller.conversations.count == 2)                            // every model's chats again
        #expect(controller.draft(for: .page).text == "half a thought")
        #expect(controller.runner.activeAttachedFolders == ["/tmp/mt22-folder"])
    }

    @Test func aPageTurnSurvivesTheSheetBeingRefused() async throws {
        let dir = MT22Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let modelDir = try MT22ModelDir(); defer { try? FileManager.default.removeItem(at: modelDir) }
        let session = MT22OpenSession()
        let controller = MT22Generating(dir, session, modelDir)
        controller.acquire(.page, model: modelA)
        controller.setDraft(ChatDraft(text: "next"), for: .page)
        await controller.send("hello", from: .page)
        #expect(await MT22Wait { controller.runner.isRunning })
        _ = controller.acquire(.bench, model: modelB)
        controller.release(.bench)                                              // the sheet goes away: "onDisappear"
        controller.stop(from: .bench)
        #expect(controller.runner.isRunning)
        #expect(controller.draft(for: .page).text == "next")
        controller.stop(from: .page)
        session.finish()
    }

    @Test func aModelScopedPrepareAfterThePageReloadsItsOwnList() {
        let dir = MT22Directory(); defer { try? FileManager.default.removeItem(at: dir) }
        _ = MT22Saved(dir, model: "model-a", title: "a", at: 1000)
        _ = MT22Saved(dir, model: "model-b", title: "b", at: 2000)
        let runner = ModelRunner()
        runner.prepare(scope: .all, in: dir)
        #expect(runner.conversations.count == 2)
        runner.prepare(for: modelA, in: dir)
        #expect(runner.conversations.map(\.modelID) == ["model-a"])
    }
}
