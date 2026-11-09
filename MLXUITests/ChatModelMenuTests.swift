import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// MT2-4: the composer model menu and switching models between turns. Helpers are prefixed `MT24`.
@MainActor
@Suite(.serialized)
struct ChatModelMenuTests {

    private let modelA = makeEntry(id: "model-a", displayName: "Alpha 9B")
    private let modelB = makeEntry(id: "model-b", displayName: "beta 4B")
    private let speech = makeEntry(id: "asr-1", displayName: "Zed Speech", modelType: .asr)

    private final class MT24Session: ChatStreamingSession, @unchecked Sendable {
        let reply: String
        let holdOpen: Bool
        init(reply: String, holdOpen: Bool = false) { self.reply = reply; self.holdOpen = holdOpen }
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
            continuation.yield(reply)
            if !holdOpen { continuation.finish() }
            return stream
        }
    }

    private final class MT24Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var calls: [(String, [Chat.Message])] = []
        func record(_ id: String, _ history: [Chat.Message]) { lock.withLock { calls.append((id, history)) } }
        var all: [(String, [Chat.Message])] { lock.withLock { calls } }
        private var loads = 0
        func load() { lock.withLock { loads += 1 } }
        var loadCount: Int { lock.withLock { loads } }
    }

    private func MT24Directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("mt24-\(UUID().uuidString)", isDirectory: true)
    }

    private func MT24Wait(_ condition: () -> Bool) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while ContinuousClock.now < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    /// A page controller on a conversation that A already answered once, with a recording seam.
    private func MT24Page(reply: String = "reply", hold: Bool = false)
        throws -> (ChatController, MT24Recorder, URL, URL) {
        let dir = MT24Directory()
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("mt24-m-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let earlier = Conversation(modelID: "model-a", title: "earlier", items: [
            .message(ChatMessage(role: .user, text: "first question")),
            .message(ChatMessage(role: .assistant, text: "first answer", modelID: "model-a")),
        ])
        ConversationStore.save(earlier, in: dir)
        let recorder = MT24Recorder()
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir,
                                        factsLoader: { _ in recorder.load(); return ChatModelFacts(canUseTools: true, usableTokens: 32768) })
        controller.runner.modelDirectoryForTesting = { _ in modelDir }
        controller.runner.sessionRequestForTesting = { id, history in recorder.record(id, history) }
        controller.runner.streamingSessionForTesting = { MT24Session(reply: reply, holdOpen: hold) }
        controller.refreshPageModel(installedChatModels: [modelA, modelB])
        controller.acquire(.page)
        return (controller, recorder, dir, modelDir)
    }

    // MARK: - the menu's items and facts

    @Test func itemsAreInstalledChatModelsOnlyByNameWithPlainFacts() {
        let facts = ["model-a": ChatModelFacts(canUseTools: true, usableTokens: 32768),
                     "model-b": ChatModelFacts(canUseTools: false, usableTokens: nil)]
        let items = ChatModelMenu.items(installed: [modelB, speech, modelA], selectedID: "model-b", facts: facts)
        #expect(items.map(\.id) == ["model-a", "model-b"])                    // sorted by name; the ASR model is not a chat model
        #expect(items.map(\.isSelected) == [false, true])
        #expect(items[0].facts == "Can use tools · 32K tokens of context on this Mac")
        #expect(items[1].facts == "Can't use tools")
        #expect(ChatModelMenu.items(installed: [modelA], selectedID: nil, facts: [:])[0].facts == "")   // not loaded yet
        #expect(ChatModelMenu.items(installed: [], selectedID: nil, facts: [:]).isEmpty)
    }

    @Test func tokenTextIsPlain() {
        #expect(ChatModelMenu.tokenText(900) == "900")
        #expect(ChatModelMenu.tokenText(8192) == "8K")
        #expect(ChatModelMenu.tokenText(131_072) == "128K")
        #expect(ChatModelMenu.tokenText(1_048_576) == "1M")
    }

    @Test func factsAreLoadedOncePerModel() throws {
        let (controller, recorder, dir, modelDir) = try MT24Page()
        defer { try? FileManager.default.removeItem(at: dir); try? FileManager.default.removeItem(at: modelDir) }
        controller.loadModelFacts(for: [modelA, modelB])
        controller.loadModelFacts(for: [modelA, modelB])
        #expect(recorder.loadCount == 2)
        #expect(controller.modelFacts["model-a"]?.canUseTools == true)
    }

    @Test func getMoreOpensBrowseOnTheChatSection() {
        #expect(ChatModelMenu.getMoreTarget(visibleSectionIDs: ["audio", "chat-text"]) == .browse("chat-text"))
        #expect(ChatModelMenu.getMoreTarget(visibleSectionIDs: ["audio"]) == nil)
        #expect(ChatModelMenu.getMoreTitle == "Get more chat models…")
    }

    // MARK: - switching

    @Test func switchingAppliesToTheNextTurnAndHistoryCarriesOver() async throws {
        let (controller, recorder, dir, modelDir) = try MT24Page(reply: "second answer")
        defer { try? FileManager.default.removeItem(at: dir); try? FileManager.default.removeItem(at: modelDir) }
        #expect(controller.model(for: .page)?.id == "model-a")                  // the last conversation's model

        // switching changes nothing that was already written
        #expect(controller.selectPageModel(modelB))
        #expect(controller.model(for: .page)?.id == "model-b")
        #expect(controller.conversations.first?.modelID == "model-a")           // still "last model used" until a turn runs
        guard case .message(let before) = controller.runner.transcript.items[1] else { Issue.record("no reply"); return }
        #expect(before.modelID == "model-a")
        #expect(recorder.all.isEmpty)

        await controller.send("follow-up", from: .page)
        #expect(await MT24Wait { !controller.runner.isRunning && controller.runner.transcript.items.count == 4 })

        // the new model got the earlier messages
        let call = try #require(recorder.all.first)
        #expect(call.0 == "model-b")
        #expect(call.1.map(\.content) == ["first question", "first answer"])
        #expect(call.1.map(\.role) == [.user, .assistant])

        // and only now do the record and the stamps change
        #expect(controller.conversations.first?.modelID == "model-b")
        guard case .message(let old) = controller.runner.transcript.items[1],
              case .message(let new) = controller.runner.transcript.items[3] else { Issue.record("shape"); return }
        #expect(old.modelID == "model-a")
        #expect(new.modelID == "model-b")
        #expect(new.text == "second answer")
    }

    @Test func switchingBackSendsTheWholeConversationToTheOldModel() async throws {
        let (controller, recorder, dir, modelDir) = try MT24Page()
        defer { try? FileManager.default.removeItem(at: dir); try? FileManager.default.removeItem(at: modelDir) }
        controller.selectPageModel(modelB)
        await controller.send("two", from: .page)
        #expect(await MT24Wait { !controller.runner.isRunning && controller.runner.transcript.items.count == 4 })
        controller.selectPageModel(modelA)
        await controller.send("three", from: .page)
        #expect(await MT24Wait { !controller.runner.isRunning && controller.runner.transcript.items.count == 6 })
        let last = try #require(recorder.all.last)
        #expect(last.0 == "model-a")
        #expect(last.1.count == 4)                                              // q1, a1, q2, a2
        #expect(last.1.last?.content == "reply")
    }

    @Test func theMenuIsRefusedWhileATurnRuns() async throws {
        let (controller, _, dir, modelDir) = try MT24Page(hold: true)
        defer { try? FileManager.default.removeItem(at: dir); try? FileManager.default.removeItem(at: modelDir) }
        await controller.send("hello", from: .page)
        #expect(await MT24Wait { controller.runner.isRunning })
        #expect(!controller.selectPageModel(modelB))
        #expect(controller.model(for: .page)?.id == "model-a")                  // nothing changed under the running turn
        controller.stop(from: .page)
        #expect(controller.selectPageModel(modelB))
        #expect(controller.model(for: .page)?.id == "model-b")
    }

    @Test func anUninstalledLastModelPreselectsTheInstalledPageModel() throws {
        let (_, _, dir, modelDir) = try MT24Page()
        defer { try? FileManager.default.removeItem(at: dir); try? FileManager.default.removeItem(at: modelDir) }
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.refreshPageModel(installedChatModels: [modelB])              // the conversation's model-a is not installed
        controller.acquire(.page)
        #expect(controller.model(for: .page)?.id == "model-b")
        let badge = ChatConversationBadge.make(modelID: "model-a", catalog: ["model-a": "Alpha 9B"], installedIDs: ["model-b"])
        #expect(badge.text == "Alpha 9B · not installed")
    }
}
