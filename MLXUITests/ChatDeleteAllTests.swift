import Foundation
import Testing
@testable import MLXUI

/// CP-2: Delete All Conversations. Helpers `CP2*`.
@MainActor
@Suite(.serialized)
struct ChatDeleteAllTests {
    private let modelA = makeEntry(id: "model-a", displayName: "Alpha")
    private let modelB = makeEntry(id: "model-b", displayName: "Beta")

    private func CP2Directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("cp2-\(UUID().uuidString)", isDirectory: true)
    }

    @discardableResult
    private func CP2Seed(_ dir: URL, model: String, title: String, at seconds: TimeInterval) -> Conversation {
        let c = Conversation(modelID: model, title: title, updatedAt: Date(timeIntervalSince1970: seconds),
                             items: [.message(ChatMessage(role: .user, text: title))])
        ConversationStore.save(c, in: dir)
        return c
    }

    private func CP2Files(_ dir: URL) -> Int {
        ConversationStore.allConversations(in: dir).count
    }

    // MARK: - pure wording

    @Test func titleWording() {
        #expect(ChatDeleteAllText.title(count: 3, modelName: nil) == "Delete all 3 conversations?")
        #expect(ChatDeleteAllText.title(count: 1, modelName: nil) == "Delete the conversation?")
        #expect(ChatDeleteAllText.title(count: 3, modelName: "Alpha") == "Delete all 3 conversations with Alpha?")
        #expect(ChatDeleteAllText.title(count: 1, modelName: "Alpha") == "Delete the conversation with Alpha?")
        #expect(ChatDeleteAllText.message(modelName: "Alpha").contains("only the conversations with Alpha"))
        #expect(!ChatDeleteAllText.message(modelName: nil).contains("only"))
    }

    // MARK: - runner

    @Test func benchKeepsOneFreshMemoryOnlyChatAndRemovesOnlyThisModelsFiles() {
        let dir = CP2Directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        for i in 0..<3 { CP2Seed(dir, model: "model-a", title: "a\(i)", at: Double(i + 1)) }
        CP2Seed(dir, model: "model-b", title: "b", at: 10)
        let runner = ModelRunner()
        runner.prepare(for: modelA, in: dir)
        #expect(runner.conversations.count == 3)
        runner.deleteAllConversations()
        #expect(runner.conversations.count == 1)
        #expect(runner.conversations[0].items.isEmpty)
        #expect(runner.conversations[0].modelID == "model-a")
        #expect(runner.activeConversationID == runner.conversations[0].id)
        let left = ConversationStore.allConversations(in: dir)
        #expect(left.count == 1 && left[0].modelID == "model-b")      // a blank chat is never written
    }

    @Test func pageRemovesEveryModelsChatsAndSelectsNothing() {
        let dir = CP2Directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        CP2Seed(dir, model: "model-a", title: "a", at: 1)
        CP2Seed(dir, model: "model-b", title: "b", at: 2)
        let runner = ModelRunner()
        runner.prepare(scope: .all, in: dir)
        #expect(runner.conversations.count == 2)
        runner.deleteAllConversations()
        #expect(runner.conversations.isEmpty)
        #expect(runner.activeConversationID == nil)
        #expect(CP2Files(dir) == 0)
    }

    // MARK: - controller

    @Test func wrongSurfaceIsANoOpAndTheHolderDeletes() {
        let dir = CP2Directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        CP2Seed(dir, model: "model-a", title: "a", at: 1)
        CP2Seed(dir, model: "model-b", title: "b", at: 2)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        _ = controller.acquire(.page)
        #expect(controller.conversations.count == 2)
        controller.deleteAll(on: .bench)                 // not the holder
        #expect(controller.conversations.count == 2)
        #expect(CP2Files(dir) == 2)
        controller.deleteAll(on: .page)
        #expect(controller.conversations.isEmpty)
        #expect(CP2Files(dir) == 0)
    }

    // MARK: - CP-2-FIX-1: deleted on one surface, gone from both

    private func CP2PageThenBench(_ dir: URL) -> ChatController {
        ChatGate.overrideForTesting = true
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.acquire(.page)                          // the page loads every model's chats
        controller.acquire(.bench, model: modelA)          // the sheet opens: the page's set is parked
        return controller
    }

    @Test func deleteAllOnTheBenchDoesNotComeBackWhenThePageTakesTheRunnerBack() {
        let dir = CP2Directory()
        defer { ChatGate.overrideForTesting = nil; try? FileManager.default.removeItem(at: dir) }
        let a = (0..<3).map { CP2Seed(dir, model: "model-a", title: "a\($0)", at: Double($0 + 1)).id }
        let b = CP2Seed(dir, model: "model-b", title: "b", at: 10).id
        let controller = CP2PageThenBench(dir)
        #expect(controller.runner.conversations.count == 3)
        controller.deleteAll(on: .bench)
        controller.release(.bench)
        controller.acquire(.page)
        let ids = Set(controller.runner.conversations.map(\.id))
        #expect(ids.contains(b))
        for id in a { #expect(!ids.contains(id)) }
        let onDisk = Set(ConversationStore.allConversations(in: dir).map(\.id))
        #expect(onDisk == [b])
        // and a later save of the page's active chat must not write a deleted file back
        controller.runner.selectConversation(b)
        #expect(Set(ConversationStore.allConversations(in: dir).map(\.id)) == [b])
    }

    @Test func singleDeleteOnTheBenchDoesNotComeBackEither() {
        let dir = CP2Directory()
        defer { ChatGate.overrideForTesting = nil; try? FileManager.default.removeItem(at: dir) }
        let a1 = CP2Seed(dir, model: "model-a", title: "a1", at: 1).id
        let a2 = CP2Seed(dir, model: "model-a", title: "a2", at: 2).id
        let controller = CP2PageThenBench(dir)
        controller.delete(a1, on: .bench)
        controller.release(.bench)
        controller.acquire(.page)
        let ids = Set(controller.runner.conversations.map(\.id))
        #expect(!ids.contains(a1) && ids.contains(a2))
        #expect(Set(ConversationStore.allConversations(in: dir).map(\.id)) == [a2])
    }

    @Test func deletingOnThePageIsNotRevivedByTheBenchesParkedSet() {
        let dir = CP2Directory()
        defer { ChatGate.overrideForTesting = nil; try? FileManager.default.removeItem(at: dir) }
        let a1 = CP2Seed(dir, model: "model-a", title: "a1", at: 1).id
        let a2 = CP2Seed(dir, model: "model-a", title: "a2", at: 2).id
        let controller = CP2PageThenBench(dir)
        controller.release(.bench)
        controller.acquire(.page)                          // the sheet's last set is parked
        controller.delete(a1, on: .page)
        controller.acquire(.bench, model: modelA)
        let ids = Set(controller.runner.conversations.map(\.id))
        #expect(!ids.contains(a1) && ids.contains(a2))
    }
}
