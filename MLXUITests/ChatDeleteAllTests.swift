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
}
