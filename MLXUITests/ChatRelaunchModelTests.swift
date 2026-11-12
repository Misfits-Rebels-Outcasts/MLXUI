import Foundation
import Testing
@testable import MLXUI

/// MT4-7-FIX-3: on relaunch the page reopens on the model the chat last ran on, not the default router's pick.
@MainActor
@Suite(.serialized)
struct ChatRelaunchModelTests {

    private let modelA = makeEntry(id: "model-a", displayName: "Alpha")
    private let modelB = makeEntry(id: "model-b", displayName: "Beta")

    private func directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("fix3-\(UUID().uuidString)", isDirectory: true)
    }

    private func save(_ dir: URL, model: String, at seconds: TimeInterval) -> Conversation {
        let c = Conversation(modelID: model, title: model, updatedAt: Date(timeIntervalSince1970: seconds),
                             items: [.message(ChatMessage(role: .user, text: "hi"))])
        ConversationStore.save(c, in: dir)
        return c
    }

    private func controller(_ dir: URL, readiness: Readiness?) -> ChatController {
        ChatController(runner: ModelRunner(), conversationDirectory: dir, readinessProvider: { readiness })
    }

    @Test func relaunchKeepsTheModelTheNewestChatRanOnWhileAppleIsReady() {
        let dir = directory(); defer { try? FileManager.default.removeItem(at: dir) }
        _ = save(dir, model: "model-b", at: 2000)
        let c = controller(dir, readiness: .ready)
        c.refreshPageModel(installedChatModels: [modelA, modelB])
        #expect(c.model(for: .page)?.id == "model-b")
    }

    @Test func relaunchKeepsAppleIntelligenceWhenThatWasTheLastModelAndItIsReady() {
        let dir = directory(); defer { try? FileManager.default.removeItem(at: dir) }
        _ = save(dir, model: AppleIntelligenceChat.entry.id, at: 2000)
        let c = controller(dir, readiness: .ready)
        c.refreshPageModel(installedChatModels: [modelA])
        #expect(AppleIntelligenceChat.isApple(c.model(for: .page)?.id ?? ""))
    }

    @Test func aSavedModelThatIsNoLongerInstalledFallsThroughToTheDefaultRouter() {
        let dir = directory(); defer { try? FileManager.default.removeItem(at: dir) }
        _ = save(dir, model: "gone", at: 2000)
        let c = controller(dir, readiness: .ready)
        c.refreshPageModel(installedChatModels: [modelA])
        // Default router: Apple Intelligence while ready (rule 1), unchanged.
        #expect(AppleIntelligenceChat.isApple(c.model(for: .page)?.id ?? ""))
    }

    @Test func aSavedAppleChatWithAppleNotReadyFallsThrough() {
        let dir = directory(); defer { try? FileManager.default.removeItem(at: dir) }
        _ = save(dir, model: AppleIntelligenceChat.entry.id, at: 2000)
        let c = controller(dir, readiness: .unavailable(reason: "off"))
        c.refreshPageModel(installedChatModels: [modelA])
        #expect(c.model(for: .page)?.id == "model-a")
    }

    @Test func noConversationsUsesTheDefaultRouterUnchanged() {
        let c = controller(directory(), readiness: .ready)
        c.refreshPageModel(installedChatModels: [modelA])
        #expect(AppleIntelligenceChat.isApple(c.model(for: .page)?.id ?? ""))
    }
}
