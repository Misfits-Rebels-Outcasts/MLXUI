import Foundation
import Testing
@testable import MLXUI

/// HC-3 (ruling F): Home is the launch screen and shows the Chat page, so what the page does on first appearance and
/// on a re-selection of Home must equal what it did when Chat was a sidebar item. On a first appearance
/// `ChatPageView.onAppear` calls `AppState.refreshChatPage()`; a sidebar selection additionally calls the plain
/// `refreshPageModel` (the setter in `SidebarView`). Both reduce to `ChatController.refreshPageModel`, pinned here.
@MainActor
@Suite(.serialized)
struct HomeChatLaunchTests {
    private let modelA = makeEntry(id: "model-a", displayName: "Alpha")
    private let modelB = makeEntry(id: "model-b", displayName: "Beta")

    private func directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("hc3-\(UUID().uuidString)", isDirectory: true)
    }

    @Test func launchReopensTheLastChatsModelAndDoesNotStartAFreshChat() {
        let dir = directory(); defer { try? FileManager.default.removeItem(at: dir) }
        ConversationStore.save(Conversation(modelID: "model-a", title: "old", updatedAt: Date(timeIntervalSince1970: 1000),
                                            items: [.message(ChatMessage(role: .user, text: "old"))]), in: dir)
        ConversationStore.save(Conversation(modelID: "model-b", title: "new", updatedAt: Date(timeIntervalSince1970: 2000),
                                            items: [.message(ChatMessage(role: .user, text: "new"))]), in: dir)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.refreshPageModel(installedChatModels: [modelA, modelB])      // first appearance at launch
        #expect(controller.model(for: .page)?.id == "model-b")                 // the chat last used, as pre-HC
        #expect(!controller.startsFreshPageChat)                               // a launch refresh never opens a new chat
        #expect(!ChatController.shouldStartFreshChat(hadNoModel: false, hasModelNow: true))
    }

    @Test func reselectingHomeKeepsTheModelTheConversationChoiceAndTheDraft() {
        let dir = directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir)
        controller.refreshPageModel(installedChatModels: [modelA, modelB])
        controller.acquire(.page, model: modelB)
        let draft = ChatDraft(text: "half-typed", files: [URL(fileURLWithPath: "/tmp/x.txt")])
        controller.setDraft(draft, for: .page)
        let active = controller.activeConversationID
        // The sidebar setter's refresh, run again with the page already showing.
        controller.refreshPageModel(installedChatModels: [modelA, modelB])
        controller.refreshPageModel(installedChatModels: [modelA, modelB])
        #expect(controller.model(for: .page)?.id == "model-b")
        #expect(controller.draft(for: .page) == draft)
        #expect(controller.activeConversationID == active)
        #expect(!controller.startsFreshPageChat)
    }
}
