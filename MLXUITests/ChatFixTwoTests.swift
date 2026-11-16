import Foundation
import Testing
@testable import MLXUI

/// CD-FIX-2 — the no-model Download control, the fresh chat after the first install, and the equal-height grid's column rule.
@MainActor
@Suite(.serialized)
struct ChatFixTwoTests {
    private let modelA = makeEntry(id: "model-a", displayName: "Alpha")

    private func directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("cdfix2-\(UUID().uuidString)", isDirectory: true)
    }

    // MARK: A — Download control

    @Test func theControlFollowsTheInstallState() {
        #expect(ChatNoModelDownload.presentation(for: nil) == .button(note: nil))
        #expect(ChatNoModelDownload.presentation(for: .idle) == .button(note: nil))
        #expect(ChatNoModelDownload.presentation(for: .resolving) == .starting)          // queued: no bytes yet
        #expect(ChatNoModelDownload.presentation(for: .verifying) == .verifying)
        #expect(ChatNoModelDownload.presentation(for: .downloading(progress: 0.5, downloaded: 500_000_000, total: 1_000_000_000))
                == .progress(fraction: 0.5, detail: "50% · 0.5 GB of 1.0 GB"))
        #expect(ChatNoModelDownload.presentation(for: .error("No disk", canRetry: true)) == .button(note: "No disk"))
        #expect(ChatNoModelDownload.presentation(for: .needsAuth("Token needed")) == .button(note: "Token needed"))
    }

    @Test func cancelReturnsToTheButtonBecauseTheEngineGoesIdle() {
        #expect(ChatNoModelDownload.presentation(for: .resolving) != .button(note: nil))
        #expect(ChatNoModelDownload.presentation(for: .idle) == .button(note: nil))      // InstallManager.cancel sets .idle
    }

    // MARK: B — fresh chat on the first model

    @Test func onlyTheNoModelToModelTransitionStartsAFreshChat() {
        #expect(ChatController.shouldStartFreshChat(hadNoModel: true, hasModelNow: true))
        #expect(!ChatController.shouldStartFreshChat(hadNoModel: false, hasModelNow: true))
        #expect(!ChatController.shouldStartFreshChat(hadNoModel: true, hasModelNow: false))
        #expect(!ChatController.shouldStartFreshChat(hadNoModel: false, hasModelNow: false))
    }

    @Test func installingTheFirstModelOpensANewEmptyChatAndKeepsTheOldOne() {
        let dir = directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let old = Conversation(modelID: "gone-model", title: "old", updatedAt: Date(timeIntervalSince1970: 1000),
                               items: [.message(ChatMessage(role: .user, text: "old"))])
        ConversationStore.save(old, in: dir)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir, readinessProvider: { nil })
        controller.refreshPageModel(installedChatModels: [])               // the no-model screen
        #expect(controller.model(for: .page) == nil)
        controller.refreshPageModel(installedChatModels: [modelA])         // the first model arrives
        #expect(controller.startsFreshPageChat)
        controller.acquire(.page)
        #expect(!controller.startsFreshPageChat)
        let active = controller.conversations.first { $0.id == controller.activeConversationID }
        #expect(active?.items.isEmpty == true)
        #expect(active?.id != old.id)
        #expect(controller.conversations.contains { $0.id == old.id })     // the old conversation is still listed
    }

    @Test func aModelInstalledWhileTheLaunchPageAlreadyHadOneChangesNothing() {
        let dir = directory(); defer { try? FileManager.default.removeItem(at: dir) }
        let saved = Conversation(modelID: "model-a", title: "chat", updatedAt: Date(timeIntervalSince1970: 1000),
                                 items: [.message(ChatMessage(role: .user, text: "chat"))])
        ConversationStore.save(saved, in: dir)
        let controller = ChatController(runner: ModelRunner(), conversationDirectory: dir, readinessProvider: { nil })
        controller.refreshPageModel(installedChatModels: [modelA])         // launch: a model is there from the first refresh
        #expect(!controller.startsFreshPageChat)
        controller.acquire(.page)
        #expect(controller.activeConversationID == saved.id)
        controller.refreshPageModel(installedChatModels: [modelA, makeEntry(id: "model-b", displayName: "Beta")])
        #expect(!controller.startsFreshPageChat)
        #expect(controller.activeConversationID == saved.id)
    }

    // MARK: C — grid columns

    @Test func columnsAreTheAdaptiveCountAndAlwaysAtLeastOne() {
        #expect(ChatGridColumns.count(width: 0) == 1)
        #expect(ChatGridColumns.count(width: 150) == 1)
        #expect(ChatGridColumns.count(width: 200) == 1)
        #expect(ChatGridColumns.count(width: 410) == 2)      // 2 × 200 + 10
        #expect(ChatGridColumns.count(width: 409) == 1)
        #expect(ChatGridColumns.count(width: 850) == 4)      // 4 × 200 + 3 × 10 = 830
        #expect(ChatGridColumns.count(width: .infinity) == 1)
    }
}
