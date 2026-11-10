import Testing
@testable import MLXUI

/// MT3-FIX-1 — the header subtitle and the per-answer model badge.
@MainActor
struct ChatModelLabelsTests {
    private let catalog = ["mlx-community--Qwen3.5-9B-4bit": "Qwen3.5 9B"]
    private let qwen = "mlx-community--Qwen3.5-9B-4bit"

    @Test func theSubtitleNeverSaysMLXForAppleIntelligence() {
        #expect(ChatModelLabels.headerSubtitle(modelID: AppleIntelligenceChat.modelID) == "Running on this Mac · Apple Intelligence")
        #expect(ChatModelLabels.headerSubtitle(modelID: qwen) == "Running locally · MLX")
    }

    @Test func anAssistantAnswerNamesItsModel() {
        let installed: Set<String> = [qwen, AppleIntelligenceChat.modelID]
        #expect(ChatModelLabels.answerBadge(role: .assistant, modelID: qwen, catalog: catalog, installedIDs: installed) == "Qwen3.5 9B")
        #expect(ChatModelLabels.answerBadge(role: .assistant, modelID: AppleIntelligenceChat.modelID, catalog: catalog, installedIDs: installed) == "Apple Intelligence")
    }

    @Test func aModelThatIsGoneIsNotInstalledAndAnUnknownIdShowsItsTail() {
        #expect(ChatModelLabels.answerBadge(role: .assistant, modelID: qwen, catalog: catalog, installedIDs: []) == "Qwen3.5 9B · not installed")
        #expect(ChatModelLabels.answerBadge(role: .assistant, modelID: "mlx-community--Other-4bit", catalog: catalog, installedIDs: []) == "Other-4bit · not installed")
        #expect(ChatModelLabels.answerBadge(role: .assistant, modelID: AppleIntelligenceChat.modelID, catalog: catalog, installedIDs: []) == "Apple Intelligence · not installed")
    }

    @Test func userTurnsAndUnstampedAnswersShowNoBadge() {
        #expect(ChatModelLabels.answerBadge(role: .user, modelID: qwen, catalog: catalog, installedIDs: [qwen]) == nil)
        #expect(ChatModelLabels.answerBadge(role: .assistant, modelID: nil, catalog: catalog, installedIDs: [qwen]) == nil)
        #expect(ChatModelLabels.answerBadge(role: .assistant, modelID: "", catalog: catalog, installedIDs: [qwen]) == nil)
    }
}
