import Testing
@testable import MLXUI

@MainActor
struct ChatCapabilityMenuTextTests {
    private func model(_ name: String) -> ModelEntry {
        ModelEntry(
            id: name, family: "F", displayName: name, paramSize: "", paramCountB: nil, modelType: .llm, source: .mlx, format: "mlx",
            platforms: ["macos"], minMacOSVersion: nil, hfRepo: "", hfModelId: "", ramGB: 4, downloadSizeGB: 1, contextWindow: 4096,
            license: nil, licenseUrl: nil, description: nil, summary: nil, descriptionSource: nil, architecture: nil, languages: nil,
            lastUpdated: nil, taskTags: nil, benchmarks: nil, speedTokensPerSec: nil, speedHardware: nil, speedEstimated: nil,
            communityDownloads: nil, communityLikes: nil, variants: nil)
    }

    private var cards: [ChatCapabilityCard] {
        let statuses: [String: ModelToolStatus] = [
            "ocr_image": .ready(model("GLM-OCR")),
            "transcribe_audio": .needsDownload(model("Parakeet"), gb: 0.6),
            "speak": .cantRun(reason: "Too big."),
        ]
        let r = ChatCapabilityCards.cards(specs: ModelToolSpec.modelTools, statuses: statuses)
        return r.primary + r.more
    }

    @Test func labelCarriesTheSummaryOnlyWhenToolsWork() {
        #expect(ChatCapabilityMenuText.label(modelName: "Qwen3 4B", cards: cards, canUseTools: true) == ("Qwen3 4B", "2 tools"))
        #expect(ChatCapabilityMenuText.label(modelName: "Qwen3 4B", cards: cards, canUseTools: false) == ("Qwen3 4B", nil))
        #expect(ChatCapabilityMenuText.label(modelName: "Qwen3 4B", cards: [], canUseTools: true) == ("Qwen3 4B", nil))
    }

    @Test func rowsAreTitleDashStatusAndEmptyWhenTheRouterCannotUseTools() {
        #expect(ChatCapabilityMenuText.toolRows(cards: cards, canUseTools: true) == [
            "Read text in an image — Ready · GLM-OCR",
            "Transcribe a recording — 0.6 GB download on first use",
            "Read text aloud — Too big.",
        ])
        #expect(ChatCapabilityMenuText.toolRows(cards: cards, canUseTools: false).isEmpty)
        #expect(ChatCapabilityMenuText.toolRows(cards: [], canUseTools: true).isEmpty)
    }

    @Test func copyIsTheBacklogs() {
        #expect(ChatCapabilityMenuText.sectionTitle == "Uses these models as tools")
        #expect(ChatCapabilityMenuText.dropCaption == "Chat will pick a model for this")
        #expect(ChatCapabilityMenuText.pagePlaceholder == "Ask, or drop a photo, PDF or recording…")
    }

    @Test func resolveCoversEveryModelTool() {
        let runtime = ModelToolRuntime(catalog: [], installedModelIDs: [], claimableModelIDs: [], totalRAMGB: 16,
                                       makeExecutor: { _, _ in fatalError("unused") })
        let statuses = ChatCapabilityStatuses.resolve(runtime: runtime)
        #expect(Set(statuses.keys) == Set(ModelToolSpec.modelTools.map(\.name)))
        #expect(statuses.values.allSatisfy { $0 == .none })   // empty catalog: nothing to offer
    }
}
