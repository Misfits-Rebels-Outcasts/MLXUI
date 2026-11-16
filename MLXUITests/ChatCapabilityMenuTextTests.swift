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

    // MARK: CD-FIX-1

    @Test func placeholderFollowsSurfaceAndTools() {
        #expect(ChatCapabilityMenuText.placeholder(isPage: true, modelName: "Q", canUseTools: true) == "Ask, or drop a photo, PDF or recording…")
        #expect(ChatCapabilityMenuText.placeholder(isPage: true, modelName: "Q", canUseTools: false) == "Message Q…")
        #expect(ChatCapabilityMenuText.placeholder(isPage: false, modelName: "Q", canUseTools: true) == "Message Q…")
        #expect(ChatCapabilityMenuText.placeholder(isPage: false, modelName: "Q", canUseTools: false) == "Message Q…")
    }

    @Test func menuWidthLeavesTheTextFieldItsShare() {
        #expect(ChatCapabilityMenuText.menuMaxWidth(containerWidth: nil) == nil)
        #expect(ChatCapabilityMenuText.menuMaxWidth(containerWidth: 0) == nil)
        #expect(ChatCapabilityMenuText.menuMaxWidth(containerWidth: 400) == 108)       // 400 - 292
        #expect(ChatCapabilityMenuText.menuMaxWidth(containerWidth: 300) == 80)        // floor
        #expect(ChatCapabilityMenuText.menuMaxWidth(containerWidth: 2000) == 360)      // ceiling
        // whatever the width, the menu plus everything else leaves the field at least its reserved minimum
        for width in stride(from: 380.0, through: 1200.0, by: 20) {
            let menu = ChatCapabilityMenuText.menuMaxWidth(containerWidth: width) ?? 0
            #expect(width - menu - ChatCapabilityMenuText.composerReserved >= 0 || menu == 80)
        }
    }

    @Test func labelFallsBackFromSummaryToNameToTruncatedName() {
        let wide = ChatCapabilityMenuText.fitLabel(name: "Qwen3 4B", summary: "6 tools", maxWidth: 360)
        #expect(wide.summary == "6 tools" && wide.fitsWhole)
        let mid = ChatCapabilityMenuText.fitLabel(name: "Qwen3 4B", summary: "6 tools", maxWidth: 108)
        #expect(mid.summary == nil && mid.fitsWhole)
        let tight = ChatCapabilityMenuText.fitLabel(name: "Qwen3 4B Instruct 2507", summary: "6 tools", maxWidth: 108)
        #expect(tight.summary == nil && !tight.fitsWhole)
        let unlimited = ChatCapabilityMenuText.fitLabel(name: "Qwen3 4B", summary: "6 tools", maxWidth: nil)
        #expect(unlimited.summary == "6 tools" && unlimited.fitsWhole)
    }
}
