import Testing
import Foundation
@testable import MLXUI

/// MC2-1: the TranslateGemma 4B catalog entry and the new `translate` routing kind. Facts are from the pinned
/// `mlx-community/translategemma-4b-it-4bit` revision `5788ec08c047f3f2e17808101b8d9566ac930d58` (fetched 2026-11-04).
///
/// MC2-2 extends this file with the About text and the full field-by-field facts. The point of MC2-1 is that the entry
/// routes as `.translate` everywhere — never `.llm` — so the chat sheet, the Local Server and every generic LLM workflow
/// task leave it out without a special case (owner decision 3).
@MainActor
struct TranslateGemmaCatalogTests {

    static let hfID = "mlx-community/translategemma-4b-it-4bit"
    static let cardID = "mlx-community--translategemma-4b-it-4bit"

    static func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    static func entry() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == hfID })
    }

    /// The registry the app builds at launch (`installedModules`, `AppState.init`).
    private func registry() -> ModelRegistry {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return registry
    }

    // MARK: - The catalog

    @Test func loadBrowserDataDecodesWithTheNewEntry() throws {
        let state = AppState()
        state.loadBrowserData()
        #expect(state.loadError == nil, "browser.json must decode: \(state.loadError ?? "")")
        let data = try #require(state.browserData)
        let all = data.domains.flatMap { $0.allModels }
        #expect(all.contains { $0.hfModelId == Self.hfID })
    }

    @Test func kindRoutesToTranslate() throws {
        let e = try Self.entry()
        #expect(e.runnerKind == .translate)
        #expect(e.modelType == .llm, "the entry stays modelType llm; the kind carries the routing")
        #expect(RunnerKind.translate.rawValue == "translate")
    }

    @Test func livesInChatAndTextAppendedLastToTheLLMDomain() throws {
        let data = try Self.catalog()
        let llm = try #require(data.domains.first { $0.id == "llm" })
        #expect(llm.allModels.last?.hfModelId == Self.hfID, "appended last to the llm domain, per decision 3")
        #expect(llm.allModels.filter { $0.hfModelId == Self.hfID }.count == 1)
        let chatText = try #require(data.sidebarSections.first { $0.id == "chat-text" })
        #expect(chatText.domainIds == ["llm"])
    }

    // MARK: - Facts (MC2-1; the About text and the rest of the rendered facts are MC2-2's)

    @Test func entryFactsMatchThePinnedRevision() throws {
        let e = try Self.entry()
        #expect(e.id == Self.cardID)
        #expect(e.displayName == "TranslateGemma 4B")
        #expect(e.family == "translategemma")
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-4bit")
        #expect(e.architecture == "Gemma3ForConditionalGeneration")
        // The pinned repo's safetensors report total_parameters 3,880,263,168 (3.88B).
        #expect(e.paramCountB == 3.88)
        // Owner decision 8: the card's 2K input limit, not config.json's 131,072 positions.
        #expect(e.contextWindow == 2048)
        // The whole pinned repo is 2,222,628,201 B = 2.22 GB (decimal, the catalog's convention).
        #expect(e.downloadSizeGB == 2.22)
        // MC2-1: provisional 1.5 x downloadSizeGB; the maintainers set the final figure from their own run (MC2-7).
        #expect(e.ramGB == 3.33)
        #expect(e.license == "gemma-terms-of-use")
        #expect(DataNormalizer.normalizeLicense(e.license) == "Gemma Terms of Use")
        #expect(e.licenseUrl == "https://ai.google.dev/gemma/terms")
        #expect(e.lastUpdated == "2026-01-16")
        #expect(e.variants?.count == 1)
        let variant = try #require(e.variants?.first)
        #expect(variant.quantization == "4-bit")
        #expect(variant.hfModelId == Self.hfID)
        #expect(variant.ramGB == e.ramGB)
        #expect(variant.downloadSizeGB == e.downloadSizeGB)
        // Implementer's call, pending owner confirmation: every catalog 4-bit entry uses 90; the card publishes no
        // benchmark figure, so this is the table's convention, not a measurement.
        #expect(variant.qualityPercent == 90)
        #expect(variant.recommended == true)
    }

    // MARK: - Routing (Trap 5)

    @Test func chatModuleDoesNotClaimTheTranslator() throws {
        let e = try Self.entry()
        #expect(registry().bestModule(for: e)?.descriptor.id != "chat",
                "TranslateGemma must never resolve to the chat module")
        // The UI claim mirrors the SDK's, so the Run sheet cannot fall back to chat either.
        #expect(ChatSDK().claim(e) == .no)
        #expect(ChatUI().claim(e) == .no)
    }

    @Test func runDoesNotOpenTheChatSheet() throws {
        let e = try Self.entry()
        let state = AppState()
        state.runModel(e)
        #expect(state.runningModel?.id != Self.cardID, "Run must not send the translator to the chat sheet")
    }

    // MARK: - The Local Server never offers it (decision 9)

    @Test func theServeToggleIsHidden() throws {
        let e = try Self.entry()
        #expect(!ServeToggleVisibility.shouldShow(state: .installed, isInstalledOnDisk: true,
                                                   runnerKind: e.runnerKind, gateAvailable: true))
        // Control: the same state/kind the server does serve.
        #expect(ServeToggleVisibility.shouldShow(state: .installed, isInstalledOnDisk: true,
                                                 runnerKind: .llm, gateAvailable: true))
    }

    @Test func theServersInstalledLLMListExcludesIt() throws {
        let e = try Self.entry()
        // The kind the Local Server's `isInstalledLLM` / model list filter reads (`InstalledModelIndex`), mirrored from
        // `ModelEntry.runnerKind` so the nonisolated resolver agrees with the app.
        let kind = InstalledModelIndex.kind(modelType: e.modelType.rawValue, family: e.family)
        #expect(kind == .translate)
        let index = InstalledModelIndex(entries: [
            .init(id: e.id, hfModelId: e.hfModelId, kind: kind, ramGB: e.ramGB,
                  downloadSizeGB: e.downloadSizeGB),
        ])
        let listed = index.entries.filter { $0.kind == .llm }   // what LocalServerController / ServerView list
        #expect(listed.isEmpty, "the translator must not appear in the Local Server's model list")
    }

    // MARK: - Memory booking (Trap 8 / decision 3)

    @Test func theTranslateRowBooksLikeTheLLMRow() {
        let llm = ModelLoadProfile.entry(for: .llm)
        let translate = ModelLoadProfile.entry(for: .translate)
        #expect(translate == llm)
        #expect(translate.workingSetBytes == 0, "text-only: same MLXLLM runner, no image working set")
        #expect(translate.factor == ModelWeightCharge.loadFactor)
    }
}
