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

// MC2-2: the About text and every fact the model page and Compare render, plus the pinned HF file list the installer
// must fetch. Facts from the pinned HF revision (2026-11-04) and its model card.
extension TranslateGemmaCatalogTests {

    /// The pinned repo's siblings (`/api/models/.../revision/5788ec08?blobs=true`), in the tree's order.
    static let siblings = [
        ".gitattributes", "README.md", "added_tokens.json", "chat_template.jinja", "config.json",
        "generation_config.json", "model.safetensors", "model.safetensors.index.json",
        "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    ]

    /// The whole pinned repo: 2,222,628,201 B = 2.22 GB (decimal, the catalog's convention).
    static let repoBytes: Int64 = 2_222_628_201

    @Test func entryFactsMatchTheCardAndCompare() throws {
        let e = try Self.entry()
        #expect(e.hfModelId == "mlx-community/translategemma-4b-it-4bit")
        // Card header and Compare: display name, family, size, source.
        #expect(e.displayName == "TranslateGemma 4B")
        #expect(e.family == "translategemma")
        #expect(e.paramSize == "4B")
        #expect(e.source == .mlx)
        #expect(e.source.rawValue.uppercased() == "MLX")                 // Compare's Source row
        // Detail tiles: Architecture, Context Window, RAM, Download, License, Updated, Downloads.
        #expect(DataNormalizer.normalizeArchitecture(e.architecture) == "Gemma3")
        #expect(e.contextWindow == 2048)                                 // detail/compare render "2K"
        #expect(String(format: "%.1f", e.ramGB) == "3.3")
        #expect(DataNormalizer.downloadSize(gb: e.downloadSizeGB) == "2.2 GB")
        #expect(DataNormalizer.normalizeLicense(e.license) == "Gemma Terms of Use")
        #expect(e.lastUpdated == "2026-01-16")
        #expect(e.formattedDownloads() == "3.8K")
        // No speed figure (the card publishes none; meaningfulness aside, a translator's tokens/s is not a quality
        // figure) and no benchmarks — the sanctioned dashes, as SAM-Audio's facts test has it.
        #expect(e.speedTokensPerSec == nil)
        #expect(e.displaySpeed == "—")
        #expect(e.benchmarks == nil)
        #expect(e.qualityScore == nil)
        #expect(e.taskTags == ["translation", "text-generation"])
    }

    /// The About text's claims, pinned — the issue asks for the card's own claims and nothing it doesn't make.
    @Test func aboutTextSaysWhatItDoesAndNothingMore() throws {
        let e = try Self.entry()
        let d = try #require(e.description)
        // What it does.
        #expect(d.contains("translating text"))
        #expect(d.contains("TranslateGemma 4B"))
        // Decision 2: text only — the About text must not claim image translation.
        #expect(d.contains("**Text only.**"))
        #expect(d.localizedCaseInsensitiveContains("no image input"))
        #expect(d.localizedCaseInsensitiveContains("translating text inside an image is not available"))
        // Decision 8: the 2K-token input limit, counted on the rendered request.
        #expect(d.contains("2K-token total input limit"))
        #expect(d.contains("2,048 tokens"))
        #expect(d.localizedCaseInsensitiveContains("rendered request"))
        // Decision 7: deterministic, stops at the end of the turn.
        #expect(d.contains("greedy (temperature 0)"))
        #expect(d.localizedCaseInsensitiveContains("stops at the end of the model's turn"))
        // The model card's 55-language claim, and that the other listed languages are untested.
        #expect(d.contains("55 languages"))
        #expect(d.localizedCaseInsensitiveContains("WMT24++"))
        #expect(d.localizedCaseInsensitiveContains("only the 55 evaluated languages were tested"))
        // Decision 11: the licence notice, on the page, both editions (the catalog is shared).
        #expect(d.contains("Gemma Terms of Use"))
        #expect(d.contains("https://ai.google.dev/gemma/terms"))
        // MC2-1: the RAM figure is an estimate here; the maintainers set it from their own measurement (MC2-7).
        #expect(d.contains("RAM figure is an estimate"))
        #expect(d.contains("download × 1.5"))
        // No quality claim the card doesn't make.
        for claim in ["state-of-the-art", "best-in-class", "outperforms", "highest quality", "most accurate",
                      "better than"] {
            #expect(!d.localizedCaseInsensitiveContains(claim), "About text must not claim \(claim)")
        }
        // The short blurb (cards/lists) also stays text-only and honest.
        let summary = try #require(e.summary)
        #expect(summary.contains("TranslateGemma 4B"))
        #expect(summary.contains("text only"))
        #expect(summary.contains("2.2 GB"))
        #expect(!summary.localizedCaseInsensitiveContains("chat"))
        #expect(!summary.localizedCaseInsensitiveContains("quality"))
    }

    /// The pinned file list: the installer must fetch the weights, the configs, the chat template and the tokenizer —
    /// the `.jinja` file is this repo's only copy of the template (tokenizer_config.json has no `chat_template` key).
    @Test func theInstallerFetchesEveryFileTheEngineNeeds() {
        let selected = ModelFileSelector.filesToDownload(siblings: Self.siblings)
        #expect(selected == [
            "added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json",
            "model.safetensors", "model.safetensors.index.json", "special_tokens_map.json",
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        ])
        // Named one by one for the review: every file an MLX Gemma 3 load reaches for.
        for needed in ["model.safetensors", "config.json", "generation_config.json", "chat_template.jinja",
                       "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json"] {
            #expect(selected.contains(needed), "the installer must fetch \(needed)")
        }
        // Docs and git metadata are not installed.
        #expect(!selected.contains("README.md"))
        #expect(!selected.contains(".gitattributes"))
        // The single-file checkpoint is fully accounted for in the catalog's download figure.
        #expect(Self.repoBytes == 2_183_295_977 + 33_384_568 + 4_689_074 + 1_155_413 + 79_768 + 16_982 + 2_815 + 128 + 662 + 35 + 1_209 + 1_570)
    }
}

