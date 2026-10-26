import Testing
import Foundation
@testable import MLXUI

/// MP6-1: the LFM2.5-Audio 1.5B catalog entry and the new `voice` kind. Facts are from the pinned HF files
/// (RSI/notes/modelpack/hf/mlx-community_LFM2.5-Audio-1.5B-6bit @ c875bcf). Until MP6-3 registers a module the model
/// installs but resolves to "no runner yet", and no workflow pool offers it.
@MainActor
struct LFMAudioCatalogTests {

    private static let hfID = "mlx-community/LFM2.5-Audio-1.5B-6bit"

    /// The pinned repo's siblings (api.json), in the tree's order.
    private static let siblings = [
        ".gitattributes", "README.md",
        "audio_detokenizer/config.json", "audio_detokenizer/model.safetensors",
        "audio_detokenizer/tokenizer.json", "audio_detokenizer/tokenizer_config.json",
        "config.json", "model.safetensors", "model.safetensors.index.json", "special_tokens_map.json",
        "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer.json", "tokenizer_config.json",
    ]

    /// Byte sizes from the pinned tree.json, for the files the selector keeps.
    private static let sizes: [String: Int64] = [
        "audio_detokenizer/config.json": 1_150, "audio_detokenizer/model.safetensors": 314_137_512,
        "audio_detokenizer/tokenizer.json": 4_732_426, "audio_detokenizer/tokenizer_config.json": 91_509,
        "config.json": 3_331, "model.safetensors": 1_410_710_961, "model.safetensors.index.json": 110_587,
        "special_tokens_map.json": 434, "tokenizer.json": 4_732_404, "tokenizer_config.json": 91_566,
    ]

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func lfm() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    @Test func entryFactsMatchThePinnedRepo() throws {
        let e = try lfm()
        #expect(e.id == "mlx-community--LFM2.5-Audio-1.5B-6bit")
        #expect(e.displayName == "LFM2.5-Audio 1.5B")
        #expect(e.family == "LFM2.5-Audio")
        #expect(e.modelType == .voice)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-6bit")
        #expect(e.downloadSizeGB == 1.73)
        // MP-0 G (implementer's call to round up, pending owner confirmation): download x 1.5 = 2.595, shown as 2.6, an estimate.
        #expect(e.ramGB == 2.6)
        #expect(e.ramGB >= e.downloadSizeGB * 1.5)
        #expect(e.description?.contains("RAM figure is an estimate (download size × 1.5)") == true)
        #expect(e.speedTokensPerSec == nil)
        #expect(e.languages == ["en"])
        #expect(e.variants?.count == 1)
        #expect(e.variants?.first?.quantization == "6-bit")
        #expect(e.variants?.first?.qualityPercent == 90)   // same as the other sub-8-bit quantizations
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.variants?.first?.downloadSizeGB == e.downloadSizeGB)
        #expect(e.variants?.first?.hfModelId == Self.hfID)
    }

    @Test func kindRoutesToVoiceAndSamAudioStaysUnsupported() throws {
        let e = try lfm()
        #expect(e.runnerKind == .voice)
        #expect(ModelType.voice.rawValue == "voice")
        #expect(RunnerKind.voice.rawValue == "voice")
        #expect(ModelType.voice.sfSymbol == "waveform.and.mic")
        // The family override wins over whatever modelType the catalog writes; sam-audio is not in this change.
        #expect(makeEntry(family: "LFM2.5-Audio", modelType: .asr).runnerKind == .voice)
        #expect(makeEntry(family: "sam-audio", modelType: .asr).runnerKind == .unsupported)
    }

    @Test func liveInTheAudioSectionsVoiceLeaf() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "voice" })
        #expect(leaf.allModels.map(\.hfModelId) == [Self.hfID])
        let audio = try #require(data.sidebarSections.first { $0.id == "audio" })
        #expect(audio.domainIds.contains("voice"))
        #expect(data.sidebarSections.filter { $0.domainIds.contains("voice") }.map(\.id) == ["audio"])
        #expect(data.domainIndex["voice"] != nil)
    }

    /// No module registered yet, so the registry claims nothing and Run falls to the "no runner yet" notice.
    @Test func noModuleClaimsItAndTheGapFlagsIt() throws {
        let e = try lfm()
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        #expect(full.bestModule(for: e) == nil, "MP6-3 owns the LFM module; remove this expectation and the ModelSupport gap there")
        #expect(ModelSupport.unsupportedReason(for: e)?.contains("no runner in this build yet") == true)
    }

    @Test func stageFactoryRefusesTheKindWithAnExplicitCase() throws {
        let e = try lfm()
        #expect(throws: StageError.self) {
            _ = try StageFactory.make(for: e, installedModelIDs: [e.id], availableRAMGB: 16)
        }
    }

    @Test func noWorkflowPoolOffersIt() throws {
        let e = try lfm()
        for task in TaskCatalog.allTasks().map(\.name) {
            #expect(TaskModels.runnerKind(forTask: task) != .voice, "\(task) names the voice kind")
            let pool = TaskModels.derivedModels(for: task, catalog: [e], claimableModelIDs: [e.id])
            #expect(!pool.contains { $0.id == e.id }, "\(task) offers LFM2.5-Audio")
        }
        #expect(CatalogBridge.entries.allSatisfy { $0.display != "LFM2.5-Audio 1.5B" }, "the bridge row is MP6-W")
    }

    /// T7: the root `tokenizer-e351c8d8-checkpoint125.safetensors` (384,644,900 B) is skipped, because the standard weights
    /// are present and the top-level fallback is suppressed. The audio detokenizer component comes down whole.
    @Test func selectorSkipsTheRootTokenizerCheckpoint() {
        let out = ModelFileSelector.filesToDownload(siblings: Self.siblings)
        #expect(!out.contains("tokenizer-e351c8d8-checkpoint125.safetensors"))
        #expect(Set(out) == Set(Self.sizes.keys))
        #expect(!out.contains("README.md"))
        #expect(out.contains("model.safetensors"))
        #expect(out.contains("audio_detokenizer/model.safetensors"))
        let bytes = out.reduce(Int64(0)) { $0 + (Self.sizes[$1] ?? 0) }
        #expect(bytes == 1_734_611_880)
        #expect((Double(bytes) / 1e9 * 100).rounded() / 100 == 1.73)
    }
}

// MP6-2: the description and metadata (facts from the pinned model card only).
extension LFMAudioCatalogTests {

    /// Every field `ModelDetailView` and `ComparisonTableView` render, pinned so the only dashes are the sanctioned nulls:
    /// Speed (no tokens/s for an audio model, MP-0 G), Context and benchmarks (the card states none).
    @Test func metadataRendersWithNoUnsanctionedDash() throws {
        let e = try lfm()
        #expect(e.license == "lfm1.0")
        #expect(DataNormalizer.normalizeLicense(e.license) == "LFM Open License v1.0")
        #expect(DataNormalizer.normalizeLicense(e.license) != "Custom")
        // Detail renders the architecture string through the normalizer; it leaves a descriptive string as written.
        let arch = try #require(e.architecture)
        #expect(arch == "Conformer audio encoder + LFM backbone (1.5B) + Depthformer audio decoder + ISTFT neural detokenizer")
        #expect(DataNormalizer.normalizeArchitecture(arch) == arch)
        #expect(e.descriptionSource == "mlx-community/LFM2.5-Audio-1.5B-6bit")
        #expect(e.taskTags == ["speech-to-speech", "audio-to-audio"])
        // api.json: lastModified 2026-01-15, downloads 34, likes 0.
        #expect(e.lastUpdated == "2026-01-15")
        #expect(e.communityLikes == 0)
        #expect(e.communityDownloads == 34)
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US_POSIX")
        day.dateFormat = "yyyy-MM-dd"
        day.isLenient = false
        #expect(e.lastUpdated.flatMap { day.date(from: $0) } != nil)
        #expect(e.formattedDownloads() != "—")
        #expect(e.speedTokensPerSec == nil)
        #expect(e.contextWindow == nil)
        #expect(e.benchmarks == nil)
        #expect(e.languages == ["en"])
        #expect(!e.paramSize.isEmpty)
        #expect(e.summary?.isEmpty == false)
    }

    @Test func descriptionCarriesOnlyWhatTheCardSupports() throws {
        let e = try lfm()
        let description = try #require(e.description)
        #expect(description.hasPrefix("# LFM2.5-Audio 1.5B"))
        #expect(description.contains("LiquidAI/LFM2.5-Audio-1.5B"))
        #expect(description.contains("English only"))
        #expect(description.contains("| **Voice conversation** |"))
        #expect(description.contains("| **Speech to text** |"))
        #expect(description.contains("| **Text to speech** |"))
        #expect(description.contains("| **License** | LFM Open License v1.0 |"))
        #expect(description.contains("Read the license before using the model commercially."))
        #expect(description.contains("| **Audio output** | 24 kHz |"))
        #expect(description.contains("RAM figure is an estimate (download size × 1.5), not a measurement."))
        // The MP6-1 placeholder is gone, and nothing says the Run screen is missing.
        #expect(!description.contains("Run isn't available"))
        // The owner has not read the license's commercial terms; the text must not paraphrase them. The card states no
        // quality, benchmark, latency or voice-choice claim either.
        for forbidden in ["revenue", "royalt", "free for", "accuracy", "state-of-the-art", "outperform", "benchmark",
                          "real-time", "realtime", "low-latency", "voice clon", "multilingual"] {
            #expect(!description.localizedCaseInsensitiveContains(forbidden), "\(forbidden)")
        }
        #expect((e.summary ?? "").contains("English only"))
    }
}
