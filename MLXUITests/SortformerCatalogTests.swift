import Testing
import Foundation
@testable import MLXUI

/// MP4-1: the Sortformer 4spk v2.1 catalog entry and the new `diarization` kind. Facts are from the pinned HF files
/// (RSI/notes/modelpack/hf/mlx-community_diar_streaming_sortformer_4spk-v2.1-fp16 @ e23e6404). Until MP4-3 registers a
/// module the model installs but resolves to "no runner yet", and no workflow pool offers it. (MP4-3a registered the module.)
@MainActor
struct SortformerCatalogTests {

    private static let hfID = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func sortformer() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    @Test func entryFactsMatchThePinnedRepo() throws {
        let e = try sortformer()
        #expect(e.id == "mlx-community--diar_streaming_sortformer_4spk-v2.1-fp16")
        #expect(e.displayName == "Sortformer 4spk v2.1")
        #expect(e.modelType == .diarization)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-fp16")
        // MP-0 H: NVIDIA's CC-BY-4.0 (the mlx-community repo itself declares no license in its card or API record).
        #expect(e.license == "cc-by-4.0")
        // Pinned tree listing: model.safetensors 236,108,132 B + config.json 1,702 B = 236,109,834 B = 0.24 GB.
        #expect(e.downloadSizeGB == 0.24)
        // MP4-3d: ramGB is the measured process peak (phys_footprint, 1,673 MiB = 1.754 GB) at the 60 s offline cap, the most any
        // accepted clip needs, rounded up to 0.1 GB. Provenance is in the description.
        #expect(e.ramGB == 1.8)
        #expect(e.ramGB >= 1_673.0 * 1_048_576 / 1e9, "never below the measured peak")
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.speedTokensPerSec == nil)
        #expect(e.variants?.count == 1)
        #expect(e.variants?.first?.quantization == "fp16")
        #expect(e.description?.contains("RAM figure is measured") == true)
    }

    @Test func kindRoutesToDiarization() throws {
        let e = try sortformer()
        #expect(e.runnerKind == .diarization)
        #expect(ModelType.diarization.rawValue == "diarization")
        #expect(RunnerKind.diarization.rawValue == "diarization")
        #expect(ModelType.diarization.sfSymbol == "person.2.wave.2")
        // The catalog-mislabel overrides do not touch it.
        #expect(e.family == "Sortformer")
    }

    @Test func liveInTheAudioSectionsDiarizationLeaf() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "diarization" })
        #expect(leaf.allModels.map(\.hfModelId) == [Self.hfID])
        let audio = try #require(data.sidebarSections.first { $0.id == "audio" })
        #expect(audio.domainIds.contains("diarization"))
        #expect(data.sidebarSections.filter { $0.domainIds.contains("diarization") }.map(\.id) == ["audio"])
        #expect(data.domainIndex["diarization"] != nil)
    }

    /// MP4-3a: the Sortformer module is the only one that claims it (before MP4-3a none did, and the `ModelSupport` gap flagged it).
    @Test func exactlyOneModuleClaimsItAndTheMP41GapIsGone() throws {
        let e = try sortformer()
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        #expect(full.bestModule(for: e)?.descriptor.id == "sortformer")
        var claimers: [String] = []
        for module in installedModules {
            let one = ModelRegistry()
            module.register(into: one)
            if one.bestModule(for: e) != nil { claimers.append(module.descriptor.id) }
        }
        #expect(claimers == ["sortformer"], "a chat/ASR module must not be handed a diarization model: \(claimers)")
        #expect(ModelSupport.unsupportedReason(for: e) == nil)
    }

    @Test func stageFactoryRefusesTheKindWithAnExplicitCase() throws {
        let e = try sortformer()
        #expect(throws: StageError.self) {
            _ = try StageFactory.make(for: e, installedModelIDs: [e.id], availableRAMGB: 16)
        }
    }

    @Test func noWorkflowPoolOffersIt() throws {
        let e = try sortformer()
        for task in TaskCatalog.allTasks().map(\.name) {
            #expect(TaskModels.runnerKind(forTask: task) != .diarization, "\(task) names the diarization kind")
            let pool = TaskModels.derivedModels(for: task, catalog: [e], claimableModelIDs: [e.id])
            // Provider / system slots (remote or on-device models) are not catalog entries; only the catalog matters here.
            #expect(!pool.contains { $0.id == e.id }, "\(task) offers Sortformer")
        }
    }
}

// MP4-2: the description and metadata.
extension SortformerCatalogTests {

    /// Every field `ModelDetailView` and `ComparisonTableView` render, pinned so the only dashes are the sanctioned nulls:
    /// Speed (diarization has no tokens/s, MP-0 G), Context, benchmarks and languages (the card states none).
    @Test func metadataRendersWithNoUnsanctionedDash() throws {
        let e = try sortformer()
        // Detail "Architecture" tile appears only when non-nil; the string is the card's "Model Details" line.
        #expect(e.architecture == "FastConformer (17 layers) + Transformer encoder (18 layers) + Sortformer modules")
        #expect(e.descriptionSource == "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16")
        #expect(e.taskTags == ["speaker-diarization", "voice-activity-detection"])
        // Presence only (regenerate_catalog.py rewrites these from the HF API).
        #expect((e.communityDownloads ?? 0) > 0)   // else the detail page and comparison print "—"
        #expect(e.communityLikes != nil)
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US_POSIX")
        day.dateFormat = "yyyy-MM-dd"
        day.isLenient = false
        #expect(e.lastUpdated.flatMap { day.date(from: $0) } != nil)
        #expect(e.formattedDownloads() != "—")
        // Sanctioned nulls.
        #expect(e.speedTokensPerSec == nil)
        #expect(e.contextWindow == nil)
        #expect(e.benchmarks == nil)
        #expect(e.languages == nil)
        // The comparison table's License / Size / Download rows.
        #expect(DataNormalizer.normalizeLicense(e.license) != "—")
        #expect(!e.paramSize.isEmpty)
        #expect(e.summary?.isEmpty == false)
    }

    @Test func descriptionCarriesOnlyWhatTheCardSupports() throws {
        let e = try sortformer()
        let description = try #require(e.description)
        #expect(description.hasPrefix("# Sortformer 4spk v2.1"))
        #expect(description.contains("up to 4 speakers"))
        #expect(description.contains("16 kHz mono audio"))
        #expect(description.contains("streaming"))
        #expect(description.contains("| **Max speakers** | 4 |"))
        #expect(description.contains("| **License** | CC-BY-4.0 |"))
        #expect(description.contains("Original model by NVIDIA, licensed under"))
        #expect(description.contains("nvidia/diar_streaming_sortformer_4spk-v2.1"))
        #expect(description.contains("RAM figure is measured: process peak (phys_footprint) 1.63 GiB (1.75 GB, listed as 1.8)"))
        #expect(description.contains("replaces the download × 1.5 estimate"))
        #expect((e.summary ?? "").contains("up to 4 speakers"))
        // The card states no accuracy, DER, overlap or language claim, so the text must not make one.
        for forbidden in ["accuracy", "state-of-the-art", "outperform", "overlap", "language", "real-time", "realtime"] {
            #expect(!description.localizedCaseInsensitiveContains(forbidden), "\(forbidden)")
        }
        #expect(description.range(of: #"\bDER\b"#, options: .regularExpression) == nil)
    }
}
