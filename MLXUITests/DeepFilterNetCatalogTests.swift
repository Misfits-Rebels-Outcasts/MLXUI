import Testing
import Foundation
@testable import MLXUI

/// MP5-1: the DeepFilterNet 3 catalog entry and the new `enhance` kind. Facts are from the pinned HF files
/// (RSI/notes/modelpack/hf/mlx-community_DeepFilterNet-mlx @ 220d5dfb). Until MP5-3 registers a module the model
/// installs but resolves to "no runner yet", and no workflow pool offers it.
@MainActor
struct DeepFilterNetCatalogTests {

    private static let hfID = "mlx-community/DeepFilterNet-mlx"

    /// The pinned repo's siblings (api.json), in the tree's order.
    private static let siblings = [
        ".gitattributes", "README.md", "convert_deepfilternet.py",
        "v1/README.md", "v1/config.json", "v1/model.safetensors",
        "v2/README.md", "v2/config.json", "v2/model.safetensors",
        "v3/README.md", "v3/config.json", "v3/model.safetensors",
    ]

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func deepFilterNet() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    @Test func entryFactsMatchThePinnedRepo() throws {
        let e = try deepFilterNet()
        #expect(e.id == "mlx-community--DeepFilterNet-mlx")
        #expect(e.displayName == "DeepFilterNet 3")
        #expect(e.family == "DeepFilterNet")
        #expect(e.modelType == .enhance)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-fp32")
        #expect(e.license == "mit")   // api.json cardData.license
        // Pinned tree: v1+v2+v3 model.safetensors 25,585,573 B + three config.json 2,568 B = 25,588,141 B = 0.0256 GB.
        #expect(e.downloadSizeGB == 0.03)
        // MP5-3d: ramGB is the measured process peak (phys_footprint 2.349 GB on a 60-minute clip, chunked 8 s / 1 s, the whole
        // Run UI path), rounded up to 0.01 GB. Provenance is in the description. It replaced the MP5-1 estimate (0.045).
        #expect(e.ramGB == 2.35)
        #expect(e.ramGB >= 2.349, "never below the measured peak")
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.description?.contains("RAM figure is measured") == true)
        #expect(e.speedTokensPerSec == nil)
        #expect(e.variants?.count == 1)
        #expect(e.variants?.first?.quantization == "fp32")
        // v3 model.safetensors 8,682,709 B / 4 B per fp32 weight = 2.17 M parameters (derived).
        #expect(e.paramCountB == 0.0022)
    }

    @Test func kindRoutesToEnhance() throws {
        let e = try deepFilterNet()
        #expect(e.runnerKind == .enhance)
        #expect(ModelType.enhance.rawValue == "enhance")
        #expect(RunnerKind.enhance.rawValue == "enhance")
        #expect(ModelType.enhance.sfSymbol == "waveform.badge.minus")
    }

    @Test func liveInTheAudioSectionsEnhanceLeaf() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "enhance" })
        #expect(leaf.allModels.map(\.hfModelId) == [Self.hfID])
        let audio = try #require(data.sidebarSections.first { $0.id == "audio" })
        #expect(audio.domainIds.contains("enhance"))
        #expect(data.sidebarSections.filter { $0.domainIds.contains("enhance") }.map(\.id) == ["audio"])
        #expect(data.domainIndex["enhance"] != nil)
    }

    /// MP5-3a: the DeepFilterNet module is the only one that claims it (before MP5-3a none did, and the `ModelSupport` gap flagged it).
    @Test func exactlyOneModuleClaimsItAndTheMP51GapIsGone() throws {
        let e = try deepFilterNet()
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        #expect(full.bestModule(for: e)?.descriptor.id == "deepfilternet")
        var claimers: [String] = []
        for module in installedModules {
            let one = ModelRegistry()
            module.register(into: one)
            if one.bestModule(for: e) != nil { claimers.append(module.descriptor.id) }
        }
        #expect(claimers == ["deepfilternet"], "a chat/ASR module must not be handed an enhance model: \(claimers)")
        #expect(ModelSupport.unsupportedReason(for: e) == nil)
    }

    @Test func stageFactoryRefusesTheKindWithAnExplicitCase() throws {
        let e = try deepFilterNet()
        #expect(throws: StageError.self) {
            _ = try StageFactory.make(for: e, installedModelIDs: [e.id], availableRAMGB: 16)
        }
    }

    @Test func noWorkflowPoolOffersIt() throws {
        let e = try deepFilterNet()
        for task in TaskCatalog.allTasks().map(\.name) {
            #expect(TaskModels.runnerKind(forTask: task) != .enhance, "\(task) names the enhance kind")
            let pool = TaskModels.derivedModels(for: task, catalog: [e], claimableModelIDs: [e.id])
            #expect(!pool.contains { $0.id == e.id }, "\(task) offers DeepFilterNet")
        }
    }

    /// The repo has no root weights: v1, v2 and v3 each hold `config.json` + `model.safetensors`. The selector treats
    /// each as a component and takes all three (25.6 MB, harmless, backlog MP5). `InstallManager` creates the
    /// subfolders, so the engine finds `<dir>/v3/{config.json,model.safetensors}`.
    @Test func selectorTakesAllThreeVersionsAndNoDocs() {
        let out = Set(ModelFileSelector.filesToDownload(siblings: Self.siblings))
        #expect(out == [
            "v1/config.json", "v1/model.safetensors",
            "v2/config.json", "v2/model.safetensors",
            "v3/config.json", "v3/model.safetensors",
        ])
        #expect(!out.contains("convert_deepfilternet.py"))
    }
}

// MP5-2: the description and metadata.
extension DeepFilterNetCatalogTests {

    /// Every field `ModelDetailView` and `ComparisonTableView` render, pinned so the only dashes are the sanctioned nulls:
    /// Speed (no tokens/s for an audio filter, MP-0 G), Context, benchmarks and languages (the card states none).
    @Test func metadataRendersWithNoUnsanctionedDash() throws {
        let e = try deepFilterNet()
        #expect(e.architecture == "Deep-filtering speech enhancement (FFT 960, hop 480, 32 ERB bands, 96 DF bins, DF order 5)")
        #expect(e.descriptionSource == "mlx-community/DeepFilterNet-mlx")
        #expect(e.taskTags == ["speech-enhancement", "noise-suppression"])
        // api.json: lastModified 2026-03-11, likes 7. Downloads is 0 at the pin (renders "0", not a dash).
        #expect(e.lastUpdated == "2026-03-11")
        #expect(e.communityLikes == 7)
        #expect(e.communityDownloads != nil)
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US_POSIX")
        day.dateFormat = "yyyy-MM-dd"
        day.isLenient = false
        #expect(e.lastUpdated.flatMap { day.date(from: $0) } != nil)
        #expect(e.formattedDownloads() != "—")
        #expect(e.speedTokensPerSec == nil)
        #expect(e.contextWindow == nil)
        #expect(e.benchmarks == nil)
        #expect(e.languages == nil)
        #expect(DataNormalizer.normalizeLicense(e.license) != "—")
        #expect(!e.paramSize.isEmpty)
        #expect(e.summary?.isEmpty == false)
    }

    @Test func descriptionCarriesOnlyWhatTheCardSupports() throws {
        let e = try deepFilterNet()
        let description = try #require(e.description)
        #expect(description.hasPrefix("# DeepFilterNet 3"))
        #expect(description.contains("full-band 48 kHz audio"))
        #expect(description.contains("MLXUI uses v3"))
        #expect(description.contains("| **License** | MIT |"))
        #expect(description.contains("Hendrik Schroeter"))
        #expect(description.contains("Rikorose/DeepFilterNet"))
        #expect(description.contains("| **Sample rate** | 48 kHz in and out |"))
        #expect(description.contains("no fine-tuning or quantization"))
        #expect(description.contains("RAM figure is measured: process peak (phys_footprint) 2.35 GB (2.349 GB) on a 60-minute clip"))
        #expect(description.contains("replaces the download × 1.5 estimate (0.045 GB)"))
        #expect((e.summary ?? "").contains("48 kHz"))
        // The card states no quality, PESQ, real-time or language claim, so the text must not make one.
        for forbidden in ["accuracy", "state-of-the-art", "outperform", "PESQ", "real-time", "realtime", "language"] {
            #expect(!description.localizedCaseInsensitiveContains(forbidden), "\(forbidden)")
        }
    }
}
