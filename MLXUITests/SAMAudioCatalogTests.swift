import Testing
import Foundation
@testable import MLXUI

/// MC1-1: the SAM-Audio Base catalog entry and the new `separate` kind. Facts are from the pinned HF repos
/// (`mlx-community/sam-audio-base` sha `614abc87`, `google-t5/t5-base`, fetched 2026-11-02).
///
/// MC1-1 pins the kind and the entry (it decodes and the Audio section offers the leaf); MC1-2 adds the about text and
/// the full field-by-field facts.
@MainActor
struct SAMAudioCatalogTests {

    static let hfID = "mlx-community/sam-audio-base"

    /// The repo's siblings (api.json `blobs=true`), in the tree's order.
    static let siblings = [
        ".gitattributes", "README.md", "config.json",
        "model-00001-of-00001.safetensors", "model.safetensors.index.json",
    ]

    /// `google-t5/t5-base`'s siblings, in the tree's order: the PyTorch / TF / Flax / Rust weights it also ships are
    /// not installed.
    static let t5Siblings = [
        ".gitattributes", "README.md", "config.json", "flax_model.msgpack", "generation_config.json",
        "model.safetensors", "pytorch_model.bin", "rust_model.ot", "spiece.model", "tf_model.h5", "tokenizer.json",
    ]

    static func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    static func samAudio() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == hfID })
    }

    @Test func loadBrowserDataDecodesWithTheNewEntry() throws {
        let state = AppState()
        state.loadBrowserData()
        #expect(state.loadError == nil, "browser.json must decode: \(state.loadError ?? "")")
        let data = try #require(state.browserData)
        let all = data.domains.flatMap { $0.allModels }
        #expect(all.contains { $0.hfModelId == Self.hfID })
    }

    @Test func kindRoutesToSeparate() throws {
        let e = try Self.samAudio()
        #expect(e.runnerKind == .separate)
        #expect(ModelType.separate.rawValue == "separate")
        #expect(RunnerKind.separate.rawValue == "separate")
        #expect(ModelType.separate.sfSymbol == "waveform.path.ecg.rectangle")
    }

    @Test func liveInTheAudioSectionsSoundIsolationLeaf() throws {
        let data = try Self.catalog()
        let leaf = try #require(data.domains.first { $0.id == "separate" })
        #expect(leaf.name == "Sound Isolation")
        #expect(leaf.sfSymbol == "waveform.path.ecg.rectangle")
        #expect(leaf.allModels.map(\.hfModelId) == [Self.hfID])
        let audio = try #require(data.sidebarSections.first { $0.id == "audio" })
        #expect(audio.domainIds.contains("separate"))
        #expect(data.sidebarSections.filter { $0.domainIds.contains("separate") }.map(\.id) == ["audio"])
        #expect(data.domainIndex["separate"] != nil)
    }
}

// MC1-2: the description and the facts the model page and Compare render.
extension SAMAudioCatalogTests {

    /// Every field `ModelDetailView` and `ComparisonTableView` render, pinned so the only dashes are the sanctioned
    /// nulls: Speed (no tokens/s for a sound isolator), Context and benchmarks (the card states no benchmark figures).
    @Test func entryFactsMatchThePinnedRepos() throws {
        let e = try Self.samAudio()
        #expect(e.id == "mlx-community--sam-audio-base")
        #expect(e.displayName == "SAM-Audio Base")
        #expect(e.family == "sam-audio")
        #expect(e.modelType == .separate)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-fp32")
        #expect(e.paramCountB == 1.258)   // 5,031,977,488 B of float32 weights / 4
        // The download is SAM-Audio's 5,032,108,828 B (config.json 754 + index 55,842 + weights 5,032,052,232) plus
        // T5-base's selected 893,828,754 B (config 1,208 + generation config 147 + weights 891,646,390 +
        // spiece.model 791,656 + tokenizer.json 1,389,353) = 5,925,937,582 B = 5.93 GB.
        #expect(e.downloadSizeGB == 5.93)
        #expect(e.variants?.count == 1)
        #expect(e.variants?.first?.quantization == "fp32")
        #expect(e.variants?.first?.hfModelId == Self.hfID)
        #expect(e.variants?.first?.recommended == true)
        #expect(e.variants?.first?.qualityPercent == 100)
        #expect(e.variants?.first?.downloadSizeGB == e.downloadSizeGB)
        #expect(e.license == "sam-license")
        #expect(DataNormalizer.normalizeLicense(e.license) == "SAM License")
        #expect(e.licenseUrl == "https://huggingface.co/facebook/sam-audio-base/blob/main/LICENSE")
    }

    /// MC1-1: the RAM figure is the provisional 1.5 × weights estimate, and the description says so.
    @Test func ramGBIsTheProvisionalEstimateAndSaysSo() throws {
        let e = try Self.samAudio()
        #expect(e.ramGB == 8.89)
        // 5,925,937,582 B of selected files = 5.925937582 GB x 1.5 = 8.8889 GB, rounded up to 0.01.
        let exactGB = Double(5_925_937_582) / 1e9
        #expect(e.ramGB == (exactGB * 1.5 * 100).rounded() / 100)
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.description?.contains("is an estimate") == true)
        #expect(e.description?.contains("5.93 GB download") == true)
    }

    @Test func metadataRendersWithNoUnsanctionedDash() throws {
        let e = try Self.samAudio()
        #expect(e.architecture == "DAC-VAE audio codec + diffusion transformer (DiT) with a T5-base text encoder")
        #expect(e.descriptionSource == Self.hfID)
        #expect(e.taskTags == ["sound-isolation", "audio-separation", "voice-isolation"])
        #expect(e.languages == ["en"])
        #expect(e.lastUpdated == "2026-02-06")   // HF lastModified
        #expect(e.communityLikes == 2)
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
        #expect(!e.paramSize.isEmpty)
        #expect(e.summary?.isEmpty == false)
    }

    /// What it isolates, that it returns two tracks, that long files take minutes, the limit, the licence notice and
    /// where the text encoder comes from. No quality claim the model card does not make.
    @Test func descriptionCarriesOnlyWhatTheCardSupports() throws {
        let e = try Self.samAudio()
        let d = try #require(e.description)
        #expect(d.hasPrefix("# SAM-Audio Base"))
        #expect(d.contains("two tracks"))
        #expect(d.contains("30-second windows with a 3-second overlap"), "MC1-10: the shipped window length")
        #expect(!d.contains("10-second"))
        #expect(!d.localizedCaseInsensitiveContains("clean speech"), "R8: no promise of a clean result")
        #expect(d.contains("a speech track plus a track of cups, chairs and chatter"))
        #expect(d.contains("takes minutes"))
        #expect(d.contains("up to 30 minutes"))
        #expect(d.contains("48 kHz"))
        #expect(d.contains("| **License** | SAM License |"))
        #expect(d.contains("SAM License"))
        #expect(d.contains("https://huggingface.co/facebook/sam-audio-base/blob/main/LICENSE"))
        #expect(d.contains("t5/"))
        #expect(d.contains("google-t5/t5-base"))
        #expect(d.contains("nothing is downloaded at run time"))
        for forbidden in ["state-of-the-art", "best in class", "outperform", "accuracy", "real-time", "real time", "PESQ", "SOTA"] {
            #expect(!d.localizedCaseInsensitiveContains(forbidden), "unsupported claim: \(forbidden)")
        }
        #expect((e.summary ?? "").contains("SAM-Audio"))
        #expect((e.summary ?? "").contains("48 kHz"))
    }

    /// The selector's view of both repos: SAM-Audio's own files only, and T5's text encoder — never the PyTorch / TF /
    /// Flax / ONNX / Rust weights `google-t5/t5-base` also ships.
    @Test func selectorTakesTheSAMAudioFilesAndTheT5EncoderOnly() {
        let sam = Set(ModelFileSelector.filesToDownload(siblings: Self.siblings))
        #expect(sam == ["config.json", "model-00001-of-00001.safetensors", "model.safetensors.index.json"])
        let t5 = Set(ModelFileSelector.filesToDownload(siblings: Self.t5Siblings))
        #expect(t5 == ["config.json", "generation_config.json", "model.safetensors", "spiece.model", "tokenizer.json"])
        for excluded in ["pytorch_model.bin", "tf_model.h5", "flax_model.msgpack", "rust_model.ot"] {
            #expect(!t5.contains(excluded), "\(excluded) must not be downloaded")
        }
    }

    @Test func noWorkflowPoolOffersIt() throws {
        let e = try Self.samAudio()
        for task in TaskCatalog.allTasks().map(\.name) {
            #expect(TaskModels.runnerKind(forTask: task) != .separate, "\(task) names the separate kind")
            let pool = TaskModels.derivedModels(for: task, catalog: [e], claimableModelIDs: [e.id])
            #expect(!pool.contains { $0.id == e.id }, "\(task) offers SAM-Audio")
        }
    }
}
