import Testing
import Foundation
@testable import MLXUI

/// FA-2 for issue #13: the Qwen3-ForcedAligner catalog entry, the new `align` kind, the pinned HF file list and the
/// download selection. Facts come from `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` at revision
/// `0e1a68e91d815300c7c9754b2a7639378b23db15` (fetched 2026-11-19; see `Fixtures/ForcedAligner/repo-files-ForcedAligner.json`),
/// and from the base model's card `Qwen/Qwen3-ForcedAligner-0.6B`.
///
/// The runner-declaring tests (claim, stage, resolution) land with the module in FA-5; the download side effect
/// (the library writes `tokenizer.json` into the folder on first load, readme T5 / trap 1) is in
/// `ForcedAlignerInstallTests.swift`. The parity fixtures land in FA-4.
@MainActor
struct ForcedAlignerCatalogTests {

    private static let hfID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

    private static var fixtureDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .appendingPathComponent("Fixtures/ForcedAligner")
    }

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func aligner() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    /// The pinned revision the fixtures and the loader read from, and trap 7: the entry's `hfModelId` is **exactly**
    /// this conversion, never a 4-, 5- or 6-bit sibling and never a third-party requantisation.
    @Test func entryFactsMatchTheRepo() throws {
        let fixtureURL = Self.fixtureDirectory.appendingPathComponent("repo-files-ForcedAligner.json")
        let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL)) as? [String: Any]
        #expect(fixture?["repository"] as? String == Self.hfID, "the pinned file-list fixture is the same repo")
        #expect(fixture?["revision"] as? String == "0e1a68e91d815300c7c9754b2a7639378b23db15")

        let e = try aligner()
        #expect(e.id == "mlx-community--Qwen3-ForcedAligner-0.6B-8bit")
        // Trap 7 — the licence lookalikes: the checkpoint mlx-audio's own docs use, as converted (8-bit text decoder).
        #expect(e.hfModelId == "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
                "owner decision 1: not the 4/5/6-bit siblings and not a third-party requantisation")
        #expect(e.hfRepo == "mlx-community")
        #expect(e.displayName == "Qwen3 Aligner 0.6B")
        // Trap 3 — the Qwen3-ASR SDK claims families containing `qwen3-asr`; this family must not contain it.
        #expect(e.family == "Qwen3-ForcedAligner")
        #expect(!e.family.lowercased().contains("qwen3-asr"))
        #expect(e.modelType == .align)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-8bit")
        #expect(e.license == "apache-2.0", "the HF card's licence: Apache-2.0")
        #expect(e.licenseUrl == "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B")
        // 1,276,475,979 bytes for the 11 files = 1.28 GB; the weights alone are 1,271,924,386 bytes.
        #expect(e.downloadSizeGB == 1.28)
        // FA-1's provisional figure, 1.5 x the weights: see the About text's closing line.
        #expect(e.ramGB == 1.91)
        #expect(e.paramCountB == 0.6)
        #expect(e.paramSize == "0.6B")
        #expect(e.contextWindow == nil, "an aligner has no token context window")
        #expect(e.speedTokensPerSec == nil, "not a token generator")
        #expect(e.speedHardware == nil)
        #expect(e.speedEstimated == nil)
        #expect(e.variants?.count == 1, "one variant, 8-bit as converted (owner decision 1)")
        #expect(e.variants?.first?.quantization == "8-bit")
        #expect(e.variants?.first?.format == "mlx-8bit")
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.variants?.first?.downloadSizeGB == e.downloadSizeGB)
        #expect(e.variants?.first?.qualityPercent == 100)
        #expect(e.variants?.first?.hfModelId == Self.hfID)
        #expect(e.variants?.first?.recommended == true)
    }

    @Test func kindRoutesToAlign() throws {
        let e = try aligner()
        #expect(e.runnerKind == .align)
        #expect(ModelType.align.rawValue == "align")
        #expect(RunnerKind.align.rawValue == "align")
        #expect(ModelType.align.sfSymbol == "captions.bubble")
    }

    /// Owner decision 3: a Browse leaf of its own — "Forced Alignment" (domain id `align`) in the Audio section, after
    /// Sound Isolation (`separate`) — so it never appears under Speech-to-Text.
    @Test func livesInItsOwnLeafInAudioAfterSoundIsolation() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "align" })
        #expect(leaf.name == "Forced Alignment")
        #expect(leaf.sfSymbol == "captions.bubble")
        #expect(leaf.allModels.map(\.hfModelId) == [Self.hfID])
        let section = try #require(data.sidebarSections.first { $0.id == "audio" })
        #expect(section.domainIds.contains("align"))
        #expect(section.domainIds.last == "align", "owner decision 3: after Sound Isolation")
        #expect(data.sidebarSections.filter { $0.domainIds.contains("align") }.map(\.id) == ["audio"])
        #expect(data.domainIndex["align"] != nil)
        // Sound Isolation and Speech-to-Text are untouched.
        let separate = try #require(data.domains.first { $0.id == "separate" })
        #expect(!separate.allModels.contains { $0.hfModelId == Self.hfID })
        let asr = try #require(data.domains.first { $0.id == "asr" })
        #expect(!asr.allModels.contains { $0.hfModelId == Self.hfID }, "the aligner is not an ASR model")
    }

    /// Owner decision 6: the 11 language codes `Transcribe` already uses.
    @Test func carriesTheElevenLanguageCodes() throws {
        let e = try aligner()
        #expect(e.languages == ["zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"])
    }

    /// The About text: what it does, the 11 languages, the 300 s limit, the 80 ms steps, "the transcript must match the
    /// audio", the attribution and the provisional RAM note. No accuracy claim the card doesn't make.
    @Test func descriptionStatesTheContractTheLimitsAndTheLicence() throws {
        let d = try #require(try aligner().description)
        #expect(d.contains("forced-alignment"))
        #expect(d.contains("11 languages"))
        #expect(d.contains("Chinese") && d.contains("Japanese"))
        #expect(d.contains("80 ms"))
        #expect(d.contains("5 minutes") || d.contains("five minutes"))
        #expect(d.contains("The transcript must match the audio"))
        #expect(d.contains("not transcription"))
        #expect(d.contains("Apache-2.0"))
        #expect(d.contains("Qwen team"))
        #expect(d.contains("mlx-community"))
        for banned in ["state-of-the-art", "SOTA", "best ", "outperforms", "WER", "accuracy of"] {
            #expect(!d.contains(banned), "unsupported claim: \(banned)")
        }
        // FA-1's provisional ramGB carries a note saying it is an estimate.
        #expect(d.lowercased().contains("provisional estimate"))
    }

    /// Every field `ModelDetailView` and `ComparisonTableView` render, pinned so the only dashes are the sanctioned
    /// nulls: Speed (not a token generator), Context and benchmarks (none on the card).
    @Test func metadataRendersWithNoUnsanctionedDash() throws {
        let e = try aligner()
        #expect(e.descriptionSource == Self.hfID)
        #expect(e.taskTags == ["speech-to-text", "forced-alignment"])
        #expect(e.lastUpdated == "2026-01-29")   // HF lastModified
        #expect(e.communityLikes == 4)
        #expect(e.communityDownloads == 4997)
        #expect(e.formattedDownloads() == "5.0K")
        #expect(e.contextWindow == nil)
        #expect(e.benchmarks == nil)
        #expect(DataNormalizer.normalizeLicense(e.license) == "Apache 2.0")
        #expect(e.summary?.isEmpty == false)
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US_POSIX")
        day.dateFormat = "yyyy-MM-dd"
        day.isLenient = false
        #expect(e.lastUpdated.flatMap { day.date(from: $0) } != nil)
    }

    /// Trap 2: the repo's siblings at the pinned revision, from the fixture. `ModelFileSelector` takes every file the
    /// loader reads, plus three it does not read, which the selector's earlier rules pick for any repo that has them.
    /// `README.md` and `.gitattributes` are not downloaded, and nothing was added to `metadataNames` for this model.
    ///
    /// What `Qwen3ForcedAlignerModel.fromModelDirectory` reads (mlx-audio-swift @ 5c4c12b, swift-transformers 1.3.3):
    /// - `Qwen3ForcedAligner.swift:557-558` `config.json`, and `:572-577` every `*.safetensors`;
    /// - `:566` `generateTokenizerJSONIfMissing` (`Qwen3ASR.swift:1662-1696`): `vocab.json`, `merges.txt` and
    ///   `tokenizer_config.json`, from which it writes `tokenizer.json`;
    /// - `:569` `AutoTokenizer.from(modelFolder:)` (swift-transformers `Hub.swift:247-290`): `config.json`, that
    ///   `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja` or, without one (this repo), `chat_template.json`.
    ///
    /// `preprocessor_config.json`: the pinned aligner (mlx-audio-swift @ 5c4c12b) does **not** read it.
    /// `preprocessAudio` in `Qwen3ForcedAligner.swift` hard-codes 16 kHz, nFft 400 and hop 160, and takes the mel-bin
    /// count from `config.json`. The existing `metadataNames` rule in `ModelFileSelector.swift` still downloads it.
    /// This differs from issue #13 trap 2, which lists it among the files the loader reads.
    /// Owner ruling 24 (Boo, 2026-11-28): "Keep the download as it is, because the selection rule is shared and out of
    /// scope here. Change the file-list test's comment to say the pinned aligner doesn't read the file and the existing
    /// rule still downloads it. Note the difference from issue #13 in the PR description. [...]"
    @Test func selectorTakesTheWeightsAndMetadataOnly() throws {
        let url = Self.fixtureDirectory.appendingPathComponent("repo-files-ForcedAligner.json")
        let object = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let files = try #require(object["files"] as? [[String: Any]])
        let siblings = files.compactMap { $0["path"] as? String }
        #expect(siblings == [".gitattributes", "README.md", "chat_template.json", "config.json", "generation_config.json",
                             "merges.txt", "model.safetensors", "model.safetensors.index.json",
                             "preprocessor_config.json", "tokenizer_config.json", "vocab.json"])
        let selected = Set(ModelFileSelector.filesToDownload(siblings: siblings))
        let loaderReads: Set<String> = ["config.json", "model.safetensors", "vocab.json", "merges.txt",
                                        "tokenizer_config.json", "chat_template.json"]
        #expect(loaderReads.isSubset(of: selected), "missing: \(loaderReads.subtracting(selected).sorted())")
        // Not read by the aligner. They come from rules that existed before this model: `metadataNames` in
        // `ModelFileSelector.swift` (line 20 `generation_config.json`, line 23 `preprocessor_config.json`) and the
        // sharded-index rule (line 224 `model.safetensors.index.json`).
        #expect(selected.subtracting(loaderReads)
                == ["generation_config.json", "preprocessor_config.json", "model.safetensors.index.json"])
        #expect(!selected.contains("tokenizer.json"), "the repo ships none; the library writes one on first load")
        #expect(files.first { $0["path"] as? String == "model.safetensors" }?["size"] as? Int == 1271924386)
    }

    /// `AppState.loadBrowserData()` decodes the shipped catalog with the new entry (FA-1's done-when), and the model has
    /// no support gap.
    @Test func bundledCatalogDecodesThroughAppState() throws {
        let state = AppState()
        state.loadBrowserData()
        #expect(state.loadError == nil, "browser.json must decode: \(state.loadError ?? "")")
        let data = try #require(state.browserData)
        let entry = try #require(data.domains.flatMap { $0.allModels }.first { $0.hfModelId == Self.hfID })
        #expect(ModelSupport.unsupportedReason(for: entry) == nil)
        #expect(throws: StageError.self) {
            _ = try StageFactory.make(for: entry, installedModelIDs: [entry.id], availableRAMGB: 16)
        }
    }
}
