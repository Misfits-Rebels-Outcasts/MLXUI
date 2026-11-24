import Testing
import Foundation
@testable import MLXUI

/// PF-2: the Privacy Filter catalog entry, its About text, and the pinned download list. Facts come from the official
/// `openai/privacy-filter` repo at revision `7ffa9a043d54d1be65afb281eddf0ffbe629385b` (HF API, fetched 2026-11-18; see
/// `Fixtures/PrivacyFilter/privacy-repo-files.json`) and its model card. PF-1 added the entry and the `.redact` kind;
/// this file pins every field the model page and Compare render, the owner-decided licence/about wording, and Trap 8/9.
@MainActor
struct PrivacyFilterCatalogTests {

    private static let hfID = "openai/privacy-filter"
    private static let cardID = "openai--privacy-filter"
    private static let revision = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"

    private static var fixtureDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .appendingPathComponent("Fixtures/PrivacyFilter")
    }

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func entry() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    /// The pinned HF listing, as `{path, size}` pairs. The fixture is the authority for the file facts (Trap 9).
    private func repoFileList() throws -> [[String: Any]] {
        let url = Self.fixtureDirectory.appendingPathComponent("privacy-repo-files.json")
        let object = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        #expect(object["repository"] as? String == Self.hfID, "the pinned file-list fixture is the official repo")
        #expect(object["revision"] as? String == Self.revision)
        return try #require(object["files"] as? [[String: Any]])
    }

    // MARK: - The catalog entry (the model page and Compare)

    @Test func loadBrowserDataDecodesWithTheNewEntry() throws {
        let state = AppState()
        state.loadBrowserData()
        #expect(state.loadError == nil, "browser.json must decode: \(state.loadError ?? "")")
        let data = try #require(state.browserData)
        #expect(data.domains.flatMap { $0.allModels }.contains { $0.hfModelId == Self.hfID })
    }

    @Test func entryFactsMatchTheRepo() throws {
        let e = try entry()
        // Trap 8: the exact official repo. Never an mlx-community conversion or a fine-tune.
        #expect(e.hfModelId == "openai/privacy-filter")
        #expect(e.id == Self.cardID)
        #expect(e.hfRepo == "openai")
        #expect(e.displayName == "Privacy Filter")
        #expect(e.family == "privacy-filter")
        #expect(e.modelType == .redact)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-bf16")
        #expect(e.minMacOSVersion == "14.0")
        // Owner decision 2: the official bf16 checkpoint run as shipped.
        #expect(e.variants?.count == 1)
        let variant = try #require(e.variants?.first)
        #expect(variant.quantization == "bf16")
        #expect(variant.format == "mlx-bf16")
        #expect(variant.hfModelId == Self.hfID, "the variant is the official repo")
        #expect(variant.ramGB == e.ramGB)
        #expect(variant.downloadSizeGB == e.downloadSizeGB)
        #expect(variant.qualityPercent == 100, "bf16 as published: no quantisation loss to report")
        #expect(variant.recommended == true)
        // Model facts from the card/config.json.
        #expect(e.architecture == "OpenAIPrivacyFilterForTokenClassification")
        #expect(e.paramCountB == 1.5)
        #expect(e.paramSize == "1.5B")
        #expect(e.contextWindow == 131_072, "the model's documented 128K context")
        // Decision 9: the card's licence.
        #expect(e.license == "Apache-2.0")
        #expect(DataNormalizer.normalizeLicense(e.license) == "Apache 2.0")
        #expect(e.licenseUrl == "https://huggingface.co/openai/privacy-filter")
        // Speed is meaningless for a one-pass token classifier — null, not invented.
        #expect(e.speedTokensPerSec == nil)
        #expect(e.speedHardware == nil)
        #expect(e.speedEstimated == nil)
        // Rendered values on the detail page and in Compare.
        #expect(DataNormalizer.downloadSize(gb: e.downloadSizeGB) == "2.8 GB")
        #expect(e.lastUpdated == "2026-04-22")   // the pinned revision's HF lastModified
        #expect(e.formattedDownloads() == "104.1K")
        #expect(e.languages == ["English"], "the card's primary language")
        #expect(e.benchmarks == nil)
        #expect(e.qualityScore == nil)
        #expect(e.descriptionSource == "curated")
        #expect(e.summary?.isEmpty == false)
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US_POSIX")
        day.dateFormat = "yyyy-MM-dd"
        day.isLenient = false
        #expect(e.lastUpdated.flatMap { day.date(from: $0) } != nil)
    }

    @Test func kindRoutesToRedact() throws {
        let e = try entry()
        #expect(e.runnerKind == .redact)
        #expect(ModelType.redact.rawValue == "redact")
        #expect(RunnerKind.redact.rawValue == "redact")
        #expect(ModelType.redact.sfSymbol == "eye.slash")   // owner decision 3
        // The nonisolated mirror `MLXTools` keeps must agree with `ModelEntry.runnerKind` (MC2's precedent).
        #expect(InstalledModelIndex.kind(modelType: e.modelType.rawValue, family: e.family) == .redact)
        #expect(ModelSupport.unsupportedReason(for: e) == nil)
    }

    @Test func livesInItsOwnLeafInInfrastructureAfterRerank() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "redact" })
        #expect(leaf.name == "PII Redaction")
        #expect(leaf.sfSymbol == "eye.slash")
        #expect(leaf.allModels.map(\.hfModelId) == [Self.hfID])
        let section = try #require(data.sidebarSections.first { $0.id == "infrastructure" })
        #expect(section.domainIds.contains("redact"))
        #expect(section.domainIds.last == "redact", "owner decision 3: after Rerank")
        #expect(data.sidebarSections.filter { $0.domainIds.contains("redact") }.map(\.id) == ["infrastructure"])
        #expect(data.domainIndex["redact"] != nil)
        // The Rerank leaf is untouched.
        let rerank = try #require(data.domains.first { $0.id == "rerank" })
        #expect(!rerank.allModels.contains { $0.hfModelId == Self.hfID })
    }

    /// Owner ruling R1 (Boo, 2026-11-24): ramGB is the measured figure — 7.50 GB peak at the 128k-token worst case,
    /// 5.09 GB for load plus a sentence — rounded to 7.7. `ModelLoadProfile`'s `.redact` row books 7.679 GB.
    @Test func ramGBIsTheMeasuredPeak() throws {
        let e = try entry()
        #expect(e.ramGB == 7.7)
        #expect(e.variants?.first?.ramGB == 7.7)
        let d = try #require(e.description)
        #expect(!d.localizedCaseInsensitiveContains("provisional"))
        #expect(d.contains("7.50 GB"))
        #expect(d.contains("7.679 GB"))
    }

    /// Decision 9's About text: what it does, the eight categories, primarily English, can miss personal data, a
    /// redaction aid and not an anonymisation/compliance/safety guarantee — and **no** accuracy claim the card doesn't
    /// make.
    @Test func aboutTextStatesWhatItDoesAndTheCardLimitations() throws {
        let e = try entry()
        let d = try #require(e.description)
        // What it does: detection + the typed placeholder redaction.
        #expect(d.localizedCaseInsensitiveContains("personal data"))
        #expect(d.contains("<PRIVATE_PERSON>"))
        #expect(d.contains("<SECRET>"))
        // The eight categories, by model label.
        for category in ["private_person", "private_email", "private_phone", "private_url",
                         "private_address", "private_date", "account_number", "secret"] {
            #expect(d.contains(category), "About text must name the category \(category)")
        }
        // Decision 9's three limitations, from the model card.
        #expect(d.contains("Primarily English"))
        #expect(d.localizedCaseInsensitiveContains("can miss personal data"))
        #expect(d.localizedCaseInsensitiveContains("not an anonymisation, compliance or safety guarantee"))
        #expect(d.localizedCaseInsensitiveContains("redaction and data-minimisation aid")
                || d.localizedCaseInsensitiveContains("redaction aid"))
        // Decision 5's limit, and the text-only contract.
        #expect(d.contains("128,000"))
        #expect(d.localizedCaseInsensitiveContains("text only"))
        // The licence notice, on the page (the catalog is shared by both editions).
        #expect(d.contains("Apache-2.0"))
        #expect(d.contains("https://huggingface.co/openai/privacy-filter"))
        // No quality claim the card doesn't make.
        for claim in ["state-of-the-art", "SOTA", "best-in-class", "outperforms", "most accurate",
                      "highest accuracy", "accuracy of", "F1 score", "precision and recall"] {
            #expect(!d.localizedCaseInsensitiveContains(claim), "About text must not claim \(claim)")
        }
        // The short blurb stays honest and text-only.
        let summary = try #require(e.summary)
        #expect(summary.localizedCaseInsensitiveContains("Privacy Filter"))
        #expect(summary.localizedCaseInsensitiveContains("redaction aid"))
        #expect(!summary.localizedCaseInsensitiveContains("state-of-the-art"))
    }

    // MARK: - Download selection (Trap 9)

    /// Trap 9: the repo also ships `original/model.safetensors` (the `opf` project's own layout, ~2.8 GB) and a whole
    /// `onnx/` family. The loader reads the transformers layout only, so the installer must take exactly these five files.
    @Test func selectorTakesExactlyTheFilesTheLoaderReads() throws {
        let files = try repoFileList()
        let siblings = files.compactMap { $0["path"] as? String }
        // Control: the duplicate directories really are in the pinned listing (a fixture change would hide the bug).
        #expect(siblings.contains("original/model.safetensors"))
        #expect(siblings.contains { $0.hasPrefix("onnx/") })
        let selected = ModelFileSelector.filesToDownload(siblings: siblings)
        #expect(selected == [
            "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "viterbi_calibration.json",
        ])
        // Nothing under the upstream's alternate-packaging `original/` folder, no ONNX export, no signature, no docs.
        #expect(!selected.contains("original/model.safetensors"),
                "the 2.8 GB duplicate under original/ must not download")
        #expect(!selected.contains { $0.hasPrefix("original/") })
        #expect(!selected.contains { $0.hasPrefix("onnx/") })
        #expect(!selected.contains("model.sig"))
        #expect(!selected.contains("README.md"))
        #expect(!selected.contains(".gitattributes"))
        // …and `viterbi_calibration.json` is now an install metadata name at all (Trap 9's first half).
        #expect(ModelFileSelector.metadataNames.contains("viterbi_calibration.json"))
    }

    /// The catalog's `downloadSizeGB` is the pinned listing's own bytes for those five files (decimal GB, the catalog's
    /// convention), so the displayed size matches what the install actually fetches.
    @Test func downloadSizeIsTheSelectedFilesByteTotal() throws {
        let files = try repoFileList()
        let byPath = Dictionary(uniqueKeysWithValues: files.compactMap { file -> (String, Int64)? in
            guard let path = file["path"] as? String, let size = file["size"] as? Int64 else { return nil }
            return (path, size)
        })
        let selected = ModelFileSelector.filesToDownload(siblings: files.compactMap { $0["path"] as? String })
        let selectedBytes = selected.reduce(Int64(0)) { $0 + (byPath[$1] ?? 0) }
        let gigabytes = Double(selectedBytes) / 1_000_000_000.0
        let rounded = (gigabytes * 100).rounded() / 100
        let e = try entry()
        #expect(rounded == e.downloadSizeGB, "catalog \(e.downloadSizeGB) vs selected files \(rounded) GB")
        #expect(byPath["model.safetensors"] == 2_798_989_498,
                "the checkpoint itself is 2.798 GB (the fixture pins the pinned revision's size)")
    }

    /// The Trap 9 fix is narrow: it only drops `original/` in repos that also ship standard top-level weights. A repo
    /// whose only weights live under `original/` keeps the pre-existing component behaviour, and a component subfolder
    /// that the engine genuinely needs (Qwen3-TTS's `speech_tokenizer/`) is untouched.
    @Test func theAlternatePackagingRuleDoesNotChangeOtherRepoShapes() {
        // Only-original repo: unchanged (component rule still takes it).
        let onlyOriginal = ModelFileSelector.filesToDownload(siblings: [
            "config.json", "original/model.safetensors", "original/config.json",
        ])
        #expect(onlyOriginal.contains("original/model.safetensors"))
        // Standard weights + a genuinely-needed component: untouched.
        let component = ModelFileSelector.filesToDownload(siblings: [
            "config.json", "model.safetensors",
            "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors",
        ])
        #expect(component.contains("speech_tokenizer/model.safetensors"))
        #expect(component.contains("model.safetensors"))
        // A standard repo with an alternate-packaging folder: the folder is dropped, the rest is not.
        let standard = ModelFileSelector.filesToDownload(siblings: [
            "config.json", "model.safetensors", "original/model.safetensors",
            "original/config.json", "original/viterbi_calibration.json",
        ])
        #expect(standard == ["config.json", "model.safetensors"])
    }

    /// PF-8: the `.redact` row books `files x 1.10 + 4.60 GB`, so the shared-budget reservation is never under the
    /// highest measured peak — the 128,000-token phase, 7.582 GB (`PrivacyFilterFootprintTests`, 2026-11-18; phases 1/2
    /// peaked at 5.081 / 7.414 GB, an earlier run recorded 5.063 / 6.707 / 7.418 GB). `measured` is true: this is a
    /// measurement, not the conservative default.
    @Test func theRedactRowBooksFilesPlusTheMeasuredWorkingSet() {
        let redact = ModelLoadProfile.entry(for: .redact)
        #expect(redact.measured)
        #expect(redact.factor == ModelWeightCharge.loadFactor)
        #expect(redact.workingSetBytes == 4_600_000_000, "PF-8's measured per-call working set")
        // 2,798,989,498 B of safetensors x 1.10 + 4.60 GB, against the 7.582 GB worst-case peak.
        let booked = ModelWeightCharge.charge(onDiskBytes: 2_798_989_498) + redact.workingSetBytes
        #expect(booked >= 7_582_000_000, "booked \(booked) must cover the measured peak")
        // R1 (Boo, 2026-11-24): the catalog ramGB is the measured 7.7.
        let entries = try? catalog().domains.flatMap { $0.allModels }
        #expect(entries?.first { $0.hfModelId == Self.hfID }?.ramGB == 7.7)
    }

    /// The StageFactory's legacy kind switch refuses `.redact` by kind (the module's own SDK is what runs it — PF-7).
    @Test func stageFactoryRefusesTheKindExplicitly() throws {
        let e = try entry()
        #expect(throws: StageError.self) {
            _ = try StageFactory.make(for: e, installedModelIDs: [e.id], availableRAMGB: 16)
        }
    }
}
