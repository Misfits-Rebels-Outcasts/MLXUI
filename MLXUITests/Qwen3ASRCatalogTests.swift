import Testing
import Foundation
@testable import MLXUI

/// MP3-1: the Qwen3-ASR 1.7B (8-bit) catalog entry. Pins its facts (from the pinned HF files,
/// RSI/notes/modelpack/hf/mlx-community_Qwen3-ASR-1.7B-8bit @ a8379a2e) and proves its `family`
/// string collides with none of the existing ASR family checks (readme T3).
@MainActor
struct Qwen3ASRCatalogTests {

    private func loadASRLeaf() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data)
        return try #require(catalog.domains.first { $0.id == "asr" }).allModels
    }

    private func qwen() throws -> ModelEntry {
        try #require(loadASRLeaf().first { $0.hfModelId == "mlx-community/Qwen3-ASR-1.7B-8bit" })
    }

    @Test func entryFactsMatchThePinnedRepo() throws {
        let e = try qwen()
        #expect(e.id == "mlx-community--Qwen3-ASR-1.7B-8bit")
        #expect(e.modelType == .asr)
        #expect(e.source == .mlx)
        // Card front matter: `license: apache-2.0`.
        #expect(e.license == "apache-2.0")
        // Sum of the pinned tree listing's 9 selected files (2,467,856,503 B) = 2.47 GB; 2,467,859,030 B for all 11.
        #expect(e.downloadSizeGB == 2.47)
        // MP-0 G: ramGB is an estimate, download size × 1.5 (nobody has measured it).
        #expect(abs(e.ramGB - e.downloadSizeGB * 1.5) < 0.01)
        #expect(e.languages?.count == 30)
        #expect(e.variants?.count == 1)
        #expect(e.variants?.first?.quantization == "8-bit")
        #expect(e.speedTokensPerSec == nil)
    }

    @Test func isTheLastEntryInTheAsrLeaf() throws {
        let asr = try loadASRLeaf()
        #expect(asr.last?.hfModelId == "mlx-community/Qwen3-ASR-1.7B-8bit")
        // Appended after Parakeet, not in front of it.
        #expect(asr.dropLast().last?.hfModelId == "mlx-community/parakeet-tdt-0.6b-v3")
    }

    @Test func familyCollidesWithNoOtherASRFamily() throws {
        let e = try qwen()
        let family = e.family.lowercased()
        #expect(!family.contains("whisper"))
        #expect(!family.contains("voxtral"))
        #expect(!family.contains("parakeet"))
        #expect(WhisperSDK().claim(e) == .no)
        #expect(MLXWhisperSDK().claim(e) == .no)
        #expect(VoxtralSDK().claim(e) == .no)
        #expect(ParakeetSDK().claim(e) == .no)
    }
}

// MP3-2: the description and metadata.
extension Qwen3ASRCatalogTests {

    /// `support_languages` from the pinned config.json (mlx-community/Qwen3-ASR-1.7B-8bit @ a8379a2e), in file order.
    /// Copied here because the pinned folder is outside the repository.
    private static let configLanguages = [
        "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese", "Indonesian",
        "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch",
        "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian",
        "Macedonian",
    ]

    @Test func descriptionListsTheThirtyConfigLanguagesVerbatim() throws {
        let e = try qwen()
        let description = e.description ?? ""
        #expect(Self.configLanguages.count == 30)
        #expect(description.contains("30 languages, as listed in the model's configuration: "
                                     + Self.configLanguages.joined(separator: ", ") + "."))
        #expect(e.languages?.count == 30)
    }

    @Test func descriptionSaysLanguageIsDetectedAutomaticallyAndShowsTextOnly() throws {
        let e = try qwen()
        let description = e.description ?? ""
        #expect(description.contains("## Language detection"))
        #expect(description.contains("identifies the language of the audio itself"))
        #expect(description.contains("no timestamps"))
        #expect((e.summary ?? "").contains("automatic language detection"))
    }

    /// The card (README.md) makes no accuracy claim, and the forced aligner is a separate model that is out of scope.
    @Test func descriptionMakesNoAccuracyOrAlignerClaim() throws {
        let description = try qwen().description ?? ""
        for forbidden in ["word error", "accuracy", "state-of-the-art", "outperform", "aligner", "word-level"] {
            #expect(!description.localizedCaseInsensitiveContains(forbidden), "\(forbidden)")
        }
        // "WER" as a word, not as a substring (it sits inside ordinary words).
        #expect(description.range(of: #"\bWER\b"#, options: .regularExpression) == nil)
    }

    @Test func attributionLicenseAndEstimateLines() throws {
        let e = try qwen()
        let description = e.description ?? ""
        #expect(e.displayName == "Qwen3-ASR 1.7B")
        #expect(description.contains("Original model by the Qwen team, licensed under"))
        #expect(description.contains("Apache-2.0"))
        #expect(description.contains("Qwen/Qwen3-ASR-1.7B"))
        #expect(description.contains("RAM figure is an estimate (download size × 1.5)"))
    }
}
