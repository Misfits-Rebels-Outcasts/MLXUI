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
