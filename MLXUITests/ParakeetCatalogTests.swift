import Testing
import Foundation
@testable import MLXUI

/// MP2-1: the Parakeet TDT 0.6B v3 catalog entry. The module lands in MP2-3; until then this
/// pins the entry's facts (from the pinned HF files, RSI/notes/modelpack/hf/) and proves its
/// `family` string collides with none of the existing ASR family checks (readme T3).
@MainActor
struct ParakeetCatalogTests {

    private func parakeet() throws -> ModelEntry {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data)
        let asr = try #require(catalog.domains.first { $0.id == "asr" })
        return try #require(asr.allModels.first { $0.hfModelId == "mlx-community/parakeet-tdt-0.6b-v3" })
    }

    @Test func entryFactsMatchThePinnedRepo() throws {
        let e = try parakeet()
        #expect(e.id == "mlx-community--parakeet-tdt-0.6b-v3")
        #expect(e.modelType == .asr)
        #expect(e.source == .mlx)
        #expect(e.license == "cc-by-4.0")
        #expect(e.downloadSizeGB == 2.51)
        // MP-0 G: ramGB is an estimate, download size × 1.5.
        #expect(abs(e.ramGB - e.downloadSizeGB * 1.5) < 0.01)
        #expect(e.languages?.count == 25)
    }

    @Test func isTheLastEntryInTheAsrLeaf() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data)
        let asr = try #require(catalog.domains.first { $0.id == "asr" })
        #expect(asr.allModels.last?.hfModelId == "mlx-community/parakeet-tdt-0.6b-v3")
    }

    @Test func familyCollidesWithNeitherWhisperNorVoxtral() throws {
        let e = try parakeet()
        let family = e.family.lowercased()
        #expect(!family.contains("whisper"))
        #expect(!family.contains("voxtral"))
        #expect(WhisperSDK().claim(e) == .no)
        #expect(MLXWhisperSDK().claim(e) == .no)
        #expect(VoxtralSDK().claim(e) == .no)
    }

    // MP2-2: display name (owner ruling L, 2026-10-24) and NVIDIA / CC-BY-4.0 attribution (MP-0 H).
    @Test func displayNameAndAttribution() throws {
        let e = try parakeet()
        #expect(e.displayName == "Parakeet TDT 0.6B v3")
        let description = e.description ?? ""
        #expect(description.contains("Original model by NVIDIA, licensed under"))
        #expect(description.contains("CC-BY-4.0"))
        #expect(description.contains("nvidia/parakeet-tdt-0.6b-v3"))
        #expect(description.contains("RAM figure is an estimate (download size × 1.5)"))
        #expect(description.contains("Timestamps"))
    }
}
