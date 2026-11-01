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
