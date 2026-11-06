import Testing
import Foundation
@testable import MLXUI

/// MC4-1: the Qwen3-VL Embedding 2B catalog entry. Facts are from the pinned mlx-community repo
/// (`mlx-community/Qwen3-VL-Embedding-2B-4bit` @ c713a965, fetched 2026-11-07) and the issue's
/// decisions. The embedder itself (dispatch, forward pass, Images tab) lands in MC4-4…MC4-6; this
/// file starts with what the catalog must say and which module must resolve the entry.
@MainActor
struct Qwen3VLEmbeddingCatalogTests {

    private static let hfID = "mlx-community/Qwen3-VL-Embedding-2B-4bit"

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

    @Test func catalogDecodesWithTheEntry() throws {
        let all = try catalog().domains.flatMap { $0.allModels }
        #expect(all.count == 56, "every MC4-1/MC4-8 step appends one entry; the count guard lives in CatFlowChatModuleTests")
        #expect(try entry().id == "mlx-community--Qwen3-VL-Embedding-2B-4bit")
    }

    @Test func entryFactsMatchTheRepo() throws {
        let e = try entry()
        #expect(e.displayName == "Qwen3-VL Embedding 2B")
        #expect(e.family == "qwen3-vl-embedding")
        #expect(e.modelType == .embedding)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-4bit")
        #expect(e.architecture == "Qwen3VLForConditionalGeneration")
        #expect(e.paramCountB == 2.0)
        // Decision 4: this version caps input at 8,192 tokens after rendering (the model supports 32K).
        #expect(e.contextWindow == 8192)
        // 1,796,337,015 B for the whole pinned repo = 1.80 GB.
        #expect(e.downloadSizeGB == 1.8)
        // Decision 10: Apache-2.0 with the model page as licenseUrl.
        #expect(e.license == "apache-2.0")
        #expect(e.licenseUrl == "https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B")
        #expect(e.speedTokensPerSec == nil)
        #expect(e.benchmarks == nil)
        #expect(e.communityDownloads == 367)
        #expect(e.communityLikes == 2)
        #expect(e.lastUpdated == "2026-03-13")
    }

    /// Decision 9: one 4-bit variant, `ramGB` provisional (1.5× weights ≈ 2.7) until MC4-7
    /// measures the peak — the maintainers then set the final value, and no `.embedding` row is added.
    @Test func oneProvisionalVariantUntilThePeakIsMeasured() throws {
        let e = try entry()
        #expect(e.ramGB == 2.7)
        #expect(e.ramGB == (e.downloadSizeGB * 1.5 * 10).rounded() / 10, "provisional 1.5× the weights")
        #expect(e.variants?.count == 1)
        #expect(e.variants?.first?.quantization == "4-bit")
        #expect(e.variants?.first?.format == "mlx-4bit")
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.variants?.first?.downloadSizeGB == e.downloadSizeGB)
        #expect(e.variants?.first?.hfModelId == Self.hfID)
        #expect(e.variants?.first?.recommended == true)
        // Implementer's call, pending owner confirmation: qualityPercent 90 — a strong multimodal
        // embedder below the text-only 8B class, above the small 300M/0.6B embedders.
        #expect(e.variants?.first?.qualityPercent == 90)
        #expect(e.description?.contains("provisional estimate") == true)
    }

    @Test func appendedLastInTheEmbeddingLeaf() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "embedding" })
        #expect(leaf.allModels.count == 5)
        #expect(leaf.allModels.last?.hfModelId == Self.hfID, "append, never reorder")
        #expect(leaf.allModels.dropLast().map(\.hfModelId) == [
            "mlx-community/all-MiniLM-L6-v2-4bit",
            "mlx-community/embeddinggemma-300m-4bit",
            "mlx-community/nomicai-modernbert-embed-base-bf16",
            "mlx-community/bge-m3-mlx-fp16",
        ])
        let section = try #require(data.sidebarSections.first { $0.id == "infrastructure" })
        #expect(section.domainIds.contains("embedding"))
        #expect(data.domainIndex["embedding"] != nil)
    }

    /// Trap 1: the chat model Qwen3-VL 4B shares `model_type` `qwen3_vl` and the architecture string.
    /// The embedder must be claimed by the existing Embedding module, and the 4B chat entry must keep
    /// resolving to the VLM module — never route by `model_type`/architecture anywhere else.
    @Test func onlyTheEmbeddingModuleClaimsItAndQwen3VL4BStaysVLM() throws {
        let e = try entry()
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        #expect(full.bestModule(for: e)?.descriptor.id == "embedding")
        var claimers: [String] = []
        for module in installedModules {
            let one = ModelRegistry()
            module.register(into: one)
            if one.bestModule(for: e) != nil { claimers.append(module.descriptor.id) }
        }
        #expect(claimers == ["embedding"], "a Qwen3-VL embedder must not be handed to the VLM module: \(claimers)")

        let all = try catalog().domains.flatMap { $0.allModels }
        let chatVLM = try #require(all.first { $0.hfModelId == "mlx-community/Qwen3-VL-4B-Instruct-4bit" })
        #expect(full.bestModule(for: chatVLM)?.descriptor.id == "vlm", "Qwen3-VL 4B must keep opening its VLM screen")
    }
}
