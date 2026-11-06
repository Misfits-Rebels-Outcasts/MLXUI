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

// MC4-2: the About text, the metadata and the download file list.
extension Qwen3VLEmbeddingCatalogTests {

    /// The pinned repo's file list (`/api/models/mlx-community/Qwen3-VL-Embedding-2B-4bit?blobs=true`
    /// at c713a96, fetched 2026-11-07), in the tree's order: name and size in bytes.
    static let pinnedFiles: [(name: String, bytes: Int)] = [
        (".gitattributes", 1_570),
        ("README.md", 1_240),
        ("added_tokens.json", 707),
        ("chat_template.jinja", 5_519),
        ("config.json", 1_949),
        ("model.safetensors", 1_782_040_155),
        ("model.safetensors.index.json", 82_780),
        ("preprocessor_config.json", 783),
        ("special_tokens_map.json", 613),
        ("tokenizer.json", 11_423_701),
        ("tokenizer_config.json", 348),
        ("video_preprocessor_config.json", 817),
        ("vocab.json", 2_776_833),
    ]

    /// Every field `ModelDetailView` and `ComparisonTableView` render, pinned so the only dashes
    /// are the sanctioned nulls: Speed (no tokens/s figure for an embedder), benchmarks (the card's
    /// MTEB-style comparisons are not the catalog's quality benchmarks), and nothing else.
    @Test func metadataRendersWithNoUnsanctionedDash() throws {
        let e = try entry()
        #expect(e.architecture == "Qwen3VLForConditionalGeneration")
        #expect(e.descriptionSource == "mlx-community/Qwen3-VL-Embedding-2B-4bit")
        #expect(e.taskTags == ["feature-extraction", "image-text-retrieval", "multimodal-embedding"])
        #expect(e.languages == ["multilingual"])
        #expect(e.lastUpdated == "2026-03-13")
        #expect(e.communityLikes == 2)
        #expect(e.communityDownloads == 367)
        #expect(e.formattedDownloads() != "—")
        #expect(e.speedTokensPerSec == nil)
        #expect(e.contextWindow != nil, "the input cap is a rendered fact, not a dash")
        #expect(e.benchmarks == nil)
        #expect(DataNormalizer.normalizeLicense(e.license) != "—")
        #expect(!e.paramSize.isEmpty)
        #expect(e.summary?.isEmpty == false)
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US_POSIX")
        day.dateFormat = "yyyy-MM-dd"
        day.isLenient = false
        #expect(e.lastUpdated.flatMap { day.date(from: $0) } != nil)
    }

    /// The About text says what the model does, its dimensions and input limits, that its vectors
    /// are not interchangeable with other embedders', and the licence — and makes no quality claim
    /// the card does not make.
    @Test func descriptionStatesTheSpaceTheDimensionsTheLimitsAndTheLicence() throws {
        let d = try #require(try entry().description)
        #expect(d.contains("2048-dimension"))
        #expect(d.contains("one image"))
        #expect(d.contains("8,192 tokens"))
        #expect(d.contains("32K"))
        #expect(d.contains("4,096 to 1,310,720 pixels"))
        #expect(d.contains("1,280 vision tokens"))
        #expect(d.contains("no video"))
        #expect(d.contains("refused by the dimension check"))
        #expect(d.contains("rebuilt"))
        #expect(d.contains("Apache-2.0"))
        #expect(d.contains("provisional estimate"))
        for banned in ["best", "state-of-the-art", "SOTA", "outperforms", "beats", "top of the leaderboard"] {
            #expect(!d.lowercased().contains(banned.lowercased()), "unsupported claim: \(banned)")
        }
    }

    /// MC4-2's download contract: the pinned file list resolves to exactly the 4-bit checkpoint,
    /// its index, configs, chat template and tokenizer — not the video processor (video is out of
    /// scope and the Swift `Qwen3VLProcessor` reads only `preprocessor_config.json`), the README or
    /// `.gitattributes`.
    @Test func selectorPicksTheCheckpointTheConfigsAndTheTokenizer() throws {
        let siblings = Self.pinnedFiles.map(\.name)
        let out = Set(ModelFileSelector.filesToDownload(siblings: siblings))
        #expect(out == [
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ])
        #expect(!out.contains("video_preprocessor_config.json"))
        #expect(!out.contains("README.md"))
        #expect(!out.contains(".gitattributes"))
        // The whole repo is 1,796,337,015 B; the selected files are the same 1.80 GB after the
        // docs and the video processor are dropped.
        let repoBytes = Self.pinnedFiles.reduce(0) { $0 + $1.bytes }
        #expect(repoBytes == 1_796_337_015)
        #expect(try entry().downloadSizeGB == 1.8)
    }
}
