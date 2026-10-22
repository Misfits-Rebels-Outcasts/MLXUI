import Testing
import Foundation
@testable import MLXUI

/// MP1-1 (`RSI/DelegateModelPackBacklog.md`): Gemma 4 E4B is a Chat & Text catalog entry over
/// `mlx-community/gemma-4-e4b-it-4bit`. MP1-R-FIX-1 (owner ruling 2026-10-22) withdrew its Vision & OCR twin:
/// MLXVLM's `Gemma4.swift` in mlx-swift-lm 3.31.4 cannot load this checkpoint (see `RSI/backlog.md`).
/// Every number asserted here comes from the pinned HF files in
/// `RSI/notes/modelpack/hf/mlx-community_gemma-4-e4b-it-4bit/` (revision 475b9088d2…).
/// MP1-1-FIX-1: ids use HF's canonical casing (`api.json` `id`, readme T11), all lowercase here.
struct Gemma4CatalogTests {

    private static let repo = "mlx-community/gemma-4-e4b-it-4bit"
    private static let llmID = "mlx-community--gemma-4-e4b-it-4bit"

    private func loadLeaves() throws -> [DomainNode] {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()              // MLXUITests
            .deletingLastPathComponent()              // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.leafDomains }
    }

    private func entry(_ id: String, in leaves: [DomainNode]) throws -> ModelEntry {
        try #require(leaves.flatMap { $0.models ?? [] }.first { $0.id == id }, "\(id) is not in the bundled catalog")
    }

    @Test func entryDecodes() throws {
        let leaves = try loadLeaves()
        let llm = try entry(Self.llmID, in: leaves)

        #expect(llm.modelType == .llm)
        #expect(llm.displayName == "Gemma 4 E4B")
        #expect(llm.hfModelId == Self.repo)
        #expect(llm.source == .mlx)
    }

    /// MP1-R-FIX-1: the Vision twin is withdrawn until mlx-swift-lm's VLM Gemma 4 loads this checkpoint.
    @Test func visionEntryIsWithdrawn() throws {
        let leaves = try loadLeaves()
        let ids = leaves.flatMap { $0.models ?? [] }.map(\.id)
        #expect(!ids.contains { $0.hasPrefix(Self.llmID) && $0 != Self.llmID })
        #expect(!ids.contains(Self.llmID + "-vision"))
    }

    @Test func idAndFamilyFollowHFCanonicalCasingAndTheGemmaPrecedent() throws {
        let leaves = try loadLeaves()
        let llm = try entry(Self.llmID, in: leaves)
        #expect(llm.hfModelId == "mlx-community/gemma-4-e4b-it-4bit")
        // Existing Gemma 2/3 entries use family "gemma" (confirmed by owner 2026-10-21).
        #expect(llm.family == "gemma")
    }

    /// The pinned `api.json` lives under the gitignored `RSI/`: it exists in the owner's main folder (and
    /// beside a sibling worktree) but not in a clean clone, where this reports "skipped", not "passed".
    private static let pinnedAPIJSON: URL? = {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let rel = "RSI/notes/modelpack/hf/mlx-community_gemma-4-e4b-it-4bit/api.json"
        return [root, root.deletingLastPathComponent().appendingPathComponent("PipelineStudio")]
            .map { $0.appendingPathComponent(rel) }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }()

    @Test(.enabled(if: Gemma4CatalogTests.pinnedAPIJSON != nil))
    func hfModelIdEqualsThePinnedAPIJSONId() throws {
        let url = try #require(Self.pinnedAPIJSON)
        let api = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let pinned = try #require(api["id"] as? String)
        let leaves = try loadLeaves()
        #expect(try entry(Self.llmID, in: leaves).hfModelId == pinned)
    }

    @Test func numbersMatchThePinnedRepo() throws {
        let leaves = try loadLeaves()
        for id in [Self.llmID] {
            let e = try entry(id, in: leaves)
            #expect(e.downloadSizeGB == 5.18, "\(id): 5,179,241,512 bytes in the pinned tree")
            #expect(e.ramGB == 7.77, "\(id): MP-0 G, 5.18 × 1.5 (an estimate, not measured)")
            #expect(e.contextWindow == 131072, "\(id): text_config.max_position_embeddings")
            #expect(e.license == "gemma")
            #expect(e.speedTokensPerSec == nil)
            let variants = try #require(e.variants)
            #expect(variants.count == 1)
            let v = try #require(variants.first)
            #expect(v.quantization == "4-bit")
            #expect(v.hfModelId == Self.repo)
            #expect(v.downloadSizeGB == 5.18 && v.ramGB == 7.77)
        }
    }

    @Test func entryLandsInItsLeafAfterMiMo() throws {
        let leaves = try loadLeaves()
        let llmLeaf = try #require(leaves.first { $0.id == "llm" })
        let visionLeaf = try #require(leaves.first { $0.id == "vision" })
        let llmIDs = (llmLeaf.models ?? []).map(\.id)
        let visionIDs = (visionLeaf.models ?? []).map(\.id)

        // Backlog MP1-1: llm leaf after MiMo.
        let mimo = try #require(llmIDs.firstIndex(of: "mlx-community--MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit"))
        #expect(llmIDs[mimo + 1] == Self.llmID)
        #expect(!visionIDs.contains(Self.llmID))
    }

    @Test func descriptionsClaimNoAudioInput() throws {
        // Gemma4.swift (VLM) drops audio_tower / embed_audio at sanitize: image + text only in MLXUI.
        let leaves = try loadLeaves()
        for id in [Self.llmID] {
            let e = try entry(id, in: leaves)
            // An explicit negation ("so there is no audio input") is the one allowed mention.
            let text = [e.description, e.summary].compactMap { $0 }.joined(separator: " ").lowercased()
                .replacingOccurrences(of: "no audio input", with: "")
            for claim in ["audio input", "speech", "voice", "transcri"] {
                #expect(!text.contains(claim), "\(id): must not claim or offer \(claim)")
            }
        }
    }

    /// MP1-2: tool use and thinking are MP1-S's to decide, so the copy claims neither; benchmarks stay null
    /// because the pinned README states none; and every field the detail/comparison views show is present.
    @Test func metadataRendersWithoutGapsAndClaimsNoToolsThinkingOrBenchmarks() throws {
        let leaves = try loadLeaves()
        for id in [Self.llmID] {
            let e = try entry(id, in: leaves)
            let text = [e.description, e.summary].compactMap { $0 }.joined(separator: " ").lowercased()
            for claim in ["tool", "thinking", "reasoning", "function call", "benchmark score", "mmlu", "%"] {
                #expect(!text.contains(claim), "\(id): must not claim \(claim)")
            }
            #expect(e.benchmarks == nil)
            #expect(!(e.description ?? "").isEmpty && !(e.summary ?? "").isEmpty)
            #expect(DataNormalizer.normalizeLicense(e.license) == "Gemma License")
            #expect(DataNormalizer.normalizeArchitecture(e.architecture) == "Gemma4")
            #expect(e.lastUpdated == "2026-07-06")
            #expect(e.communityDownloads != nil && e.communityLikes != nil)
        }
        let llm = try entry(Self.llmID, in: leaves)
        #expect(llm.taskTags == ["text-generation"])
    }
}
