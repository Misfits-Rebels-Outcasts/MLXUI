import Testing
import Foundation
@testable import MLXUI

/// MP1-1 (`RSI/DelegateModelPackBacklog.md`): Gemma 4 E4B is two catalog entries — Chat & Text and
/// Vision & OCR — over one `mlx-community/gemma-4-e4b-it-4bit` install (MP-0 E, ruled 2026-10-21).
/// Every number asserted here comes from the pinned HF files in
/// `RSI/notes/modelpack/hf/mlx-community_gemma-4-e4b-it-4bit/` (revision 475b9088d2…).
/// MP1-1-FIX-1: ids use HF's canonical casing (`api.json` `id`, readme T11), all lowercase here.
struct Gemma4CatalogTests {

    private static let repo = "mlx-community/gemma-4-e4b-it-4bit"
    private static let llmID = "mlx-community--gemma-4-e4b-it-4bit"
    private static let visionID = "mlx-community--gemma-4-e4b-it-4bit-vision"

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

    @Test func bothEntriesDecodeAndShareOneInstall() throws {
        let leaves = try loadLeaves()
        let llm = try entry(Self.llmID, in: leaves)
        let vision = try entry(Self.visionID, in: leaves)

        #expect(llm.modelType == .llm)
        #expect(vision.modelType == .vision)
        #expect(llm.displayName == "Gemma 4 E4B")
        #expect(vision.displayName == "Gemma 4 E4B Vision")
        // The shared-install key (MoC-5) is the repo id, so it must be identical.
        #expect(llm.hfModelId == Self.repo)
        #expect(vision.hfModelId == Self.repo)
        #expect(llm.source == .mlx && vision.source == .mlx)
    }

    @Test func idsAndFamilyFollowHFCanonicalCasingAndTheGemmaPrecedent() throws {
        let leaves = try loadLeaves()
        let llm = try entry(Self.llmID, in: leaves)
        let vision = try entry(Self.visionID, in: leaves)
        #expect(llm.hfModelId == "mlx-community/gemma-4-e4b-it-4bit")
        #expect(Self.visionID == Self.llmID + "-vision")
        // Existing Gemma 2/3 entries use family "gemma" (confirmed by owner 2026-10-21).
        #expect(llm.family == "gemma" && vision.family == "gemma")
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
        #expect(try entry(Self.visionID, in: leaves).hfModelId == pinned)
    }

    @Test func numbersMatchThePinnedRepo() throws {
        let leaves = try loadLeaves()
        for id in [Self.llmID, Self.visionID] {
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

    @Test func entriesLandInTheirLeavesAfterTheirNeighbours() throws {
        let leaves = try loadLeaves()
        let llmLeaf = try #require(leaves.first { $0.id == "llm" })
        let visionLeaf = try #require(leaves.first { $0.id == "vision" })
        let llmIDs = (llmLeaf.models ?? []).map(\.id)
        let visionIDs = (visionLeaf.models ?? []).map(\.id)

        // Backlog MP1-1: llm leaf after MiMo, vision leaf after gemma-3-12b-it-4bit.
        let mimo = try #require(llmIDs.firstIndex(of: "mlx-community--MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit"))
        #expect(llmIDs[mimo + 1] == Self.llmID)
        let gemma3 = try #require(visionIDs.firstIndex(of: "mlx-community--gemma-3-12b-it-4bit"))
        #expect(visionIDs[gemma3 + 1] == Self.visionID)
        #expect(!llmIDs.contains(Self.visionID) && !visionIDs.contains(Self.llmID))
    }

    @Test func descriptionsClaimNoAudioInput() throws {
        // Gemma4.swift (VLM) drops audio_tower / embed_audio at sanitize: image + text only in MLXUI.
        let leaves = try loadLeaves()
        for id in [Self.llmID, Self.visionID] {
            let e = try entry(id, in: leaves)
            let text = [e.description, e.summary].compactMap { $0 }.joined(separator: " ").lowercased()
            for claim in ["audio input", "speech", "voice", "transcri"] {
                #expect(!text.contains(claim), "\(id): must not claim or offer \(claim)")
            }
        }
        let vision = try entry(Self.visionID, in: leaves)
        #expect(vision.taskTags == ["image-text-to-text"])
    }
}
