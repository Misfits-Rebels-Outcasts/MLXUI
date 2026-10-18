import Foundation
import Testing
@testable import MLXUI

/// MM1 — the bundled catalog carries `mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit` as
/// "MiMo V2.6 9B": a text-only `qwen3_5` LLM next to `Qwen3.5-9B` in Chat & Text. Numbers are from
/// the pinned HF files (`RSI/notes/mimo/hf/`, MiMo @ `60061fc`); the RAM figure is the owner's
/// MM-0 B ruling (2026-10-17): 9.0 × 6.34/8 × 1.5 ≈ 10.70 at the real OptiQ bits per weight.
@MainActor
struct MiMoCatalogEntryTests {

    private static let id = "mlx-community--MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit"
    private static let hfID = "mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit"

    private func entry() throws -> ModelEntry {
        let state = AppState()
        state.loadBrowserData()
        #expect(state.loadError == nil, "browser.json must decode: \(state.loadError ?? "")")
        let data = try #require(state.browserData)
        return try #require(data.domains.flatMap { $0.allModels }.first { $0.id == Self.id })
    }

    @Test func entryDecodesWithTheRuledShape() throws {
        let e = try entry()
        #expect(e.displayName == "MiMo V2.6 9B")
        #expect(e.modelType == .llm)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-4bit")
        #expect(e.hfModelId == Self.hfID)
        #expect(e.architecture == "Qwen3_5ForConditionalGeneration")
        #expect(e.paramCountB == 9.0)
        #expect(e.ramGB == 10.70)
        #expect(e.downloadSizeGB == 7.10)
        #expect(e.contextWindow == 262144)
        #expect(e.license == "mit")
        #expect(e.taskTags == ["text-generation"])
        #expect(e.speedTokensPerSec == nil)
        #expect(e.benchmarks == nil)
    }

    @Test func entryHasASingleRecommended4BitVariantMatchingTheEntry() throws {
        let e = try entry()
        let variants = try #require(e.variants)
        #expect(variants.count == 1)
        let v = try #require(variants.first)
        #expect(v.quantization == "4-bit")
        #expect(v.format == "mlx-4bit")
        #expect(v.hfModelId == Self.hfID)
        #expect(v.ramGB == e.ramGB)
        #expect(v.downloadSizeGB == e.downloadSizeGB)
        #expect(v.qualityPercent == 90)  // the catalog's 4-bit tier figure, not a benchmark
        #expect(v.recommended == true)
    }

    /// The description is where the OptiQ provenance lives (the validator limits `format`).
    @Test func descriptionRecordsTheMixedPrecisionAndTheRAMDerivation() throws {
        let e = try entry()
        let text = e.description ?? ""
        #expect(text.contains("6.34"))
        #expect(text.contains("8-bit"))
        #expect(text.contains("10.70"))
    }

    /// Text-only: one entry for this repo, and no Vision card.
    @Test func thereIsNoVisionCardForThisRepo() throws {
        let state = AppState()
        state.loadBrowserData()
        let data = try #require(state.browserData)
        let mine = data.domains.flatMap { $0.allModels }.filter { $0.hfModelId == Self.hfID }
        #expect(mine.count == 1)
        #expect(mine.first?.modelType == .llm)
    }

    /// It sits in Chat & Text, immediately after `Qwen3.5-9B`.
    @Test func entryIsInTheChatDomainNextToQwen35() throws {
        let state = AppState()
        state.loadBrowserData()
        let data = try #require(state.browserData)
        let llm = try #require(data.domains.first { $0.id == "llm" })
        let ids = llm.allModels.map(\.id)
        let q = try #require(ids.firstIndex(of: "mlx-community--Qwen3.5-9B-MLX-4bit"))
        #expect(ids[q + 1] == Self.id)
    }

    /// A `qwen3_5` LLM carries no support gap and resolves to the LLM runner.
    @Test func entryHasNoSupportGapAndRoutesToLLM() throws {
        let e = try entry()
        #expect(ModelSupport.unsupportedReason(for: e) == nil)
        #expect(e.runnerKind == .llm)
    }

    // MARK: - MM2 — what ModelDetailView and ComparisonTableView read

    /// MM2 done-when: every field the detail tiles and the comparison rows read is present, so a
    /// "—" can only come from a genuinely null field (speed — unmeasured — and benchmarks).
    @Test func detailAndComparisonFieldsArePopulated() throws {
        let e = try entry()
        #expect(e.contextWindow == 262144)               // Context tile / row ("262K")
        #expect(e.lastUpdated == "2026-09-27")           // Updated tile
        #expect(e.communityDownloads != nil)
        #expect(e.formattedDownloads() != "—")           // Downloads tile / row
        #expect(DataNormalizer.normalizeLicense(e.license) == "MIT")
        #expect(DataNormalizer.normalizeArchitecture(e.architecture) == "Qwen3_5")
        // The genuinely null ones, which the views render as an absent tile / "—":
        #expect(e.speedTokensPerSec == nil)
        #expect(e.benchmarks == nil)
    }

    /// The About section only renders when the description differs from the card summary, and the
    /// summary is what Browse cards show. Both must be real prose, and the unverified "agentic /
    /// tool-use" tag claims from the card must not be repeated (MM2 contract).
    @Test func summaryAndDescriptionAreDistinctProseWithoutUnverifiedClaims() throws {
        let e = try entry()
        let summary = try #require(e.summary)
        let description = try #require(e.description)
        #expect(!summary.isEmpty)
        #expect(description != e.displaySummary)
        #expect(e.descriptionSource == Self.hfID)
        for text in [summary, description] {
            let lower = text.lowercased()
            #expect(!lower.contains("agentic"))
            #expect(!lower.contains("tool-use"))
            #expect(!lower.contains("tool use"))
        }
        // Lineage, hybrid attention, OptiQ provenance, license, no benchmarks.
        #expect(description.contains("XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"))
        #expect(description.contains("24") && description.contains("8 hold a full KV cache"))
        #expect(description.contains("transferred from mlx-community/Qwen3.5-9B-OptiQ-4bit"))
        #expect(description.contains("not measured on this model"))
        #expect(description.contains("MIT"))
        #expect(description.contains("unmeasured"))
    }
}
