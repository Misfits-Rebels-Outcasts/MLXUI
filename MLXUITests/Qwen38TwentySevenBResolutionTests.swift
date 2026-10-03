import Foundation
import Testing
@testable import MLXUI

/// Q38-2 — verifies the `lukaskremla/Qwen3.8-27B-3bit-MLX` catalog entry (`qwen3_5`,
/// `modelType: llm`, `format: mlx-3bit`) decodes and resolves to the LLM run path, matching the
/// `Qwen35NineBResolutionTests` / `TernaryBonsaiResolutionTests` precedent for the same
/// architecture family (`RSI/DelegateQwen38Backlog.md` Q38-2, ruled 2026-10-03 A1/B1/C1/D1;
/// evidence in journal `RSI/journal/2026-379-ds-q38-1-verify.md`).
///
/// The architecture is SUPPORTED (`LLMModelFactory` registers `qwen3_5` → `Qwen35Model`), so this
/// entry must carry no `ModelSupport` gap and must route through `ChatModule`, never `VLMModule`.
/// It is the 3-bit card only; the 4-bit `Qwen3.8-27B` entry is Phase B (Q38-5).
@MainActor
struct Qwen38TwentySevenBResolutionTests {

    private static let hfModelId = "lukaskremla/Qwen3.8-27B-3bit-MLX"

    /// The catalog entry's own JSON, copied field-for-field from `MLXUI/Resources/browser.json`.
    /// Kept as a literal (rather than read from the bundle) so the test proves the *shape* the
    /// shipped file must keep, the `Qwen35NineBResolutionTests` precedent. The values are the
    /// Q38-1 evidence: ramGB 19.08 = 1.5 × downloadSizeGB 12.72 (readme F6), context 262,144 from
    /// `config.json` `text_config.max_position_embeddings`, revision recorded in the description.
    private static let catalogJSON = """
    {
        "id": "lukaskremla--Qwen3.8-27B-3bit-MLX",
        "family": "Qwen3.8",
        "displayName": "Qwen3.8-27B (3-bit)",
        "paramSize": "27B",
        "paramCountB": 27.0,
        "modelType": "llm",
        "source": "mlx",
        "format": "mlx-3bit",
        "platforms": ["macOS 13+"],
        "minMacOSVersion": "13.0",
        "hfRepo": "lukaskremla",
        "hfModelId": "lukaskremla/Qwen3.8-27B-3bit-MLX",
        "ramGB": 19.08,
        "downloadSizeGB": 12.72,
        "contextWindow": 262144,
        "license": "apache-2.0",
        "licenseUrl": "https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "taskTags": ["text-generation"],
        "benchmarks": null,
        "speedTokensPerSec": null,
        "speedHardware": null,
        "speedEstimated": true,
        "communityDownloads": 1414,
        "communityLikes": 5,
        "variants": [
            {
                "quantization": "3-bit",
                "format": "mlx-3bit",
                "ramGB": 19.08,
                "downloadSizeGB": 12.72,
                "qualityPercent": 80,
                "hfModelId": "lukaskremla/Qwen3.8-27B-3bit-MLX",
                "recommended": true
            }
        ]
    }
    """

    private func qwen38Entry() -> ModelEntry {
        makeEntry(
            id: "lukaskremla--Qwen3.8-27B-3bit-MLX",
            family: "Qwen3.8",
            displayName: "Qwen3.8-27B (3-bit)",
            modelType: .llm,
            source: .mlx,
            ramGB: 19.08,
            downloadSizeGB: 12.72,
            hfRepo: "lukaskremla",
            hfModelId: Self.hfModelId
        )
    }

    /// The **shipped** catalog, decoded the way `AppState.loadBrowserData()` does. Loading the
    /// real file (rather than only a literal) is what makes the tests below fail if the entry is
    /// removed from `browser.json` — verified by deleting it and re-running (G3).
    private func bundledCatalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)   // .../MLXUITests/Qwen38…Tests.swift
            .deletingLastPathComponent()                 // MLXUITests
            .deletingLastPathComponent()                 // PipelineStudio-q38
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func bundledEntry() throws -> ModelEntry {
        let all = try bundledCatalog().domains.flatMap { $0.allModels }
        return try #require(
            all.first { $0.hfModelId == Self.hfModelId },
            "the shipped browser.json has no \(Self.hfModelId) entry")
    }

    /// Q38-2's core deliverable: the entry is **in the shipped catalog** and decodes with the
    /// shape Q38-1's evidence calls for.
    @Test func qwen38EntryIsInTheBundledCatalog() throws {
        let entry = try bundledEntry()
        #expect(entry.id == "lukaskremla--Qwen3.8-27B-3bit-MLX")
        #expect(entry.modelType == .llm)
        #expect(entry.source == .mlx)
        #expect(entry.format == "mlx-3bit")
        #expect(entry.family == "Qwen3.8")
        #expect(entry.displayName == "Qwen3.8-27B (3-bit)")
        #expect(entry.hfRepo == "lukaskremla")
        #expect(entry.hfModelId == Self.hfModelId)
        #expect(entry.contextWindow == 262144)
        #expect(entry.license == "apache-2.0")
        #expect(entry.architecture == "Qwen3_5ForConditionalGeneration")
        #expect(entry.ramGB == 19.08)
        #expect(entry.downloadSizeGB == 12.72)
        #expect(entry.speedTokensPerSec == nil)
        #expect(entry.speedEstimated == true)
        #expect(entry.variants?.count == 1)
        #expect(entry.variants?.first?.quantization == "3-bit")
        #expect(entry.variants?.first?.format == "mlx-3bit")
        #expect(entry.variants?.first?.recommended == true)
        // Rule 11 / readme T5: the description records the commit the entry was verified against.
        #expect(entry.description?.contains("b3503b31b36125107e12a2b57fbe690c37bfc053") == true)
        // The audience sentence the contract asks for, and the text-only claim.
        #expect(entry.description?.contains("18–24 GB Macs with other apps closed") == true)
        #expect(entry.description?.contains("text only") == true)
    }

    /// The bundled 3-bit entry carries no support gap and resolves to the chat module.
    @Test func bundledQwen38ResolvesToChatModule() throws {
        let registry = ModelRegistry()
        for module in installedModules {
            module.register(into: registry)
        }
        let resolved = try #require(registry.bestModule(for: try bundledEntry()))
        #expect(resolved.descriptor.id == "chat")
        #expect(resolved.sdk.claim(try bundledEntry()) == .exact)
    }

    /// The catalog entry decodes with the expected 3-bit / LLM / mlx shape.
    @Test func qwen38EntryDecodes() throws {
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(Self.catalogJSON.utf8))
        #expect(entry.id == "lukaskremla--Qwen3.8-27B-3bit-MLX")
        #expect(entry.modelType == .llm)
        #expect(entry.source == .mlx)
        #expect(entry.format == "mlx-3bit")
        #expect(entry.family == "Qwen3.8")
        #expect(entry.displayName == "Qwen3.8-27B (3-bit)")
        #expect(entry.hfModelId == Self.hfModelId)
        #expect(entry.contextWindow == 262144)
        #expect(entry.license == "apache-2.0")
        #expect(entry.architecture == "Qwen3_5ForConditionalGeneration")
        #expect(entry.variants?.count == 1)
        #expect(entry.variants?.first?.quantization == "3-bit")
        #expect(entry.variants?.first?.recommended == true)
    }

    /// The 3-bit card keeps the formula value (readme T2 / rule 12's neighbourhood): rounding it
    /// up would put it over the owner's 19.33 GB Mac and make the stream's only testable model
    /// untestable.
    @Test func qwen38RAMIsTheFormulaValueNotRoundedUp() throws {
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(Self.catalogJSON.utf8))
        #expect(entry.ramGB == 19.08)
        #expect(entry.ramGB <= 19.33)
        // The 1.5× rule applied to downloadSizeGB, as `Ternary-Bonsai-27B` was sized (readme F6).
        #expect(abs(entry.ramGB - entry.downloadSizeGB * 1.5) < 0.01)
    }

    /// `qwen3_5` is a supported architecture in the pinned `mlx-swift-lm` (3.31.3), so this entry
    /// must carry no support gap and must not render `UnsupportedModelView`.
    @Test func qwen38HasNoSupportGap() throws {
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(Self.catalogJSON.utf8))
        #expect(ModelSupport.unsupportedReason(for: entry) == nil)
        #expect(ModelSupport.unsupportedReason(for: qwen38Entry()) == nil)
    }

    /// `runnerKind` maps `.llm` → `.llm`.
    @Test func qwen38RoutesToRunnableKind() throws {
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(Self.catalogJSON.utf8))
        #expect(entry.runnerKind == .llm)
    }

    /// `ChatSDK` claims it exactly and resolves through `ChatModule`'s text path — the text-tower
    /// entry never reaches `VLMSDK`, which would need a vision-typed card (out of scope, Q38-0).
    @Test func qwen38ResolvesToChatModule() throws {
        let registry = ModelRegistry()
        for module in installedModules {
            module.register(into: registry)
        }
        let resolved = try #require(registry.bestModule(for: qwen38Entry()))
        #expect(resolved.descriptor.id == "chat")
        #expect(resolved.sdk.claim(qwen38Entry()) == .exact)
    }

    /// `makeStage` produces a text → text stage (an `LLMStage` over the text tower).
    @Test func qwen38MakeStageProducesTextToTextStage() throws {
        let stage = try ChatSDK().makeStage(for: qwen38Entry(), config: .default)
        #expect(stage.accepts == .text)
        #expect(stage.produces == .text)
    }
}
