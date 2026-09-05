import Foundation
import Testing
@testable import MLXUI

/// MoC-2-3 — verifies the `mlx-community/Qwen3.5-9B-MLX-4bit` catalog entry (`qwen3_5`,
/// `modelType: llm`) decodes and resolves to the LLM run path, matching the
/// `TernaryBonsaiResolutionTests` precedent for the same architecture family. This is the
/// **text-tower** entry only (Decision A) — `LLMModelFactory` registers `qwen3_5` →
/// `Qwen35Model`, so it must carry no `ModelSupport` gap and must route through `ChatModule`,
/// never `VLMModule` (that path is MoC-6's, on a separate catalog `id`).
@MainActor
struct Qwen35NineBResolutionTests {

    private func qwen35Entry() -> ModelEntry {
        makeEntry(
            id: "mlx-community--Qwen3.5-9B-MLX-4bit",
            family: "Qwen3.5",
            displayName: "Qwen3.5-9B",
            modelType: .llm,
            source: .mlx,
            ramGB: 8.96,
            downloadSizeGB: 5.97
        )
    }

    /// The catalog entry decodes with the expected 4-bit / LLM / mlx shape.
    @Test func qwen35EntryDecodes() throws {
        let json = """
        {
            "id": "mlx-community--Qwen3.5-9B-MLX-4bit",
            "family": "Qwen3.5",
            "displayName": "Qwen3.5-9B",
            "paramSize": "9B",
            "paramCountB": 9.0,
            "modelType": "llm",
            "source": "mlx",
            "format": "mlx-4bit",
            "platforms": ["macOS 13+"],
            "minMacOSVersion": "13.0",
            "hfRepo": "mlx-community",
            "hfModelId": "mlx-community/Qwen3.5-9B-MLX-4bit",
            "ramGB": 8.96,
            "downloadSizeGB": 5.97,
            "contextWindow": 262144,
            "license": "apache-2.0",
            "architecture": "Qwen3_5ForConditionalGeneration",
            "taskTags": ["text-generation"],
            "speedTokensPerSec": null,
            "speedEstimated": true,
            "variants": [
                {
                    "quantization": "4-bit",
                    "format": "mlx-4bit",
                    "ramGB": 8.96,
                    "downloadSizeGB": 5.97,
                    "qualityPercent": 90,
                    "hfModelId": "mlx-community/Qwen3.5-9B-MLX-4bit",
                    "recommended": true
                }
            ]
        }
        """
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
        #expect(entry.modelType == .llm)
        #expect(entry.source == .mlx)
        #expect(entry.format == "mlx-4bit")
        #expect(entry.contextWindow == 262144)
        #expect(entry.license == "apache-2.0")
        #expect(entry.architecture == "Qwen3_5ForConditionalGeneration")
        #expect(entry.variants?.first?.quantization == "4-bit")
    }

    /// A `qwen3_5` LLM (`source == .mlx`) carries no support gap → it routes to the LLM
    /// runner (`LLMModelFactory` has `qwen3_5`), not `UnsupportedModelView`.
    @Test func qwen35HasNoSupportGap() {
        #expect(ModelSupport.unsupportedReason(for: qwen35Entry()) == nil)
    }

    /// `runnerKind` maps `.llm` → `.llm`.
    @Test func qwen35RoutesToRunnableKind() {
        #expect(qwen35Entry().runnerKind == .llm)
    }

    /// `ChatSDK` claims it exactly and resolves through `ChatModule`'s text path — the
    /// text-tower entry never reaches `VLMSDK` (that requires the separate vision-typed
    /// catalog entry MoC-6 adds on the same `hfModelId`).
    @Test func qwen35ResolvesToChatModule() throws {
        let registry = ModelRegistry()
        for module in installedModules {
            module.register(into: registry)
        }
        let resolved = try #require(registry.bestModule(for: qwen35Entry()))
        #expect(resolved.descriptor.id == "chat")
        #expect(resolved.sdk.claim(qwen35Entry()) == .exact)
    }

    /// `makeStage` produces a text → text stage (an `LLMStage` over the text tower).
    @Test func qwen35MakeStageProducesTextToTextStage() throws {
        let stage = try ChatSDK().makeStage(for: qwen35Entry(), config: .default)
        #expect(stage.accepts == .text)
        #expect(stage.produces == .text)
    }
}
