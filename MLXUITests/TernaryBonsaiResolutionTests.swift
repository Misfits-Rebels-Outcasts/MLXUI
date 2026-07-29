import Foundation
import Testing
@testable import MLXUI

/// TB1 — verifies the `prism-ml/Ternary-Bonsai-27B-mlx-2bit` catalog entry (`qwen3_5`)
/// decodes and resolves to the LLM run path. The architecture is supported natively by the
/// pinned mlx-swift-lm (`LLMModelFactory` registers `qwen3_5`), so it must carry no
/// `ModelSupport` gap. Uses an inline fixture (the bundled catalog isn't reachable from the
/// test bundle) mirroring the `browser.json` entry.
struct TernaryBonsaiResolutionTests {

    /// The catalog entry decodes with the expected 2-bit / LLM / mlx shape.
    @Test func ternaryBonsaiEntryDecodes() throws {
        let json = """
        {
            "id": "prism-ml--Ternary-Bonsai-27B-mlx-2bit",
            "family": "Ternary-Bonsai",
            "displayName": "Ternary-Bonsai-27B",
            "paramSize": "27B",
            "paramCountB": 27.0,
            "modelType": "llm",
            "source": "mlx",
            "format": "mlx-2bit",
            "platforms": ["macOS 13+"],
            "minMacOSVersion": "13.0",
            "hfRepo": "prism-ml",
            "hfModelId": "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
            "ramGB": 12.66,
            "downloadSizeGB": 8.44,
            "contextWindow": 262144,
            "architecture": "Qwen3_5ForConditionalGeneration",
            "speedTokensPerSec": 30,
            "speedEstimated": true,
            "variants": [
                {
                    "quantization": "2-bit",
                    "format": "mlx-2bit",
                    "ramGB": 12.66,
                    "downloadSizeGB": 8.44,
                    "qualityPercent": 85,
                    "hfModelId": "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
                    "recommended": true
                }
            ]
        }
        """
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
        #expect(entry.modelType == .llm)
        #expect(entry.source == .mlx)
        #expect(entry.format == "mlx-2bit")
        #expect(entry.contextWindow == 262144)
        #expect(entry.variants?.first?.quantization == "2-bit")
    }

    /// A `qwen3_5` LLM (`source == .mlx`) carries no support gap → it routes to the LLM
    /// runner (`LLMModelFactory.shared` has `qwen3_5`), not `UnsupportedModelView`.
    @Test func ternaryBonsaiHasNoSupportGap() {
        let entry = makeEntry(
            id: "prism-ml--Ternary-Bonsai-27B-mlx-2bit",
            family: "Ternary-Bonsai",
            displayName: "Ternary-Bonsai-27B",
            modelType: .llm,
            source: .mlx
        )
        #expect(ModelSupport.unsupportedReason(for: entry) == nil)
    }
}
