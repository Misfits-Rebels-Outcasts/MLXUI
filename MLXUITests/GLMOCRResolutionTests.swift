import Foundation
import Testing
@testable import MLXUI

/// MoC-1-3 — verifies the `mlx-community/GLM-OCR-4bit` catalog entry (`glm_ocr`) decodes and
/// resolves to the OCR run path. The architecture is registered natively by the pinned
/// mlx-swift-lm (`VLMModelFactory` registers `glm_ocr` → `GlmOcr`, processor `Glm46VProcessor`),
/// so it must carry no `ModelSupport` gap and must route through the generic `OCRModule` (the
/// same path `olmOCR-2 7B` uses — no custom module, unlike PaddleOCR-VL/dots.ocr/DeepSeek-OCR-2).
/// Uses an inline fixture (the bundled catalog isn't reachable from the test bundle) mirroring
/// the `browser.json` entry added in MoC-1-1. Precedent: `TernaryBonsaiResolutionTests.swift`.
@MainActor
struct GLMOCRResolutionTests {

    private func glmOCREntry() -> ModelEntry {
        makeEntry(
            id: "mlx-community--GLM-OCR-4bit",
            family: "GLM",
            displayName: "GLM-OCR",
            modelType: .ocr,
            source: .mlx,
            ramGB: 1.87,
            downloadSizeGB: 1.25
        )
    }

    /// The catalog entry decodes with the expected 4-bit / OCR / mlx shape.
    @Test func glmOCREntryDecodes() throws {
        let json = """
        {
            "id": "mlx-community--GLM-OCR-4bit",
            "family": "GLM",
            "displayName": "GLM-OCR",
            "paramSize": "0.5B",
            "paramCountB": 0.5,
            "modelType": "ocr",
            "source": "mlx",
            "format": "mlx-4bit",
            "platforms": ["macOS 13+"],
            "minMacOSVersion": "13.0",
            "hfRepo": "mlx-community",
            "hfModelId": "mlx-community/GLM-OCR-4bit",
            "ramGB": 1.87,
            "downloadSizeGB": 1.25,
            "contextWindow": null,
            "license": "mit",
            "architecture": "GlmOcrForConditionalGeneration",
            "taskTags": ["image-text-to-text"],
            "speedEstimated": true,
            "variants": [
                {
                    "quantization": "4-bit",
                    "format": "mlx-4bit",
                    "ramGB": 1.87,
                    "downloadSizeGB": 1.25,
                    "qualityPercent": 90,
                    "hfModelId": "mlx-community/GLM-OCR-4bit",
                    "recommended": true
                }
            ]
        }
        """
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
        #expect(entry.modelType == .ocr)
        #expect(entry.source == .mlx)
        #expect(entry.format == "mlx-4bit")
        #expect(entry.license == "mit")
        #expect(entry.architecture == "GlmOcrForConditionalGeneration")
        #expect(entry.variants?.first?.quantization == "4-bit")
    }

    /// A `glm_ocr` OCR model (`source == .mlx`) carries no support gap → it routes to the OCR
    /// runner (`VLMModelFactory` has `glm_ocr`), not `UnsupportedModelView`.
    @Test func glmOCRHasNoSupportGap() {
        #expect(ModelSupport.unsupportedReason(for: glmOCREntry()) == nil)
    }

    /// `runnerKind` maps `.ocr` → `.ocr` (OC1 — no longer `.unsupported`).
    @Test func glmOCRRoutesToRunnableKind() {
        #expect(glmOCREntry().runnerKind == .ocr)
    }

    /// `OCRSDK` claims it exactly, and it is not architecture-specific like PaddleOCR-VL/
    /// dots.ocr/DeepSeek-OCR-2 — it takes the same generic `OCRModule` path `olmOCR-2 7B` does.
    @Test func glmOCRResolvesToOCRModule() throws {
        let registry = ModelRegistry()
        for module in installedModules {
            module.register(into: registry)
        }
        let resolved = try #require(registry.bestModule(for: glmOCREntry()))
        #expect(resolved.descriptor.id == "ocr")
        #expect(resolved.sdk.claim(glmOCREntry()) == .exact)
    }

    /// `makeStage` produces an image → text stage (a `VLMStage` under the fixed OCR prompt).
    @Test func glmOCRMakeStageProducesImageToText() throws {
        let stage = try OCRSDK().makeStage(for: glmOCREntry(), config: .default)
        #expect(stage.accepts == .image)
        #expect(stage.produces == .text)
    }
}
