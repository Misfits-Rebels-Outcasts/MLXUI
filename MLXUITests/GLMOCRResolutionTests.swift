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

    // MARK: - R1 carry-over (a): the two done-when clauses MoC-1-4/1-5 shipped without

    /// The silent failure R1 was told to look for: an `XcodeWrite` that never synced the
    /// manifest into the app target. `Bundle.main` resolves to `MLXUI.app` here because
    /// `MLXUITests` is host-app-hosted (`TEST_HOST` in the project settings) — see the
    /// precedent at `Bundle.main.url(forResource: "browser", …)` in `CatFlowNotRunnableTests`.
    @Test func glmOCRManifestLoadsFromTheBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "glm-ocr-4bit.json"))
        #expect(manifest.id == "mlx-community/GLM-OCR-4bit")
        #expect(manifest.display == "GLM-OCR")
    }

    /// DA-7 (`RSI/DelegateDeciderBacklog.md`) — Decision C **amended**: `GLM-OCR` is now
    /// first in `taskModels["OCR"]` and the seed for a new OCR row. Pins the promoted default
    /// so a future pool reorder can't move the seed silently — the exact regression
    /// `defaultModel(forTask:)`'s callers (task rows + gallery 22/30/55) would never notice.
    @MainActor
    @Test func ocrDefaultModelIsGLMOCRAfterDA7() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
            .domains.flatMap { $0.allModels }
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        #expect(TaskModels.defaultModel(forTask: "OCR", catalog: catalog, claimableModelIDs: claimable) == "GLM-OCR")
    }
}
