import Foundation
import Testing
@testable import MLXUI

/// Covers M2 routing: `ModelEntry.runnerKind` normalization (incl. catalog mislabel
/// overrides) and `StageFactory`'s install + RAM gates. See RSI/evals/eval-plan.md G2.
/// OC1 (2026-06-30): `.ocr` now routes to `.ocr` (MLXVLM), no longer `.unsupported`.
struct RoutingTests {

    // MARK: runnerKind — straight modelType mapping

    @Test func runnerKindMapsLLM()       { #expect(makeEntry(modelType: .llm).runnerKind == .llm) }
    @Test func runnerKindMapsASR()       { #expect(makeEntry(modelType: .asr).runnerKind == .asr) }
    @Test func runnerKindMapsTTS()       { #expect(makeEntry(modelType: .tts).runnerKind == .tts) }
    @Test func runnerKindMapsEmbedding() { #expect(makeEntry(modelType: .embedding).runnerKind == .embedding) }
    @Test func runnerKindMapsVision()    { #expect(makeEntry(modelType: .vision).runnerKind == .vision) }

    // MARK: runnerKind — OCR now runs on MLXVLM (OC1), video routes to .video (WAN-AM1)

    @Test func runnerKindMapsOCRToOCR() {
        // OC1: catalog OCR repos are MLX VLMs → run via MLXVLM (OCRModule), not Apple Vision.
        #expect(makeEntry(modelType: .ocr).runnerKind == .ocr)
    }

    @Test func runnerKindMapsVideoToVideo() {
        // WAN-AM1: video generation routes to .video (engine: Modules/WanVideo, WAN-AM4).
        #expect(makeEntry(modelType: .video).runnerKind == .video)
    }

    @Test func runnerKindMapsSegmentation() {
        // SA-AM1: segmentation routes to .segmentation (engine: Modules/SegmentAnything, SA-AM4).
        #expect(makeEntry(modelType: .segmentation).runnerKind == .segmentation)
    }

    @Test func sam3CatalogEntryDecodes() throws {
        // Verifies the mlx-community/sam3-4bit entry shape decodes without throwing.
        let json = """
        {
            "id": "mlx-community--sam3-4bit",
            "family": "SAM3",
            "displayName": "SAM3",
            "paramSize": "~1.2B",
            "paramCountB": null,
            "modelType": "segmentation",
            "source": "mlx",
            "format": "mlx-4bit",
            "platforms": ["macOS 13+"],
            "minMacOSVersion": "13.0",
            "hfRepo": "mlx-community",
            "hfModelId": "mlx-community/sam3-4bit",
            "ramGB": 0.93,
            "downloadSizeGB": 0.62,
            "contextWindow": null,
            "variants": [
                {
                    "quantization": "4-bit",
                    "format": "mlx-4bit",
                    "ramGB": 0.93,
                    "downloadSizeGB": 0.62,
                    "qualityPercent": 85,
                    "hfModelId": "mlx-community/sam3-4bit",
                    "recommended": true
                }
            ]
        }
        """
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
        #expect(entry.modelType == .segmentation)
        #expect(entry.runnerKind == .segmentation)
        #expect(entry.source == .mlx)
        #expect(entry.paramCountB == nil)
        #expect(entry.ramGB == 0.93)
    }

    @Test func wan21CatalogEntryDecodes() throws {
        // WAN-AM1: verifies the Wan-AI/Wan2.1-T2V-1.3B entry shape decodes and routes correctly.
        let json = """
        {
            "id": "Wan-AI--Wan2.1-T2V-1.3B",
            "family": "Wan2.1",
            "displayName": "Wan2.1-T2V-1.3B",
            "paramSize": "1.3B",
            "paramCountB": 1.3,
            "modelType": "video",
            "source": "mlx",
            "format": "mlx-bf16",
            "platforms": ["macOS 13+"],
            "minMacOSVersion": "13.0",
            "hfRepo": "Wan-AI",
            "hfModelId": "Wan-AI/Wan2.1-T2V-1.3B",
            "ramGB": 10.0,
            "downloadSizeGB": 11.0,
            "contextWindow": null,
            "variants": [
                {
                    "quantization": "bfloat16",
                    "format": "mlx-bf16",
                    "ramGB": 10.0,
                    "downloadSizeGB": 11.0,
                    "qualityPercent": 100,
                    "hfModelId": "Wan-AI/Wan2.1-T2V-1.3B",
                    "recommended": true
                }
            ]
        }
        """
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
        #expect(entry.modelType == .video)
        #expect(entry.runnerKind == .video)
        #expect(entry.source == .mlx)
        #expect(entry.paramCountB == 1.3)
        #expect(entry.ramGB == 10.0)
    }

    @Test func runnerKindMapsUpscaleToUpscale() {
        // SV-AM1: image upscaling routes to .upscale (engine: Modules/SeedVR2, SV-AM4).
        #expect(makeEntry(modelType: .upscale).runnerKind == .upscale)
    }

    @Test func seedvr2CatalogEntryDecodes() throws {
        // SV-AM1: verifies the mlx-community/SeedVR2-3B-mlx-int8 entry shape decodes and routes correctly.
        let json = """
        {
            "id": "mlx-community--SeedVR2-3B-mlx-int8",
            "family": "SeedVR2",
            "displayName": "SeedVR2 3B",
            "paramSize": "3B",
            "paramCountB": 3.0,
            "modelType": "upscale",
            "source": "mlx",
            "format": "mlx-int8",
            "platforms": ["macOS 14+"],
            "minMacOSVersion": "14.0",
            "hfRepo": "mlx-community",
            "hfModelId": "mlx-community/SeedVR2-3B-mlx-int8",
            "ramGB": 4.5,
            "downloadSizeGB": 3.0,
            "contextWindow": null,
            "variants": [
                {
                    "quantization": "int8",
                    "format": "mlx-int8",
                    "ramGB": 4.5,
                    "downloadSizeGB": 3.0,
                    "qualityPercent": 95,
                    "hfModelId": "mlx-community/SeedVR2-3B-mlx-int8",
                    "recommended": true
                }
            ]
        }
        """
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
        #expect(entry.modelType == .upscale)
        #expect(entry.runnerKind == .upscale)
        #expect(entry.source == .mlx)
        #expect(entry.paramCountB == 3.0)
        #expect(entry.ramGB == 4.5)
    }

    // MARK: runnerKind — catalog mislabel overrides (family beats modelType)

    @Test func runnerKindOverridesOuteTTSFromASRToTTS() {
        // Llama-OuteTTS is tagged `asr` in the catalog but is really TTS.
        let entry = makeEntry(family: "Llama-OuteTTS", modelType: .asr)
        #expect(entry.runnerKind == .tts)
    }

    @Test func runnerKindOverridesSpeechToSpeechToUnsupported() {
        #expect(makeEntry(family: "LFM2.5-Audio", modelType: .asr).runnerKind == .unsupported)
        #expect(makeEntry(family: "sam-audio", modelType: .asr).runnerKind == .unsupported)
    }

    // MARK: StageFactory — install gate

    @Test func factoryThrowsWhenModelNotInstalled() {
        let entry = makeEntry(id: "missing", ramGB: 4)
        #expect(throws: StageError.self) {
            _ = try StageFactory.make(for: entry, installedModelIDs: [], availableRAMGB: 16)
        }
    }

    // MARK: StageFactory — RAM gate

    @Test func factoryThrowsWhenModelExceedsRAM() {
        let entry = makeEntry(id: "big", ramGB: 20)
        #expect {
            _ = try StageFactory.make(for: entry, installedModelIDs: ["big"], availableRAMGB: 16)
        } throws: { error in
            guard case let StageError.insufficientRAM(required, available) = error else { return false }
            return required == 20 && available == 16
        }
    }

    @Test func factoryPassesGatesAtExactRAMBoundary() {
        // ramGB == availableRAMGB must NOT trip the gate; it falls through to the
        // (not-yet-implemented) stage switch, i.e. `.unsupportedModel` for now.
        let entry = makeEntry(id: "fits", ramGB: 16)
        #expect {
            _ = try StageFactory.make(for: entry, installedModelIDs: ["fits"], availableRAMGB: 16)
        } throws: { error in
            guard case StageError.unsupportedModel = error else { return false }
            return true
        }
    }
}
