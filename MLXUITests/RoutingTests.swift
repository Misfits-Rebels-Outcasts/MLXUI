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

    // MARK: runnerKind — OCR now runs on MLXVLM (OC1), video stays unsupported

    @Test func runnerKindMapsOCRToOCR() {
        // OC1: catalog OCR repos are MLX VLMs → run via MLXVLM (OCRModule), not Apple Vision.
        #expect(makeEntry(modelType: .ocr).runnerKind == .ocr)
    }

    @Test func runnerKindMapsVideoToUnsupported() {
        // No video stage in the linear v1 pipeline.
        #expect(makeEntry(modelType: .video).runnerKind == .unsupported)
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
