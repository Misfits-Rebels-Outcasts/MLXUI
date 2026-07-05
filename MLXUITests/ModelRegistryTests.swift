import Testing
@testable import MLXUI

/// Tests for the model registry / AIUI template path (`plan-whisper-aiui.md`).
/// W1 covers `ClaimScore` ordering; W3 adds Whisper size mapping + registry resolution.
@MainActor
struct ModelRegistryTests {

    // MARK: - ClaimScore (W1)

    @Test func claimScoreOrdering() {
        #expect(ClaimScore.exact > ClaimScore.specific)
        #expect(ClaimScore.specific > ClaimScore.generic)
        #expect(ClaimScore.generic > ClaimScore.no)
        #expect(ClaimScore.exact > ClaimScore.no)
    }

    @Test func claimScoreSortsStrongestLast() {
        let sorted: [ClaimScore] = [.no, .exact, .generic, .specific].sorted()
        #expect(sorted.first == .no)
        #expect(sorted.last == .exact)
    }

    @Test func emptyRegistryResolvesNothing() {
        let registry = ModelRegistry()
        #expect(registry.bestModule(for: makeEntry(modelType: .asr)) == nil)
    }

    // MARK: - WhisperKitSize (W3)

    @Test func whisperSizeFromTinyId() {
        #expect(WhisperKitSize.from(makeEntry(id: "mlx-community--whisper-tiny-asr-fp16")) == "tiny")
    }

    @Test func whisperSizeFromSmallId() {
        #expect(WhisperKitSize.from(makeEntry(id: "mlx-community--whisper-small-asr-fp16")) == "small")
    }

    @Test func whisperSizeFromLargeV3Id() {
        #expect(WhisperKitSize.from(makeEntry(id: "mlx-community--whisper-large-v3-asr-fp16")) == "large-v3")
    }

    @Test func whisperSizeUnknownDefaultsToBase() {
        #expect(WhisperKitSize.from(makeEntry(id: "mlx-community--whisper-asr-fp16")) == "base")
    }

    // MARK: - Registry resolution (W3)

    private func whisperRegistry() -> ModelRegistry {
        let registry = ModelRegistry()
        WhisperModule.register(into: registry)
        return registry
    }

    private func whisperEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--whisper-small-asr-fp16", family: "whisper", modelType: .asr)
    }

    @Test func whisperModuleResolvesWhisperASR() throws {
        let entry = whisperEntry()
        let resolved = try #require(whisperRegistry().bestModule(for: entry))
        #expect(resolved.descriptor.id == "whisper")
        #expect(resolved.sdk.claim(entry) == .exact)
    }

    @Test func voxtralAsrDoesNotResolve() {
        let voxtral = makeEntry(id: "mlx-community--Voxtral-asr", family: "Voxtral", modelType: .asr)
        #expect(whisperRegistry().bestModule(for: voxtral) == nil)
    }

    @Test func llmDoesNotResolve() {
        #expect(whisperRegistry().bestModule(for: makeEntry(modelType: .llm)) == nil)
    }

    // MARK: - makeStage (W3) — do not call .run (real WhisperKit; ASRStageTests covers run)

    @Test func makeStageReturnsConfiguredASRStage() throws {
        let stage = try WhisperSDK().makeStage(for: whisperEntry(), config: .default)
        #expect(stage.accepts == .audio)
        #expect(stage.produces == .text)
        #expect(stage.id == "whisper.small")
    }

    // MARK: - MLX-native ASR resolution (A2)

    /// Mirrors the real app registration order from `installedModules`: MLX first, then
    /// WhisperKit, so the `.exact` tie resolves to MLX for `source == .mlx` whisper.
    private func fullRegistry() -> ModelRegistry {
        let registry = ModelRegistry()
        MLXWhisperModule.register(into: registry)
        WhisperModule.register(into: registry)
        VoxtralModule.register(into: registry)
        KokoroModule.register(into: registry)
        MLXAudioTTSModule.register(into: registry)
        VLMModule.register(into: registry)
        PaddleOCRModule.register(into: registry)
        DotsOCRModule.register(into: registry)
        DeepSeekOCRModule.register(into: registry)
        OCRModule.register(into: registry)
        EmbeddingModule.register(into: registry)
        return registry
    }

    @Test func mlxWhisperWinsForMlxSourceWhisper() throws {
        let entry = makeEntry(id: "mlx-community--whisper-small-asr-fp16",
                              family: "whisper", modelType: .asr, source: .mlx)
        let resolved = try #require(fullRegistry().bestModule(for: entry))
        #expect(resolved.descriptor.id == "mlx-whisper")
        #expect(resolved.sdk.claim(entry) == .exact)
    }

    @Test func whisperKitClaimsNonMlxWhisper() {
        let entry = makeEntry(id: "openai--whisper-small", family: "whisper",
                              modelType: .asr, source: .coreml)
        // MLX SDK refuses non-mlx; WhisperKit still claims it.
        #expect(MLXWhisperSDK().claim(entry) == .no)
        #expect(WhisperSDK().claim(entry) == .exact)
    }

    @Test func mlxWhisperMakeStageProducesASRStage() throws {
        let entry = makeEntry(id: "mlx-community--whisper-small-asr-fp16",
                              family: "whisper", modelType: .asr, source: .mlx)
        let stage = try MLXWhisperSDK().makeStage(for: entry, config: .default)
        #expect(stage.accepts == .audio)
        #expect(stage.produces == .text)
        #expect(stage.id.hasPrefix("mlx-whisper."))
    }

    // MARK: - Kokoro TTS resolution (K1b-2)

    private func kokoroEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--Kokoro-82M-bf16", family: "Kokoro", modelType: .tts, source: .mlx)
    }

    @Test func kokoroTTSResolvesToKokoroModule() throws {
        let resolved = try #require(fullRegistry().bestModule(for: kokoroEntry()))
        #expect(resolved.descriptor.id == "kokoro")
        #expect(resolved.sdk.claim(kokoroEntry()) == .exact)
    }

    // MARK: - Generic MLX TTS resolution (AT1)

    private func orpheusEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--orpheus-3b-0.1-ft-bf16",
                  family: "orpheus", modelType: .tts, source: .mlx)
    }

    @Test func nonKokoroTTSResolvesToMLXAudioTTS() throws {
        // Other catalog TTS families (orpheus, chatterbox, Qwen3-TTS) aren't Kokoro →
        // claimed by the generic MLXAudioTTS module at `.generic`; Kokoro's SDK declines them.
        let orpheus = orpheusEntry()
        #expect(KokoroSDK().claim(orpheus) == .no)
        #expect(MLXAudioTTSSDK().claim(orpheus) == .generic)
        let resolved = try #require(fullRegistry().bestModule(for: orpheus))
        #expect(resolved.descriptor.id == "mlx-audio-tts")
    }

    @Test func kokoroBeatsGenericTTSForKokoroRepo() {
        // exact (Kokoro) outranks generic (MLXAudioTTS); the generic SDK declines Kokoro.
        #expect(MLXAudioTTSSDK().claim(kokoroEntry()) == .no)
        #expect(KokoroSDK().claim(kokoroEntry()) == .exact)
    }

    @Test func mlxAudioTTSMakeStageProducesTTSStage() throws {
        let stage = try MLXAudioTTSSDK().makeStage(for: orpheusEntry(), config: .default)
        #expect(stage.accepts == .text)
        #expect(stage.produces == .audio)
        #expect(stage.id.hasPrefix("mlx-audio-tts."))
    }

    // MARK: - Generic MLX TTS voice presets (AT2)

    @Test func orpheusFamilyHasPresetVoices() {
        let voices = MLXAudioTTSVoices.presets(family: "orpheus")
        #expect(voices.contains("tara"))
        #expect(voices.count == MLXAudioTTSVoices.orpheusPresets.count)
        // Case-insensitive match on the family name.
        #expect(MLXAudioTTSVoices.presets(family: "Orpheus") == voices)
    }

    @Test func nonPresetFamiliesHaveNoEnumeratedVoices() {
        // chatterbox (zero-shot) and Qwen3-TTS (checkpoint-specific) → free-form field, no list.
        #expect(MLXAudioTTSVoices.presets(family: "chatterbox").isEmpty)
        #expect(MLXAudioTTSVoices.presets(family: "Qwen3").isEmpty)
    }

    @Test func kokoroMakeStageProducesTTSStage() throws {
        let stage = try KokoroSDK().makeStage(for: kokoroEntry(), config: .default)
        #expect(stage.accepts == .text)
        #expect(stage.produces == .audio)
        #expect(stage.id.hasPrefix("kokoro."))
    }

    // MARK: - VLM resolution (VL4)

    private func visionEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--Qwen3-VL-4B-Instruct-4bit",
                  family: "Qwen3", modelType: .vision, source: .mlx)
    }

    @Test func vlmResolvesVisionMLX() throws {
        let resolved = try #require(fullRegistry().bestModule(for: visionEntry()))
        #expect(resolved.descriptor.id == "vlm")
        #expect(resolved.sdk.claim(visionEntry()) == .exact)
    }

    @Test func vlmDeclinesNonVisionAndNonMLX() {
        // Wrong kind, and right kind but wrong source — both rejected.
        #expect(VLMSDK().claim(makeEntry(modelType: .asr)) == .no)
        #expect(VLMSDK().claim(makeEntry(modelType: .llm)) == .no)
        #expect(VLMSDK().claim(makeEntry(modelType: .vision, source: .coreml)) == .no)
    }

    @Test func vlmMakeStageProducesImageToText() throws {
        let stage = try VLMSDK().makeStage(for: visionEntry(), config: .default)
        #expect(stage.accepts == .image)
        #expect(stage.produces == .text)
        #expect(stage.id.hasPrefix("vlm."))
    }

    // MARK: - OCR resolution (OC3)

    private func ocrEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--olmOCR-2-7B-1025-mlx-4bit",
                  family: "olmOCR", modelType: .ocr, source: .mlx)
    }

    @Test func ocrRoutesToRunnableKind() {
        // OC1: .ocr no longer maps to .unsupported.
        #expect(ocrEntry().runnerKind == .ocr)
    }

    @Test func ocrResolvesToOCRModule() throws {
        let resolved = try #require(fullRegistry().bestModule(for: ocrEntry()))
        #expect(resolved.descriptor.id == "ocr")
        #expect(resolved.sdk.claim(ocrEntry()) == .exact)
    }

    @Test func ocrDeclinedByVLMAndViceVersa() {
        // OCR and VLM stay distinct: each declines the other's kind.
        #expect(VLMSDK().claim(ocrEntry()) == .no)
        #expect(OCRSDK().claim(visionEntry()) == .no)
        #expect(OCRSDK().claim(makeEntry(modelType: .ocr, source: .coreml)) == .no)
    }

    @Test func ocrMakeStageProducesImageToText() throws {
        let stage = try OCRSDK().makeStage(for: ocrEntry(), config: .default)
        #expect(stage.accepts == .image)
        #expect(stage.produces == .text)
    }

    // MARK: - PaddleOCR-VL resolution (SUP-1)

    private func paddleOCREntry() -> ModelEntry {
        makeEntry(id: "mlx-community--PaddleOCR-VL-0.9B-4bit",
                  family: "PaddleOCR", modelType: .ocr, source: .mlx)
    }

    @Test func paddleOCRWinsTieOverGenericOCR() throws {
        // Both OCRSDK and PaddleOCRSDK score .exact for PaddleOCR-VL; PaddleOCRModule is
        // registered first, so the registry's earliest-wins tie routes it to the vendored
        // PaddleOCR-VL pipeline instead of the MLXVLM path (which can't load its architecture).
        let entry = paddleOCREntry()
        #expect(OCRSDK().claim(entry) == .exact)
        #expect(PaddleOCRSDK().claim(entry) == .exact)
        let resolved = try #require(fullRegistry().bestModule(for: entry))
        #expect(resolved.descriptor.id == "paddleocr")
    }

    @Test func paddleOCRDeclinesOtherOCRAndNonMLX() {
        // Scoped to the paddleocr family: olmOCR (and any other OCR repo) still routes to OCRModule.
        #expect(PaddleOCRSDK().claim(ocrEntry()) == .no)
        #expect(PaddleOCRSDK().claim(makeEntry(id: "mlx-community--PaddleOCR-VL-0.9B-4bit",
                                               modelType: .ocr, source: .coreml)) == .no)
        // And the generic OCR path still owns olmOCR.
        let resolvedOlm = fullRegistry().bestModule(for: ocrEntry())
        #expect(resolvedOlm?.descriptor.id == "ocr")
    }

    @Test func paddleOCRMakeStageProducesImageToText() throws {
        let stage = try PaddleOCRSDK().makeStage(for: paddleOCREntry(), config: .default)
        #expect(stage.accepts == .image)
        #expect(stage.produces == .text)
        #expect(stage.id.hasPrefix("paddleocr."))
    }

    // MARK: - dots.ocr resolution (SUP-3, slice 1)

    private func dotsOCREntry() -> ModelEntry {
        makeEntry(id: "mlx-community--dots.mocr-4bit",
                  family: "dots", modelType: .ocr, source: .mlx)
    }

    @Test func dotsOCRWinsTieOverGenericOCR() throws {
        // Both OCRSDK and DotsOCRSDK score .exact for a dots checkpoint; DotsOCRModule is
        // registered first, so the registry's earliest-wins tie routes it to the dots.ocr engine
        // instead of the MLXVLM path (which can't load its architecture).
        let entry = dotsOCREntry()
        #expect(OCRSDK().claim(entry) == .exact)
        #expect(DotsOCRSDK().claim(entry) == .exact)
        let resolved = try #require(fullRegistry().bestModule(for: entry))
        #expect(resolved.descriptor.id == "dots-ocr")
    }

    @Test func dotsOCRDeclinesOtherOCRAndNonMLX() {
        // Scoped to dots checkpoints: olmOCR (and any non-dots OCR repo) still routes to OCRModule,
        // and PaddleOCR-VL is claimed by PaddleOCRSDK, not this one.
        #expect(DotsOCRSDK().claim(ocrEntry()) == .no)
        #expect(DotsOCRSDK().claim(paddleOCREntry()) == .no)
        #expect(DotsOCRSDK().claim(makeEntry(id: "mlx-community--dots.mocr-4bit",
                                             modelType: .ocr, source: .coreml)) == .no)
        let resolvedOlm = fullRegistry().bestModule(for: ocrEntry())
        #expect(resolvedOlm?.descriptor.id == "ocr")
    }

    @Test func dotsOCRMakeStageProducesImageToText() throws {
        let stage = try DotsOCRSDK().makeStage(for: dotsOCREntry(), config: .default)
        #expect(stage.accepts == .image)
        #expect(stage.produces == .text)
        #expect(stage.id.hasPrefix("dots-ocr."))
    }

    // MARK: - DeepSeek-OCR-2 resolution (SUP-2, slice 1)

    private func deepSeekOCREntry() -> ModelEntry {
        makeEntry(id: "mlx-community--DeepSeek-OCR-2-bf16",
                  family: "DeepSeek-OCR", modelType: .ocr, source: .mlx)
    }

    @Test func deepSeekOCRWinsTieOverGenericOCR() throws {
        // Both OCRSDK and DeepSeekOCRSDK score .exact; DeepSeekOCRModule is registered first, so the
        // earliest-wins tie routes it to the custom deepseekocr_2 engine, not the MLXVLM path.
        let entry = deepSeekOCREntry()
        #expect(OCRSDK().claim(entry) == .exact)
        #expect(DeepSeekOCRSDK().claim(entry) == .exact)
        let resolved = try #require(fullRegistry().bestModule(for: entry))
        #expect(resolved.descriptor.id == "deepseek-ocr")
    }

    @Test func deepSeekOCRDeclinesOtherOCRAndNonMLX() {
        // Scoped to deepseek-ocr: olmOCR still routes to OCRModule; paddle/dots are claimed by their own.
        #expect(DeepSeekOCRSDK().claim(ocrEntry()) == .no)
        #expect(DeepSeekOCRSDK().claim(paddleOCREntry()) == .no)
        #expect(DeepSeekOCRSDK().claim(dotsOCREntry()) == .no)
        #expect(DeepSeekOCRSDK().claim(makeEntry(id: "mlx-community--DeepSeek-OCR-2-bf16",
                                                 modelType: .ocr, source: .coreml)) == .no)
        let resolvedOlm = fullRegistry().bestModule(for: ocrEntry())
        #expect(resolvedOlm?.descriptor.id == "ocr")
    }

    @Test func deepSeekOCRMakeStageProducesImageToText() throws {
        let stage = try DeepSeekOCRSDK().makeStage(for: deepSeekOCREntry(), config: .default)
        #expect(stage.accepts == .image)
        #expect(stage.produces == .text)
        #expect(stage.id.hasPrefix("deepseek-ocr."))
    }

    // MARK: - Voxtral ASR resolution (VX1)

    private func voxtralEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--Voxtral-Mini-4B-Realtime-2602-4bit",
                  family: "Voxtral", modelType: .asr, source: .mlx)
    }

    @Test func voxtralResolvesToVoxtralModule() throws {
        let resolved = try #require(fullRegistry().bestModule(for: voxtralEntry()))
        #expect(resolved.descriptor.id == "voxtral")
        #expect(resolved.sdk.claim(voxtralEntry()) == .exact)
    }

    @Test func voxtralAndMLXWhisperDeclineEachOther() {
        // Family-scoped ASR adapters stay distinct: neither claims the other's family.
        #expect(MLXWhisperSDK().claim(voxtralEntry()) == .no)
        let whisper = makeEntry(id: "mlx-community--whisper-small-asr-fp16",
                                family: "whisper", modelType: .asr, source: .mlx)
        #expect(VoxtralSDK().claim(whisper) == .no)
        #expect(VoxtralSDK().claim(makeEntry(modelType: .asr, source: .coreml)) == .no)
    }

    @Test func voxtralMakeStageProducesASRStage() throws {
        let stage = try VoxtralSDK().makeStage(for: voxtralEntry(), config: .default)
        #expect(stage.accepts == .audio)
        #expect(stage.produces == .text)
        #expect(stage.id.hasPrefix("voxtral."))
    }

    // MARK: - Embedding resolution (EM4)

    private func embeddingEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--all-MiniLM-L6-v2-4bit",
                  family: "all", modelType: .embedding, source: .mlx)
    }

    @Test func embeddingResolvesToEmbeddingModule() throws {
        let resolved = try #require(fullRegistry().bestModule(for: embeddingEntry()))
        #expect(resolved.descriptor.id == "embedding")
        #expect(resolved.sdk.claim(embeddingEntry()) == .exact)
    }

    @Test func embeddingDeclinesNonEmbeddingAndNonMLX() {
        #expect(EmbeddingSDK().claim(makeEntry(modelType: .llm)) == .no)
        #expect(EmbeddingSDK().claim(makeEntry(modelType: .embedding, source: .coreml)) == .no)
    }

    @Test func embeddingMakeStageProducesTextToEmbedding() throws {
        let stage = try EmbeddingSDK().makeStage(for: embeddingEntry(), config: .default)
        #expect(stage.accepts == .text)
        #expect(stage.produces == .embedding)
        #expect(stage.id.hasPrefix("embedding."))
    }

    // MARK: - Model-specific run options (chatterbox ref audio, nomic prefixes)

    @Test func chatterboxUsesReferenceAudioOthersDoNot() {
        #expect(MLXAudioTTSVoices.supportsReferenceAudio(family: "chatterbox"))
        #expect(!MLXAudioTTSVoices.supportsReferenceAudio(family: "orpheus"))
        #expect(!MLXAudioTTSVoices.supportsReferenceAudio(family: "Qwen3"))
    }

    @Test func nomicHasQueryDocumentPrefixesSymmetricModelsDoNot() {
        let nomic = EmbeddingPrefixes.queryDocument(family: "nomicai")
        #expect(nomic?.query == "search_query: ")
        #expect(nomic?.document == "search_document: ")
        // Symmetric models get no prefixes → the toggle is hidden.
        #expect(EmbeddingPrefixes.queryDocument(family: "all") == nil)
        #expect(EmbeddingPrefixes.queryDocument(family: "bge") == nil)
    }
}
