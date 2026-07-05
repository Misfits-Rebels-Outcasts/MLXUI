import Foundation

/// The single shared list of model modules to register at launch. `AppState.init`
/// registers each into its `ModelRegistry` (W4). Append future modules here; nothing else
/// needs to change to add a new model type's UI.
/// Order matters: `MLXWhisperModule` is listed **before** `WhisperModule` so it wins the
/// `.exact` claim tie for `source == .mlx` whisper entries (the registry keeps the
/// earliest-registered on a tie). WhisperKit remains the fallback for any non-mlx whisper.
/// `KokoroModule` (`.exact`) outranks `MLXAudioTTSModule` (`.generic`) by score, so order
/// between them is not load-bearing. `VoxtralModule` claims only Voxtral-family ASR.
/// `PaddleOCRModule`, `DotsOCRModule`, and `DeepSeekOCRModule` are all listed **before** `OCRModule`:
/// each scores `.exact` only for its own checkpoint family (paddleocr / dots / deepseek-ocr), so the
/// tie routes those to their vendored / custom pipelines while all other OCR repos (olmOCR, …) still
/// resolve to `OCRModule`'s MLXVLM path. Those three predicates are disjoint, so order among them
/// isn't load-bearing.
///
/// Coverage (browser.json): chat (LLM) · ASR (Whisper, Voxtral) · TTS (Kokoro + generic) ·
/// Vision (VLM) · OCR (MLXVLM + PaddleOCR-VL + dots.ocr + DeepSeek-OCR) · Embeddings. Only video
/// (Molmo2) is intentionally unsupported.
@MainActor
let installedModules: [ModelModule.Type] = [
    MLXWhisperModule.self,
    WhisperModule.self,
    VoxtralModule.self,
    KokoroModule.self,
    MLXAudioTTSModule.self,
    VLMModule.self,
    PaddleOCRModule.self,
    DotsOCRModule.self,
    DeepSeekOCRModule.self,
    OCRModule.self,
    EmbeddingModule.self,
]
