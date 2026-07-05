import Foundation

/// Normalized routing key for dispatch. The catalog's `modelType` is unreliable
/// (some audio-LLMs are tagged `asr`, some `asr` entries are really TTS), so all
/// corrections live in `ModelEntry.runnerKind` below — the single source of truth.
/// See `Design/pipeline-stage-sketch.md` §routing.
nonisolated enum RunnerKind: String, Sendable {
    case llm, asr, tts, vision, embedding, ocr, unsupported
}

extension ModelEntry {
    /// Normalized routing key: start from `modelType`, then override known catalog
    /// mislabels by family. This is the only place dispatch decisions are made.
    var runnerKind: RunnerKind {
        // Known overrides first (mislabeled in the catalog):
        switch family {
        case "Llama-OuteTTS":               return .tts          // catalog tags it asr
        case "LFM2.5-Audio", "sam-audio":   return .unsupported  // speech-to-speech; no linear stage yet
        default:                            break
        }
        switch modelType {
        case .llm:       return .llm
        case .asr:       return .asr
        case .tts:       return .tts
        case .vision:    return .vision
        case .embedding: return .embedding
        case .ocr:       return .ocr           // catalog OCR repos are MLX VLMs → run via MLXVLM (OCRModule), no Apple Vision
        case .video:     return .unsupported   // no video stage in the linear v1 pipeline
        }
    }
}
