import Foundation

/// `ModelSDK` for Kokoro TTS, backed by `mlx-audio-swift` via `TTSStage`/`KokoroEngine`.
/// Claims `.tts` + Kokoro-family entries (other catalog TTS families — Qwen3-TTS, chatterbox,
/// orpheus — aren't Kokoro and fall through to the unsupported path). `makeStage` binds the
/// installed model directory.
nonisolated struct KokoroSDK: ModelSDK {
    let id = "kokoro"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .tts,
              model.family.lowercased().contains("kokoro") else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let voice = config.voice ?? KokoroEngine.defaultVoice
        return TTSStage(modelID: model.id, voice: voice, speed: config.speed)
    }
}
