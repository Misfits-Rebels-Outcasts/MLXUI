import Foundation

/// `ModelSDK` for non-Kokoro MLX TTS (Qwen3-TTS, chatterbox, orpheus), backed by
/// mlx-audio-swift via `MLXAudioTTSEngine`. Claims `.tts` + `source == .mlx` + **not** Kokoro
/// at `.generic`, so Kokoro's `.exact` adapter still wins for Kokoro repos (exact > generic).
nonisolated struct MLXAudioTTSSDK: ModelSDK {
    let id = "mlx-audio-tts"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .tts,
              model.source == .mlx,
              !model.family.lowercased().contains("kokoro") else {
            return .no
        }
        return .generic
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let directory = MLXAudioTTSEngine.installedModelDirectory(id: model.id)
        let voice = config.voice
        return TTSStage(id: "mlx-audio-tts.\(model.id)", name: "TTS (\(model.family))") { text in
            try await MLXAudioTTSEngine.synthesize(text, voice: voice, modelDirectory: directory)
        }
    }
}
