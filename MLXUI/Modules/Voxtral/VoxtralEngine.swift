import Foundation
import MLX
import MLXAudioSTT

/// MLX-native Voxtral ASR via mlx-audio-swift's `VoxtralRealtimeModel`. Despite the
/// "Realtime" name, the model exposes a plain offline `generate(audio:)` that takes the
/// whole 16 kHz buffer and returns transcribed text — the same shape `MLXWhisperEngine`
/// uses — so it composes onto `ASRStage` with no streaming. Loads the **installed**
/// safetensors directory (the A2 pattern). Isolates the `MLXAudioSTT` Voxtral import here.
enum VoxtralEngine {
    /// Transcribe 16 kHz mono float samples using the installed model directory.
    /// Loads the model per call (matches the other engines).
    nonisolated static func transcribe(_ samples: [Float], modelDirectory: URL) async throws -> String {
        do {
            let model = try VoxtralRealtimeModel.fromDirectory(modelDirectory)
            let output = model.generate(
                audio: MLXArray(samples),
                generationParameters: model.defaultGenerationParameters)
            return output.text.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            throw StageError.engineFailure(stage: "Voxtral ASR", underlying: error)
        }
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("AI Browser/models/\(id)", isDirectory: true)
    }
}
