import Foundation
import MLX
import MLXAudioSTT

/// MLX-native Parakeet TDT ASR via mlx-audio-swift's `ParakeetModel`. Takes the whole 16 kHz
/// mono buffer and returns transcribed text — the same shape `VoxtralEngine` uses — so it
/// composes onto `ASRStage`. Loads the **installed** directory (the A2 pattern, never
/// `fromPretrained`): the loader reads `config.json` and every `*.safetensors`; the 8192-entry
/// vocabulary comes from the config's `joint.vocabulary`, so `tokenizer.model` is unused.
/// Isolates the `MLXAudioSTT` Parakeet import here.
///
/// Long audio: default generation parameters (`chunkDuration` 1200 s), i.e. one pass for
/// anything under 20 minutes (owner ruling P, 2026-10-24).
///
/// The engine returns text, not arrays, so there is no `MLXErrorGuard.read` here; the Run UI wraps
/// the call in `MLXErrorGuard.guardedRun`.
enum ParakeetEngine {
    /// Transcribe 16 kHz mono float samples using the installed model directory.
    /// Loads the model per call (matches the other engines).
    nonisolated static func transcribe(_ samples: [Float], modelDirectory: URL) async throws -> String {
        do {
            let model = try ParakeetModel.fromDirectory(modelDirectory)
            let output = model.generate(
                audio: MLXArray(samples),
                generationParameters: model.defaultGenerationParameters)
            return output.text.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            throw StageError.engineFailure(stage: "Parakeet ASR", underlying: error)
        }
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
