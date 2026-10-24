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
/// Long audio: the library's own chunking (hard-coded 2 s overlap, token merging) at
/// `chunkSeconds`, not its 1200 s default (owner ruling X, 2026-10-24).
///
/// The engine returns text, not arrays, so there is no `MLXErrorGuard.read` here; the Run UI wraps
/// the call in `MLXErrorGuard.guardedRun`.
enum ParakeetEngine {
    /// Owner ruling X (2026-10-24): one pass over a 10-minute clip peaked at 13.0 GB on an 18 GiB Mac
    /// (journal 2026-566), and one pass up to 20 minutes can exhaust a 19 GB Mac. 120 s is a starting
    /// value, not a tuned one: the journal's chunked measurement decides.
    nonisolated static let chunkSeconds: Float = 120

    /// The model's own defaults with only `chunkDuration` replaced.
    nonisolated static func generationParameters(from base: STTGenerateParameters) -> STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: base.maxTokens,
            temperature: base.temperature,
            topP: base.topP,
            topK: base.topK,
            verbose: base.verbose,
            language: base.language,
            chunkDuration: chunkSeconds,
            minChunkDuration: base.minChunkDuration,
            repetitionPenalty: base.repetitionPenalty,
            repetitionContextSize: base.repetitionContextSize,
            kvBits: base.kvBits,
            kvGroupSize: base.kvGroupSize,
            quantizedKVStart: base.quantizedKVStart)
    }

    /// Transcribe 16 kHz mono float samples using the installed model directory.
    /// Loads the model per call (matches the other engines).
    nonisolated static func transcribe(_ samples: [Float], modelDirectory: URL) async throws -> String {
        do {
            let text = try generate(samples, modelDirectory: modelDirectory)
            // Owner ruling AB (2026-10-24): return MLX's buffer cache once the text is read. It stood at
            // 2.6-4.6 GiB after `generate` (journal 2026-566), Activity Monitor counts it as the app's memory,
            // and this engine reloads the model per call, so keeping it saves nothing. Cleared *after* the
            // model is released (it is local to `generate`): clearing while it was alive left its own 1.2 GiB
            // of weights to land in the cache on return (measured, journal 2026-566).
            Memory.clearCache()
            return text
        } catch {
            throw StageError.engineFailure(stage: "Parakeet ASR", underlying: error)
        }
    }

    nonisolated private static func generate(_ samples: [Float], modelDirectory: URL) throws -> String {
        let model = try ParakeetModel.fromDirectory(modelDirectory)
        let output = model.generate(
            audio: MLXArray(samples),
            generationParameters: generationParameters(from: model.defaultGenerationParameters))
        return output.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
