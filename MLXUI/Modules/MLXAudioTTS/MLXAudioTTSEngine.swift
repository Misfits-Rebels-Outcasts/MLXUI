import Foundation
import MLX
import MLXAudioTTS

/// Generic MLX text-to-speech for the non-Kokoro TTS families (Qwen3-TTS, chatterbox,
/// orpheus) via mlx-audio-swift's `TTS.loadModel`, which auto-detects the architecture from
/// the installed `config.json` and dispatches to the right model. Loads the **installed**
/// model directory (the A2 pattern). Kokoro keeps its own `KokoroEngine` (claimed `.exact`);
/// this is the `.generic` fallback for every other MLX TTS repo. Isolates the import to one file.
enum MLXAudioTTSEngine {
    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }

    /// Synthesize `text` to an `AudioBuffer` using the installed model directory.
    /// - `voice`: optional named voice id (nil = the model's default speaker).
    /// - `referenceAudio`: optional reference clip for zero-shot voice cloning (chatterbox).
    ///   The model's public generate path treats `refAudio` as 24 kHz, so the clip is
    ///   resampled to 24 kHz before being passed.
    /// Loads the model per call (matches the other engines).
    nonisolated static func synthesize(
        _ text: String,
        voice: String?,
        referenceAudio: AudioBuffer? = nil,
        modelDirectory: URL
    ) async throws -> AudioBuffer {
        do {
            let model = try await TTS.loadModel(modelRepo: modelDirectory.path)
            let refArray: MLXArray? = referenceAudio.map { buffer in
                let resampled = (try? AudioResampler.resample(buffer, to: 24_000)) ?? buffer
                return MLXArray(resampled.samples)
            }
            let audio = try await model.generate(
                text: text, voice: voice, refAudio: refArray, refText: nil, language: nil)
            return AudioBuffer(samples: audio.asArray(Float.self), sampleRate: model.sampleRate)
        } catch {
            throw StageError.engineFailure(stage: "MLXAudio TTS", underlying: error)
        }
    }
}
