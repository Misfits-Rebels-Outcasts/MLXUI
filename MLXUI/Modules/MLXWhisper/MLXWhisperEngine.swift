import Foundation
import MLX
import MLXAudioSTT

/// MLX-native Whisper transcription via mlx-audio-swift's `WhisperModel`. Unlike
/// `WhisperEngine` (WhisperKit/CoreML, which downloads its own model), this loads the
/// **safetensors the app already installed** — `mlx-community/whisper-*-asr-fp16` ships
/// `config.json` + `model.safetensors` + tokenizer files, exactly what `fromDirectory`
/// needs and what `InstallManager` downloads. Isolates the `MLXAudioSTT` import to one file.
enum MLXWhisperEngine {
    /// Transcribe 16 kHz mono float samples using the installed model directory.
    /// Loads the model per call (matches `WhisperEngine`); instance caching is a later
    /// optimization once ASR runs repeatedly.
    /// - `language`: Whisper language code (e.g. "en"), or `nil` to auto-detect. Multilingual
    ///   checkpoints condition on the language token; without one a weak model (e.g. tiny) can
    ///   transcribe into the wrong language, so the UI defaults this to English.
    nonisolated static func transcribe(
        _ samples: [Float], modelDirectory: URL, language: String? = nil
    ) async throws -> String {
        do {
            let model = try await WhisperModel.fromDirectory(modelDirectory)
            let params = STTGenerateParameters(language: language)
            let output = model.generate(audio: MLXArray(samples), generationParameters: params)
            return output.text.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            throw StageError.engineFailure(stage: "MLX Whisper ASR", underlying: error)
        }
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
