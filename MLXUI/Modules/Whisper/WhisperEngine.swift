import Foundation
import WhisperKit

/// The single place WhisperKit transcription happens. Isolates the WhisperKit import to
/// one file (mirrors `LLMEngine`). WhisperKit downloads and manages its own CoreML models
/// (argmaxinc/whisperkit-coreml) on init — separate from the app's MLX model registry.
enum WhisperEngine {
    /// Transcribe 16 kHz mono float samples to text using the named WhisperKit model.
    /// Note: this builds a `WhisperKit` per call (loads the model each time); a cached
    /// instance is a later optimization once ASR runs repeatedly.
    nonisolated static func transcribe(_ samples: [Float], model: String) async throws -> String {
        do {
            // Pin WhisperKit's download location to a sandbox-writable folder. Its default
            // base (~/Documents/huggingface, redirected into the container) didn't persist
            // here, so the tokenizer's `tokenizer_config.json` never resolved and surfaced
            // as `TokenizerError.missingConfig`. Giving the CoreML model *and* the tokenizer
            // one explicit writable home fixes that; `.debug` logging makes any remaining
            // download failure visible in the console.
            let downloadBase = ModelStore.shared.whisperKitDirectory
            try? FileManager.default.createDirectory(at: downloadBase, withIntermediateDirectories: true)

            let config = WhisperKitConfig(
                model: model,
                downloadBase: downloadBase,
                tokenizerFolder: downloadBase,
                verbose: true,
                logLevel: .debug,
                download: true
            )
            let pipe = try await WhisperKit(config)
            let results = try await pipe.transcribe(audioArray: samples)
            return results
                .map(\.text)
                .joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            throw StageError.engineFailure(stage: "Whisper ASR", underlying: error)
        }
    }
}
