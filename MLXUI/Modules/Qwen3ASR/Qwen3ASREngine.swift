import Foundation
import MLX
import MLXAudioSTT

/// MLX-native Qwen3-ASR via mlx-audio-swift's `Qwen3ASRModel`. Takes the whole 16 kHz mono buffer and returns
/// transcribed text, the same shape `VoxtralEngine` and `ParakeetEngine` use, so it composes onto `ASRStage`.
/// Loads the **installed** directory (the A2 pattern, never `fromPretrained`, which would download behind the
/// app's back). Isolates the `MLXAudioSTT` Qwen3-ASR import here.
///
/// **Language.** `language` is `nil`: the model identifies the language itself. The library prefixes its output with
/// `language X<asr_text>` and strips it; this engine returns only the transcript.
///
/// **Thread-bound streams (KICKOFF rule 17; journal 2026-558).** mlx-audio-swift 5c4c12b's `Qwen3ASR.swift` has the
/// array-index-after-another-index gather (`nextLast[0..., recentArr]`, lines 1326 and 1528) for its repetition
/// penalty. It runs only when `repetitionPenalty != 1.0`; the model's defaults (and so this engine's) leave it at
/// 1.0, so the gather is never built, and `generationParameters(from:)` pins that. Nothing here crosses an actor hop
/// with a lazy array: the model is built, run and read inside `generate`, and `generate` returns a `String`. So
/// `MLXAudioThread.run` is **not** used (owner-confirmed 2026-10-25; the main-actor probe in
/// `Qwen3ASRRealLoadTests` is the evidence). There is also no array handed back to read through
/// `MLXErrorGuard.read`: `STTOutput` carries text and plain dictionaries only. The input buffer is evaluated up
/// front with `MLXErrorGuard.evaluate`, so a failed evaluation throws before the library touches it (rule 17b).
///
/// **Install side effect (readme T5).** On first load the library writes `tokenizer.json` into the model folder
/// (`generateTokenizerJSONIfMissing`), built from `vocab.json` + `merges.txt` + `tokenizer_config.json`, which the
/// install fetches. `Qwen3ASRInstallTests` shows that is harmless to `InstallManager` (verify runs on the download
/// folder before the move; the `.installed` marker is all the reconcile reads).
///
/// Long audio: the library splits at low-energy points into chunks of at most `chunkSeconds` (default 1200 s), one
/// pass per chunk, and the audio encoder and the prompt's KV cache grow with the chunk. The engine uses 120 s, as
/// `ParakeetEngine` does (owner ruling X, 2026-10-24); the owner confirmed it on 2026-10-25. It is a ceiling per
/// chunk, so the peak stays bounded on long clips. Measured 2026-10-25 on a 588 s clip with
/// `Qwen3ASRLongClipMeasurement`: at 120 s the MLX peak was 3.80 GiB and the run took 47.4 s; at the library's
/// 1200 s it was 4.50 GiB and 66.2 s. Both held 2.335 GiB after load and returned to it. The in-app peak
/// (MLXUI-Direct, Activity Monitor basis) was 6.82 GiB. A 588 s clip fits inside one 1200 s chunk, so the gap on
/// a ~20 min clip is not measured.
enum Qwen3ASREngine {
    nonisolated static let chunkSeconds: Float = 120

    /// The model's own defaults with only `chunkDuration` replaced (`chunkSeconds` unless a chunk's own length says
    /// otherwise, see `generate`). `language` stays as the model has it (`nil`, auto) and `repetitionPenalty` stays 1.0
    /// (see the type comment).
    nonisolated static func generationParameters(from base: STTGenerateParameters,
                                                 chunkDuration: Float = chunkSeconds) -> STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: base.maxTokens,
            temperature: base.temperature,
            topP: base.topP,
            topK: base.topK,
            verbose: base.verbose,
            language: base.language,
            chunkDuration: chunkDuration,
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
            let text = try await generate(samples, modelDirectory: modelDirectory)
            // As ParakeetEngine (owner ruling AB, 2026-10-24): return MLX's buffer cache once the text is read, after
            // the model (a local of `generate`) is released, so its weights don't land in the cache on return.
            Memory.clearCache()
            return text
        } catch {
            throw StageError.engineFailure(stage: "Qwen3-ASR", underlying: error)
        }
    }

    nonisolated private static func generate(_ samples: [Float], modelDirectory: URL) async throws -> String {
        let model = try await Qwen3ASRModel.fromModelDirectory(modelDirectory)
        let audio = MLXArray(samples)
        try MLXErrorGuard.evaluate(audio)
        // MP3-FIX-1 (owner ruling 2026-10-26). The library's `generate` treats `maxTokens` (8192) as ONE budget across
        // every chunk (`remainingTokens -= result.generationTokens; if remainingTokens <= 0 { break }`) and drops the
        // later chunks without a word: about 40 minutes of speech, then nothing. So the split is done here, with the
        // library's own public `splitAudioIntoChunks` at `chunkSeconds` (the same cut points the library used), and each
        // chunk gets its own `generate` call and so its own full budget. A chunk can run a little past `chunkSeconds`
        // (the cut searches 5 s either side), so each call's `chunkDuration` is that chunk's length plus a margin and the
        // library never splits it again. Texts are joined with one space, as the library joins them.
        let base = model.defaultGenerationParameters
        var texts: [String] = []
        for (chunk, _) in splitAudioIntoChunks(audio, sampleRate: 16_000, chunkDuration: chunkSeconds,
                                               minChunkDuration: base.minChunkDuration) {
            let length = Float(chunk.dim(0)) / 16_000
            let parameters = generationParameters(from: base, chunkDuration: max(chunkSeconds, length + 1))
            texts.append(model.generate(audio: chunk, generationParameters: parameters).text)
            Memory.clearCache()
        }
        return texts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
