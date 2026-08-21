import Foundation
import MLX
import MLXAudioTTS

/// MLX-native text-to-speech via mlx-audio-swift's `KokoroModel`. Isolates the
/// `MLXAudioTTS` import to one file (mirrors `WhisperEngine` / `MLXWhisperEngine`).
///
/// Loads the **installed** model directory (config + `kokoro-v1_0.safetensors` + the
/// `voices/` subfolder), which `InstallManager` downloads in full — so there's no separate
/// first-use download. The model emits 24 kHz mono samples.
enum KokoroEngine {
    /// Default MLX Kokoro repo id — the catalog entry installs this (config + weights + voices).
    nonisolated static let defaultRepo = "mlx-community/Kokoro-82M-bf16"
    /// Default voice shipped in that repo.
    nonisolated static let defaultVoice = "af_heart"

    /// Synthesize `text` to a 24 kHz mono `AudioBuffer` using the installed model directory.
    /// `speed` scales pacing (1.0 = normal; higher = faster). Loads the model per call
    /// (matches the other engines); instance caching is a later optimization.
    ///
    /// **Long-text handling.** The Python `mlx_audio` Kokoro pipeline splits the input on
    /// newlines (`split_pattern=r"\n+"`, `engines/tts.py` collects the segments and
    /// concatenates) — the model itself is one-shot with a 510-token cap. This Swift engine
    /// mirrors that: `textSegments` splits on newlines, each segment is synthesized, and the
    /// audio is concatenated. A single over-510-token *segment* (a giant unbroken paragraph)
    /// still refuses — matching the Python, never a truncated wrong answer.
    nonisolated static func synthesize(
        _ text: String,
        voice: String = defaultVoice,
        speed: Float = 1.0,
        modelDirectory: URL
    ) async throws -> AudioBuffer {
        do {
            let model = try await KokoroModel.fromModelDirectory(modelDirectory)
            model.speed = speed
            let segments = textSegments(text)
            var allSamples: [Float] = []
            var sampleRate = model.sampleRate
            for segment in segments {
                let audio = try await model.generate(
                    text: segment, voice: voice, refAudio: nil, refText: nil, language: nil)
                allSamples.append(contentsOf: audio.asArray(Float.self))
                sampleRate = model.sampleRate
            }
            return AudioBuffer(samples: allSamples, sampleRate: sampleRate)
        } catch {
            throw StageError.engineFailure(stage: "Kokoro TTS", underlying: error)
        }
    }

    /// Split text into the segments the Kokoro pipeline synthesizes separately — the Swift
    /// mirror of the Python's `split_pattern=r"\n+"`. A blank line between paragraphs (one
    /// or more newlines) is the split point; whitespace-only segments are dropped. Pure so
    /// it is unit-testable without a model.
    nonisolated static func textSegments(_ text: String) -> [String] {
        text.split(whereSeparator: \.isNewline)
            .map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }

    /// Voice names available in an installed Kokoro model's `voices/` folder, sorted.
    /// Reads the directory directly (no model load), so the run view can populate a picker
    /// before any synthesis.
    nonisolated static func availableVoices(modelID: String) -> [String] {
        let voicesDir = installedModelDirectory(id: modelID)
            .appendingPathComponent("voices", isDirectory: true)
        let files = (try? FileManager.default.contentsOfDirectory(atPath: voicesDir.path)) ?? []
        return voiceNames(inDirectoryContents: files)
    }

    /// Pure mapping of a `voices/` directory listing to sorted voice names (drops the
    /// `.safetensors` extension, ignores anything else). Unit-tested (gate G2).
    nonisolated static func voiceNames(inDirectoryContents files: [String]) -> [String] {
        files
            .filter { $0.hasSuffix(".safetensors") }
            .map { String($0.dropLast(".safetensors".count)) }
            .sorted()
    }
}
