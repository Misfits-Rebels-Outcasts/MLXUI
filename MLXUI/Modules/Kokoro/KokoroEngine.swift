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
    nonisolated static func synthesize(
        _ text: String,
        voice: String = defaultVoice,
        speed: Float = 1.0,
        modelDirectory: URL
    ) async throws -> AudioBuffer {
        do {
            let model = try await KokoroModel.fromModelDirectory(modelDirectory)
            model.speed = speed
            let audio = try await model.generate(
                text: text, voice: voice, refAudio: nil, refText: nil, language: nil)
            return AudioBuffer(samples: audio.asArray(Float.self), sampleRate: model.sampleRate)
        } catch {
            throw StageError.engineFailure(stage: "Kokoro TTS", underlying: error)
        }
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("AI Browser/models/\(id)", isDirectory: true)
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
