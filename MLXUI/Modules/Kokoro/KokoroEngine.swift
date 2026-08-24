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
    /// **Long-text handling (CFM-FIX-4).** The Python `mlx_audio` Kokoro pipeline splits the
    /// input on `re.split(r"\n+", text.strip())` and synthesizes each segment; the model
    /// itself is one-shot with a 510-token cap. This Swift engine mirrors that:
    /// `textSegments` splits on runs of `\n` **after stripping the whole text** (CR, VT, FF
    /// and U+2028/9 are *not* split points — only `\n`, exactly as the Python), each segment
    /// is further `subChunks`d at sentence/clause boundaries so it stays under the cap, and
    /// the audio is concatenated. Nothing is dropped: where the Python **truncates** an
    /// over-long phoneme batch to 510 (a warning, not a refusal), the Swift speaks the whole
    /// text in sub-chunks. Empty or whitespace-only input raises "Speak produced no audio"
    /// rather than returning a zero-length WAV (the Python's own error).
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
                for chunk in subChunks(segment) {
                    let audio = try await model.generate(
                        text: chunk, voice: voice, refAudio: nil, refText: nil, language: nil)
                    allSamples.append(contentsOf: audio.asArray(Float.self))
                    sampleRate = model.sampleRate
                }
            }
            // FIX-4a: empty/whitespace input (or every segment producing nothing) is an
            // error — never a silent zero-length WAV.
            guard !allSamples.isEmpty else {
                throw SpeakProducedNoAudio()
            }
            return AudioBuffer(samples: allSamples, sampleRate: sampleRate)
        } catch {
            throw StageError.engineFailure(stage: "Kokoro TTS", underlying: error)
        }
    }

    /// Split the text into the segments the Kokoro pipeline synthesizes separately — the
    /// Swift mirror of `re.split(r"\n+", text.strip())`. The whole text is stripped first,
    /// only runs of `\n` are split points (a blank line between paragraphs), and empty
    /// segments are dropped. Pure so it is unit-testable without a model.
    nonisolated static func textSegments(_ text: String) -> [String] {
        let stripped = text.trimmingCharacters(in: .whitespacesAndNewlines)
        var result: [String] = []
        var current = String.UnicodeScalarView()
        var lastWasNewline = false
        for scalar in stripped.unicodeScalars {
            if scalar == "\n" {
                if lastWasNewline { continue }
                result.append(String(current))
                current = String.UnicodeScalarView()
                lastWasNewline = true
            } else {
                current.append(scalar)
                lastWasNewline = false
            }
        }
        result.append(String(current))
        return result.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    /// Sub-chunk one `\n+` segment at sentence/clause boundaries so each piece fits under
    /// the model's 510-token cap (CFM-FIX-4d). A single over-long "sentence" is hard-split
    /// at whitespace. Pure, so it is unit-testable without a model.
    nonisolated static func subChunks(_ text: String, maxLength: Int = 350) -> [String] {
        guard text.unicodeScalars.count > maxLength else { return [text] }
        let regex = NSRegularExpression.compiled(#"[.!?;:]\s+"#)
        let ns = NSRange(text.startIndex..<text.endIndex, in: text)
        var sentences: [String] = []
        var last = text.startIndex
        for match in regex.matches(in: text, range: ns) {
            guard let range = Range(match.range, in: text) else { continue }
            sentences.append(String(text[last...range.lowerBound]))
            last = range.upperBound
        }
        sentences.append(String(text[last...]))
        var chunks: [String] = []
        var current = ""
        for sentence in sentences {
            let sentence = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !sentence.isEmpty else { continue }
            if sentence.unicodeScalars.count > maxLength {
                if !current.isEmpty { chunks.append(current); current = "" }
                chunks.append(contentsOf: hardSplit(sentence, maxLength: maxLength))
            } else if current.isEmpty {
                current = sentence
            } else if current.unicodeScalars.count + sentence.unicodeScalars.count + 1 <= maxLength {
                current += " " + sentence
            } else {
                chunks.append(current)
                current = sentence
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks.filter { !$0.isEmpty }
    }

    /// Split a too-long string at whitespace boundaries into ≤`maxLength` pieces.
    private static func hardSplit(_ text: String, maxLength: Int) -> [String] {
        let words = text.split(whereSeparator: \.isWhitespace).map(String.init)
        var chunks: [String] = []
        var current = ""
        for word in words {
            if current.isEmpty {
                current = word
            } else if current.unicodeScalars.count + word.unicodeScalars.count + 1 <= maxLength {
                current += " " + word
            } else {
                chunks.append(current)
                current = word
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
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

/// CFM-FIX-4a: the Python's `RuntimeError("Speak produced no audio")` — empty/whitespace
/// input is an error, never a zero-length WAV. `CustomStringConvertible` so the error-voice
/// bridge surfaces the sentence, not an opaque NSError.
private struct SpeakProducedNoAudio: Error, CustomStringConvertible {
    var description: String { "Speak produced no audio" }
}
