import Foundation

/// Per-family voice options for the generic MLX TTS run view (AT2). mlx-audio-swift's
/// `TTS.loadModel` doesn't enumerate a model's voices, so these are the well-known,
/// documented preset names per family. Families without a fixed list (Qwen3-TTS speakers
/// vary by checkpoint) return `[]`, and the UI falls back to a free-form voice field that
/// defaults to the model's own speaker. Cloning families (chatterbox) take a reference clip
/// instead of a named voice — see `supportsReferenceAudio`.
enum MLXAudioTTSVoices {
    /// Documented Orpheus-3B voices (passed straight through as the `voice` id).
    static let orpheusPresets = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

    /// Known preset voice ids for `family`, or `[]` when none are enumerable.
    /// Matched case-insensitively on the family name.
    static func presets(family: String) -> [String] {
        let lower = family.lowercased()
        if lower.contains("orpheus") { return orpheusPresets }
        return []
    }

    /// Families whose voice is supplied as a **reference-audio clip** (zero-shot cloning)
    /// rather than a named id — the run view shows an audio picker for these. Chatterbox
    /// clones from a reference clip and *requires* one: its engine throws
    /// `invalidInput` when no `refAudio` is passed, so the run view makes the clip mandatory.
    static func supportsReferenceAudio(family: String) -> Bool {
        family.lowercased().contains("chatterbox")
    }
}
