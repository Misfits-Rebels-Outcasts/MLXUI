import Foundation

/// Maps a catalog `ModelEntry` to the WhisperKit model *size* name. WhisperKit ignores the
/// catalog's MLX fp16 whisper files and downloads its own CoreML model keyed by this size
/// name (see `WhisperModule.descriptor.notes`), so the entry only selects the size.
///
/// Order matters: check longer / more-specific names first so `small` / `tiny` don't
/// shadow `large` / `medium`. `large` and `large-v3` both resolve to `"large-v3"`.
nonisolated enum WhisperKitSize {
    static func from(_ model: ModelEntry) -> String {
        let haystack = (model.id + " " + model.displayName).lowercased()
        if haystack.contains("large")  { return "large-v3" }
        if haystack.contains("medium") { return "medium" }
        if haystack.contains("small")  { return "small" }
        if haystack.contains("tiny")   { return "tiny" }
        if haystack.contains("base")   { return "base" }
        return "base"
    }
}
