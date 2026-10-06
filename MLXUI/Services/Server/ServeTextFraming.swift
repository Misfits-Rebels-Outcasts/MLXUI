import Foundation

/// One rule, shared by the two framers that stream raw text: how much of the tail is still
/// undecided.
///
/// Both `ServeToolCallScanner` (looking for a start tag) and `ServeReasoningSplitter` (looking for
/// `</think>`) have the same job at the end of a buffer: emit everything that cannot be part of the
/// tag, and hold only the longest trailing suffix that could still become it. Held text is never
/// dropped — the next push, or `finish()`, hands it back.
nonisolated enum ServeTextFraming {

    /// Split `text` into what may be emitted **now** and the longest trailing suffix that is a
    /// proper prefix of `tag`. A tag of one character or less can never have a partial prefix, so
    /// everything is safe.
    static func holding(_ text: String, partialOf tag: String) -> (safe: String, held: String) {
        guard tag.count > 1, !text.isEmpty else { return (text, "") }
        for length in stride(from: min(tag.count - 1, text.count), through: 1, by: -1) {
            let suffix = text.suffix(length)
            guard suffix == tag.prefix(length) else { continue }
            return (String(text.dropLast(length)), String(suffix))
        }
        return (text, "")
    }
}
