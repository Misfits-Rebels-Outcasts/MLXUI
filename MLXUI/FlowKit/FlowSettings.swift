import Foundation

/// A minimal port of the Python `Settings` settings-string parser (`tools/_settings.py`):
/// a `.cat` row's settings is a `;`-separated list of `key=value` tokens, where a token
/// with no `=` is the **bare value** (`first_bare()`). A quoted value is unquoted and its
/// `\n \t \" \\` escapes processed — exactly the Python's `_unquote` (so `separator="\n"`
/// yields a real newline). Used by the instant tools to find their path (`memo.m4a`,
/// `memo-tldr.wav`, …) and read settings like `lang=`, `voice=`, `separator=`.
nonisolated struct FlowSettings {
    private let tokens: [(key: String, value: String)]
    private let bare: String?

    init(_ raw: String?) {
        var tokens: [(String, String)] = []
        var bare: String?
        for piece in (raw ?? "").split(separator: ";") {
            let trimmed = piece.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            if let eq = trimmed.firstIndex(of: "=") {
                let key = trimmed[..<eq].trimmingCharacters(in: .whitespacesAndNewlines)
                let rawValue = trimmed[trimmed.index(after: eq)...]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !key.isEmpty {
                    tokens.append((key, Self.unquote(rawValue)))
                }
            } else if bare == nil {
                bare = Self.unquote(trimmed)
            }
        }
        self.tokens = tokens
        self.bare = bare
    }

    /// `_unquote` — strip one surrounding pair of double quotes and process
    /// `\n \t \" \\` escapes (the Python's shared tokenizer rule).
    static func unquote(_ raw: String) -> String {
        var s = raw
        if s.count >= 2, s.first == "\"", s.last == "\"" {
            s = String(s.dropFirst().dropLast())
        }
        return s.replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")
            .replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\\\", with: "\\")
    }

    /// The value for `key`, or the bare (no-`=`) token, or nil.
    func pathValue() -> String? {
        tokens.first(where: { $0.key == "path" })?.value ?? bare
    }

    /// A typed value for `key`, else `defaultValue`.
    func value(for key: String, default defaultValue: String? = nil) -> String? {
        tokens.first(where: { $0.key == key })?.value ?? defaultValue
    }
}
