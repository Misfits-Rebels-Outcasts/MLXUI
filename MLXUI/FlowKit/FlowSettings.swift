import Foundation

/// A minimal port of the Python `Settings` settings-string parser (`catalog/registry.py`):
/// a `.cat` row's settings is a `;`-separated list of `key=value` tokens, where a token
/// with no `=` is the **bare value** (`first_bare()`). Used by the instant tools to find
/// their path (`memo.m4a`, `memo-tldr.wav`, …). Not the full resolver — enums/ranges/
/// defaults stay in `CuratedManifest` (CFM-R2-2) and R2-6's dispatch.
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
                let value = trimmed[trimmed.index(after: eq)...]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                if !key.isEmpty {
                    tokens.append((key, value))
                }
            } else if bare == nil {
                bare = trimmed
            }
        }
        self.tokens = tokens
        self.bare = bare
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
