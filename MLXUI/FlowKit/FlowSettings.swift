import Foundation

/// A port of the Python `Settings` settings-string parser (`tools/_settings.py`) — CFM-FIX-2
/// (H5). The tokenizer is `_TOKEN_RE`, which splits on **whitespace and `;`**, handles quoted
/// spans containing `;`, and lets a `lora=`/`controlnet=` value run greedy-to-`;`. `pairs`
/// keeps the **last** value of a repeated key; `multi` keeps every value in order. This is
/// load-bearing in R9, where the inspector writes settings back, so the parser must
/// round-trip what it reads. `Fixtures/CatFlow/tools/settings_golden.json` pins it against
/// the Python tokenizer byte-for-byte.
///
/// A bare `;` between tokens is a no-op delimiter, never data (P5-NA-06).
nonisolated struct FlowSettings {
    private let pairs: [String: String]
    private let multiPairs: [String: [String]]
    private let bare: [String]

    init(_ raw: String?) {
        let text = raw ?? ""
        var pairs: [String: String] = [:]
        var multiPairs: [String: [String]] = [:]
        var bare: [String] = []
        let regex = NSRegularExpression.compiled(#"(?:lora|controlnet)=[^;]*|[A-Za-z_][A-Za-z0-9_]*="(?:[^"\\]|\\.)*"|"(?:[^"\\]|\\.)*"|[^\s;]+"#)
        let ns = NSRange(text.startIndex..<text.endIndex, in: text)
        for match in regex.matches(in: text, range: ns) {
            guard let range = Range(match.range, in: text) else { continue }
            let token = String(text[range])
            if token.hasPrefix("\"") {
                bare.append(Self.unquote(token))
                continue
            }
            if let eq = token.firstIndex(of: "=") {
                let key = String(token[..<eq])
                let value = String(token[token.index(after: eq)...])
                if Self.isValidKey(key) {
                    let unquoted = Self.unquote(value)
                    pairs[key] = unquoted
                    multiPairs[key, default: []].append(unquoted)
                } else {
                    bare.append(Self.unquote(token))
                }
            } else {
                bare.append(Self.unquote(token))
            }
        }
        self.pairs = pairs
        self.multiPairs = multiPairs
        self.bare = bare
    }

    /// `re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)` — a token only becomes a pair when its
    /// key is a bare identifier.
    private static func isValidKey(_ key: String) -> Bool {
        guard let first = key.unicodeScalars.first else { return false }
        guard first == "_" || (first.value >= 0x41 && first.value <= 0x5A)
            || (first.value >= 0x61 && first.value <= 0x7A) else { return false }
        for scalar in key.unicodeScalars.dropFirst() {
            let ok = scalar == "_" || (scalar.value >= 0x41 && scalar.value <= 0x5A)
                || (scalar.value >= 0x61 && scalar.value <= 0x7A)
                || (scalar.value >= 0x30 && scalar.value <= 0x39)
            if !ok { return false }
        }
        return true
    }

    /// `_unquote` — strip one surrounding pair of double quotes and process
    /// `\n \t \" \\` escapes **only when the token is quoted**; an unquoted token is returned
    /// untouched (the Python `_settings.py:49-58` — CFM-R9-FIX-7, so a Windows path like
    /// `C:\notes\temp.txt` isn't silently mangled).
    static func unquote(_ raw: String) -> String {
        guard raw.count >= 2, raw.first == "\"", raw.last == "\"" else { return raw }
        let inner = String(raw.dropFirst().dropLast())
        return inner.replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")
            .replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\\\", with: "\\")
    }

    /// The path a Read/Save row uses: `path=` if present, else the first bare token —
    /// mirroring `tools/files.py`'s `s.get("path") or s.first_bare()`.
    func pathValue() -> String? {
        value(for: "path") ?? firstBare()
    }

    /// `Settings.first_bare()` — the first no-`=` token, or nil.
    func firstBare() -> String? {
        bare.first
    }

    /// `Settings.get(key, default)` — the last value of `key`.
    func value(for key: String, default defaultValue: String? = nil) -> String? {
        pairs[key] ?? defaultValue
    }

    /// `Settings.multi(key)` — every value `key` appears with, in source order.
    func multi(for key: String) -> [String] {
        multiPairs[key] ?? []
    }

    /// `Settings.has(key)`.
    func has(_ key: String) -> Bool {
        pairs[key] != nil
    }

    /// `raw_tokens` — the token boundaries `_TOKEN_RE` finds, without unquoting/pairing,
    /// for callers that rebuild or strip settings text verbatim (`presets.py`).
    static func rawTokens(_ raw: String?) -> [String] {
        let text = raw ?? ""
        let regex = try! NSRegularExpression(pattern: #"(?:lora|controlnet)=[^;]*|[A-Za-z_][A-Za-z0-9_]*="(?:[^"\\]|\\.)*"|"(?:[^"\\]|\\.)*"|[^\s;]+"#)
        let ns = NSRange(text.startIndex..<text.endIndex, in: text)
        var out: [String] = []
        for match in regex.matches(in: text, range: ns) {
            guard let range = Range(match.range, in: text) else { continue }
            out.append(String(text[range]))
        }
        return out
    }

    /// Test-only exposure of the parsed pair keys (pinned against the Python golden).
    var testKeys: Set<String> { Set(pairs.keys) }
    var testBare: [String] { bare }
}
