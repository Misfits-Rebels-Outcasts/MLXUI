import Foundation

/// CFM-R9-1 / CFM-R9-FIX-1 — the byte-preserving settings editor, the write-back half of the
/// QR9 gate ("settings must survive a parse → edit → serialize round-trip unchanged").
///
/// The Python has no settings *serializer* — `Settings` is a parse-only view and `Row.settings`
/// is the verbatim raw string. This editor splices exactly the edited token's source span, so
/// everything else — separators, spacing, other tokens, quoted strings — is copied through
/// byte-for-byte.
///
/// **FIX-1 (the quoted-value corruption):** tokens are found and spliced against the **same**
/// string. `_TOKEN_RE`'s `key="…"` alternative already matches a quoted value as one token, so
/// there is no quote-blanking step at all — a `key=` inside a quoted span is never a separate
/// token, and a quoted value's true extent is preserved. (The earlier blank-quoted→original
/// splice matched a short `key=` range and left the quoted tail behind.)
///
/// **FIX-2:** `quote` is the exact inverse of `FlowSettings.unquote` (escapes `\`, `"`, `\n`,
/// `\t`), so `unquote("\"\(quote(x))\"") == x` for arbitrary `x`.
nonisolated enum FlowSettingsEditor {

    /// `parse_duration` — `timeout=<duration>` with no spec grammar: one or more
    /// `<number><unit>` runs back to back (`"10m"`, `"1h30m"`, `"500ms"`), units
    /// `{ms, s, m, h}`, no whitespace (CFM-R10-FIX-4 — `TimeInterval("30s")` is nil, so
    /// every corpus-style duration silently meant "no timeout"). Ported from
    /// `engines/agent.py::parse_duration`.
    static func parseDuration(_ text: String) -> TimeInterval? {
        let regex = NSRegularExpression.compiled(#"(\d+(?:\.\d+)?)(ms|s|m|h)"#)
        let units: [String: TimeInterval] = ["ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0]
        var total: TimeInterval = 0
        var pos = text.startIndex
        var any = false
        let ns = NSRange(text.startIndex..<text.endIndex, in: text)
        for match in regex.matches(in: text, range: ns) {
            guard let full = Range(match.range, in: text), full.lowerBound == pos else { break }
            guard let numRange = Range(match.range(at: 1), in: text),
                  let unitRange = Range(match.range(at: 2), in: text) else { break }
            let number = Double(text[numRange]) ?? 0
            let unit = String(text[unitRange])
            total += number * (units[unit] ?? 0)
            pos = full.upperBound
            any = true
        }
        return (any && pos == text.endIndex) ? total : nil
    }

    /// The same `_TOKEN_RE` as `FlowSettings`, returning each token's source range so a value
    /// can be spliced in place.
    private static let tokenRegex = NSRegularExpression.compiled(#"(?:lora|controlnet)=[^;]*|[A-Za-z_][A-Za-z0-9_]*="(?:[^"\\]|\\.)*"|"(?:[^"\\]|\\.)*"|[^\s;]+"#)

    /// Every `(token, range)` the tokenizer finds in `raw`, in source order — tokenized in the
    /// **original** string (FIX-1).
    static func tokensAndRanges(_ raw: String) -> [(token: String, range: Range<String.Index>)] {
        let ns = NSRange(raw.startIndex..<raw.endIndex, in: raw)
        var out: [(token: String, range: Range<String.Index>)] = []
        for match in tokenRegex.matches(in: raw, range: ns) {
            guard let range = Range(match.range, in: raw) else { continue }
            out.append((String(raw[range]), range))
        }
        return out
    }

    /// Set (or remove, when `value` is nil) a `key=` token's value, splicing only that token's
    /// span. A missing key is appended as `key=value`; a removal with no token is a no-op.
    /// Everything outside the edited span is byte-identical (the QR9 round-trip).
    static func replace(key: String, value: String?, in raw: String?) -> String? {
        let text = raw ?? ""
        // SP-2 (`RSI/DelegateFixItBacklog.md`; SPEC-Q226 in `catflow-mlx`): an empty or
        // whitespace-only value is treated as no value at all, i.e. a removal — a picker
        // field left blank (the Direct build's duration control) can no longer write a
        // valueless `key=` (`timeout=` with nothing after the `=`). `FlowSettings` and
        // `FlowValidator.parseSettingsKV` already disagree about how to *read* one of those
        // (Q226, open); this stops the app *writing* one, which is correct under either
        // answer — it only changes what a fresh edit produces, never how an existing file's
        // text is read.
        let value = value.flatMap { $0.trimmingCharacters(in: .whitespaces).isEmpty ? nil : $0 }
        var target: (token: String, range: Range<String.Index>)?
        for entry in tokensAndRanges(text) {
            if entry.token.hasPrefix("\(key)=") {
                target = entry
            }
        }
        if let target {
            if let value {
                let eq = target.token.firstIndex(of: "=")!
                let keyPart = target.token[..<eq]
                // Preserve the token's existing quoting style: a `key="quoted"` stays quoted
                // even if the new value doesn't need it (a no-op edit must be byte-identical).
                let wasQuoted = target.token.hasSuffix("\"")
                let newValue = (wasQuoted || needsQuoting(value)) ? quote(value) : value
                return text.replacingCharacters(in: target.range, with: "\(keyPart)=\(newValue)")
            } else {
                return removingToken(in: text, range: target.range)
            }
        }
        guard let value else { return text }
        let newValue = needsQuoting(value) ? quote(value) : value
        if text.isEmpty { return "\(key)=\(newValue)" }
        return "\(text); \(key)=\(newValue)"
    }

    /// Set the row's instruction — the first quoted token (the "What should it do?" box,
    /// answer `a5`). `nil` clears it. Spliced in place; the rest of the settings survive.
    /// Reads and writes the **same** token (FIX-6): the first quoted span.
    static func replaceInstruction(_ instruction: String?, in raw: String?) -> String? {
        let text = raw ?? ""
        if let token = firstQuotedToken(in: text) {
            if let instruction {
                return text.replacingCharacters(in: token.range, with: quote(instruction))
            }
            return removingToken(in: text, range: token.range)
        }
        guard let instruction else { return text }
        if text.isEmpty { return quote(instruction) }
        return "\(text); \(quote(instruction))"
    }

    /// Replace the row's path token — a `path=` value if present, else the first bare token
    /// (the same two readings as `FlowSettings.pathValue()`), else append it. Used by the
    /// inspector's "Choose file…" to write a copied-in file's bare name. Spliced in place.
    static func replacePath(_ path: String, in raw: String?) -> String? {
        let text = raw ?? ""
        for entry in tokensAndRanges(text) where entry.token.hasPrefix("path=") {
            let wasQuoted = entry.token.hasSuffix("\"")
            let newValue = (wasQuoted || needsQuoting(path)) ? quote(path) : path
            return text.replacingCharacters(in: entry.range, with: "path=\(newValue)")
        }
        for entry in tokensAndRanges(text) where !entry.token.contains("=") {
            return text.replacingCharacters(in: entry.range, with: needsQuoting(path) ? quote(path) : path)
        }
        if text.isEmpty { return needsQuoting(path) ? quote(path) : path }
        return "\(text); \(needsQuoting(path) ? quote(path) : path)"
    }

    /// RT-1 — `Template`'s Pattern box. `Template`'s settings string **is** the whole pattern
    /// (`TextTools.unquoteWhole`, fact 11), not a single spliced token like every other
    /// `replace*` here, so this replaces the row's entire settings string outright rather than
    /// locating and splicing a span. `raw` isn't read — kept for call-site symmetry with
    /// `replaceInstruction`/`replacePath`. An empty pattern clears the settings entirely
    /// (`nil`), matching this file's other "empty box" behavior.
    static func replaceWholeSettings(_ text: String, in raw: String?) -> String? {
        text.isEmpty ? nil : quote(text)
    }

    /// The first quoted span's `(token, range)` — the instruction's true extent.
    static func firstQuotedToken(in text: String) -> (token: String, range: Range<String.Index>)? {
        let quoted = NSRegularExpression.compiled(#""(?:[^"\\]|\\.)*""#)
        let ns = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = quoted.firstMatch(in: text, range: ns),
              let range = Range(match.range, in: text) else { return nil }
        return (String(text[range]), range)
    }

    /// Whether a value must be quoted to survive the tokenizer (contains a separator, a space,
    /// or a backslash — the tokenizer's own quoting triggers).
    static func needsQuoting(_ value: String) -> Bool {
        value.contains(where: { $0 == " " || $0 == ";" || $0 == "\n" || $0 == "\t" || $0 == "\"" || $0 == "\\" })
    }

    /// Remove the token at `range` plus one adjacent `;`/space separator (whichever side has
    /// one), so a removal doesn't leave a dangling `;` or a leading space — and never collapses
    /// spaces *inside* quoted instructions (FIX-7).
    private static func removingToken(in text: String, range: Range<String.Index>) -> String {
        var start = range.lowerBound
        var end = range.upperBound
        if start > text.startIndex {
            let before = text[text.index(before: start)]
            if before == ";" || before == " " {
                start = text.index(before: start)
            }
        } else if end < text.endIndex {
            let after = text[end]
            if after == ";" || after == " " {
                end = text.index(after: end)
            }
        }
        var out = text
        out.replaceSubrange(start..<end, with: "")
        // Strip a leading separator left by the splice (a removed token was the only thing
        // after it). No global space collapse — that would reach inside quoted strings.
        while out.hasPrefix(" ") { out.removeFirst() }
        if out.hasPrefix(";") {
            out.removeFirst()
            while out.hasPrefix(" ") { out.removeFirst() }
        }
        return out
    }

    /// The exact inverse of `FlowSettings.unquote` (FIX-2): escapes `\`, `"`, newlines and tabs
    /// so the tokenizer's `_TOKEN_RE` escaping round-trips. Order matters — escape `\` first so
    /// the `\n`/`\t` escapes it inserts are not themselves re-escaped.
    static func quote(_ value: String) -> String {
        let escaped = value.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\t", with: "\\t")
        return "\"\(escaped)\""
    }
}
