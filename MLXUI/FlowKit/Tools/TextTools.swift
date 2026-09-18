import Foundation

/// The §2 text instant tools — the Swift port of `catflow-mlx/src/catflow/tools/text.py`
/// (CFM-R4-1/2), as `AssetStage`s. Pure text transforms: no engine, no filesystem, no
/// model imports (the Python's isolation rule's "tool" side). All operate on the input
/// items' inline text.
///
/// Several defaults / micro-syntax choices are the Python's own (SPEC-Q13): `Split`'s
/// `by=tokens` default with a 200-unit sliding window, `Filter`'s `length=N` op, `Count`'s
/// `items` vs `words`, `Template`'s `{1}`/`{input:K}` placeholders, `Join Text`'s default
/// `\n` separator, `Diff`'s `inline` (ndiff) vs `unified`.
nonisolated enum TextTools {

    /// `_unquote_whole` — strip one surrounding pair of double quotes and process
    /// `\n \t \" \\` escapes (used for settings that are themselves one value).
    static func unquoteWhole(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.count >= 2, s.first == "\"", s.last == "\"" {
            s = String(s.dropFirst().dropLast())
        }
        return s.replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")
            .replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\\\", with: "\\")
    }

    /// The flat inline texts of every input asset, in ref/chain order. Python's `_flat_texts`
    /// keeps non-inline items as `""` — positions stay aligned rather than shifting (M10).
    static func flatTexts(_ inputs: [Asset]) -> [String] {
        inputs.flatMap { asset in asset.items.map { $0.value ?? "" } }
    }

    /// The first input item's inline text, or "".
    static func singleText(_ inputs: [Asset]) -> String {
        inputs.first?.items.first?.value ?? ""
    }
}

// MARK: - Split

nonisolated struct SplitTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .single(.text) }
    var produces: Shape { .listOf(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let text = TextTools.singleText([input])
        let s = FlowSettings(settings)
        let by = s.value(for: "by", default: "tokens") ?? "tokens"
        let units = splitUnits(text, by: by)
        guard !units.isEmpty else { return Asset(items: [Item(kind: .text, value: "", path: nil, sourceText: nil)]) }
        // lines/sentences/headings are document boundaries — one item per unit (SPEC-Q121).
        if by == "lines" || by == "sentences" || by == "headings" {
            return Asset(items: units.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
        }
        // tokens — sliding window by chunk_size (default 200) with overlap. A non-numeric
        // chunk_size/overlap/bare token raises, matching the Python's ValueError (M6) —
        // never a silent fallback to 200/0. The shorthand `Split 500` reads the first bare
        // token, exactly as `s.get("chunk_size") or s.first_bare()` (CFM-FIX-2).
        let rawSize = s.value(for: "chunk_size") ?? s.firstBare()
        let chunkSize: Int
        if let rawSize {
            guard let v = Int(rawSize), v >= 1 else {
                throw FlowError.invalidSettings(row: "Split", setting: "chunk_size",
                                                detail: "expected a positive number")
            }
            chunkSize = v
        } else {
            chunkSize = 200
        }
        let overlap: Int
        if let rawOverlap = s.value(for: "overlap") {
            guard let v = Int(rawOverlap), v >= 0 else {
                throw FlowError.invalidSettings(row: "Split", setting: "overlap",
                                                detail: "expected a non-negative number")
            }
            overlap = v
        } else {
            overlap = 0
        }
        let stride = max(chunkSize - overlap, 1)
        var chunks: [String] = []
        var i = 0
        while i < units.count {
            let window = Array(units[i..<min(i + chunkSize, units.count)])
            chunks.append(joinUnits(window, by: by))
            if i + chunkSize >= units.count { break }
            i += stride
        }
        return Asset(items: chunks.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }

    private func splitUnits(_ text: String, by: String) -> [String] {
        switch by {
        case "lines":
            // Python: `[line for line in text.split("\n") if line.strip()]` — scalar-level
            // split keeps `\r` attached and interior blanks; only empty-after-trim lines are
            // dropped (CFM-FIX-3/M8). CRLF files split exactly as in Python.
            return pythonSplitLines(text).filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        case "sentences":
            // Python: re.split(r"(?<=[.!?])\s+", text.strip()) — splits on the *whitespace
            // after* terminal punctuation and keeps the punctuation (H4). Splitting on the
            // punctuation itself eats "Dr." → "Dr" and splits "3.14".
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let regex = NSRegularExpression.compiled(#"(?<=[.!?])\s+"#)
            let ns = NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)
            var pieces: [String] = []
            var last = trimmed.startIndex
            for match in regex.matches(in: trimmed, range: ns) {
                guard let range = Range(match.range, in: trimmed) else { continue }
                pieces.append(String(trimmed[last..<range.lowerBound]))
                last = range.upperBound
            }
            if last < trimmed.endIndex {
                pieces.append(String(trimmed[last..<trimmed.endIndex]))
            }
            return pieces.filter { !$0.isEmpty }
        case "headings":
            return splitHeadings(text)
        default:
            return text.split(whereSeparator: \.isWhitespace).map(String.init)
        }
    }

    private func splitHeadings(_ text: String) -> [String] {
        // Port of `_split_headings`: scalar split("\n") keeps interior blank lines (M8);
        // each section is joined with "\n" and `.strip()`ped, then empty sections dropped.
        let lines = pythonSplitLines(text)
        var sections: [String] = []
        var current: [String] = []
        for line in lines {
            if line.hasPrefix("#"), !current.isEmpty {
                sections.append(current.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines))
                current = [line]
            } else {
                current.append(line)
            }
        }
        if !current.isEmpty {
            sections.append(current.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return sections.filter { !$0.isEmpty }
    }

    private func joinUnits(_ units: [String], by: String) -> String {
        if by == "lines" { return units.joined(separator: "\n") }
        return units.joined(separator: " ")
    }
}

// MARK: - Filter

nonisolated struct FilterTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .listOf(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let texts = TextTools.flatTexts([input])
        let s = FlowSettings(settings)
        let contains = s.value(for: "contains")
        // Compile the regex up front: a malformed pattern raises (Python `re.error`) rather
        // than silently matching nothing (M5).
        let regex: NSRegularExpression?
        if let pattern = s.value(for: "regex") {
            do {
                regex = try NSRegularExpression(pattern: pattern)
            } catch {
                throw FlowError.invalidSettings(row: "Filter", setting: "regex",
                                                detail: "'\(pattern)' isn't a valid regular expression")
            }
        } else {
            regex = nil
        }
        let lengthMatch = parseLength(settings)

        func keep(_ text: String) -> Bool {
            if let contains, !text.contains(contains) { return false }
            if let regex, regex.firstMatch(
                in: text, range: NSRange(text.startIndex..<text.endIndex, in: text)) == nil {
                return false
            }
            if let lengthMatch {
                let (op, n) = lengthMatch
                // Python `len` counts code points, not grapheme clusters (M4).
                if !applyLength(op, length: text.unicodeScalars.count, n: n) { return false }
            }
            return true
        }
        return Asset(items: texts.filter(keep).map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }

    private func parseLength(_ settings: String) -> (String, Int)? {
        // length (>=|<=|>|<|=)? N
        let pattern = NSRegularExpression.compiled(#"length\s*(>=|<=|>|<|=)?\s*(\d+)"#)
        guard let match = pattern.firstMatch(in: settings, range: NSRange(settings.startIndex..<settings.endIndex, in: settings)),
              match.numberOfRanges >= 3,
              let numRange = Range(match.range(at: 2), in: settings),
              let n = Int(settings[numRange]) else { return nil }
        let opRange = match.range(at: 1)
        let op = opRange.location == NSNotFound ? "=" : String(settings[Range(opRange, in: settings)!])
        return (op, n)
    }

    private func applyLength(_ op: String, length: Int, n: Int) -> Bool {
        switch op {
        case ">":  return length > n
        case ">=": return length >= n
        case "<":  return length < n
        case "<=": return length <= n
        default:   return length == n
        }
    }
}

// MARK: - Dedupe

nonisolated struct DedupeTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .listOf(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let texts = TextTools.flatTexts([input])
        let s = FlowSettings(settings)
        let by = s.value(for: "by", default: "exact") ?? "exact"
        let direction = s.value(for: "direction", default: "first") ?? "first"

        func key(_ t: String) -> String {
            by == "normalized" ? t.casefoldWords : t
        }
        // Swift `Dictionary` has no insertion order, so both directions build an explicit
        // `order` array and map through it (the Python's `dict.values()` preserves insertion
        // order; the Swift port must not rely on dictionary ordering).
        var firstValue: [String: String] = [:]
        var order: [String] = []
        for t in texts {
            let k = key(t)
            if firstValue[k] == nil {
                firstValue[k] = t
                order.append(k)
            }
        }
        var result: [String]
        if direction == "last" {
            // Python: keep the LAST occurrence, ordered by that occurrence's index.
            var lastValue: [String: String] = [:]
            var lastIndex: [String: Int] = [:]
            for (i, t) in texts.enumerated() {
                let k = key(t)
                lastValue[k] = t
                lastIndex[k] = i
            }
            result = lastIndex.keys.sorted { lastIndex[$0]! < lastIndex[$1]! }.compactMap { lastValue[$0] }
        } else {
            result = order.compactMap { firstValue[$0] }
        }
        return Asset(items: result.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }
}

private nonisolated extension String {
    /// A practical `str.casefold()` — full Unicode case folding isn't in Foundation, so this
    /// covers the folds that matter in practice (sharp s, the common ligatures, Turkish
    /// dotted capital İ, final sigma, Kelvin/Angstrom signs) then lowercases (CFM-FIX-3/M3).
    /// The residual gap (obscure folding-only mappings) is documented in the FIX-3 journal.
    var pythonCasefold: String {
        var s = self
        let folds: [(String, String)] = [
            ("ẞ", "ss"), ("ß", "ss"),
            ("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl"),
            ("ﬅ", "st"), ("ﬆ", "st"),
            ("İ", "i\u{0307}"),
            ("ς", "σ"),
            ("\u{212A}", "k"),   // KELVIN SIGN
            ("\u{212B}", "å"),   // ANGSTROM SIGN
        ]
        for (from, to) in folds {
            s = s.replacingOccurrences(of: from, with: to)
        }
        return s.lowercased()
    }

    /// `" ".join(text.casefold().split())` — the Python's normalized dedupe key.
    var casefoldWords: String {
        pythonCasefold.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}

// MARK: - Sort (CFM-FIX-3/M9)

nonisolated struct SortTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .listOf(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let texts = TextTools.flatTexts([input])
        let s = FlowSettings(settings)
        let by = s.value(for: "by", default: "alpha") ?? "alpha"
        let direction = s.value(for: "direction", default: "asc") ?? "asc"
        let desc = direction == "desc"
        let sorted: [String]
        if by == "length" {
            // Python `len` counts code points (CFM-FIX-3/M9).
            sorted = texts.sorted { desc ? $0.unicodeScalars.count > $1.unicodeScalars.count
                                            : $0.unicodeScalars.count < $1.unicodeScalars.count }
        } else {
            sorted = texts.sorted { desc ? $0 > $1 : $0 < $1 }
        }
        return Asset(items: sorted.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }
}

// MARK: - Extract (CFM-FIX-3/M9)

/// `tools/text.py::extract` — the three presets plus a bare/`regex=` pattern. `re.findall`
/// semantics: with a capture group the group's content is what's returned, else the match.
nonisolated struct ExtractTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .single(.text) }
    var produces: Shape { .listOf(.text) }

    private static let presets: [String: String] = [
        "emails": #"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"#,
        "urls": #"https?://[^\s)>\]]+"#,
        "dates": #"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"#,
    ]

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let text = TextTools.singleText([input])
        let s = FlowSettings(settings)
        let raw = settings.trimmingCharacters(in: .whitespacesAndNewlines)
        let bare = raw.isEmpty ? "" : TextTools.unquoteWhole(raw)
        var pattern = s.value(for: "regex")
        if pattern == nil, let preset = Self.presets[bare.lowercased()] {
            pattern = preset
        } else if pattern == nil, !bare.isEmpty {
            pattern = bare
        }
        guard let pattern, !pattern.isEmpty else {
            return Asset(items: [])
        }
        let regex = try NSRegularExpression(pattern: pattern)   // malformed raises, like `re.error`
        let hasGroup = regex.numberOfCaptureGroups > 0
        let ns = NSRange(text.startIndex..<text.endIndex, in: text)
        var results: [String] = []
        for match in regex.matches(in: text, range: ns) {
            if hasGroup {
                let group = match.range(at: 1)
                if group.location != NSNotFound, let range = Range(group, in: text) {
                    results.append(String(text[range]))
                }
            } else if let range = Range(match.range, in: text) {
                results.append(String(text[range]))
            }
        }
        return Asset(items: results.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }
}

// MARK: - Count

nonisolated struct CountTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let texts = TextTools.flatTexts([input])
        // Python: `mode = s.first_bare() or "items"` — NOT `path=`. `Count path=x` counts
        // *items* in Python, so the Swift must not pick up the path as the mode (CFM-FIX-2).
        let mode = FlowSettings(settings).firstBare() ?? "items"
        let n = mode == "items" ? texts.count : texts.reduce(0) { $0 + $1.split(whereSeparator: \.isWhitespace).count }
        return Asset(items: [Item(kind: .text, value: String(n), path: nil, sourceText: nil)])
    }
}

// MARK: - Join Text

nonisolated struct JoinTextTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let texts = TextTools.flatTexts([input])
        let s = FlowSettings(settings)
        let sep: String
        if let value = s.value(for: "separator") {
            sep = value
        } else if !settings.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            sep = TextTools.unquoteWhole(settings)
        } else {
            sep = "\n"
        }
        return Asset(items: [Item(kind: .text, value: texts.joined(separator: sep), path: nil, sourceText: nil)])
    }
}

// MARK: - Template

nonisolated struct TemplateTool: AssetStage {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }
    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let pattern = TextTools.unquoteWhole(settings)
        let texts = TextTools.flatTexts([input])
        let joined = texts.joined(separator: "\n")

        let regex = NSRegularExpression.compiled(#"\{((input(?::(\d+))?)|\d+)\}"#)
        let nsRange = NSRange(pattern.startIndex..<pattern.endIndex, in: pattern)
        let matches = regex.matches(in: pattern, range: nsRange)
        var out = pattern
        for match in matches.reversed() {
            let token = String(out[Range(match.range, in: out)!])
            let inner = token.dropFirst().dropLast()   // strip { }
            var idx: Int?
            if inner.hasPrefix("input") {
                let sub = inner.dropFirst("input".count)
                if sub.hasPrefix(":") { idx = Int(sub.dropFirst()) }
            } else {
                idx = Int(inner)
            }
            let value: String
            if let idx {
                let i = idx - 1
                value = (i >= 0 && i < texts.count) ? texts[i] : ""
            } else {
                value = joined
            }
            out.replaceSubrange(Range(match.range, in: out)!, with: value)
        }
        return Asset(items: [Item(kind: .text, value: out, path: nil, sourceText: nil)])
    }
}

// MARK: - Diff

/// The full set of Unicode line boundaries `str.splitlines()` recognizes (beyond `\n`/`\r`).
private nonisolated let pythonLineBreakScalars: Set<Unicode.Scalar> = [
    "\u{0B}", "\u{0C}", "\u{1C}", "\u{1D}", "\u{1E}", "\u{85}", "\u{2028}", "\u{2029}",
]

extension String {
    /// `str.splitlines()` — split on universal line boundaries, drop the terminators, keep
    /// interior empty lines, and **don't** emit a phantom trailing empty element. The old
    /// `split(separator: "\n")` left a spurious `- ` delete on any file ending in `\n`
    /// (CFM-FIX-1a).
    nonisolated var pythonSplitlines: [String] {
        var result: [String] = []
        var current = String.UnicodeScalarView()
        let scalars = Array(unicodeScalars)
        var i = 0
        while i < scalars.count {
            let c = scalars[i]
            if c == "\r" {
                result.append(String(current))
                current = String.UnicodeScalarView()
                if i + 1 < scalars.count && scalars[i + 1] == "\n" { i += 1 }
            } else if c == "\n" || pythonLineBreakScalars.contains(c) {
                result.append(String(current))
                current = String.UnicodeScalarView()
            } else {
                current.append(c)
            }
            i += 1
        }
        if !current.isEmpty {
            result.append(String(current))
        }
        return result
    }
}

/// `str.split("\n")` — scalar-level split that keeps empty subsequences and leaves a `\r`
/// attached (Python semantics), so CRLF files split into two lines exactly as in Python
/// (CFM-FIX-3/M8).
nonisolated func pythonSplitLines(_ text: String) -> [String] {
    var result: [String] = []
    var current = String.UnicodeScalarView()
    for scalar in text.unicodeScalars {
        if scalar == "\n" {
            result.append(String(current))
            current = String.UnicodeScalarView()
        } else {
            current.append(scalar)
        }
    }
    result.append(String(current))
    return result
}

/// The `Diff` instant tool (CFM-FIX-1): reads `inputs[0].items[0]` / `inputs[1].items[0]`
/// exactly as the Python's `diff` does — **not** the flattened bundle — and falls back to
/// `b = ""` for a single-ref row instead of throwing. Output is `difflib.ndiff` or
/// `difflib.unified_diff` (lineterm="") via the `SwiftDifflib` port, byte-pinned by
/// `Fixtures/CatFlow/tools/diff_golden.json`.
nonisolated struct DiffTool {
    let settings: String

    init(settings: String?) {
        self.settings = settings ?? ""
    }

    func run(inputs: [Asset]) async throws -> Asset {
        let a = inputs.first?.items.first?.value ?? ""
        let b = inputs.count > 1 ? (inputs[1].items.first?.value ?? "") : ""
        let fmt = FlowSettings(settings).value(for: "format", default: "inline") ?? "inline"
        let aLines = a.pythonSplitlines
        let bLines = b.pythonSplitlines
        let out: String
        if fmt == "unified" {
            out = SwiftDifflib.unifiedDiff(aLines, bLines, lineterm: "").joined(separator: "\n")
        } else {
            out = SwiftDifflib.ndiff(aLines, bLines).joined(separator: "\n")
        }
        return Asset(items: [Item(kind: .text, value: out, path: nil, sourceText: nil)])
    }
}
