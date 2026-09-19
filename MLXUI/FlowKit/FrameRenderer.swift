import Foundation

/// Renders a frame text (`frames/*.frame.txt`) into the prompt an LLM row sends — the
/// Swift port of `catflow-mlx/src/catflow/engines/llm.py::render_frame`. The whitespace and
/// newline handling are part of the prompt and are what make outputs match the Python; do
/// not "clean up" the substitution order. See `RSI/DelegateMergeBacklog.md` CFM-R2-5.
nonisolated enum FrameRenderer {

    /// Load `frames/<name>.frame.txt` from the app bundle, verbatim (no rendering).
    /// Mirrors `engines/llm.py::load_frame`. The synchronized root group flattens
    /// `CatFlow/frames/` into `Contents/Resources/`, so the lookup is by flat name.
    static func loadFrame(named name: String, bundle: Bundle = .main) throws -> String {
        guard let url = bundle.url(forResource: name, withExtension: "frame.txt"),
              let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw FrameError.missingFrame(name: name)
        }
        return text
    }

    /// The flat, ordered text of every input item, in reference order — the `{asset}`
    /// body and `{asset list}`'s source. Inline text first; a file-backed item with a
    /// `sourceText` (a vector computed *from* text) contributes that text; anything else
    /// is a refusal, matching `_flat_texts`'s "framed rows only accept text input".
    static func flatTexts(_ asset: Asset) throws -> [String] {
        var texts: [String] = []
        for item in asset.items {
            if let value = item.value {
                texts.append(value)
            } else if let source = item.sourceText {
                texts.append(source)
            } else {
                throw FlowError.unsupportedKind(row: "frame", kind: item.kind)
            }
        }
        return texts
    }

    /// `_numbered_list` — `1. …`, `2. …`, newline-joined (Merge's `{asset list}`).
    static func numberedList(_ texts: [String]) -> String {
        texts.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n")
    }

    /// Port of `render_frame`: substitutes `{settings.KEY}`, `{input[N]}`, `{asset list}`,
    /// `{asset}`, `{tags}`, `{settings}` — in exactly the Python's order, so a frame that
    /// nests placeholders renders identically. `assets` is the full reference bundle in
    /// order: the Python's `render_frame` takes `inputs: list[Asset]` and flattens *across*
    /// all of them, so `(1,2)` rows like `Rewrite (1,2)` must see ref 1 and ref 2, not just
    /// the first input (B1). (`{candidates}` / `{tools}` / `{transcript}` belong to
    /// decider/agent renderers, which canRun refuses in R2.)
    static func render(
        frameText: String,
        settings: String?,
        assets: [Asset],
        tags: [String]? = nil
    ) throws -> String {
        let texts = try assets.flatMap { try flatTexts($0) }
        let s = FlowSettings(settings)

        // {settings.KEY} — first, before any other substitution (the Python does this via
        // _SETTINGS_KEY_RE, substituting the setting value or "" when absent).
        var rendered = frameText
        let settingsKey = try NSRegularExpression(pattern: #"\{settings\.([A-Za-z_][A-Za-z0-9_]*)\}"#)
        rendered = try settingsKey.substitute(in: rendered) { key in
            s.value(for: key) ?? ""
        }

        // {input[N]} — N is a 0-based index into the flat text list in the Python.
        let inputRegex = try NSRegularExpression(pattern: #"\{input\[(\d+)\]\}"#)
        rendered = try inputRegex.substitute(in: rendered) { numString in
            guard let idx = Int(numString) else { return "" }
            if idx >= texts.count {
                throw FlowError.frameInputOutOfRange(index: idx, count: texts.count)
            }
            return texts[idx]
        }

        rendered = rendered.replacingOccurrences(of: "{asset list}", with: numberedList(texts))
        rendered = rendered.replacingOccurrences(of: "{asset}", with: texts.joined(separator: "\n"))
        rendered = rendered.replacingOccurrences(of: "{tags}",
                                                 with: tags?.joined(separator: ", ") ?? "")
        rendered = rendered.replacingOccurrences(of: "{settings}", with: settings ?? "")
        return rendered
    }
}

/// Frame-loading failures. Error voice: one plain sentence implying the fix.
nonisolated enum FrameError: Error, CustomStringConvertible, Equatable {
    case missingFrame(name: String)
    /// FV-4-2: `task` names no `refKind == .frame` catalog entry at all — a task that
    /// legitimately has no published frame, not a file this app failed to install. Distinct
    /// from `missingFrame`, whose "reinstall to restore it" is the wrong sentence here: there
    /// is nothing to reinstall, because there was never a frame to begin with.
    case noPublishedFrame(task: String)

    var description: String {
        switch self {
        case .missingFrame(let name):
            return "The prompt frame '\(name)' isn't in the app — reinstall to restore it."
        case .noPublishedFrame(let task):
            return "\(task) has no published frame — nothing to preview."
        }
    }
}

/// Small `NSRegularExpression`-driven substitution that returns the replacement for each
/// match's first capture group, throwing if the closure does.
private extension NSRegularExpression {
    /// Replace every match, passing its first capture group to `replacement`. Throws if
    /// the closure throws. Matches are replaced right-to-left so earlier ranges stay valid.
    nonisolated func substitute(in string: String, replacement: (String) throws -> String) throws -> String {
        var output = string
        let matches = self.matches(in: string, range: NSRange(string.startIndex..<string.endIndex, in: string))
        for match in matches.reversed() {
            let fullRange = Range(match.range, in: output)!
            let group = match.range(at: 1)
            let groupString = group.location == NSNotFound ? "" : String(output[Range(group, in: output)!])
            let newText = try replacement(groupString)
            output.replaceSubrange(fullRange, with: newText)
        }
        return output
    }
}
