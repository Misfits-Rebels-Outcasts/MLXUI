import Foundation

/// CFM-R9-5 — the decider frame renderers, ported from `catflow-mlx/src/catflow/engines/llm.py`
/// (`render_frame` / `render_judge_frame` / `render_think_frame` / `render_think_tools`) and
/// `core/transcript.py::render_transcript`, plus the F004 strict-parse tag extraction from
/// `engines/provider.py::_find_tag`.
///
/// **Tag-constrained decoding divergence (SPEC-Q):** the Python's local path constrains the
/// MLX decode with a token-prefix trie logits mask (`_TagConstraint`), which `mlx-swift-lm`'s
/// `GenerateParameters` has no hook for. Swift runs the rendered frame and extracts the tag
/// with the F004 strict-parse + one-retry path instead — the same mechanism the Python uses
/// for providers (`constrained_decoding: false`), which raises rather than guesses. The fired
/// tag matches for a well-behaved model; the mechanism diverges and is logged.
nonisolated enum DeciderFrame {

    /// `render_frame` — the shared renderer for Gate/Classify/Score and every framed Generator.
    static func renderFrame(frameText: String, settings: String?, inputs: [Asset],
                            tags: [String]?, context: String? = nil) -> String {
        var frame = spliceContext(frameText, context)
        let texts = flatTexts(inputs)
        let settingsView = FlowSettings(settings)

        // `{settings.KEY}` substitutions.
        frame = settingKeyRegex.replace(in: frame) { key in
            settingsView.value(for: key) ?? ""
        }
        // `{input[N]}` (0-indexed).
        frame = inputIndexRegex.replace(in: frame) { index -> String in
            guard let i = Int(index), i < texts.count else { return "" }
            return texts[i]
        }
        frame = frame.replacingOccurrences(of: "{asset list}", with: numberedList(texts))
        frame = frame.replacingOccurrences(of: "{asset}", with: texts.joined(separator: "\n"))
        frame = frame.replacingOccurrences(of: "{tags}", with: tags?.joined(separator: ", ") ?? "")
        frame = frame.replacingOccurrences(of: "{settings}", with: settings ?? "")
        return frame
    }

    /// `render_judge_frame` — candidates labeled by their tag, in declared order.
    static func renderJudgeFrame(frameText: String, settings: String?, inputs: [Asset],
                                 tags: [String], context: String? = nil) -> String {
        var frame = spliceContext(frameText, context)
        let (source, candidates) = splitJudgeBundle(inputs, tags: tags)
        var lines: [String] = []
        if let source {
            lines.append("Source:")
            lines.append(flatTexts([source]).joined(separator: "\n"))
            lines.append("")
        }
        lines.append("Candidates:")
        for (tag, candidate) in zip(tags, candidates) {
            lines.append("\(tag): \(flatTexts([candidate]).joined(separator: "\n"))")
        }
        frame = frame.replacingOccurrences(of: "{candidates}", with: lines.joined(separator: "\n"))
        frame = frame.replacingOccurrences(of: "{tags}", with: tags.joined(separator: ", "))
        frame = frame.replacingOccurrences(of: "{settings}", with: settings ?? "")
        return frame
    }

    /// `_split_judge_bundle` — `len(inputs)==len(tags)` → no source; `==len(tags)+1` → inputs[0]
    /// is the source, the rest are candidates. Anything else raises.
    static func splitJudgeBundle(_ inputs: [Asset], tags: [String]) -> (source: Asset?, candidates: [Asset]) {
        if inputs.count == tags.count {
            return (nil, inputs)
        }
        if inputs.count == tags.count + 1 {
            return (inputs[0], Array(inputs.dropFirst()))
        }
        return (nil, inputs)
    }

    /// `render_think_frame` — the agent loop's frame (`{asset}`, `{tools}`, `{transcript}`,
    /// `{tags}` pipe-joined, `{settings}`).
    static func renderThinkFrame(frameText: String, settings: String?, inputs: [Asset],
                                 tags: [String], tools: [String], transcript: String,
                                 context: String? = nil) -> String {
        var frame = spliceContext(frameText, context)
        let texts = flatTexts(inputs)
        frame = frame.replacingOccurrences(of: "{asset}", with: texts.joined(separator: "\n"))
        frame = frame.replacingOccurrences(of: "{tools}", with: tools.joined(separator: "\n"))
        frame = frame.replacingOccurrences(of: "{transcript}", with: transcript)
        frame = frame.replacingOccurrences(of: "{tags}", with: tags.joined(separator: " | "))
        frame = frame.replacingOccurrences(of: "{settings}", with: settings ?? "")
        return frame
    }

    /// `render_think_tools` — one line per decide edge ("- {tag}: calls row N." /
    /// "finish and give your answer.").
    static func renderThinkTools(edges: [ClauseEdge]) -> [String] {
        edges.map { edge in
            switch edge.target {
            case .call(let n):
                return "- \(edge.tag): calls row \(n)."
            default:
                return "- \(edge.tag): finish and give your answer."
            }
        }
    }

    /// `render_transcript` — the Think "So far" section.
    static func renderTranscript(entries: [FlowInterpreter.TranscriptEntry]) -> String {
        if entries.isEmpty { return "(nothing yet)" }
        return entries.enumerated().map { i, e in
            "[\(i + 1)] \(e.tool)(\"\(e.input)\")  -> \"\(e.observation)\""
        }.joined(separator: "\n")
    }

    /// `_splice_context` — a `· ctx` journal inserted ahead of the first input anchor.
    static func spliceContext(_ frame: String, _ context: String?) -> String {
        guard let context else { return frame }
        for anchor in ["{asset list}", "{asset}", "{candidates}", "{input[", "{transcript}"] {
            if let range = frame.range(of: anchor) {
                var out = frame
                out.insert(contentsOf: "Shared context so far:\n\(context)\n\n", at: range.lowerBound)
                return out
            }
        }
        return frame
    }

    /// `_flat_texts` — every item's text, joined by newline.
    static func flatTexts(_ assets: [Asset]) -> [String] {
        assets.flatMap { asset in
            asset.items.map { $0.value ?? "<\($0.kind.rawValue)>" }
        }
    }

    /// `_numbered_list` — "1. …" lines.
    static func numberedList(_ texts: [String]) -> String {
        texts.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n")
    }

    /// `_find_tag` (F004 strict-parse): the first declared tag appearing in `text`,
    /// whole-word and case-insensitive; longest tag first so a substring tag can't shadow.
    static func extractTag(from text: String, tags: [String]) -> String? {
        let lowered = text.lowercased()
        for tag in tags.sorted(by: { $0.count > $1.count }) {
            let pattern = "(?<![A-Za-z0-9_])" + NSRegularExpression.escapedPattern(for: tag.lowercased()) + "(?![A-Za-z0-9_])"
            if lowered.range(of: pattern, options: .regularExpression) != nil {
                return tag
            }
        }
        return nil
    }

    /// `_SETTINGS_KEY_RE` — `{settings.KEY}`.
    private static let settingKeyRegex = NSRegularExpression.compiled(#"\{settings\.([A-Za-z_][A-Za-z0-9_]*)\}"#)
    /// `_INPUT_INDEX_RE` — `{input[N]}`.
    private static let inputIndexRegex = NSRegularExpression.compiled(#"\{input\[(\d+)\]\}"#)
}

private extension NSRegularExpression {
    nonisolated func replace(in string: String, replacement: (String) -> String) -> String {
        var output = string
        let matches = self.matches(in: string, range: NSRange(string.startIndex..<string.endIndex, in: string))
        for match in matches.reversed() {
            guard let fullRange = Range(match.range, in: output) else { continue }
            let group: String
            let g = match.range(at: 1)
            if g.location != NSNotFound, let gr = Range(g, in: output) {
                group = String(output[gr])
            } else {
                group = ""
            }
            output.replaceSubrange(fullRange, with: replacement(group))
        }
        return output
    }
}
