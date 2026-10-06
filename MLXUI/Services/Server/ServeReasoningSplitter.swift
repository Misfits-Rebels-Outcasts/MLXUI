import Foundation

/// Splits a model's raw output into its **thinking** and its **answer**, byte-exactly.
///
/// TCP-1 still applies (journal `2026-230`, readme T1): the server decodes raw token ids precisely
/// because the `.chunk` path loses text, so the split has to happen on text this code owns, and it
/// must never drop a byte of either half. Only `</think>` is ever held back, and only a genuine
/// trailing prefix of it.
///
/// **The opening is decided by the prompt, not by looking at the output** (S2-3b, review call 2).
/// The caller passes `startsInsideThink`, computed from the rendered prompt's decoded tail:
///
/// - `true` — *prompt-opened* (`qwen3_5*` templates end `<think>\n`): the output begins inside the
///   block, so it is thinking from the first character and only `</think>` has to be found;
/// - `false` — the prompt did not open a block (`qwen3` ends at the assistant header): a **literal
///   leading `<think>`** opens one, and any other opening is the whole answer. A model that answers
///   without thinking therefore produces content and no reasoning at all, instead of having its
///   first words mislabelled as thinking.
///
/// In both modes the leading tag is consumed rather than emitted, and `</think>` at the very first
/// position gives empty thinking plus the answer.
nonisolated struct ServeReasoningSplitter {
    private static let openTag = "<think>"
    private static let closeTag = "</think>"

    private enum Phase {
        /// Nothing has been decided about the output's opening yet.
        case atStart
        /// Inside the thinking block, looking for `</think>`.
        case reasoning
        /// `</think>` seen; the blank line the templates leave behind is still being trimmed.
        case trimming
        /// The answer.
        case answer
    }

    private var phase: Phase
    private var buffer = ""

    /// - Parameter startsInsideThink: whether the rendered prompt already wrote `<think>`. See the
    ///   type's documentation: it decides whether an output *without* a leading tag is thinking or
    ///   answer.
    init(startsInsideThink: Bool) {
        phase = startsInsideThink ? .reasoning : .atStart
    }

    /// Feed a text delta; returns the thinking and the answer to emit **now**. Either may be empty.
    mutating func push(_ text: String) -> (reasoning: String, content: String) {
        buffer += text
        return drain()
    }

    /// End of generation. Whatever is still held is thinking — an unclosed block runs to EOS, which
    /// is exactly what a `max_tokens` cut mid-think looks like (S2-3 decision 7): reasoning out,
    /// no content, `finish_reason: length`.
    mutating func finish() -> (reasoning: String, content: String) {
        defer { buffer = "" }
        switch phase {
        case .atStart:
            // Nothing but whitespace ever arrived: the prompt didn't open a block and no `<think>`
            // came, so there is neither thinking nor an answer to report.
            return ("", "")
        case .reasoning:
            return (buffer, "")
        case .trimming:
            return ("", "")
        case .answer:
            return ("", buffer)
        }
    }

    private mutating func drain() -> (reasoning: String, content: String) {
        var reasoning = ""
        var content = ""
        while true {
            switch phase {
            case .atStart:
                // The prompt did *not* open a block: only a literal leading `<think>` does, and
                // anything else is the answer from here on.
                let trimmed = buffer.drop(while: \.isWhitespace)
                if trimmed.isEmpty { return (reasoning, content) }        // whitespace so far: hold
                if trimmed.hasPrefix(Self.openTag) {
                    buffer = String(trimmed.dropFirst(Self.openTag.count))
                    phase = .reasoning
                    continue
                }
                if Self.openTag.hasPrefix(trimmed) { return (reasoning, content) }  // a partial `<thi`
                phase = .answer
                buffer = String(trimmed)
                continue

            case .reasoning:
                guard let end = buffer.range(of: Self.closeTag) else {
                    let (safe, held) = ServeTextFraming.holding(buffer, partialOf: Self.closeTag)
                    reasoning += safe
                    buffer = held
                    return (reasoning, content)
                }
                reasoning += String(buffer[buffer.startIndex ..< end.lowerBound])
                buffer = String(buffer[end.upperBound...])
                phase = .trimming
                continue

            case .trimming:
                // Only the newline(s) the template leaves after `</think>` are dropped, never
                // spaces: an answer that starts with indentation keeps it (S2-3b, review call 5).
                //
                // `"\r\n"` is listed as well as `"\r"`: in Swift a CRLF is **one** `Character`
                // (one grapheme cluster), so a test for the two separately misses it and the CRLF
                // would survive into the answer.
                let rest = String(buffer.drop(while: { $0 == "\n" || $0 == "\r" || $0 == "\r\n" }))
                if rest.isEmpty { buffer = ""; return (reasoning, content) }   // still newlines: hold nothing
                phase = .answer
                buffer = rest
                continue

            case .answer:
                content += buffer
                buffer = ""
                return (reasoning, content)
            }
        }
    }
}

/// The thinking path's text pipeline: detokenize → split thinking/answer → `stop`.
///
/// Shapes match `ServeTextPipeline` (S1) on purpose, so the two can be read side by side — including
/// its `if !stopped` guard: **once a stop sequence has matched, nothing is flushed**, because the
/// detokenizer may be holding the first half of a character that arrived *after* the match, and
/// text after a stop string is not this server's to send (S2-3b, review call 1).
nonisolated struct ServeThinkingTextPipeline {
    private var detokenizer: IncrementalDetokenizer
    private var splitter: ServeReasoningSplitter
    private var stop: StopMatcher
    private(set) var stopped = false

    init(decode: @escaping ([Int]) -> String, stop: [String], startsInsideThink: Bool) {
        self.detokenizer = IncrementalDetokenizer(decode: decode)
        self.splitter = ServeReasoningSplitter(startsInsideThink: startsInsideThink)
        self.stop = StopMatcher(stop)
    }

    /// One generated token: the thinking and the answer to emit now.
    mutating func push(_ id: Int) -> (reasoning: String, content: String) {
        guard !stopped else { return ("", "") }
        return apply(splitter.push(detokenizer.push(id)))
    }

    /// End of stream. Each stage is skipped once a stop has matched — the same three guards as S1's
    /// `ServeTextPipeline.finish()`.
    mutating func finish() -> (reasoning: String, content: String) {
        var reasoning = "", content = ""
        if !stopped {
            let halves = apply(splitter.push(detokenizer.flush()))
            reasoning += halves.reasoning
            content += halves.content
        }
        if !stopped {
            let halves = apply(splitter.finish())
            reasoning += halves.reasoning
            content += halves.content
        }
        if !stopped {
            content += stop.finish()          // the tail held back while a stop string was undecided
        }
        return (reasoning, content)
    }

    /// The answer half goes through `stop`; the thinking half never does (a stop string inside the
    /// thinking is not the client's stop).
    private mutating func apply(_ halves: (reasoning: String, content: String)) -> (reasoning: String, content: String) {
        guard !halves.content.isEmpty else { return halves }
        let matched = stop.push(halves.content)
        if matched.stopped { stopped = true }
        return (halves.reasoning, matched.emit)
    }
}
