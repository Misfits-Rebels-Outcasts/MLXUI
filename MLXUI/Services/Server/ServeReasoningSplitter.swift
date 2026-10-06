import Foundation

/// Splits a model's raw output into its **thinking** and its **answer**, byte-exactly.
///
/// TCP-1 still applies (journal `2026-230`, readme T1): the server decodes raw token ids precisely
/// because the `.chunk` path loses text, so the split has to happen on text this code owns, and it
/// must never drop a byte of either half. Only `</think>` is ever held back, and only a genuine
/// trailing prefix of it.
///
/// **Both generation-prompt shapes are handled without being told which one is in play** (S2-3
/// decision 3):
///
/// - *prompt-opened* (`qwen3_5*`: the prompt ends `<think>\n`, so the output starts inside the
///   block) — the output's first characters are thinking, and `</think>` ends it;
/// - *model-emitted* (`qwen3`: the prompt ends at the assistant header) — the output opens with
///   `<think>` itself, which is consumed here and never emitted as either half.
///
/// The splitter decides by looking: while the opening is undecided, leading whitespace and a
/// partial `<think>` are held; a full `<think>` is swallowed; anything else means the block was
/// already open. `</think>` at the very first position is therefore handled too (empty thinking).
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

    private var phase = Phase.atStart
    private var buffer = ""

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
            // A turn that only ever emitted whitespace is neither half.
            let trimmed = buffer.drop(while: \.isWhitespace)
            return (trimmed.isEmpty ? "" : buffer, "")
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
                let trimmed = buffer.drop(while: \.isWhitespace)
                if trimmed.isEmpty { return (reasoning, content) }        // whitespace so far: hold
                if trimmed.hasPrefix(Self.openTag) {
                    buffer = String(trimmed.dropFirst(Self.openTag.count))
                    phase = .reasoning
                    continue
                }
                if Self.openTag.hasPrefix(trimmed) { return (reasoning, content) }  // a partial `<thi`
                phase = .reasoning                                        // the block was already open
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
                let rest = String(buffer.drop(while: \.isWhitespace))
                if rest.isEmpty { buffer = ""; return (reasoning, content) }   // still whitespace: hold nothing
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
