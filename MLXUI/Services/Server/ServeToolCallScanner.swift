import Foundation
import MLXLMCommon

/// Pulls tool calls out of a model's **raw** text stream, without losing a byte of the text around
/// them.
///
/// ## Why this exists instead of `ToolCallProcessor`
///
/// TCP-1 (journal `2026-230`, readme T1): mlx-swift-lm's `ToolCallProcessor` holds back text from
/// the first `<` of a chunk (`processTaggedChunk` splits on the start tag's first character and
/// then decides), which is how `<div><span>` comes out as `<div<span>` on the `.chunk` path. S1
/// avoided the whole path by decoding **raw token ids** (`IncrementalDetokenizer`), and S2-2 keeps
/// doing that — so the tool-call parse has to happen here, on text this code owns.
///
/// The rule this scanner holds to: **text is only held once a complete start tag has been seen**
/// (plus, for an inline format, once a `{` has been seen). Anything that turns out not to be a
/// tool call is emitted again **verbatim** — never dropped. That is the property S1's
/// `<div><span>` test pins, and it is checked per format in `ServeToolCallScannerTests`.
///
/// The scanner does framing only; the semantics — which parameter is an int, what a call looks
/// like — stay in mlx-swift-lm's `ToolCallParser` for the family's `ToolCallFormat`, which is also
/// where the client's `tools` schemas make arguments typed.
nonisolated struct ServeToolCallScanner {
    private let parser: any ToolCallParser
    private let tools: [[String: any Sendable]]?
    /// Non-nil for the tagged formats (`<tool_call>`, `[TOOL_CALLS]`); nil for Llama 3, whose
    /// template renders a bare JSON object with no wrapper.
    private let startTag: String?
    private let endTag: String?
    /// Text held because it might still turn out to be a call.
    private var buffer = ""
    private var calls: [ToolCall] = []
    /// Inline (Llama 3) only: whether the turn's opening has been decided. Until it has, only the
    /// first non-whitespace text can be a call; from then on every character streams immediately.
    private var inlineDecided = false
    /// Inline only: the turn opened with `<|python_tag|>`, which is consumed and never emitted.
    private var inlineSawMarker = false

    /// Llama 3's template writes a tool call as the whole assistant turn, preceded by this special
    /// token. `Llama3ToolCallParser` strips it; the scanner must too, so it never reaches a client
    /// as content.
    private static let pythonTag = "<|python_tag|>"

    init(format: ToolCallFormat, tools: [[String: any Sendable]]?) {
        let parser = format.createParser()
        self.parser = parser
        self.tools = tools
        self.startTag = parser.startTag
        self.endTag = parser.endTag
    }

    /// Tool calls parsed so far, in order. Read them after every `push` (they are appended, never
    /// cleared — `takeCalls()` is what drains them).
    var parsedCalls: [ToolCall] { calls }

    /// Drain the calls parsed since the last drain.
    mutating func takeCalls() -> [ToolCall] {
        defer { calls = [] }
        return calls
    }

    /// Feed a text delta; returns the text that is safe to send on **now**. Text that may yet turn
    /// out to be part of a call is held and returned by a later `push` or by `finish()`.
    mutating func push(_ text: String) -> String {
        buffer += text
        return drain()
    }

    /// End of generation: whatever is still held is either a final call the stream ended inside
    /// (no end tag to close it) or plain text — parse it once, and hand back the text if it isn't
    /// a call. Never returns without emptying the buffer.
    mutating func finish() -> String {
        guard !buffer.isEmpty else { return "" }
        let held = buffer
        buffer = ""
        if let call = parser.parse(content: held, tools: tools) {
            calls.append(call)
            return ""
        }
        return held
    }

    // MARK: - Tagged formats

    private mutating func drain() -> String {
        guard let startTag else { return drainInline() }
        var out = ""
        while true {
            guard let start = buffer.range(of: startTag) else {
                return out + holdPartialTag()
            }
            guard let endTag else {
                // No end tag (`[TOOL_CALLS]`): the call runs up to the next start tag, or to EOS.
                if let next = buffer.range(of: startTag, range: start.upperBound ..< buffer.endIndex) {
                    out += buffer[buffer.startIndex ..< start.lowerBound]
                    out += emit(String(buffer[start.lowerBound ..< next.lowerBound]))
                    buffer = String(buffer[next.lowerBound...])
                    continue
                }
                out += buffer[buffer.startIndex ..< start.lowerBound]
                buffer = String(buffer[start.lowerBound...])
                return out
            }
            guard let end = buffer.range(of: endTag, range: start.upperBound ..< buffer.endIndex) else {
                // The call is still arriving; everything before its tag is safe.
                out += buffer[buffer.startIndex ..< start.lowerBound]
                buffer = String(buffer[start.lowerBound...])
                return out
            }
            out += buffer[buffer.startIndex ..< start.lowerBound]
            out += emit(String(buffer[start.lowerBound ..< end.upperBound]))
            buffer = String(buffer[end.upperBound...])
        }
    }

    /// A `{`-delimited format with no wrapper (Llama 3, whose template renders
    /// `{"name": …, "parameters": …}` as the whole assistant turn, optionally behind
    /// `<|python_tag|>`).
    ///
    /// **Only the opening of the turn can be a call** (review call 3): the first non-whitespace text
    /// of the assistant turn, or a `{` immediately after the `<|python_tag|>` marker. Anywhere else
    /// a `{` is prose and streams **immediately** — no brace counting, no delay, and a JSON example
    /// in an answer stays an answer. That matches the template, which puts the call where a
    /// generation prompt expects it, and the parser's own limitation (Llama 3.1 takes one call per
    /// turn).
    ///
    /// The marker is consumed rather than emitted, so it never reaches a client as content. Only
    /// the JSON form is detected: the parser's pythonic fallback has no delimiter to scan for.
    private mutating func drainInline() -> String {
        var out = ""
        while true {
            if inlineDecided {
                out += buffer
                buffer = ""
                return out
            }
            // Undecided: hold the whitespace too — a call-only turn's content is null, not spaces.
            let trimmed = buffer.drop(while: \.isWhitespace)
            let leading = String(buffer.prefix(buffer.count - trimmed.count))
            if trimmed.isEmpty { return out }                       // all whitespace so far: hold

            if !inlineSawMarker {
                if trimmed.hasPrefix(Self.pythonTag) {
                    buffer = String(trimmed.dropFirst(Self.pythonTag.count))
                    inlineSawMarker = true
                    continue
                }
                if Self.pythonTag.hasPrefix(trimmed) { return out }  // a partial marker: hold
            }

            guard trimmed.hasPrefix("{") else {
                // Ordinary text: it goes out, and nothing later in the turn is a candidate.
                inlineDecided = true
                out += buffer
                buffer = ""
                return out
            }
            let candidate = String(trimmed)
            guard let end = Self.endOfBalancedObject(candidate) else { return out }   // still arriving
            let segment = String(candidate[candidate.startIndex ..< end])
            buffer = String(candidate[end...])
            if let call = parser.parse(content: segment, tools: tools) {
                calls.append(call)
                inlineDecided = true                                // one call per turn is the format's rule
                continue                                            // `leading` whitespace goes with it
            }
            // Not a call — an example in prose. The whitespace before it is text again, and the
            // rest of the turn streams from here.
            inlineDecided = true
            out += leading + segment
        }
    }

    /// Parse one candidate segment: a call, or the segment handed straight back as text (S2-2
    /// requirement: a call the parser can't read is content, not an error).
    private mutating func emit(_ segment: String) -> String {
        guard let call = parser.parse(content: segment, tools: tools) else { return segment }
        calls.append(call)
        return ""
    }

    /// Hold back only a genuine prefix of the start tag at the very end of the buffer — for
    /// `<div><span>` that is the single `<` when a delta ends there, and nothing at all once the
    /// next character arrives. Everything else goes out now, byte for byte.
    ///
    /// Every path clears the buffer: this is the one function whose whole job is deciding what is
    /// still held, and returning without clearing would re-emit it on the next push.
    private mutating func holdPartialTag() -> String {
        guard let startTag else {
            let held = buffer
            buffer = ""
            return held
        }
        let (safe, held) = ServeTextFraming.holding(buffer, partialOf: startTag)
        buffer = held
        return safe
    }

    /// The index just past the `}` that closes the object starting at `buffer.first`, or `nil`
    /// while it is still open. Braces inside strings (and their escapes) don't count.
    static func endOfBalancedObject(_ text: String) -> String.Index? {
        var depth = 0
        var inString = false
        var escaped = false
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            if escaped {
                escaped = false
            } else if inString {
                if character == "\\" { escaped = true } else if character == "\"" { inString = false }
            } else if character == "\"" {
                inString = true
            } else if character == "{" {
                depth += 1
            } else if character == "}" {
                depth -= 1
                if depth == 0 { return text.index(after: index) }
            }
            index = text.index(after: index)
        }
        return nil
    }
}
