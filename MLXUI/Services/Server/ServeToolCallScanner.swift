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
    /// `{"name": …, "parameters": …}` straight after the assistant header).
    ///
    /// Detection is by balanced braces, so a JSON object the model writes in prose is held until it
    /// closes and then — if the parser doesn't recognise it as a call — handed straight back. Only
    /// the JSON form is detected: the template we serve renders that one, and the parser's
    /// pythonic fallback has no delimiter to scan for.
    private mutating func drainInline() -> String {
        var out = ""
        while true {
            guard let brace = buffer.firstIndex(of: "{") else {
                out += buffer
                buffer = ""
                return out
            }
            out += buffer[buffer.startIndex ..< brace]
            buffer = String(buffer[brace...])
            guard let end = Self.endOfBalancedObject(buffer) else { return out }   // still arriving
            let segment = String(buffer[buffer.startIndex ..< end])
            buffer = String(buffer[end...])
            out += emit(segment)
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
    private mutating func holdPartialTag() -> String {
        guard let startTag, !buffer.isEmpty else {
            defer { buffer = "" }
            return buffer
        }
        let most = min(startTag.count - 1, buffer.count)
        guard most > 0 else { return buffer }
        for length in stride(from: most, through: 1, by: -1) {
            let suffix = buffer.suffix(length)
            guard suffix == startTag.prefix(length) else { continue }
            let safe = String(buffer.dropLast(length))
            buffer = String(suffix)
            return safe
        }
        defer { buffer = "" }
        return buffer
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
