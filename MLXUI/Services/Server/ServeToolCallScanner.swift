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
        if startTag == nil {
            // S2-7: an inline turn's opening may be a *run* of calls, and only now is the run's end
            // known for certain (nothing more can arrive). Whatever is left after it is text.
            var out = drainInline(atEnd: true)
            if !buffer.isEmpty {
                out += buffer
                buffer = ""
            }
            return out
        }
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
        guard let startTag else { return drainInline(atEnd: false) }
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
    /// generation prompt expects it.
    ///
    /// **The opening may be a run of calls (S2-7).** Llama 3.1's template is written for one call
    /// per turn, but the models emit several — SV-6's Llama-3.1-8B turn produced a `write` object
    /// followed by a `glob` object, and the second one used to leak into `content` as text (the
    /// first parsed, then the "one call per turn" rule streamed everything after it). So the run is
    /// read as a whole:
    ///
    /// * objects separated by whitespace, `,` or `;`, and a `[ … ]` array of them, are all calls —
    ///   indices 0, 1, …;
    /// * if **any** object in the run fails to parse, the **whole run is text** — a client must
    ///   never be handed call #1 of a run whose #2 is prose;
    /// * the run ends at its last object: the separators after it stay in the buffer, so text after
    ///   the run streams exactly as it does today.
    ///
    /// A run is decided only when it cannot grow: at a non-`{` character after the separators, or at
    /// end of stream. That is the price of "no half-calls" — a call at the turn's opening is emitted
    /// once the run is settled rather than at its own closing brace.
    ///
    /// The marker is consumed rather than emitted, so it never reaches a client as content. Only
    /// the JSON form is detected: the parser's pythonic fallback has no delimiter to scan for.
    private mutating func drainInline(atEnd: Bool) -> String {
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

            guard trimmed.hasPrefix("{") || trimmed.hasPrefix("[") else {
                // Ordinary text: it goes out, and nothing later in the turn is a candidate.
                inlineDecided = true
                out += buffer
                buffer = ""
                return out
            }
            let candidate = String(trimmed)
            switch openingRun(candidate, atEnd: atEnd) {
            case .incomplete:
                return out                                          // still arriving: keep holding
            case .text:
                // Not a call run — prose, or a run with an unreadable object. All of it is text.
                inlineDecided = true
                out += leading
                buffer = candidate
            case .calls(let parsed, let end):
                calls.append(contentsOf: parsed)
                inlineDecided = true                                // one *run* per turn is the rule
                buffer = String(candidate[end...])
            }
            // `leading` whitespace goes with a call run, and is text otherwise.
        }
    }

    /// What the turn's opening turned out to be (S2-7).
    private enum OpeningRun {
        /// Calls, and the offset just past the last object (the separators after it stay in the buffer).
        case calls([ToolCall], end: String.Index)
        /// No calls here: prose, or a run with at least one unreadable object.
        case text
        /// The run may still grow — a partial object, or a separator that another object may follow.
        case incomplete
    }

    /// Read `candidate` as a run of tool calls (`{…}`, `{…} {…}`, `{…},{…}`, `{…};{…}`, or `[ … ]`).
    ///
    /// The separators and the array form are the owner's S2-7 call, and what they rest on is JSON
    /// itself (RFC 8259: whitespace between values is insignificant, `,` separates array elements)
    /// plus the JSON-lines style models emit with `;`. The bracketed form needs no invention at all
    /// — upstream's `parseEOS` already decodes `[LlamaFunction]` for this format, so it is the
    /// format's own sanctioned multi-call shape.
    private func openingRun(_ candidate: String, atEnd: Bool) -> OpeningRun {
        if candidate.hasPrefix("[") {
            guard let length = Self.balancedLength(candidate, open: "[", close: "]") else { return .incomplete }
            let end = candidate.index(candidate.startIndex, offsetBy: length)
            let parsed = parser.parseEOS(String(candidate[candidate.startIndex ..< end]), tools: tools)
            return parsed.isEmpty ? .text : .calls(parsed, end: end)
        }
        var cursor = candidate.startIndex
        var parsed: [ToolCall] = []
        var lastEnd = cursor
        while cursor < candidate.endIndex, candidate[cursor] == "{" {
            guard let length = Self.balancedLength(String(candidate[cursor...]), open: "{", close: "}") else {
                return .incomplete
            }
            let end = candidate.index(cursor, offsetBy: length)
            guard let call = parser.parse(content: String(candidate[cursor ..< end]), tools: tools) else {
                return .text                                        // whole run is text, no half-call
            }
            parsed.append(call)
            lastEnd = end
            var next = end
            while next < candidate.endIndex, Self.isRunSeparator(candidate[next]) {
                next = candidate.index(after: next)
            }
            if next < candidate.endIndex, candidate[next] == "{" {
                cursor = next                                       // another object: keep reading
                continue
            }
            if next == candidate.endIndex, !atEnd { return .incomplete }  // it may still grow
            break
        }
        return parsed.isEmpty ? .text : .calls(parsed, end: lastEnd)
    }

    /// What may sit between two objects of one run: whitespace, `,` (JSON's own separator) or `;`
    /// (what models that emit JSON lines use). A separator inside a string is invisible here — it is
    /// inside a balanced object.
    private static func isRunSeparator(_ character: Character) -> Bool {
        character.isWhitespace || character == "," || character == ";"
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
        balancedLength(text, open: "{", close: "}").map { text.index(text.startIndex, offsetBy: $0) }
    }

    /// How many characters the balanced `open`…`close` value at the start of `text` occupies, or
    /// `nil` while it is still open. Delimiters inside strings (and their escapes) don't count.
    /// Used for the run's objects (`{…}`) and its array form (`[…]`, S2-7).
    static func balancedLength(_ text: String, open: Character, close: Character) -> Int? {
        var depth = 0
        var inString = false
        var escaped = false
        var count = 0
        for character in text {
            count += 1
            if escaped {
                escaped = false
            } else if inString {
                if character == "\\" { escaped = true } else if character == "\"" { inString = false }
            } else if character == "\"" {
                inString = true
            } else if character == open {
                depth += 1
            } else if character == close {
                depth -= 1
                if depth == 0 { return count }
            }
        }
        return nil
    }
}
