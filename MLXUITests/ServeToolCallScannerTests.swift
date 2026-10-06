import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

// S2-2 — the streaming framer that pulls tool calls out of a model's raw text.
//
// The property under test throughout: **text is only held once a complete start tag has been
// seen**, and anything that turns out not to be a call comes back verbatim. That is what keeps
// TCP-1 (journal `2026-230`) fixed while tools are being detected.

private let weatherToolSchema: [String: any Sendable] = [
    "type": "function",
    "function": [
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": [
            "type": "object",
            "properties": [
                "location": ["type": "string"] as [String: any Sendable],
                "days": ["type": "integer"] as [String: any Sendable],
            ] as [String: any Sendable],
            "required": ["location"],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]

struct ServeToolCallScannerTests {

    /// Feed `chunks` to a fresh scanner and return everything it let through plus the calls it
    /// found. `finish()` is always called, because two of the catalog's formats have no end tag.
    private func run(_ format: ToolCallFormat, _ chunks: [String],
                     tools: [[String: any Sendable]]? = nil) -> (text: String, calls: [ToolCall]) {
        var scanner = ServeToolCallScanner(format: format, tools: tools)
        var text = ""
        for chunk in chunks { text += scanner.push(chunk) }
        text += scanner.finish()
        return (text, scanner.takeCalls())
    }

    // MARK: - TCP-1, per catalog format

    /// The S1 regression, now with tools in play: `<div><span>hi</span></div>` must come out byte
    /// for byte, for every format the catalog's tool-capable families use. One character at a time
    /// is the harshest framing there is.
    @Test func markupStreamsByteExactForEveryCatalogFormat() {
        let html = "<div><span>hi</span></div>"
        for format in [ToolCallFormat.json, .xmlFunction, .mistral, .llama3] {
            let characterWise = run(format, html.map(String.init))
            #expect(characterWise.text == html, "\(format) mangled the text one character at a time")
            #expect(characterWise.calls.isEmpty)

            // And in the chunk shapes a detokenizer actually produces.
            let chunked = run(format, ["<div>", "<span>", "hi", "</span>", "</div>"])
            #expect(chunked.text == html, "\(format) mangled the text in chunks")
            #expect(!chunked.text.contains("<div<span"))
        }
    }

    /// Text on both sides of a call survives, and only the call itself is removed.
    @Test func textAroundACallIsKeptExactly() {
        let call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>"
        let result = run(.json, ["Let me check. ", call, " Done."])
        #expect(result.text == "Let me check.  Done.")
        #expect(result.calls.count == 1)
        #expect(result.calls.first?.function.name == "get_weather")
    }

    /// A `<` that never becomes a start tag is not swallowed at end of stream.
    @Test func aLoneStartCharacterAtEndOfStreamComesBack() {
        for format in [ToolCallFormat.json, .xmlFunction, .mistral] {
            let result = run(format, ["cost < 5"])
            #expect(result.text == "cost < 5", "\(format) dropped a bare `<`")
        }
    }

    /// TCP-1's exact shape: text that arrives in the same chunk as a `<` which *might* begin a tag
    /// must survive while the tag is undecided. The mlx-swift-lm processor drops it here (it keeps
    /// `leadingToken` only on the path where the match fails immediately), so this is the test that
    /// would catch the bug being reintroduced in our framer.
    @Test func textBeforeAHeldTagIsNotLost() {
        let html = run(.json, ["Let me check the cost <", "div>ok</div>"])
        #expect(html.text == "Let me check the cost <div>ok</div>")

        let partialTag = run(.json, ["Answer: <tool", "_call is not a tag here"])
        #expect(partialTag.text == "Answer: <tool_call is not a tag here")
        #expect(partialTag.calls.isEmpty)

        let mistralPartial = run(.mistral, ["Answer: [TOOL", "_CALLS] is not a tag here"])
        #expect(mistralPartial.text == "Answer: [TOOL_CALLS] is not a tag here")
    }

    // MARK: - One call per family

    /// `.json` — the recorded Qwen3-8B-4bit output (see journal 2026-405), split mid-tag.
    @Test func jsonCallIsFoundAcrossChunkBoundaries() {
        let result = run(.json, ["<tool", "_call>\n{\"name\": \"get_weather\", \"arguments\": ",
                                 "{\"location\": \"Paris\"}}\n</tool_call>"])
        #expect(result.text.isEmpty)
        #expect(result.calls.count == 1)
        #expect(result.calls.first?.function.name == "get_weather")
        #expect(result.calls.first?.function.arguments["location"] == .string("Paris"))
    }

    /// `.xmlFunction` — the shape the Qwen3.5/Qwen3.8 template renders.
    @Test func xmlFunctionCallIsFound() {
        let output = "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>"
        let result = run(.xmlFunction, [output])
        #expect(result.text.isEmpty)
        #expect(result.calls.first?.function.name == "get_weather")
        #expect(result.calls.first?.function.arguments["location"] == .string("Paris"))
    }

    /// `.mistral` — no end tag: the call is closed by the next start tag or by end of stream.
    @Test func mistralCallIsClosedByEndOfStream() {
        let result = run(.mistral, ["[TOOL_CALLS]get_weather[ARGS]{\"location\": \"Paris\"}"])
        #expect(result.text.isEmpty)
        #expect(result.calls.count == 1)                       // only after `finish()`
        #expect(result.calls.first?.function.name == "get_weather")
    }

    /// `.llama3` — the template renders the call as the whole assistant turn, so the `{` only
    /// counts as a candidate at the turn's opening.
    @Test func llama3CallIsFoundFromBareJSON() {
        let output = "{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Paris\"}}"
        let result = run(.llama3, [output])
        #expect(result.text.isEmpty)
        #expect(result.calls.first?.function.name == "get_weather")
        #expect(result.calls.first?.function.arguments["location"] == .string("Paris"))
    }

    /// Leading whitespace before the call is not content: a calls-only turn reports none.
    @Test func llama3CallBehindLeadingWhitespaceIsStillTheTurn() {
        let result = run(.llama3, ["\n  {\"name\": \"get_weather\", \"parameters\": {}}"])
        #expect(result.text.isEmpty)
        #expect(result.calls.count == 1)
    }

    /// The `<|python_tag|>` prefix Llama 3 also emits: parsed, and **not** leaked as content.
    @Test func llama3PythonTagIsConsumedAndItsCallParsed() {
        let result = run(.llama3, ["<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Paris\"}}"])
        #expect(result.calls.count == 1)
        #expect(result.calls.first?.function.name == "get_weather")
        #expect(!result.text.contains("<|python_tag|>"))
        #expect(result.text.isEmpty)

        // Split across deltas, including mid-marker.
        let split = run(.llama3, ["<|python", "_tag|>{\"name\": \"f\", \"parameters\": {}}"])
        #expect(split.calls.count == 1)
        #expect(!split.text.contains("python"))
    }

    /// A marker with no call behind it is consumed, and the text that follows it is content.
    @Test func llama3PythonTagWithoutACallLeavesOnlyTheText() {
        let result = run(.llama3, ["<|python_tag|>I could not find a tool for that."])
        #expect(result.calls.isEmpty)
        #expect(result.text == "I could not find a tool for that.")
    }

    /// Braces inside strings don't end the object early.
    @Test func llama3BraceInsideAStringDoesNotCloseTheObject() {
        let output = #"{"name": "get_weather", "parameters": {"location": "Par}is"}}"#
        let result = run(.llama3, [output])
        #expect(result.calls.count == 1)
        #expect(result.calls.first?.function.arguments["location"] == .string("Par}is"))
    }

    /// **S2-2b, review call 3.** Away from the turn's opening a `{` is prose: it streams in the
    /// same push it arrives in — no brace counting, no delay — and an example that looks exactly
    /// like a call stays an example.
    @Test func aMidAnswerExampleStaysTextAndStreamsImmediately() {
        var scanner = ServeToolCallScanner(format: .llama3, tools: nil)
        var text = scanner.push("Here is an example: {")
        #expect(text == "Here is an example: {")                    // nothing held back
        text += scanner.push("\"a\": 1} and that is all.")
        text += scanner.finish()
        #expect(text == "Here is an example: {\"a\": 1} and that is all.")
        #expect(scanner.takeCalls().isEmpty)

        // An unmatched `{` mid-answer is not held either — it is out before the stream ends.
        var unmatched = ServeToolCallScanner(format: .llama3, tools: nil)
        #expect(unmatched.push("Use a JSON object like {") == "Use a JSON object like {")
        #expect(unmatched.finish().isEmpty)

        // A call-shaped object mid-answer is text, not a call.
        let example = run(.llama3, ["For instance {\"name\": \"f\", \"parameters\": {}} is the shape."])
        #expect(example.calls.isEmpty)
        #expect(example.text == "For instance {\"name\": \"f\", \"parameters\": {}} is the shape.")
    }


    // MARK: - Several calls in one turn

    @Test func twoTaggedCallsComeBackInOrder() {
        let result = run(.json, ["<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>",
                                 "<tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>"])
        #expect(result.calls.map(\.function.name) == ["a", "b"])
        #expect(result.text.isEmpty)
    }

    @Test func twoMistralCallsComeBackInOrder() {
        let result = run(.mistral, ["[TOOL_CALLS]a[ARGS]{}[TOOL_CALLS]b[ARGS]{}"])
        #expect(result.calls.map(\.function.name) == ["a", "b"])
        #expect(result.text.isEmpty)
    }

    // MARK: - Calls the parser can't read (S2-2 requirement 6)

    /// A `<tool_call>` block whose body isn't a call is **content**, not an error, and says so with
    /// `finish_reason: "stop"` — the scanner never throws.
    @Test func anUnreadableCallComesBackAsText() {
        let result = run(.json, ["<tool_call>not json at all</tool_call>"])
        #expect(result.calls.isEmpty)
        #expect(result.text == "<tool_call>not json at all</tool_call>")
    }

    /// A call the stream ended inside (no end tag) is text too.
    @Test func aTruncatedCallComesBackAsText() {
        let result = run(.json, ["<tool_call>{\"name\": \"get_weather\", \"argum"])
        #expect(result.calls.isEmpty)
        #expect(result.text == "<tool_call>{\"name\": \"get_weather\", \"argum")
    }

    /// An unclosed inline object is flushed at end of stream rather than dropped.
    @Test func anUnclosedInlineObjectComesBackAsText() {
        let result = run(.llama3, ["{not json"])
        #expect(result.calls.isEmpty)
        #expect(result.text == "{not json")
    }

    // MARK: - Typed arguments (S2-2 requirement 2)

    /// The client's schema is what makes a number a number: the XML parser has nothing else to go
    /// on (the S2-1 finding). Both paths are pinned so the schema-less shape can't regress into a
    /// silent type change.
    @Test func theSchemaIsWhatMakesArgumentsTyped() {
        let output = "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
        let withoutSchema = run(.xmlFunction, [output])
        #expect(withoutSchema.calls.first?.function.arguments["days"] == .string("3"))

        let withSchema = run(.xmlFunction, [output], tools: [weatherToolSchema])
        #expect(withSchema.calls.first?.function.arguments["days"] == .int(3))
        #expect(withSchema.calls.first?.function.arguments["location"] == .string("Paris"))
    }

    /// Calls are drained, not accumulated: a caller that reads them after every push sees each one
    /// exactly once.
    @Test func takeCallsDrains() {
        var scanner = ServeToolCallScanner(format: .json, tools: nil)
        _ = scanner.push("<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>")
        #expect(scanner.takeCalls().count == 1)
        #expect(scanner.takeCalls().isEmpty)
    }
}
