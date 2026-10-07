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

/// S2-7 — OpenCode's own tool schemas (the SV-6 run's shapes): `write`, `glob`, `read`, `bash`.
/// Shared with the gated real-model turn in `ServeChatTests`.
let openCodeTools: [[String: any Sendable]] = ["write", "glob", "read", "bash"].map { name in
    [
        "type": "function",
        "function": [
            "name": name,
            "description": "\(name) (OpenCode)",
            "parameters": [
                "type": "object",
                "properties": [
                    "content": ["type": "string"] as [String: any Sendable],
                    "filePath": ["type": "string"] as [String: any Sendable],
                    "path": ["type": "string"] as [String: any Sendable],
                    "pattern": ["type": "string"] as [String: any Sendable],
                    "command": ["type": "string"] as [String: any Sendable],
                ] as [String: any Sendable],
                "required": [] as [String],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]
}

/// The exact text SV-6's Llama-3.1-8B turn produced (owner, 2026-10-07): two top-level JSON objects,
/// whitespace-separated, where the Llama 3 template expects one call per turn.
private let sv6LlamaText = #"{"name": "write", "parameters": {"content": "hello world", "filePath": "notes.md"}} {"name": "glob", "parameters": {"path": "", "pattern": "notes.md"}}"#

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

    // MARK: - S2-7: a run of JSON objects at the turn's opening

    /// **The SV-6 reproduction** (S2-7): the exact Llama-3.1-8B turn, one push, OpenCode's schemas.
    /// Before the fix it produced `calls=["write"]` plus the `glob` object as **content** — the first
    /// object parsed and the "one call per turn" rule streamed everything after it (case (a)).
    @Test func theSV6StringIsTwoCallsInOnePush() {
        let result = run(.llama3, [sv6LlamaText], tools: openCodeTools)
        print("S2-7 MEASURE one push: text=\(result.text.debugDescription) calls=\(result.calls.map(\.function.name))")
        #expect(result.calls.map(\.function.name) == ["write", "glob"])
        #expect(result.text.isEmpty)
        #expect(result.calls.first?.function.arguments["content"] == .string("hello world"))
        #expect(result.calls.first?.function.arguments["filePath"] == .string("notes.md"))
        #expect(result.calls.last?.function.arguments["pattern"] == .string("notes.md"))
    }

    /// The same string arriving a token at a time — the real streaming shape — and behind each
    /// opening the model has been seen to emit.
    @Test func theSV6StringIsTwoCallsTokenByTokenAndBehindEveryOpening() {
        for opening in ["", "\n", "\n\n", "  ", "<|python_tag|>", "\n<|python_tag|>"] {
            let text = opening + sv6LlamaText
            let one = run(.llama3, [text], tools: openCodeTools)
            #expect(one.calls.map(\.function.name) == ["write", "glob"], "\(opening.debugDescription)")
            #expect(one.text.isEmpty, "\(opening.debugDescription): \(one.text.debugDescription)")

            let tokens = run(.llama3, text.map(String.init), tools: openCodeTools)
            print("S2-7 MEASURE token-sized (\(opening.debugDescription)): text=\(tokens.text.debugDescription) "
                  + "calls=\(tokens.calls.map(\.function.name))")
            #expect(tokens.calls.map(\.function.name) == ["write", "glob"], "\(opening.debugDescription)")
            #expect(tokens.text.isEmpty, "\(opening.debugDescription): \(tokens.text.debugDescription)")
        }
    }

    /// **The real turn** (S2-7 requirement 3, gated test `aRealLlamaToolTurnIsReadAsCalls`): the
    /// installed `Meta-Llama-3.1-8B-Instruct-4bit` emitted three objects behind `<|python_tag|>`,
    /// separated by `;` — not the space the SV-6 report showed, which is why `;` is in the separator
    /// set on evidence rather than in the abstract. Pinned here so the default suite covers it.
    @Test func theRealLlamaShapeIsThreeSemicolonSeparatedCalls() {
        let raw = #"<|python_tag|>{"name": "read", "parameters": {"filePath": "hello.txt"}}; {"name": "write", "parameters": {"content": "Hello World", "filePath": "notes.md"}}; {"name": "glob", "parameters": {"pattern": "*.txt"}}"#
        let result = run(.llama3, [raw], tools: openCodeTools)
        #expect(result.calls.map(\.function.name) == ["read", "write", "glob"])
        #expect(result.text.isEmpty)

        // The same turn one token at a time.
        let tokens = run(.llama3, raw.map(String.init), tools: openCodeTools)
        #expect(tokens.calls.map(\.function.name) == ["read", "write", "glob"])
        #expect(tokens.text.isEmpty)
    }

    /// Every separator the run accepts, plus the bracketed form upstream's `parseEOS` already decodes.
    @Test func everyRunSeparatorAndTheArrayFormAreAccepted() {
        let a = #"{"name": "write", "parameters": {"content": "hi", "filePath": "notes.md"}}"#
        let b = #"{"name": "glob", "parameters": {"pattern": "notes.md"}}"#
        for separator in [" ", "\n", "\n\n", "\t", ", ", "; ", ",\n", ";\n", ",", ";"] {
            let result = run(.llama3, [a + separator + b], tools: openCodeTools)
            #expect(result.calls.map(\.function.name) == ["write", "glob"], "\(separator.debugDescription)")
            #expect(result.text.isEmpty, "\(separator.debugDescription): \(result.text.debugDescription)")
        }
        let array = run(.llama3, ["[" + a + ", " + b + "]"], tools: openCodeTools)
        #expect(array.calls.map(\.function.name) == ["write", "glob"])
        #expect(array.text.isEmpty)
        // …and a bracketed run behind the marker, split across deltas.
        let marked = run(.llama3, ["<|python_tag|>[" + a + ",", b + "]"], tools: openCodeTools)
        #expect(marked.calls.map(\.function.name) == ["write", "glob"])
        #expect(marked.text.isEmpty)
    }

    /// **No half-calls**: one unreadable object in the run makes the whole run text — the readable
    /// one is not handed to the client on its own.
    @Test func oneBadObjectMakesTheWholeRunText() {
        let good = #"{"name": "write", "parameters": {"content": "hi"}}"#
        // A **complete** object that is not a call: it has no `name` for the parser to read. (An
        // *incomplete* object is the truncation path instead — `anUnclosedInlineObjectComesBackAsText`.)
        let unreadable = #"{"parameters": {}}"#
        let result = run(.llama3, [good + " " + unreadable + " done"], tools: openCodeTools)
        print("S2-7 MEASURE bad run: text=\(result.text.debugDescription) calls=\(result.calls.map(\.function.name))")
        #expect(result.calls.isEmpty)                       // no half-call: the readable one too
        #expect(result.text.contains("\"name\": \"write\""))
        #expect(result.text.contains("done"))

        // The same, one token at a time.
        let tokens = run(.llama3, (good + " " + unreadable + " done").map(String.init), tools: openCodeTools)
        #expect(tokens.calls.isEmpty)
        #expect(tokens.text.contains("\"name\": \"write\""))

        // And an unreadable object *second* in a two-object run does not strand the first.
        let stranded = run(.llama3, [good + " " + #"{"name": 42}"#], tools: openCodeTools)
        #expect(stranded.calls.isEmpty)
        #expect(stranded.text.contains("\"name\": \"write\""))
    }

    /// A single call is unchanged (the S2-2 cases), and so are the mid-answer `{` rules: a run is
    /// only ever read at the turn's **opening**.
    @Test func aSingleCallAndTheMidAnswerRulesAreUnchanged() {
        let one = run(.llama3, [#"{"name": "get_weather", "parameters": {"location": "Paris"}}"#])
        #expect(one.calls.count == 1 && one.text.isEmpty)

        // Prose first: the whole turn is text, including anything that looks like a run.
        let prose = run(.llama3, ["Sure! " + sv6LlamaText], tools: openCodeTools)
        #expect(prose.calls.isEmpty)
        #expect(prose.text.hasPrefix("Sure! "))

        // A run *after* an answer is an example, not calls (S2-2b's rule, at the run's scale).
        var scanner = ServeToolCallScanner(format: .llama3, tools: openCodeTools)
        var text = scanner.push("Here is an example: ")
        text += scanner.push(sv6LlamaText)
        text += scanner.finish()
        #expect(scanner.takeCalls().isEmpty)
        #expect(text.contains("{\"name\": \"write\"") && text.contains("{\"name\": \"glob\""))
    }

    /// The trailing separators after a run stay in the buffer, so text after the run streams as it
    /// does today (and a calls-only turn's whitespace is what S2-2b's content rule then nulls).
    @Test func textAfterTheRunStreamsAsToday() {
        let result = run(.llama3, [sv6LlamaText + "\n\nAll done."], tools: openCodeTools)
        #expect(result.calls.map(\.function.name) == ["write", "glob"])
        #expect(result.text == "\n\nAll done.")
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
