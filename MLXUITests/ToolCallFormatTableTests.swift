import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

// S2-1 — the tool-call format table, the per-family "accepts tools" answer, and the per-family
// `system` placement. Model-free except for the last suite, which is gated on an installed model.
//
// Every expectation here is the mapping the S2-1 journal records from each family's own chat
// template: `RSI/journal/2026-405-s2-1-tool-call-formats.md`. Implementer's call, pending owner
// confirmation.

// MARK: - The 12 catalog chat models (readme F15) and the `model_type` each checkpoint reports

/// A catalog chat model and the `model_type` its `config.json` carries (read 2026-10-06).
private let catalogChatModels: [(repo: String, modelType: String)] = [
    ("prism-ml/Ternary-Bonsai-27B-mlx-2bit", "qwen3_5"),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "qwen3_5"),
    ("lukaskremla/Qwen3.8-27B-3bit-MLX", "qwen3_5"),
    ("mlx-community/Qwen3.8-27B-4bit", "qwen3_5"),
    ("mlx-community/gemma-3-1b-it-qat-4bit", "gemma3_text"),
    ("mlx-community/Ministral-3-3B-Instruct-2512-4bit", "mistral3"),
    ("mlx-community/Qwen3-8B-4bit", "qwen3"),
    ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "llama"),
    ("mlx-community/gemma-2-9b-it-4bit", "gemma2"),
    ("mlx-community/Qwen2.5-14B-Instruct-4bit", "qwen2"),
    ("mlx-community/gpt-oss-20b-MXFP4-Q8", "gpt_oss"),
    ("mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", "mistral3"),
]

/// A `config.json` shaped like the checkpoint's, for the types whose inference needs secondary
/// signals. Qwen3.5's `vocab_size` really lives under `text_config` and is not needed here.
private func configJSON(_ object: [String: Any]) -> Data {
    try! JSONSerialization.data(withJSONObject: object)
}

private let llama31Config = configJSON([
    "model_type": "llama", "vocab_size": 128256,
    "rope_scaling": ["rope_type": "llama3"],
])
private let qwen35Config = configJSON(["model_type": "qwen3_5", "text_config": ["vocab_size": 248320]])

// MARK: - Format resolution

/// The app's side of the mapping, for every `model_type` the catalog's 12 chat models report.
///
/// The app carries **no** override for any catalog family, and since mlx-swift-lm 3.32 it no longer
/// repeats upstream's `model_type` → format resolution (it is package-internal and also reads
/// `tool_parser_type` and the chat template), so `ToolCallFormatTable.resolved` is the override alone.
/// WW2-2-FIX-1, owner-approved 2026-10-22: the per-family format expectations that pinned 3.31's mapping
/// moved to `RealModelToolFormatTests` (WW2-3 G1), which asserts the loaded container's own format.
struct ToolCallFormatMappingTests {

    /// No catalog chat model needs an app override; the container's own resolved format applies.
    @Test func everyCatalogChatModelDefersToUpstream() {
        for (repo, modelType) in catalogChatModels {
            let config = modelType == "llama" ? llama31Config : qwen35Config
            #expect(ToolCallFormatTable.override(forModelType: modelType, configData: config) == nil,
                    "\(repo) (\(modelType)) must not need an app override")
            #expect(ToolCallFormatTable.resolved(forModelType: modelType, configData: config) == nil,
                    "\(repo) (\(modelType)): with no override the app leaves the format to the container")
        }
    }
}

// MARK: - "Can this family call tools at all?"

struct ToolSupportTests {

    /// The answer S2-2 and S2-4 branch on, per catalog family.
    @Test func catalogFamiliesAnswerAcceptsTools() {
        for modelType in ["qwen3_5", "qwen3", "qwen2", "llama", "mistral3"] {
            #expect(ToolCallFormatTable.toolSupport(forModelType: modelType) == .supported,
                    "\(modelType) renders tool schemas and has a parser")
            #expect(ToolCallFormatTable.acceptsTools(forModelType: modelType))
        }
        // Gemma 2 and Gemma 3 1B: the string "tool" appears zero times in either template.
        for modelType in ["gemma2", "gemma3_text"] {
            #expect(ToolCallFormatTable.toolSupport(forModelType: modelType) == .templateIgnoresTools,
                    "\(modelType)'s template has no tools branch")
            #expect(!ToolCallFormatTable.acceptsTools(forModelType: modelType))
        }
        #expect(ToolCallFormatTable.toolSupport(forModelType: "phi3") == .unknown)
        #expect(!ToolCallFormatTable.acceptsTools(forModelType: "phi3"),
                "an unknown family is not served tools")
        #expect(!ToolCallFormatTable.acceptsTools(forModelType: ""))
    }

    /// Families this table has no evidence for are not claimed. `qwen3_moe` is the documented AG0
    /// gap (Coder and 30B-A3B share it, upstream maps neither); `gemma3` is Gemma 3's VLM spelling
    /// (only `gemma3_text`, the 1B's, was read); `llama4` is a different template from `llama`.
    @Test func unmeasuredFamiliesAreNotClaimed() {
        for modelType in ["qwen3_moe", "gemma3", "llama4", "mistral", "gemma"] {
            #expect(ToolCallFormatTable.toolSupport(forModelType: modelType) == .unknown,
                    "\(modelType) was not measured and must not be claimed")
            #expect(!ToolCallFormatTable.acceptsTools(forModelType: modelType))
        }
        // `qwen3_next` is upstream's own `.xmlFunction` family — the one non-catalog entry the
        // table claims, and only because upstream maps it.
        #expect(ToolCallFormatTable.toolSupport(forModelType: "qwen3_next") == .supported)
    }

    /// Task instruction 3: gpt-oss's harmony format has no `ToolCallFormat` case, so S2-1 records
    /// it as "no supported tool format" rather than writing a parser. This test is the pin that
    /// keeps S2-2/S2-4 from mistaking it for supported.
    @Test func gptOssIsRecordedAsHavingNoSupportedToolFormat() {
        #expect(ToolCallFormatTable.toolSupport(forModelType: "gpt_oss") == .unsupportedFormat)
        #expect(!ToolCallFormatTable.acceptsTools(forModelType: "gpt_oss"))
        // The template does render tool schemas — that is exactly why the reason matters.
        #expect(ToolCallFormatTable.toolSupport(forModelType: "gpt_oss") != .templateIgnoresTools)
        // No case parses it, and none was added for S2-1.
        #expect(!ToolCallFormat.allCases.contains { $0.rawValue.lowercased().contains("harmony") })
    }
}

// MARK: - Where a `system` message may sit

struct SystemPlacementTests {

    @Test func catalogFamiliesGetTheirTemplatesPlacement() {
        // Raise on a `system` message that is not `messages[0]`.
        for modelType in ["qwen3_5", "mistral3", "gemma3_text", "gpt_oss"] {
            #expect(ToolCallFormatTable.systemPlacement(forModelType: modelType) == .firstOnly,
                    "\(modelType) only honours a leading system message")
        }
        // Gemma 2 refuses one even in the leading slot.
        #expect(ToolCallFormatTable.systemPlacement(forModelType: "gemma2") == .unsupported)
        // These render or ignore a `system` message wherever it is.
        for modelType in ["qwen3", "qwen2", "llama", "phi3", ""] {
            #expect(ToolCallFormatTable.systemPlacement(forModelType: modelType) == .anywhere,
                    "\(modelType) preserves the client's order")
        }
    }

    /// Ministral 3's `text_config.model_type` is `ministral3`; the app reads the top-level
    /// `mistral3`, but the nested spelling must not fall through to `.anywhere`.
    @Test func nestedMistralSpellingIsAlsoFirstOnly() {
        #expect(ToolCallFormatTable.systemPlacement(forModelType: "ministral3") == .firstOnly)
        #expect(ToolCallFormatTable.systemPlacement(forModelType: "mistral3_text") == .firstOnly)
    }
}

// MARK: - One parser test per catalog family, with a real output string

/// The recorded Qwen3-8B-4bit run's tool call, verbatim — see `RecordedToolCallTests` for the run
/// that produced it and the journal for the machine. **Recorded**, not a fixture.
private let recordedQwen3ToolCall = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>"

struct ToolCallParserTests {

    /// `.json`, via `qwen3` — the one catalog family with an installed model, so this is the one
    /// **recorded** string in this file.
    @Test func jsonParserReadsTheRecordedQwen3ToolCall() throws {
        let parser = ToolCallFormat.json.createParser()
        let call = try #require(parser.parse(content: recordedQwen3ToolCall, tools: nil),
                                "the recorded Qwen3 output must parse as .json")
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["location"] == .string("Paris"))
    }

    /// `.xmlFunction`, via `qwen3_5`. **Fixture** — mlx-lm's `qwen3_coder` case, which is the
    /// parser this Swift case mirrors (`XMLFunctionParser` cites `tool_parsers/qwen3_coder.py`):
    /// https://github.com/ml-explore/mlx-lm/blob/5cfec4cb39deba54210b3ff4d86f2337c7bc10b5/tests/test_tool_parsing.py#L43
    @Test func xmlFunctionParserReadsTheQwen35Fixture() throws {
        let parser = ToolCallFormat.xmlFunction.createParser()
        let output = "<function=multiply>\n<parameter=a>\n12234585\n</parameter>\n<parameter=b>\n48838483920\n</parameter>\n</function>"
        let call = try #require(parser.parse(content: output, tools: nil))
        #expect(call.function.name == "multiply")
        // Without a tool schema the XML parser has nothing to type a parameter with, so it hands
        // back the raw text. The harness's own `tools` array is what makes arguments typed — which
        // is exactly what S2-2 will pass into `UserInput(tools:)`.
        #expect(call.function.arguments["a"] == .string("12234585"))
        let typed = try #require(parser.parse(content: output, tools: [multiplyToolSchema]))
        #expect(typed.function.arguments["a"] == .int(12234585))
        #expect(typed.function.arguments["b"] == .int(48838483920))
    }

    /// `.mistral`, via `mistral3` (Ministral 3 and Devstral Small 2). **Fixture** — mlx-lm's
    /// `test_parsers`, the `get_current_temperature` case:
    /// https://github.com/ml-explore/mlx-lm/blob/5cfec4cb39deba54210b3ff4d86f2337c7bc10b5/tests/test_tool_parsing.py#L129
    @Test func mistralParserReadsTheMinistralFixture() throws {
        let parser = ToolCallFormat.mistral.createParser()
        let output = "get_current_temperature[ARGS]{\"location\": \"London\"}"
        let call = try #require(parser.parse(content: output, tools: nil))
        #expect(call.function.name == "get_current_temperature")
        #expect(call.function.arguments["location"] == .string("London"))
    }

    /// `.mistral` on the numeric fixture (mlx-lm's `test_parsers` at
    /// .../tests/test_tool_parsing.py#L59) and on the template's own render form (which carries the
    /// `[TOOL_CALLS]` marker the template emits; the fixture above does not).
    @Test func mistralParserReadsTheNumericFixtureAndTheMarkerForm() throws {
        let parser = ToolCallFormat.mistral.createParser()
        let numeric = try #require(parser.parse(content: "multiply[ARGS]{\"a\": 12234585, \"b\": 48838483920}",
                                                tools: nil))
        #expect(numeric.function.name == "multiply")
        #expect(numeric.function.arguments["a"] == .int(12234585))

        let marked = try #require(parser.parse(content: "[TOOL_CALLS]get_weather[ARGS]{\"location\": \"Tokyo\"}",
                                               tools: nil))
        #expect(marked.function.name == "get_weather")
        #expect(marked.function.arguments["location"] == .string("Tokyo"))
    }

    /// `.llama3`, via `llama`. **Fixture** — mlx-lm ships no llama3 parser (its parser registry
    /// has no `<|python_tag|>` case at all), so the only citable string is upstream's own
    /// documented example in the pinned package we build against:
    /// mlx-swift-lm 3.31.3 `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift:98`
    /// (`Example: `<|python_tag|>{ "name": "func", "parameters": {...} }``).
    /// The model is not installed on this Mac (27B and 8B chat models only), so this one is a
    /// fixture, not a recorded run — never presented as one.
    @Test func llama3ParserReadsTheUpstreamDocumentedExample() throws {
        let parser = ToolCallFormat.llama3.createParser()
        let output = "<|python_tag|>{ \"name\": \"get_weather\", \"parameters\": { \"location\": \"Paris\" } }"
        let call = try #require(parser.parse(content: output, tools: nil))
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["location"] == .string("Paris"))
    }
}

// MARK: - Recording suite (gated on an installed MLX chat model)

/// The run that produced `recordedQwen3ToolCall`: Qwen3-8B-4bit, the only catalog chat model
/// installed on the owner's Mac (4.3 GB; the other three installed folders are not catalog chat
/// models, and the 27B is out of scope for 19 GB of RAM).
///
/// Give the test host the folder with `TEST_RUNNER_MLXUI_TEST_LLM_DIR=<path>`, exactly as
/// `MLXChatBackendRealModelTests` does. It prints the raw completion once so the string can be
/// re-recorded after a model update — a gated test's stdout is not the server's request log
/// (rule 13 constrains the server, and nothing here is written to disk).
struct RecordedToolCallTests {
    private static let modelDir = ProcessInfo.processInfo.environment["MLXUI_TEST_LLM_DIR"]

    @Test(.enabled(if: RecordedToolCallTests.modelDir != nil))
    func recordsAndParsesAQwen3ToolCall() async throws {
        let dir = URL(fileURLWithPath: try #require(Self.modelDir))
        let pool = ModelContainerPool(budgetBytes: 40 << 30)
        let container = try await pool.container(directory: dir, architecture: nil)

        // Upstream decides for `qwen3` (there is no app override): nil → the container's `.json`
        // default. Pinned here on a real load, which is what F3b corrected.
        #expect(await container.configuration.toolCallFormat == nil)

        try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(
                    chat: [
                        .system("You are a helpful assistant with access to tools. "
                                + "Use get_weather for any weather question."),
                        .user("What is the weather in Paris? Use the get_weather tool."),
                    ],
                    tools: [weatherToolSchema],
                    additionalContext: LLMEngine.noThinkingContext))
            let stream = try MLXLMCommon.generateTokens(
                input: input,
                parameters: GenerateParameters(maxTokens: 120, temperature: 0),
                context: context)
            var ids: [Int] = []
            for await generation in stream {
                if case .token(let id) = generation { ids.append(id) }
            }
            // Raw token ids, not the `.chunk` path: `.chunk` runs mlx-swift-lm's ToolCallProcessor
            // (TCP-1), which would hand back a parsed call instead of the model's own text.
            let text = context.tokenizer.decode(tokenIds: ids)
            print("S2-1 RECORDED qwen3-8b-4bit tool call >>>\(text)<<<")
            // Also print it as a single Swift literal, so the recorded constant can be re-copied
            // exactly (newlines included) after a model update.
            let escaped = text
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\t", with: "\\t")
            print("S2-1 RECORDED LITERAL >>>\"\(escaped)\"<<<")

            let call = ToolCallFormat.json.createParser().parse(content: text, tools: nil)
            #expect(call?.function.name == "get_weather")
            #expect(call?.function.arguments["location"] != nil)
        }
    }
}

/// The same OpenAI function-schema dictionary the server hands `UserInput(tools:)`.
private let weatherToolSchema: ToolSpec = [
    "type": "function",
    "function": [
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": [
            "type": "object",
            "properties": [
                "location": [
                    "type": "string",
                    "description": "The city name, e.g. San Francisco",
                ]
            ] as [String: any Sendable],
            "required": ["location"],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]

/// The schema for mlx-lm's `multiply` fixture, used to show that a schema is what types the
/// arguments a schema-less parse leaves as text.
private let multiplyToolSchema: ToolSpec = [
    "type": "function",
    "function": [
        "name": "multiply",
        "parameters": [
            "type": "object",
            "properties": [
                "a": ["type": "integer"],
                "b": ["type": "integer"],
            ] as [String: any Sendable],
            "required": ["a", "b"],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]
