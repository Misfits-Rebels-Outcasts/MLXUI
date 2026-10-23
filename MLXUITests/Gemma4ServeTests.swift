import Foundation
import Jinja
import MLXLMCommon
import Testing
@testable import MLXUI

/// MP1-S (`RSI/DelegateModelPackBacklog.md`) — "Gemma 4 E4B" on the Local Server. Every decision here is
/// **implementer's call, pending owner confirmation**; the evidence is in `RSI/journal/2026-543-mp1-s-gemma4-serve.md`.
///
/// What is proven, and from what:
///  * the chat template, `config.json` and Python's rendering of the template are pinned in `Fixtures/Gemma4/`
///    (revision `475b9088d2…`, see `PROVENANCE.md`), so nothing here needs the 5.18 GB install;
///  * the serve path and the chat path hand the template the same messages and produce the same tokens as Python;
///  * tool calls stay `.unknown` because mlx-swift-lm's `.gemma4` parser does not read nested values;
///  * thinking stays off because `ServeReasoningSplitter` does not know Gemma 4's channel form;
///  * memory admission never underestimates this checkpoint's KV cache.
struct Gemma4ServeTests {

    private static let hfID = "mlx-community/gemma-4-e4b-it-4bit"
    private static let modelType = "gemma4"

    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }
    private func fixture(_ name: String) throws -> Data {
        try Data(contentsOf: root.appendingPathComponent("Fixtures/Gemma4/\(name)"))
    }
    private func configData() throws -> Data { try fixture("config.json") }

    // MARK: - The goldens (Python jinja2 renderings of the pinned template)

    private struct Goldens: Decodable {
        struct Case: Decodable { let name: String; let expected: String }
        let cases: [Case]
    }
    private func golden(_ name: String) throws -> String {
        let goldens = try JSONDecoder().decode(Goldens.self, from: fixture("prompt_goldens.json"))
        return try #require(goldens.cases.first { $0.name == name }, "no golden named \(name)").expected
    }

    // MARK: - 1. Tool calls: `.unknown`, and why

    @Test func tableAnswersAreTheDocumentedOnes() {
        #expect(ToolCallFormatTable.toolSupport(forModelType: Self.modelType) == .unknown)
        #expect(ToolCallFormatTable.acceptsTools(forModelType: Self.modelType) == false)
        // Left to mlx-swift-lm. The container's format (`.gemma4` under 3.31.4) is asserted in
        // `RealModelToolFormatTests` (WW2-3 G1); the app resolves nothing of its own.
        #expect(ToolCallFormatTable.resolved(forModelType: Self.modelType, configData: nil) == nil)
        #expect(ToolCallFormatTable.override(forModelType: Self.modelType, configData: nil) == nil)
        #expect(ToolCallFormatTable.systemPlacement(forModelType: Self.modelType) == .anywhere)
        #expect(ToolCallHistoryTable.oneCallPerTurn(forModelType: Self.modelType) == false)
        #expect(ToolCallHistoryTable.rendersFirstCallOnly(forModelType: Self.modelType) == false)
    }

    /// The text between the tags of the (only) tool call in a rendered history.
    private func renderedCall(_ prompt: String) throws -> String {
        let start = try #require(prompt.range(of: "<|tool_call>"))
        let end = try #require(prompt.range(of: "<tool_call|>"))
        return String(prompt[start.lowerBound ..< end.upperBound])
    }

    private func parsed(_ prompt: String) throws -> ToolCall.Function? {
        ToolCallFormat.gemma4.createParser().parse(content: try renderedCall(prompt), tools: nil)?.function
    }

    /// The calls the **template** writes, parsed by mlx-swift-lm's `.gemma4` parser. Only string arguments, and no
    /// arguments at all, come back exactly.
    @Test(arguments: [
        ("h_call_string_with_specials", ["q": "a, b: {c} and \"d\""] as [String: any Sendable]),
        ("h_call_no_arguments", [:] as [String: any Sendable]),
    ] as [(String, [String: any Sendable])])
    func stringAndEmptyArgumentsRoundTripThroughTheGemma4Parser(name: String, arguments: [String: any Sendable]) throws {
        let function = try #require(try parsed(try golden(name)), "\(name): the parser found no call")
        #expect(function == ToolCall.Function(name: "do_thing", arguments: arguments), "\(name)")
    }

    /// A number or a boolean does **not** come back typed: the parser tries `JSONSerialization` on the bare text, which
    /// refuses a fragment (`3`, `true`), and falls back to the string. The scanner does no coercion from the client's
    /// schema, so a client would receive `"days": "3"` for an integer parameter. Pinned as evidence for `.unknown`.
    @Test func numbersAndBooleansComeBackAsStrings() throws {
        let function = try #require(try parsed(try golden("h_call_flat_scalars")))
        #expect(function == ToolCall.Function(name: "do_thing", arguments: ["city": "Tokyo", "days": "3", "metric": "true"]))
        #expect(function != ToolCall.Function(name: "do_thing", arguments: ["city": "Tokyo", "days": 3, "metric": true]),
                "the parser now types scalars: reconsider ToolCallFormatTable (see the nested-arguments test's comment)")
    }

    /// **The evidence behind `.unknown`.** An array or an object as an argument value is what the template writes
    /// (`tags:[<|"|>a<|"|>,<|"|>b<|"|>]`, `opts:{j:<|"|>x<|"|>,k:1}`) and what coding-agent tool schemas are full
    /// of, but `GemmaFunctionParser` reads a value up to the next comma, so the result is a garbled string plus
    /// dropped arguments. That is not an exact match, and a wrong argument is worse than an ignored `tools` array.
    /// This test records the current behaviour. **If it starts failing, the parser improved:** re-read the decision
    /// in `ToolCallFormatTable.toolSupport` (and add SV-6/7/8 rows) before flipping it; do not just update the test.
    @Test(arguments: [
        ("h_call_array_value", ["tags": ["a", "b"], "n": 1] as [String: any Sendable]),
        ("h_call_object_value", ["opts": ["k": 1, "j": "x"]] as [String: any Sendable]),
        ("h_call_array_of_objects", ["edits": [["path": "a.txt", "line": 3], ["path": "b.txt", "line": 9]]] as [String: any Sendable]),
    ] as [(String, [String: any Sendable])])
    func nestedToolArgumentsDoNotRoundTripThroughTheGemma4Parser(name: String, arguments: [String: any Sendable]) throws {
        let function = try parsed(try golden(name))
        #expect(function != ToolCall.Function(name: "do_thing", arguments: arguments),
                "\(name): the parser now reads nested values: reconsider ToolCallFormatTable (see this test's comment)")
    }

    /// The tool traffic a client replays is flattened to text, exactly as for any family that does not accept tools
    /// (R20), so a harness that sends `tools` to this model still gets a coherent reply, not an error.
    @Test func toolTrafficIsFlattenedWhenToolsAreNotAccepted() {
        let call = ServeToolCall(id: "c1", name: "get_weather", argumentsJSON: "{\"city\":\"Tokyo\"}")
        let history = [ServeMessage(role: .user, text: "Weather?"),
                       ServeMessage(role: .assistant, text: "", toolCalls: [call]),
                       ServeMessage(role: .tool, text: "sunny", toolCallID: "c1")]
        let out = ServeMessageAssembly.assembling(
            history,
            acceptsTools: ToolCallFormatTable.acceptsTools(forModelType: Self.modelType),
            oneCallPerTurn: ToolCallHistoryTable.oneCallPerTurn(forModelType: Self.modelType),
            placement: ToolCallFormatTable.systemPlacement(forModelType: Self.modelType))
        #expect(out.allSatisfy { $0.toolCalls.isEmpty && $0.role != .tool })
        #expect(out.contains { $0.text.contains("[tool result for get_weather: sunny]") })
    }

    // MARK: - 2. Thinking: off, and why

    @Test func thinkingIsNotOfferedAndAskingForItIsIgnored() throws {
        #expect(ReasoningTemplateTable.thinkShape(forModelType: Self.modelType) == .none)
        #expect(ReasoningTemplateTable.readsEnableThinking(forModelType: Self.modelType) == false)
        #expect(ReasoningTemplateTable.thinkingOn(requested: true, forModelType: Self.modelType) == false)
        #expect(ReasoningTemplateTable.thinkingOn(requested: nil, forModelType: Self.modelType) == false)
        // The template itself would switch thinking on: that is the control being deliberately left unwired.
        #expect(try golden("g_thinking_on").contains("<|think|>"))
    }

    /// Why: the model's thinking is `<|channel>thought\n…<channel|>`, and `ServeReasoningSplitter` only knows
    /// `<think>…</think>`, so it would hand the whole channel to the client as the answer.
    @Test func theReasoningSplitterDoesNotKnowGemma4sChannelForm() {
        var splitter = ServeReasoningSplitter(startsInsideThink: false)
        let text = "<|channel>thought\nworking it out<channel|>The answer is 4."
        let first = splitter.push(text)
        let rest = splitter.finish()
        #expect(first.reasoning + rest.reasoning == "")
        #expect(first.content + rest.content == text)
    }

    // MARK: - 3. Memory + context: never underestimate

    private func textConfig() throws -> [String: Any] {
        let root = try #require(JSONSerialization.jsonObject(with: configData()) as? [String: Any])
        return try #require(root["text_config"] as? [String: Any])
    }

    @Test func modelTypeAndContextWindowAreReadFromTheConfig() throws {
        #expect(ModelContainerPool.modelTypeString(configData: try configData()) == Self.modelType)
        #expect(ModelContextWindow.limit(configData: try configData()) == 131_072)     // under `text_config`
    }

    /// What `ServeMemoryAdmission` assumes: every one of the 42 layers is full attention at `head_dim` 256.
    @Test func theProfileIsNonNilAndCountsEveryLayerAtHeadDim256() throws {
        let kv = try #require(ServeMemoryAdmission.kvCost(configData: try configData()))
        #expect(kv.perToken == 42 * 2 * 256 * 2 * 2)          // layers × kv heads × head_dim × (K,V) × fp16 = 86,016
        #expect(kv.perToken == 86_016)
        #expect(kv.fixed == 0)
        let profile = try #require(ServeMemoryAdmission.profile(configData: try configData(),
                                                                weightsBytes: 5_146_800_534, roomBytes: 12_000_000_000))
        #expect(profile.bytesPerToken == 86_016 * ServeMemoryAdmission.overheadFactor)
    }

    /// The KV bytes mlx-swift-lm 3.31.4 really allocates for `tokens` of context (fp16), worked out independently of
    /// `ServeMemoryAdmission` from the same `config.json`, following `Gemma4TextModel.newCache`
    /// (`Gemma4Text.swift:755-767`): only the first `num_hidden_layers − num_kv_shared_layers` layers own a cache;
    /// a `full_attention` one is a growing `StandardKVCache` at `global_head_dim`; a `sliding_attention` one is a
    /// `RotatingKVCache(maxSize: sliding_window)` at `head_dim`, which during prefill holds up to
    /// `maxSize + prefillStep − 1` tokens (`KVCache.swift:582`, `GenerateParameters.prefillStepSize` 512).
    private func trueKVBytes(tokens: Int) throws -> Int {
        let text = try textConfig()
        func int(_ key: String) -> Int { (text[key] as? NSNumber)?.intValue ?? 0 }
        #expect(text["attention_k_eq_v"] as? Bool == false)     // otherwise full layers would use num_global_key_value_heads
        let layers = int("num_hidden_layers"), shared = int("num_kv_shared_layers")
        let kvHeads = int("num_key_value_heads"), headDim = int("head_dim"), globalHeadDim = int("global_head_dim")
        let window = int("sliding_window"), prefillStep = 512
        let types = try #require(text["layer_types"] as? [String])
        var total = 0
        for index in 0 ..< (layers - shared) {
            if types[index] == "full_attention" {
                total += tokens * 2 * kvHeads * globalHeadDim * 2
            } else {
                total += min(tokens, window + prefillStep - 1) * 2 * kvHeads * headDim * 2
            }
        }
        return total
    }

    /// **The check that decides whether MP1-S must stop and ask: the estimate is never below the truth.**
    /// For every prompt-plus-reply length from one token to the full 131,072 window the profile's KV bytes
    /// (before the 1.75 overhead factor, which scales both sides alike) are at least what the cache holds.
    @Test func memoryAdmissionNeverUnderestimatesTheKVCache() throws {
        let kv = try #require(ServeMemoryAdmission.kvCost(configData: try configData()))
        for tokens in [1, 16, 100, 511, 512, 513, 1023, 1024, 2048, 4096, 8192, 32_768, 65_536, 131_072] {
            let estimate = kv.perToken * tokens + kv.fixed
            let truth = try trueKVBytes(tokens: tokens)
            #expect(estimate >= truth, "\(tokens) tokens: estimated \(estimate) B < true \(truth) B")
        }
    }

    /// The size of the overestimate, so the journal's number is a tested one. Beyond the window the sliding layers
    /// stop growing: the true growth is the 4 owning full-attention layers' 2 × 2 × 512 × 2 = 4,096 B each, against
    /// the profile's 86,016 B per token for every layer.
    @Test func theOverestimateIsAboutFiveTimesOnceTheSlidingWindowIsFull() throws {
        let kv = try #require(ServeMemoryAdmission.kvCost(configData: try configData()))
        let growth = try trueKVBytes(tokens: 100_000) - (try trueKVBytes(tokens: 99_000))
        #expect(growth / 1_000 == 4 * 4_096)                       // 16,384 B per token
        #expect(Double(kv.perToken) / Double(growth / 1_000) == 5.25)
    }

    /// The Connect sheet's facts for this model, from the pinned config, on a Mac whose room is 11.6 GiB (the 18 GiB
    /// Mac S3-5's calibration used): tools are not claimed, thinking is not offered, the window is 131,072, and the
    /// usable context is memory-limited well below it, because admission is conservative (see the test above).
    @Test func connectFactsForThisModel() throws {
        let weights: Int64 = 5_146_800_534
        let tight = ServeModelFacts.facts(modelID: Self.hfID, configData: try configData(), weightsBytes: weights,
                                          roomBytes: Int64(11.6 * 1_073_741_824))
        #expect(tight.modelID == Self.hfID)
        #expect(tight.supportsTools == false)
        #expect(tight.thinkingAvailable == false)
        #expect(tight.contextLength == 131_072)
        let usable = try #require(tight.usableTokens)
        #expect(usable > 0 && usable < 131_072)
        // A Mac with room to spare is limited by the window alone.
        let roomy = ServeModelFacts.facts(modelID: Self.hfID, configData: try configData(), weightsBytes: weights,
                                          roomBytes: Int64(60 * 1_073_741_824))
        #expect(roomy.usableTokens == 131_072)
    }

    // MARK: - 4. Prompt cache: off ("not tested")

    @Test func promptCacheStaysOffForGemma4() {
        #expect(PromptCacheAllowlist.allows("gemma4") == false)
        #expect(PromptCacheAllowlist.allows("gemma4_text") == false)
        #expect(PromptCacheAllowlist.tested.contains { $0.hasPrefix("gemma4") } == false)
    }

    // MARK: - 5. The serve path renders the same prompt as chat (and as Python)

    /// A throwaway tokenizer directory: the **real** pinned chat template (and its sanitizer), plus a synthetic
    /// character-level `tokenizer.json` (every ASCII character one token, the template's special tokens as added
    /// tokens). It is not Gemma's vocabulary; it is injective on ASCII text, which is all that is needed to compare
    /// a rendered prompt for exact equality. The real tokenizer is loaded through `HFTokenizerLoader`, the one
    /// loader the chat engine, the VLM engine and the server's prompt counter all use.
    private func makeTokenizerDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-serve-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try fixture("chat_template.jinja").write(to: directory.appendingPathComponent("chat_template.jinja"))
        let specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|turn>", "<turn|>", "<|image|>", "<|tool>", "<tool|>",
                        "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|\"|>", "<|think|>",
                        "<|channel>", "<channel|>"]
        var vocab: [String: Int] = [:]
        for scalar in (32 ... 126).map({ String(UnicodeScalar($0)!) }) + ["\n"] { vocab[scalar] = vocab.count }
        let added: [[String: Any]] = specials.enumerated().map { offset, token in
            ["id": vocab.count + offset, "content": token, "single_word": false, "lstrip": false, "rstrip": false,
             "normalized": false, "special": true]
        }
        let tokenizer: [String: Any] = [
            "version": "1.0", "truncation": NSNull(), "padding": NSNull(), "added_tokens": added,
            "normalizer": NSNull(), "pre_tokenizer": NSNull(), "post_processor": NSNull(), "decoder": NSNull(),
            "model": ["type": "BPE", "vocab": vocab, "merges": [String](), "unk_token": "<unk>"] as [String: Any],
        ]
        try JSONSerialization.data(withJSONObject: tokenizer).write(to: directory.appendingPathComponent("tokenizer.json"))
        let config: [String: Any] = ["tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<bos>", "eos_token": "<eos>",
                                     "unk_token": "<unk>", "pad_token": "<pad>"]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("tokenizer_config.json"))
        return directory
    }

    @Test func serveAndChatGiveTheSameTokensAsPython() async throws {
        let directory = try makeTokenizerDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenizer = try await HFTokenizerLoader().load(from: directory)

        // The conversation, as the chat path builds it (LLMUserInputProcessor → DefaultMessageGenerator) …
        let chatMessages = DefaultMessageGenerator().generate(messages: [
            .system("Be brief."), .user("Hi"), .assistant("Hello!"), .user("Name a river."),
        ])
        // … and as the server builds it (ServeMessageAssembly → MLXChatBackend.templateMessages).
        let served = ServeMessageAssembly.assembling(
            [ServeMessage(role: .system, text: "Be brief."), ServeMessage(role: .user, text: "Hi"),
             ServeMessage(role: .assistant, text: "Hello!"), ServeMessage(role: .user, text: "Name a river.")],
            acceptsTools: ToolCallFormatTable.acceptsTools(forModelType: Self.modelType),
            oneCallPerTurn: ToolCallHistoryTable.oneCallPerTurn(forModelType: Self.modelType),
            placement: ToolCallFormatTable.systemPlacement(forModelType: Self.modelType))
        let serveMessages = MLXChatBackend.templateMessages(served)

        // Same dictionaries going in …
        #expect(chatMessages.count == serveMessages.count)
        for (chat, serve) in zip(chatMessages, serveMessages) {
            #expect(chat["role"] as? String == serve["role"] as? String)
            #expect(chat["content"] as? String == serve["content"] as? String)
        }

        // … same context (thinking off for both) …
        let chatContext: [String: any Sendable] = ["enable_thinking": false]
        let serveContext: [String: any Sendable] = [
            "enable_thinking": ReasoningTemplateTable.thinkingOn(requested: nil, forModelType: Self.modelType)]

        // … same tokens out, and they are Python's prompt, character for character.
        let chatIDs = try tokenizer.applyChatTemplate(messages: chatMessages, tools: nil, additionalContext: chatContext)
        let serveIDs = try tokenizer.applyChatTemplate(messages: serveMessages, tools: nil, additionalContext: serveContext)
        let pythonIDs = tokenizer.encode(text: try golden("e_serve_conversation"), addSpecialTokens: false)
        #expect(!chatIDs.isEmpty)
        #expect(chatIDs == serveIDs)
        #expect(serveIDs == pythonIDs)
        let unk = try #require(tokenizer.convertTokenToId("<unk>"))
        #expect(!serveIDs.contains(unk), "the synthetic vocabulary must cover the whole prompt, or equality proves nothing")
    }

    /// FINDING (MP1-S): the **tool declaration** the template writes differs from Python's by whitespace. swift-jinja
    /// keeps a space after a `{` that is followed by a `-` tag (`properties:{ city:{ description:…` where Python
    /// gives `properties:{city:{description:…`; minimal repro `a{ {{- 'x' }}` → `a{ x`, Python `a{x`). Every plain-chat
    /// golden is byte-identical, so chat and Serve prompts are unaffected; only a request carrying `tools` would
    /// see it, and `tools` are not accepted for this family (`.unknown`) so none reaches the template today. It is a
    /// second reason, besides the parser, not to switch tools on yet. If this test starts failing, swift-jinja
    /// was fixed: tighten `CommentWhitespaceControlTests`' `f_tools_declared` case to exact equality.
    @Test func theToolDeclarationDiffersFromPythonOnlyByWhitespace() throws {
        let template = try String(contentsOf: root.appendingPathComponent("Fixtures/Gemma4/chat_template.jinja"), encoding: .utf8)
        let sanitized = ChatTemplateSanitizer.sanitized(template) ?? template
        let tool: [String: Any] = ["type": "function", "function": [
            "name": "get_weather", "description": "Get the weather for a city.",
            "parameters": ["type": "object",
                           "properties": ["city": ["type": "string", "description": "City name"], "days": ["type": "integer"]],
                           "required": ["city"]]]]
        let out = try Template(sanitized, with: .init(lstripBlocks: true, trimBlocks: true)).render([
            "bos_token": "<bos>", "add_generation_prompt": true,
            "messages": try Value(any: [["role": "user", "content": "What is the weather in Tokyo?"]]),
            "tools": try Value(any: [tool]),
        ])
        let python = try golden("f_tools_declared")
        #expect(out.filter { !$0.isWhitespace } == python.filter { !$0.isWhitespace })
        #expect(out != python, "swift-jinja now matches Python exactly on tool declarations: tighten the exact-equality exception")
        #expect(out.contains("properties:{ city:"))
    }
}
