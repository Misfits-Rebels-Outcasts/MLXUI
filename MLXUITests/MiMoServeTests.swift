import Foundation
import Jinja
import MLXLMCommon
import Testing
@testable import MLXUI

/// MM-S — "MiMo V2.6 9B" on the Local Server. Every call here is implementer's call, pending owner
/// confirmation (the backlog contract says so); the evidence is the checkpoint's own files pinned in
/// `Fixtures/MiMo/` (MiMo @ `60061fc`) run through the real server seams.
///
/// The server's per-family tables are keyed on `model_type`, so this model **inherits Qwen3.5's rows**
/// (`qwen3_5`). The existing `qwen3_5` rows are not edited and no table is split (KICKOFF trap 2):
/// where MiMo's template differs from Qwen3.5's, a test here shows the behaviour is still right.
struct MiMoServeTests {

    private static let hfID = "mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit"
    /// The two shards (`7,100,710,126` bytes), as `MLXChatBackend.weightsBytes(in:)` would count them.
    private static let weightsBytes: Int64 = 7_100_710_126

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
    }

    private func fixture(_ name: String) throws -> Data {
        try Data(contentsOf: repoRoot.appendingPathComponent("Fixtures/MiMo/\(name)"))
    }

    private var configData: Data { get throws { try fixture("config.json") } }

    private func template() throws -> Template {
        try Template(#require(String(data: fixture("chat_template.jinja"), encoding: .utf8)))
    }

    /// The `model_type` the server's tables are keyed on, read the way the backend reads it.
    private func modelType() throws -> String {
        try #require(ModelContainerPool.modelTypeString(configData: configData))
    }

    // MARK: - 1. Serve visibility

    /// `.llm` → the Serve toggle shows once the model is installed (and the Local Server is on).
    @Test func theServeToggleShowsOnceInstalled() {
        func shows(_ state: InstallState, onDisk: Bool, gate: Bool = true) -> Bool {
            ServeToggleVisibility.shouldShow(state: state, isInstalledOnDisk: onDisk, runnerKind: .llm, gateAvailable: gate)
        }
        #expect(shows(.installed, onDisk: true))
        #expect(shows(.idle, onDisk: true))
        #expect(!shows(.idle, onDisk: false))
        #expect(!shows(.installed, onDisk: true, gate: false))     // the hide switch wins
    }

    // MARK: - 2. Tools

    /// `qwen3_5` → `.supported` / `.xmlFunction` matches this template's `<tool_call><function=…>` form,
    /// so there is no override; the format comes out of the real `config.json`.
    @Test func toolCallsUseTheQwen35RowsUnchanged() throws {
        let type = try modelType()
        #expect(type == "qwen3_5")
        #expect(ToolCallFormatTable.toolSupport(forModelType: type) == .supported)
        #expect(ToolCallFormatTable.toolSupport(forModelType: type).acceptsTools)
        #expect(ToolCallFormatTable.resolved(forModelType: type, configData: try configData) == .xmlFunction)
        // `system` placement inherits `.firstOnly` (harmless for this template, which allows `system`
        // anywhere); the history tables stay clear of it: it renders every call, so no split.
        #expect(ToolCallFormatTable.systemPlacement(forModelType: type) == .firstOnly)
        #expect(!ToolCallHistoryTable.oneCallPerTurn(forModelType: type))
        #expect(!ToolCallHistoryTable.rendersFirstCallOnly(forModelType: type))
    }

    private static let weatherTool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": [
                "type": "object",
                "properties": ["location": ["type": "string"] as [String: any Sendable]] as [String: any Sendable],
                "required": ["location"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    /// The model's own call syntax, one character at a time (the real streaming shape), becomes calls
    /// with typed arguments and nothing leaks into the content.
    @Test func theScannerReadsMiMosCallsTokenByToken() throws {
        let format = try #require(ToolCallFormatTable.resolved(forModelType: try modelType(), configData: try configData))
        let text = "<tool_call><function=get_weather><parameter=location>Paris</parameter></function></tool_call>"
            + "<tool_call><function=get_weather><parameter=location>Rome</parameter></function></tool_call>"
        var scanner = ServeToolCallScanner(format: format, tools: [Self.weatherTool])
        var content = ""
        for character in text { content += scanner.push(String(character)) }
        content += scanner.finish()
        let calls = scanner.takeCalls()
        #expect(content.isEmpty)
        #expect(calls.map(\.function.name) == ["get_weather", "get_weather"])
        #expect(calls.map { $0.function.arguments["location"] } == [.string("Paris"), .string("Rome")])
    }

    /// The untested case the backlog names: a **two-call loop** whose results are bare `tool` turns.
    /// Run through the server's own path: `ServeMessageAssembly` → `MLXChatBackend.templateMessages` →
    /// the checkpoint's template. (The live loop is smoke rows SV-6/7.)
    @Test func aTwoCallLoopRendersThroughTheServersOwnPath() throws {
        let type = try modelType()
        let history = [
            ServeMessage(role: .system, text: "Be brief."),
            ServeMessage(role: .user, text: "Weather in Paris and Rome?"),
            ServeMessage(role: .assistant, text: "", toolCalls: [
                ServeToolCall(id: "call_1", name: "get_weather", argumentsJSON: #"{"location":"Paris"}"#),
                ServeToolCall(id: "call_2", name: "get_weather", argumentsJSON: #"{"location":"Rome"}"#),
            ]),
            ServeMessage(role: .tool, text: "Paris: 18C", toolCallID: "call_1", toolName: "get_weather"),
            ServeMessage(role: .tool, text: "Rome: 24C", toolCallID: "call_2", toolName: "get_weather"),
        ]
        let assembled = ServeMessageAssembly.assembling(
            history, acceptsTools: ToolCallFormatTable.acceptsTools(forModelType: type),
            oneCallPerTurn: ToolCallHistoryTable.oneCallPerTurn(forModelType: type),
            placement: ToolCallFormatTable.systemPlacement(forModelType: type))
        #expect(assembled == history)                  // nothing to hoist, merge or split
        let out = try template().render([
            "messages": Value(any: MLXChatBackend.templateMessages(assembled)),
            "tools": Value(any: [Self.weatherTool]),
            "add_generation_prompt": true,
        ])
        #expect(out.contains("<tool_call><function=get_weather><parameter=location>Paris</parameter></function></tool_call>"))
        #expect(out.contains("<tool_call><function=get_weather><parameter=location>Rome</parameter></function></tool_call>"))
        #expect(out.contains("<|im_start|>tool\nParis: 18C<|im_end|>"))
        #expect(out.contains("<|im_start|>tool\nRome: 24C<|im_end|>"))
        #expect(!out.contains("<tool_response>"))      // bare `tool` role, unlike Qwen3.5's wrapped results
        #expect(out.components(separatedBy: "<|im_start|>assistant\n").count - 1 == 2)   // the turn, then the generation prompt
        #expect(out.hasSuffix("<|im_start|>assistant\n"))
    }

    /// `.firstOnly` hoists a mid-conversation `system` message to the front. This template accepts one
    /// anywhere, so the hoist is harmless: the render succeeds and the instruction is kept.
    @Test func aMidConversationSystemMessageIsHoistedHarmlessly() throws {
        let type = try modelType()
        let history = [
            ServeMessage(role: .user, text: "hi"),
            ServeMessage(role: .assistant, text: "hello"),
            ServeMessage(role: .system, text: "Answer in French."),
            ServeMessage(role: .user, text: "and now?"),
        ]
        let assembled = ServeMessageAssembly.assembling(
            history, acceptsTools: true, placement: ToolCallFormatTable.systemPlacement(forModelType: type))
        #expect(assembled.first?.role == .system)
        let out = try template().render([
            "messages": Value(any: MLXChatBackend.templateMessages(assembled)),
            "add_generation_prompt": true,
        ])
        #expect(out.hasPrefix("<|im_start|>system\nAnswer in French."))
        #expect(out.contains("and now?"))
    }

    // MARK: - 3. Thinking (a test, not a table split)

    /// The table still says `.promptOpened` (Qwen3.5's), though this template is model-emitted. The
    /// shape value has **no consumer** but `readsEnableThinking`, which is true for both, so the
    /// control reaches the template exactly as intended and no row is split.
    @Test func thinkingControlStillReachesTheTemplate() throws {
        let type = try modelType()
        #expect(ReasoningTemplateTable.readsEnableThinking(forModelType: type))
        #expect(ReasoningTemplateTable.thinkingOn(requested: true, forModelType: type))
        #expect(!ReasoningTemplateTable.thinkingOn(requested: nil, forModelType: type))     // off unless asked (R30)
        #expect(!ReasoningTemplateTable.thinkingOn(requested: false, forModelType: type))
        #expect(ReasoningTemplateTable.thinkShape(forModelType: type) == .promptOpened)      // unchanged, deliberately
    }

    /// Where the opening comes from is decided by the **rendered prompt** (`promptOpensThinkBlock`),
    /// not by the table: MiMo's thinking-on prompt ends at `assistant\n` (the model opens `<think>`), so
    /// the splitter must *not* start inside the block; Qwen3.5's ends `<think>\n`, so it must.
    @Test func thePromptTailDecidesWhetherTheOutputStartsInsideTheBlock() throws {
        let thinkingOn = try template().render([
            "messages": [["role": "user", "content": "Hi"]], "add_generation_prompt": true])
        let thinkingOff = try template().render([
            "messages": [["role": "user", "content": "Hi"]], "add_generation_prompt": true, "enable_thinking": false])
        #expect(!MLXChatBackend.promptOpensThinkBlock(decodedTail: thinkingOn))
        #expect(!MLXChatBackend.promptOpensThinkBlock(decodedTail: thinkingOff))     // pre-closed, nothing to split
        // The contrast: a Qwen3.5-style prompt that opens the block itself.
        #expect(MLXChatBackend.promptOpensThinkBlock(decodedTail: "<|im_start|>assistant\n<think>\n"))
    }

    /// The case the contract asks for: a `qwen3_5` model whose **output** carries the model-emitted
    /// `<think>` block, split into reasoning and answer, byte-exactly, at every chunking. The
    /// splitter's mode comes from the real MiMo prompt, as in `MLXChatBackend`.
    @Test func aModelEmittedThinkBlockIsSplitIntoReasoningAndAnswer() throws {
        let prompt = try template().render(["messages": [["role": "user", "content": "17 times 23?"]], "add_generation_prompt": true])
        let startsInside = MLXChatBackend.promptOpensThinkBlock(decodedTail: prompt)
        #expect(!startsInside)
        let output = "<think>\nNeed 17 * 23: 17*20 = 340, 17*3 = 51.\n</think>\n\n17 × 23 = 391."
        for size in [1, 2, 3, 7, output.count] {
            var splitter = ServeReasoningSplitter(startsInsideThink: startsInside)
            var reasoning = "", answer = ""
            var rest = Substring(output)
            while !rest.isEmpty {
                let piece = rest.prefix(size); rest = rest.dropFirst(size)
                let halves = splitter.push(String(piece))
                reasoning += halves.reasoning; answer += halves.content
            }
            let tail = splitter.finish()
            reasoning += tail.reasoning; answer += tail.content
            #expect(reasoning.trimmingCharacters(in: .whitespacesAndNewlines) == "Need 17 * 23: 17*20 = 340, 17*3 = 51.", "chunk \(size)")
            #expect(answer == "17 × 23 = 391.", "chunk \(size)")
        }
    }

    /// With thinking on, a model that skips the block and just answers is all answer, no mislabelled
    /// reasoning (the other half of why the prompt, not the family, decides).
    @Test func anAnswerWithoutAThinkBlockIsNotMislabelledAsReasoning() {
        var splitter = ServeReasoningSplitter(startsInsideThink: false)
        let a = splitter.push("391.")
        let b = splitter.finish()
        #expect(a.reasoning + b.reasoning == "")
        #expect(a.content + b.content == "391.")
    }

    // MARK: - 4. Memory and context

    /// The window comes from the nested `text_config` (the top level has none): 262,144.
    @Test func theWindowIs262144FromTheTextConfig() throws {
        #expect(ModelContextWindow.limit(configData: try configData) == 262_144)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mimo-ctx-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try configData.write(to: dir.appendingPathComponent("config.json"))
        #expect(ContextWindow.tokens(for: nil, modelDirectory: dir) == 262_144)
    }

    /// 32 layers, 8 full-attention (`layer_types`): K and V grow for 8 × 4 KV heads × head_dim 256 × 2 × 2 B
    /// = 32,768 B/token. The other 24 are linear layers with a fixed state each of
    /// `32·128·128·2` (recurrent) + `(4−1)·(2·16·128 + 32·128)·2` (conv) = 1,048,576 + 49,152 B.
    @Test func theKVProfileComesFromHeadDimKVHeadsAndTheLinearKeys() throws {
        let kv = try #require(ServeMemoryAdmission.kvCost(configData: try configData))
        #expect(kv.perToken == 8 * 4 * 256 * 2 * 2)
        #expect(kv.perToken == 32_768)
        #expect(kv.fixed == 24 * (1_048_576 + 49_152))
        let profile = try #require(ServeMemoryAdmission.profile(
            configData: try configData, weightsBytes: Self.weightsBytes, roomBytes: 11_600_000_000))
        #expect(profile.bytesPerToken == 32_768 * ServeMemoryAdmission.overheadFactor)
        #expect(profile.weightsBytes == Self.weightsBytes)
    }

    /// On a 19 GB-class Mac (about 11.6 GB of shared budget) a short prompt is admitted and a prompt
    /// filling the whole 262k window is refused **before prefill**, instead of freezing the machine
    /// (S3-5); the usable context the clients are told sits strictly between.
    @Test func admissionAdmitsAShortPromptAndRefusesAFullWindowOnASmallMac() throws {
        let profile = try #require(ServeMemoryAdmission.profile(
            configData: try configData, weightsBytes: Self.weightsBytes, roomBytes: 11_600_000_000))
        if case .refuse = ServeMemoryAdmission.decide(profile: profile, promptTokens: 8_192) { Issue.record("8k prompt refused") }
        guard case .refuse = ServeMemoryAdmission.decide(profile: profile, promptTokens: 262_144) else {
            Issue.record("a full 262k prompt should be refused on 11.6 GB of room"); return
        }
        let usable = try #require(ServeMemoryAdmission.usableTokens(profile: profile, contextLength: 262_144))
        #expect(usable > 8_192 && usable < 262_144)
        // A big Mac (no memory limit in sight) is told the model's own window.
        let roomy = try #require(ServeMemoryAdmission.profile(
            configData: try configData, weightsBytes: Self.weightsBytes, roomBytes: 200_000_000_000))
        #expect(ServeMemoryAdmission.usableTokens(profile: roomy, contextLength: 262_144) == 262_144)
    }

    // MARK: - 5. Prompt cache (MM-0 D)

    /// The cache is inherited by `model_type` (`qwen3_5`, on the allowlist since the Qwen3.5-9B pass).
    /// There is **no per-repo exclusion** (owner, MM-0 D); the owner's `PromptCacheRealModelTests` run
    /// with these weights is the gate that closes MM-S.
    @Test func thePromptCacheIsInheritedByModelTypeWithNoRepoExclusion() throws {
        let type = try modelType()
        #expect(PromptCacheAllowlist.tested.contains(type))
        #expect(PromptCachePolicy.bypass(enabled: true, modelType: type, seed: nil, kvBits: nil, maxKVSize: nil) == nil)
        // The only per-request reasons to bypass are the generic ones.
        #expect(PromptCachePolicy.bypass(enabled: true, modelType: type, seed: 1, kvBits: nil, maxKVSize: nil) == .seeded)
    }

    // MARK: - 6. The Connect sheet

    /// What the Connect sheet is generated from, for this model, on a 19 GB-class Mac: tools ✓, thinking ✓,
    /// the model's window and the usable context. Plain facts only (R56): no rating anywhere.
    @Test func theConnectFactsAndSnippetsForThisModel() throws {
        let facts = ServeModelFacts.facts(
            modelID: Self.hfID, configData: try configData, weightsBytes: Self.weightsBytes, roomBytes: 11_600_000_000)
        #expect(facts.supportsTools)
        #expect(facts.thinkingAvailable)
        #expect(facts.contextLength == 262_144)
        let usable = try #require(facts.usableTokens)
        #expect(usable > 8_192 && usable < 262_144)

        let claude = ConnectSnippets.text(for: .claudeCode, port: 1212, model: facts)
        #expect(claude.contains(Self.hfID))
        #expect(claude.contains("CLAUDE_CODE_MAX_CONTEXT_TOKENS=\(usable)"))
        #expect(!claude.contains("doesn't support tool calls"))
        let codex = ConnectSnippets.text(for: .codex, port: 1212, model: facts)
        #expect(codex.contains(Self.hfID))
        #expect(codex.contains("model_context_window = \(usable)"))
        #expect(!codex.contains("doesn't support tool calls"))
        let openCode = ConnectSnippets.text(for: .openCode, port: 1212, model: facts)
        #expect(openCode.contains("tool_call"))
        #expect(openCode.contains("reasoning"))        // a thinking family: capability only (R30)
        for text in [claude, codex, openCode] {
            #expect(!text.lowercased().contains("agent-ready"))
        }
    }
}
