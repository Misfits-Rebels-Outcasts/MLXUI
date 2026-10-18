import Foundation
import Jinja
import MLXLMCommon
import Testing
@testable import MLXUI

/// MM3 — "MiMo V2.6 9B" is SUPPORTED: nothing here ports anything, it only proves the pieces the
/// real load will meet. The checkpoint's three small files are pinned verbatim in `Fixtures/MiMo/`
/// (`PROVENANCE.md`; `mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit` @ `60061fc`), so none of
/// this needs the 7.1 GB install. The real per-layer 4/8-bit weight load is owner-run:
/// `MiMoRealLoadTests`.
@MainActor
struct MiMoRunPathTests {

    private static let id = "mlx-community--MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit"

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
    }

    private func fixture(_ name: String) throws -> Data {
        try Data(contentsOf: repoRoot.appendingPathComponent("Fixtures/MiMo/\(name)"))
    }

    private func json(_ name: String) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: fixture(name)) as? [String: Any])
    }

    private func catalogEntry() throws -> ModelEntry {
        let state = AppState()
        state.loadBrowserData()
        let data = try #require(state.browserData)
        return try #require(data.domains.flatMap { $0.allModels }.first { $0.id == Self.id })
    }

    // MARK: - Resolution

    /// `bestModule(for:)` → `ChatModule` (id "chat"), exactly, the same as Qwen3.5-9B.
    @Test func resolvesToTheChatModule() throws {
        let entry = try catalogEntry()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let resolved = try #require(registry.bestModule(for: entry))
        #expect(resolved.descriptor.id == "chat")
        #expect(resolved.sdk.claim(entry) == .exact)
    }

    @Test func routesToTheLLMRunnerWithNoSupportGap() throws {
        let entry = try catalogEntry()
        #expect(entry.runnerKind == .llm)
        #expect(ModelSupport.unsupportedReason(for: entry) == nil)
    }

    @Test func makeStageProducesATextToTextStage() throws {
        let stage = try ChatSDK().makeStage(for: try catalogEntry(), config: .default)
        #expect(stage.accepts == .text)
        #expect(stage.produces == .text)
    }

    // MARK: - ModelFileSelector

    /// The repo's file list at `60061fc` (names from the HF tree). The selector takes everything
    /// the load needs. It does **not** take `optiq/metadata.json`: the file is not a recognised
    /// metadata name (the per-layer bit map the runtime uses is already in `config.json`), so the
    /// install skips it. That is fine: nothing in mlx-swift-lm reads it (MM3 records which way
    /// this goes; changing the selector for it would be scope creep).
    @Test func fileSelectorTakesTheShardsAndSkipsOptiQMetadata() {
        let siblings = [
            ".gitattributes", "README.md", "chat_template.jinja", "config.json",
            "generation_config.json",
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            "model.safetensors.index.json", "optiq/metadata.json",
            "tokenizer.json", "tokenizer_config.json",
        ]
        let out = Set(ModelFileSelector.filesToDownload(siblings: siblings))
        #expect(out == [
            "chat_template.jinja", "config.json", "generation_config.json",
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
        ])
        #expect(!out.contains("optiq/metadata.json"))
    }

    // MARK: - OptiQ per-layer quantization (the first such catalog entry)

    private func baseConfiguration() throws -> BaseConfiguration {
        try JSONDecoder().decode(BaseConfiguration.self, from: fixture("config.json"))
    }

    /// `BaseConfiguration` reads the 250 per-layer overrides from `config.json`'s `quantization`
    /// block: 4-bit/group 64 default, 134 layers at 8-bit, 116 at 4-bit (README: 134 / 116).
    @Test func configDecodesIntoThe250PerLayerOverrides() throws {
        let config = try baseConfiguration()
        #expect(config.modelType == "qwen3_5")
        let per = try #require(config.perLayerQuantization)
        #expect(per.quantization?.groupSize == 64)
        #expect(per.quantization?.bits == 4)
        #expect(per.perLayerQuantization.count == 250)
        var bits: [Int: Int] = [:]
        for option in per.perLayerQuantization.values {
            guard case .quantize(let q) = option else {
                Issue.record("an override is `.skip`; OptiQ lists only quantized layers")
                continue
            }
            #expect(q.groupSize == 64)
            bits[q.bits, default: 0] += 1
        }
        #expect(bits == [8: 134, 4: 116])
        #expect(per.quantization(layer: "language_model.model.embed_tokens")?.bits == 8)
        #expect(per.quantization(layer: "language_model.lm_head")?.bits == 8)
        // A layer the map does not name falls back to the default (and has no `.scales` anyway).
        #expect(per.quantization(layer: "language_model.model.norm")?.bits == 4)
    }

    /// `loadWeights` quantizes a module only where `weights["<path>.scales"]` exists, then asks the
    /// per-layer map for `<path>`. So the override keys must equal the `.scales` tensor prefixes,
    /// and `Qwen35Model.sanitize` must leave those paths unchanged. Its rewrite is a no-op for keys
    /// that already start `language_model.` (it only prefixes bare ones and drops `vision_tower`,
    /// `model.visual`, `mtp.`). Statically, with no weights: the two sets are identical (250 = 250).
    @Test func overrideKeysAreExactlyTheScalesTensorsAndNeedNoRemapping() throws {
        let per = try #require(try baseConfiguration().perLayerQuantization)
        let weightMap = try #require(try json("model.safetensors.index.json")["weight_map"] as? [String: String])

        let scaled = Set(weightMap.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(".scales".count)) })
        #expect(scaled == Set(per.perLayerQuantization.keys))
        #expect(scaled.count == 250)

        // All keys are already in the form `Qwen35Model.sanitize` leaves alone.
        #expect(weightMap.keys.allSatisfy { $0.hasPrefix("language_model.") })
        #expect(!weightMap.keys.contains { $0.contains("mtp.") || $0.contains("vision_tower") || $0.contains("visual") })

        // Untied head: a real, quantized `lm_head` ships and is not dropped by sanitize.
        #expect(try json("config.json")["tie_word_embeddings"] as? Bool == false)
        #expect(weightMap.keys.contains("language_model.lm_head.weight"))
        #expect(weightMap.keys.contains("language_model.lm_head.scales"))
    }

    /// `total_size` in the index is the figure behind the catalog's 7.10 GB.
    @Test func indexTotalSizeIsTheCatalogDownloadSize() throws {
        let meta = try #require(try json("model.safetensors.index.json")["metadata"] as? [String: Any])
        let total = try #require(meta["total_size"] as? Int)
        #expect(total == 7_100_589_056)
        #expect((Double(total) / 1e9 * 100).rounded() / 100 == (try catalogEntry()).downloadSizeGB)
    }

    // MARK: - The MiMo chat template through swift-jinja

    private func template() throws -> String {
        try #require(String(data: fixture("chat_template.jinja"), encoding: .utf8))
    }

    private static let weatherTool: Value = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": [
                "type": "object",
                "properties": ["location": ["type": "string"]],
                "required": ["location"],
            ],
        ],
    ]

    /// Tools + `enable_thinking: false` (what a flow row and a no-thinking request send): renders
    /// without throwing, tools go in a leading system block, and the prompt closes an empty think
    /// block. `{%- generation -%}` mask tags, `is iterable`, `tojson(ensure_ascii=False)` all parse.
    @Test func rendersWithToolsAndThinkingOff() throws {
        let out = try Template(template()).render([
            "messages": [
                ["role": "system", "content": "Be brief."],
                ["role": "user", "content": "Weather in Paris?"],
            ],
            "tools": [Self.weatherTool],
            "add_generation_prompt": true,
            "enable_thinking": false,
        ])
        #expect(out.hasPrefix("<|im_start|>system\nYou are provided with the following tools:\n\n<tools>\n"))
        #expect(out.contains("get_weather"))
        #expect(out.contains("<|im_start|>system\nBe brief.<|im_end|>"))
        #expect(out.hasSuffix("<|im_start|>user\nWeather in Paris?<|im_end|><|im_start|>assistant\n<think></think>"))
    }

    /// Thinking on (no flag): the prompt ends at `assistant\n` and does **not** write `<think>`:
    /// the model opens it (`.modelEmitted`). Pinned so the Qwen3.5 `.promptOpened` table row is
    /// known to differ for this template (MM-S covers the splitter).
    @Test func thinkingOnLeavesTheOpeningToTheModel() throws {
        let out = try Template(template()).render([
            "messages": [["role": "user", "content": "Hi"]],
            "add_generation_prompt": true,
        ])
        #expect(out.hasSuffix("<|im_start|>user\nHi<|im_end|><|im_start|>assistant\n"))
        #expect(!out.contains("<think>"))
    }

    /// No `raise_exception`: `system` may appear anywhere, and the strict Qwen3.5 "no user query"
    /// restart (`ChatSession` re-rendering with only the tool result) does not throw.
    @Test func systemAnywhereAndAToolOnlyRestartDoNotThrow() throws {
        let midSystem = try Template(template()).render([
            "messages": [
                ["role": "user", "content": "a"],
                ["role": "assistant", "content": "b"],
                ["role": "system", "content": "later system"],
                ["role": "user", "content": "c"],
            ],
            "add_generation_prompt": true,
        ])
        #expect(midSystem.contains("<|im_start|>system\nlater system<|im_end|>"))

        let restart = try Template(template()).render([
            "messages": [["role": "tool", "content": "17797"]],
            "add_generation_prompt": true,
        ])
        #expect(restart.contains("<|im_start|>tool\n17797<|im_end|>"))
        #expect(restart.hasSuffix("<|im_start|>assistant\n"))
    }

    /// A two-call turn: every call is rendered (xml-function form), reasoning is kept on the
    /// assistant turn, and each result is a bare `tool` role, with no `<tool_response>` wrapper.
    @Test func aTwoCallToolLoopRendersEveryCallAndBareToolResults() throws {
        let out = try Template(template()).render([
            "messages": [
                ["role": "user", "content": "Weather in Paris and Rome?"],
                [
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Need both cities.",
                    "tool_calls": [
                        ["function": ["name": "get_weather", "arguments": ["location": "Paris"]]],
                        ["function": ["name": "get_weather", "arguments": ["location": "Rome"]]],
                    ],
                ],
                ["role": "tool", "content": "Paris: 18C"],
                ["role": "tool", "content": "Rome: 24C"],
            ],
            "add_generation_prompt": true,
        ])
        #expect(out.contains("<|im_start|>assistant\n<think>Need both cities.</think>"))
        #expect(out.contains("<tool_call><function=get_weather><parameter=location>Paris</parameter></function></tool_call>"))
        #expect(out.contains("<tool_call><function=get_weather><parameter=location>Rome</parameter></function></tool_call>"))
        #expect(out.contains("<|im_start|>tool\nParis: 18C<|im_end|>"))
        #expect(out.contains("<|im_start|>tool\nRome: 24C<|im_end|>"))
        #expect(!out.contains("<tool_response>"))
    }

    /// `ChatTemplateSanitizer` has nothing to remove from this template (no strict guard), so
    /// `HFTokenizerLoader` hands the checkpoint's own template to the tokenizer untouched.
    @Test func theSanitizerIsNotNeeded() throws {
        #expect(ChatTemplateSanitizer.sanitized(try template()) == nil)
    }

    /// The installed model's folder when its `tokenizer.json` is present here, else nil.
    nonisolated static func installedModelDirectory() -> URL? {
        let dir = ModelStore.shared.directory(forModelID: id)   // `ModelStore` is nonisolated
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path) ? dir : nil
    }

    /// The full app path (`HFTokenizerLoader` → `applyChatTemplate` with tools and
    /// `LLMEngine.noThinkingContext`) when the model is installed here. `.enabled(if:)` (MM3
    /// follow-up, owner 2026-10-19) makes the run report "skipped" instead of passing silently.
    @Test(.enabled(if: installedModelDirectory() != nil))
    func theInstalledTokenizerRendersWithToolsAndThinkingOff() async throws {
        let dir = try #require(Self.installedModelDirectory())
        let tokenizer = try await HFTokenizerLoader().load(from: dir)
        let messages: [[String: any Sendable]] = [["role": "user", "content": "Weather in Paris?"]]
        let tool: [String: any Sendable] = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": ["type": "object", "properties": ["location": ["type": "string"]], "required": ["location"]],
            ] as [String: any Sendable],
        ]
        let ids = try tokenizer.applyChatTemplate(
            messages: messages, tools: [tool], additionalContext: LLMEngine.noThinkingContext)
        let text = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
        #expect(text.contains("get_weather"))
        #expect(text.hasSuffix("<|im_start|>assistant\n<think></think>"))
    }
}
