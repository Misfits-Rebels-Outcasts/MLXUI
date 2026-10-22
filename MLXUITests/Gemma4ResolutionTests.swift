import Testing
import Foundation
import Jinja
@testable import MLXUI

/// MP1-3 (`RSI/DelegateModelPackBacklog.md`) — Gemma 4 E4B is SUPPORTED as a chat model: `gemma4` is registered in
/// mlx-swift-lm 3.31.4's LLM factory, so there is no port. These tests pin the wiring around that: the registry
/// resolution, the absence of a `ModelSupport` flag, the install, the directory the chat engine loads from, and
/// the files the installer fetches. None of them runs the model; that is smoke rows in
/// `RSI/evals/smoke-checklist.md` (MP — Model Pack). MP1-R-FIX-1 (owner ruling 2026-10-22) withdrew the Vision
/// twin: the VLM factory registers `gemma4` but its `Gemma4.swift` declares k_proj/v_proj on the KV-shared
/// layers 24–41 that the checkpoint does not ship, so the load fails (`RSI/backlog.md`).
@MainActor
struct Gemma4ResolutionTests {

    private static let llmID = "mlx-community--gemma-4-e4b-it-4bit"

    private func loadBrowserData() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
    }

    private func llmEntry(from data: BrowserData) throws -> ModelEntry {
        try #require(data.domains.flatMap { $0.allModels }.first { $0.id == Self.llmID })
    }

    private func fullRegistry() -> ModelRegistry {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return registry
    }

    private func makeTempStore() -> (store: ModelStore, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mp1-3-\(UUID().uuidString)", isDirectory: true)
        return (ModelStore(baseDirectory: dir), { try? FileManager.default.removeItem(at: dir) })
    }

    /// The on-disk result of a completed install, exactly where `downloadModel`'s atomic move puts it.
    private func simulateInstalled(_ hfModelId: String, in store: ModelStore) throws {
        try FileManager.default.createDirectory(at: store.directory(forHFModelID: hfModelId),
                                                withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: store.installedMarker(forHFModelID: hfModelId).path, contents: nil)
    }

    // MARK: - Registry resolution

    @Test func llmEntryResolvesToChat() throws {
        let llm = try llmEntry(from: loadBrowserData())
        let registry = fullRegistry()

        let chat = try #require(registry.bestModule(for: llm))
        #expect(chat.descriptor.id == "chat")
        #expect(chat.sdk.claim(llm) == .exact)
        #expect(llm.runnerKind == .llm)
    }

    @Test func theEntryIsClaimedByExactlyOneModule() throws {
        let llm = try llmEntry(from: loadBrowserData())
        // `ModelRegistry.modules` is private, so ask each module on its own: a registry holding only
        // that module resolves the entry iff the module's SDK claims it.
        let claimants = installedModules.filter { module in
            let solo = ModelRegistry()
            module.register(into: solo)
            return solo.bestModule(for: llm) != nil
        }.map { $0.descriptor.id }
        #expect(claimants == ["chat"], "\(llm.id) is claimed by \(claimants), expected only [chat]")
    }

    @Test func stageHasTheRightShape() throws {
        let llm = try llmEntry(from: loadBrowserData())
        let chat = try ChatSDK().makeStage(for: llm, config: .default)
        #expect(chat.accepts == .text && chat.produces == .text)
        #expect(chat.id == llm.id)
    }

    @Test func theEntryIsNotFlaggedInModelSupport() throws {
        #expect(ModelSupport.unsupportedReason(for: try llmEntry(from: loadBrowserData())) == nil)
    }

    // MARK: - Install

    @Test func theEntrySurvivesARelaunchAndRecordsTheRepoPath() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let data = try loadBrowserData()
        let llm = try llmEntry(from: data)
        let catalog = data.domains.flatMap { $0.allModels }
        #expect(llm.id == ModelStore.repoSlug(for: llm.hfModelId))
        let manager = InstallManager(store: store)

        #expect(!manager.isInstalled(llm))
        try simulateInstalled(llm.hfModelId, in: store)
        #expect(manager.isInstalled(llm))

        let verified = InstallManager(store: store).loadInstalled(modelIDs: [llm.id], catalog: catalog)
        #expect(verified == [llm.id])

        InstallManager(store: store).saveRegistry(installedIDs: [llm.id], browserData: data)
        let registry = try JSONDecoder().decode(InstalledModels.self, from: Data(contentsOf: store.installedRegistryURL))
        #expect(registry.models[llm.id]?.path == "models/\(ModelStore.repoSlug(for: llm.hfModelId))")
    }

    @Test func theChatEngineLoadsFromTheRepoDirectory() throws {
        let llm = try llmEntry(from: loadBrowserData())
        #expect(LLMEngine.modelDirectory(for: llm.id).lastPathComponent == "mlx-community--gemma-4-e4b-it-4bit")
    }

    // MARK: - What the installer fetches

    /// The pinned repo's file list (revision 475b9088d2…, `tree.json`). Every file the two loaders read is
    /// downloaded, and only `.gitattributes` / `README.md` are skipped.
    @Test func installerFetchesEveryFileTheLoadersReadAndSkipsOnlyDocs() {
        let siblings = [".gitattributes", "README.md", "chat_template.jinja", "config.json",
                        "generation_config.json", "model.safetensors", "model.safetensors.index.json",
                        "processor_config.json", "tokenizer.json", "tokenizer_config.json"]
        let picked = Set(ModelFileSelector.filesToDownload(siblings: siblings))
        #expect(picked == Set(siblings).subtracting([".gitattributes", "README.md"]))
        // The VLM processor reads processor_config.json; the tokenizer reads chat_template.jinja.
        #expect(picked.contains("processor_config.json") && picked.contains("chat_template.jinja"))
    }

    // MARK: - The chat template (pinned in Fixtures/Gemma4, no install needed)

    private func pinnedTemplate() throws -> String {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        return try String(contentsOf: root.appendingPathComponent("Fixtures/Gemma4/chat_template.jinja"), encoding: .utf8)
    }

    /// The options swift-transformers compiles every chat template with (`Tokenizer.swift` `compiledTemplate`),
    /// which match HF Python's `trim_blocks`/`lstrip_blocks`. A bare `Template(string)` leaves them off and
    /// adds stray newlines to this template, so the tests must not use the defaults.
    private func compile(_ template: String) throws -> Template {
        try Template(template, with: .init(lstripBlocks: true, trimBlocks: true))
    }

    /// The template as the app renders it: through `ChatTemplateSanitizer` (which applies the `-` on comment
    /// tags swift-jinja ignores, MP1-3-FIX-1) and the swift-transformers options. The Python-reference parity
    /// for this lives in `CommentWhitespaceControlTests`.
    private func render(_ context: [String: Value]) throws -> String {
        let template = try pinnedTemplate()
        return try compile(ChatTemplateSanitizer.sanitized(template) ?? template).render(context)
    }

    @Test func pinnedTemplateIsFixedByTheSanitizerAndRendersAPlainTurn() throws {
        // Ten dashed comments: the sanitizer now has something to apply (it had nothing before MP1-3-FIX-1).
        #expect(ChatTemplateSanitizer.sanitized(try pinnedTemplate()) != nil)
        let out = try render([
            "bos_token": "<bos>",
            "messages": [["role": "user", "content": "Hello"]],
            "add_generation_prompt": true,
        ])
        // No `enable_thinking` in the context → no system turn, no thinking switch in the prompt.
        #expect(out == "<bos><|turn>user\nHello<turn|>\n<|turn>model\n", "got \(out.debugDescription)")
    }

    @Test func pinnedTemplateRendersASystemTurnAndAnImagePlaceholder() throws {
        let out = try render([
            "bos_token": "<bos>",
            "messages": [
                ["role": "system", "content": "Be brief."],
                ["role": "user", "content": [["type": "image"], ["type": "text", "text": "What is this?"]]],
            ],
            "add_generation_prompt": true,
        ])
        #expect(out == "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\n<|image|>What is this?<turn|>\n<|turn>model\n",
                "got \(out.debugDescription)")
    }

    /// Thinking is opt-in via `enable_thinking`; MP1-S decides what MLXUI does with it. Recorded here so the
    /// smoke row's "no thinking tags in the visible text" has a stated cause: the default prompt never asks.
    @Test func thinkingIsOffUnlessTheContextAsksForIt() throws {
        let base: [String: Value] = ["bos_token": "<bos>",
                                     "messages": [["role": "user", "content": "Hi"]],
                                     "add_generation_prompt": true]
        #expect(!(try render(base)).contains("<|think|>"))
        var on = base
        on["enable_thinking"] = true
        #expect(try render(on).contains("<|think|>"))
    }
}
