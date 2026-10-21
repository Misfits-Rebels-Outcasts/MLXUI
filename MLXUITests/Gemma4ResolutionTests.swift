import Testing
import Foundation
import Jinja
@testable import MLXUI

/// MP1-3 (`RSI/DelegateModelPackBacklog.md`) — Gemma 4 E4B is SUPPORTED: `gemma4` is registered in
/// mlx-swift-lm 3.31.4's LLM *and* VLM factories, so there is no port. These tests pin the wiring around
/// that: the registry resolution, the absence of a `ModelSupport` flag, the one shared install (the
/// MoC-5 path Qwen3.5 9B uses), the directory both engines load from, and the files the installer fetches.
/// None of them runs the model; that is smoke rows in `RSI/evals/smoke-checklist.md` (MP — Model Pack).
@MainActor
struct Gemma4ResolutionTests {

    private static let llmID = "mlx-community--gemma-4-e4b-it-4bit"
    private static let visionID = "mlx-community--gemma-4-e4b-it-4bit-vision"

    private func loadBrowserData() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
    }

    private func pair(from data: BrowserData) throws -> (llm: ModelEntry, vision: ModelEntry) {
        let catalog = data.domains.flatMap { $0.allModels }
        let llm = try #require(catalog.first { $0.id == Self.llmID })
        let vision = try #require(catalog.first { $0.id == Self.visionID })
        return (llm, vision)
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

    @Test func llmEntryResolvesToChatAndVisionEntryToVLM() throws {
        let (llm, vision) = try pair(from: loadBrowserData())
        let registry = fullRegistry()

        let chat = try #require(registry.bestModule(for: llm))
        #expect(chat.descriptor.id == "chat")
        #expect(chat.sdk.claim(llm) == .exact)

        let vlm = try #require(registry.bestModule(for: vision))
        #expect(vlm.descriptor.id == "vlm")
        #expect(vlm.sdk.claim(vision) == .exact)

        #expect(llm.runnerKind == .llm)
        #expect(vision.runnerKind == .vision)
    }

    @Test func eachEntryIsClaimedByExactlyItsOwnModule() throws {
        let (llm, vision) = try pair(from: loadBrowserData())
        for (entry, owner) in [(llm, "chat"), (vision, "vlm")] {
            // `ModelRegistry.modules` is private, so ask each module on its own: a registry holding only
            // that module resolves the entry iff the module's SDK claims it.
            let claimants = installedModules.filter { module in
                let solo = ModelRegistry()
                module.register(into: solo)
                return solo.bestModule(for: entry) != nil
            }.map { $0.descriptor.id }
            #expect(claimants == [owner], "\(entry.id) is claimed by \(claimants), expected only [\(owner)]")
        }
    }

    @Test func stagesHaveTheRightShapes() throws {
        let (llm, vision) = try pair(from: loadBrowserData())
        let chat = try ChatSDK().makeStage(for: llm, config: .default)
        #expect(chat.accepts == .text && chat.produces == .text)
        #expect(chat.id == llm.id)
        let vlm = try VLMSDK().makeStage(for: vision, config: .default)
        #expect(vlm.accepts == .image && vlm.produces == .text)
        #expect(vlm.id == "vlm.\(vision.id)")
    }

    @Test func neitherEntryIsFlaggedInModelSupport() throws {
        let (llm, vision) = try pair(from: loadBrowserData())
        #expect(ModelSupport.unsupportedReason(for: llm) == nil)
        #expect(ModelSupport.unsupportedReason(for: vision) == nil)
    }

    // MARK: - Shared install (MoC-5)

    @Test func installingOneCardMakesBothReadInstalled() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let (llm, vision) = try pair(from: loadBrowserData())
        #expect(llm.hfModelId == vision.hfModelId)
        #expect(vision.id != ModelStore.repoSlug(for: vision.hfModelId))   // the diverging-id case MoC-5 exists for
        #expect(llm.id == ModelStore.repoSlug(for: llm.hfModelId))
        let manager = InstallManager(store: store)

        #expect(!manager.isInstalled(llm) && !manager.isInstalled(vision))
        try simulateInstalled(llm.hfModelId, in: store)
        #expect(manager.isInstalled(llm))
        #expect(manager.isInstalled(vision), "the vision card shares the llm card's download")
    }

    @Test func installingEitherCardInstallsTheOther() throws {
        // The marker is keyed by repo, so it does not matter which card triggered the download.
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let (llm, vision) = try pair(from: loadBrowserData())
        try simulateInstalled(vision.hfModelId, in: store)   // as if the Vision card had been installed
        let manager = InstallManager(store: store)
        #expect(manager.isInstalled(vision) && manager.isInstalled(llm))
    }

    @Test func bothSurviveARelaunchAndRecordTheRepoPath() throws {
        let (store, cleanup) = makeTempStore()
        defer { cleanup() }
        let data = try loadBrowserData()
        let (llm, vision) = try pair(from: data)
        let catalog = data.domains.flatMap { $0.allModels }
        try simulateInstalled(llm.hfModelId, in: store)

        let verified = InstallManager(store: store).loadInstalled(modelIDs: [llm.id, vision.id], catalog: catalog)
        #expect(verified == [llm.id, vision.id])

        InstallManager(store: store).saveRegistry(installedIDs: [llm.id, vision.id], browserData: data)
        let registry = try JSONDecoder().decode(InstalledModels.self, from: Data(contentsOf: store.installedRegistryURL))
        let slug = ModelStore.repoSlug(for: llm.hfModelId)
        #expect(registry.models[llm.id]?.path == "models/\(slug)")
        #expect(registry.models[vision.id]?.path == "models/\(slug)")
    }

    @Test func bothEnginesLoadFromTheSharedDirectory() throws {
        let (llm, vision) = try pair(from: loadBrowserData())
        // ChatSDK loads `LLMEngine.modelDirectory(for: llm.id)`; VLMSDK loads the directory keyed by hfModelId.
        let chatDir = LLMEngine.modelDirectory(for: llm.id)
        let vlmDir = VLMEngine.installedModelDirectory(hfModelID: vision.hfModelId)
        #expect(chatDir == vlmDir)
        #expect(chatDir.lastPathComponent == "mlx-community--gemma-4-e4b-it-4bit")
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
