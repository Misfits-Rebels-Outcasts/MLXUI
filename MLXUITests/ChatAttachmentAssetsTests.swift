//
//  ChatAttachmentAssetsTests.swift
//  MT4-2 (ruling I): attachments are asset handles, with no eager OCR, when Model Tools is on; behaviour is unchanged when
//  it is off. Pure registration and rendering first, then the whole send both ways through `ModelRunner`.
//

import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

// MARK: - pure

@Suite("MT4-2 attachments as assets (pure)")
struct ChatAttachmentAssetsPureTests {

    private func temp() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mt42a-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func attachment(_ name: String, in dir: URL, bytes: Data, mode: ChatAttachmentMode = .reference,
                            inline: String? = nil) throws -> ChatAttachment {
        let file = dir.appendingPathComponent(name)
        try bytes.write(to: file)
        return ChatAttachment(name: name, storedPath: file.path, mode: mode, byteCount: bytes.count, inlineText: inline)
    }

    @Test func aByReferenceTextFileBecomesATextAssetNamedByHandleWithNoPath() throws {
        let dir = try temp(); defer { try? FileManager.default.removeItem(at: dir) }
        let store = ChatAssetStore(conversationID: UUID(), in: dir.appendingPathComponent("conversations"))
        let body = String(repeating: "alpha beta gamma\n", count: 8_000)              // ~136 KB
        let a = try attachment("rates.csv", in: dir, bytes: Data(body.utf8))
        let registered = ChatAttachmentAssets.register(a, in: store)
        #expect(registered.assetHandle == "a1" && registered.assetKind == "text")
        #expect(store.text(for: "a1") == body)
        #expect(store.record("a1")?.words == 24_000)
        let facing = ChatAttachment.modelFacing(registered)
        #expect(facing == "<attachment name=\"rates.csv\" asset=\"a1\" kind=\"text\" mode=\"reference\"/>")
        #expect(!facing.contains(dir.path))                                           // no path: nothing to read_file whole
        let note = try #require(ChatAttachment.assetNote(for: [registered], assets: store))
        #expect(note.contains("asset a1 (text, 24000 words)") && note.contains("read_asset") && note.contains("a page at a time"))
        #expect(ChatAttachment.environmentNote(for: [registered]) == nil)            // the old read_file line is not also given
        #expect(registered.badgeLabel == "asset")
    }

    @Test func aBigAttachmentIsReadAPageAtATimeNotWhole() async throws {
        let dir = try temp(); defer { try? FileManager.default.removeItem(at: dir) }
        let store = ChatAssetStore(conversationID: UUID(), in: dir.appendingPathComponent("conversations"))
        let body = String(repeating: "x", count: 2_200_000)                           // the 2.2 MB case: ~550k tokens whole
        let a = ChatAttachmentAssets.register(try attachment("big.csv", in: dir, bytes: Data(body.utf8)), in: store)
        let host = ModelToolHost(assets: store)
        let page = await ModelToolHost.$current.withValue(host) {
            try? await ModelTool(spec: ModelToolSpec.spec(named: "read_asset")!, advertisesAssets: true)
                .execute(arguments: ["asset": .string(try! #require(a.assetHandle))])
        }
        let text = try #require(page)
        #expect(text.contains("\"length\":4000") && text.contains("\"next\":4000") && text.contains("\"total\":2200000"))
        #expect(text.count < 4_300)                                                   // one page (~1,000 tokens), not 65k
    }

    @Test func imagesAndAudioAreRegisteredWithoutOCRAndKeepTheirExtension() throws {
        let dir = try temp(); defer { try? FileManager.default.removeItem(at: dir) }
        let store = ChatAssetStore(conversationID: UUID(), in: dir.appendingPathComponent("conversations"))
        let image = ChatAttachmentAssets.register(try attachment("receipt.PNG", in: dir, bytes: Data([1, 2, 3])), in: store)
        let audio = ChatAttachmentAssets.register(try attachment("memo.m4a", in: dir, bytes: Data([4, 5, 6])), in: store)
        #expect(image.assetKind == "image" && image.assetHandle == "a1")
        #expect(audio.assetKind == "audio" && audio.assetHandle == "a2")
        #expect(store.record("a1")?.fileName == "a1.png" && store.record("a2")?.fileName == "a2.m4a")
        #expect(ChatAttachment.modelFacing(image) == "<attachment name=\"receipt.PNG\" asset=\"a1\" kind=\"image\" mode=\"reference\"/>")
        let note = try #require(ChatAttachment.assetNote(for: [image, audio], assets: store))
        #expect(note.contains("Attached image receipt.PNG is asset a1 (image)"))
        #expect(note.contains("Attached audio memo.m4a is asset a2 (audio)") && note.contains("transcribe_audio"))
        // The "no OCR installed" note is for a CW image that went by reference; a registered image is not that.
        #expect(!image.isUnreadImage)
        #expect(ChatAttachment.environmentNote(for: [image]) == nil)
    }

    @Test func aFileThatIsNeitherMediaNorTextIsNotRegistered() throws {
        let dir = try temp(); defer { try? FileManager.default.removeItem(at: dir) }
        let store = ChatAssetStore(conversationID: UUID(), in: dir.appendingPathComponent("conversations"))
        let zip = try attachment("a.zip", in: dir, bytes: Data([0x50, 0x4B, 0, 3, 0, 0]))
        let result = ChatAttachmentAssets.register(zip, in: store)
        #expect(result == zip && result.assetHandle == nil)
        #expect(store.all.isEmpty)
        #expect(ChatAttachment.modelFacing(result).contains("path="))                 // keeps its CW form
    }

    @Test func anInlineAttachmentKeepsItsTextAndGainsItsHandle() throws {
        let dir = try temp(); defer { try? FileManager.default.removeItem(at: dir) }
        let store = ChatAssetStore(conversationID: UUID(), in: dir.appendingPathComponent("conversations"))
        let a = try attachment("n.txt", in: dir, bytes: Data("hello".utf8), mode: .inline, inline: "hello")
        let registered = ChatAttachmentAssets.register(a, in: store)
        #expect(ChatAttachment.modelFacing(registered) == "<attachment name=\"n.txt\" asset=\"a1\">\nhello\n</attachment>")
        #expect(ChatAttachment.assetNote(for: [registered], assets: store) == nil)    // already in the message
        #expect(registered.badgeLabel == "inline")
    }

    @Test func withoutAHandleTheCWRenderingIsByteForByteTheSame() {
        let ref = ChatAttachment(name: "big.txt", storedPath: "/c/1-big.txt", mode: .reference, byteCount: 9)
        #expect(ChatAttachment.modelFacing(ref) == "<attachment name=\"big.txt\" path=\"/c/1-big.txt\" mode=\"reference\"/>")
        let inline = ChatAttachment(name: "a.txt", storedPath: "/c/0-a.txt", mode: .inline, byteCount: 1, inlineText: "x")
        #expect(ChatAttachment.modelFacing(inline) == "<attachment name=\"a.txt\">\nx\n</attachment>")
        #expect(ChatAttachment.assetNote(for: [ref, inline], assets: nil) == nil)
    }

    @Test func aConversationSavedBeforeAssetsStillDecodes() throws {
        let old = Data(#"{"id":"6F9619FF-8B86-D011-B42D-00C04FC964FF","name":"a.txt","storedPath":"/c/a.txt","mode":"reference","byteCount":3}"#.utf8)
        let decoded = try JSONDecoder().decode(ChatAttachment.self, from: old)
        #expect(decoded.assetHandle == nil && decoded.assetKind == nil)
        let registered = decoded.registered(handle: "a4", kind: "text")
        let again = try JSONDecoder().decode(ChatAttachment.self, from: JSONEncoder().encode(registered))
        #expect(again == registered && again.assetHandle == "a4")
        let plain = try JSONEncoder().encode(decoded)
        #expect(!String(decoding: plain, as: UTF8.self).contains("asset"))            // nothing new is written for a CW attachment
    }

    @Test func theEnvironmentBlockKeepsAssetHandlesEvenWithNoFolderAndNoFileTool() {
        let empty = FileScope(folders: [], workingFolder: nil)
        let none = ChatEnvironment.instructions(scope: empty, edition: .appStore, fileToolsEnabled: [], writeEnabled: false)
        #expect(none == nil)
        let withNote = ChatEnvironment.instructions(scope: empty, edition: .appStore, fileToolsEnabled: [], writeEnabled: false,
                                                    assetNote: "Attached image r.png is asset a1 (image).")
        #expect(withNote == "Attached image r.png is asset a1 (image).")
        // The CW line is still dropped when there is nothing else to say (unchanged).
        let cw = ChatEnvironment.instructions(scope: empty, edition: .appStore, fileToolsEnabled: [], writeEnabled: false,
                                              attachmentsNote: "Attached file x is at /p; read it with read_file.")
        #expect(cw == nil)
    }
}

// MARK: - the whole send, both ways

@MainActor
@Suite(.serialized)
struct ChatAttachmentAssetsSendTests {

    final class OneShot: ChatStreamingSession, @unchecked Sendable {
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            AsyncThrowingStream { $0.yield("reply"); $0.finish() }
        }
    }

    final class Box<T: Sendable>: @unchecked Sendable {
        private let lock = NSLock()
        private var value: T
        init(_ value: T) { self.value = value }
        var get: T { lock.withLock { value } }
        func set(_ new: T) { lock.withLock { value = new } }
    }

    private func llm() -> ModelEntry {
        ModelEntry(
            id: "mlx-community--assets", family: "Q", displayName: "Qwen3-8B", paramSize: "", paramCountB: nil, modelType: .llm, source: .mlx,
            format: "mlx", platforms: ["macos"], minMacOSVersion: nil, hfRepo: "", hfModelId: "", ramGB: 4, downloadSizeGB: 1,
            contextWindow: 32_000, license: nil, licenseUrl: nil, description: nil, summary: nil, descriptionSource: nil,
            architecture: nil, languages: nil, lastUpdated: nil, taskTags: nil, benchmarks: nil, speedTokensPerSec: nil,
            speedHardware: nil, speedEstimated: nil, communityDownloads: nil, communityLikes: nil, variants: nil)
    }

    private func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    /// A runtime with the GLM OCR model and Parakeet installed, as the real app would report.
    private func runtime() throws -> ModelToolRuntime {
        let cat = try catalog()
        let ids = Set(try ["mlx-community/GLM-OCR-4bit", "mlx-community/parakeet-tdt-0.6b-v3"].map { name in
            try #require(cat.first { $0.hfModelId == name }, "\(name)").id })
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(cat.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        return ModelToolRuntime(catalog: cat, installedModelIDs: ids, claimableModelIDs: claimable, totalRAMGB: 19,
                                makeExecutor: { _, _ in MT13Executor.text("x") })
    }

    private struct Rig {
        let runner: ModelRunner
        let store: URL
        let files: URL
        let turn: Box<RouterTurn?>
        let ocrCalls: Box<Int>
        let dirs: [URL]
    }

    private func rig(toolsOn: Bool) throws -> Rig {
        let fm = FileManager.default
        let store = fm.temporaryDirectory.appendingPathComponent("mt42s-\(UUID().uuidString)")
        let modelDir = fm.temporaryDirectory.appendingPathComponent("mt42s-m-\(UUID().uuidString)")
        let files = fm.temporaryDirectory.appendingPathComponent("mt42s-f-\(UUID().uuidString)")
        for d in [store, modelDir, files] { try fm.createDirectory(at: d, withIntermediateDirectories: true) }
        let runner = ModelRunner()
        let model = llm()
        runner.prepare(for: model, in: store)
        runner.modelDirectoryForTesting = { _ in modelDir }
        runner.streamingSessionForTesting = { OneShot() }
        runner.usableTokensForTesting = { _ in 32_000 }
        let turn = Box<RouterTurn?>(nil)
        runner.routerTurnForTesting = { turn.set($0) }
        let ocrCalls = Box(0)
        runner.attachmentOCRForTesting = { _ in ocrCalls.set(ocrCalls.get + 1); return "TOTAL 42.00" }
        let runtime = try runtime()
        runner.modelToolRuntimeProvider = { runtime }
        runner.modelToolsEnabled = toolsOn
        return Rig(runner: runner, store: store, files: files, turn: turn, ocrCalls: ocrCalls, dirs: [store, modelDir, files])
    }

    private func settle(_ r: ModelRunner) async {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while r.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }

    private func attachments(in dir: URL) throws -> [URL] {
        let big = dir.appendingPathComponent("rates.csv")
        try Data(String(repeating: "alpha,beta,gamma\n", count: 6_000).utf8).write(to: big)          // ~100 KB: by reference
        let png = dir.appendingPathComponent("receipt.png")
        try Data([0x89, 0x50, 0x4E, 0x47, 0]).write(to: png)
        let memo = dir.appendingPathComponent("memo.m4a")
        try Data([0, 1, 2, 3]).write(to: memo)               // a NUL byte, as in a real audio file: not text
        return [png, big, memo]
    }

    @Test func modelToolsOnRegistersAssetsListsHandlesSkipsOCRAndOffersTheModelTools() async throws {
        let r = try rig(toolsOn: true); defer { r.dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        let model = llm()
        #expect(await r.runner.send("what's the total?", attachments: try attachments(in: r.files), for: model))
        await settle(r.runner)

        #expect(r.ocrCalls.get == 0)                                                   // no eager OCR
        let turn = try #require(r.turn.get)
        let note = try #require(turn.instructions)
        #expect(note.contains("Attached image receipt.png is asset a1 (image)"))
        #expect(note.contains("Attached file rates.csv is asset a2 (text, 6000 words") && note.contains("read_asset"))
        #expect(note.contains("Attached audio memo.m4a is asset a3 (audio)"))
        #expect(!note.contains("is at "))                                              // no read_file path line
        #expect(turn.prompt.contains("<attachment name=\"receipt.png\" asset=\"a1\" kind=\"image\" mode=\"reference\"/>"))
        #expect(!turn.prompt.contains("path="))

        let names = turn.tools.map(\.name)
        #expect(names.first == "list_capabilities")
        #expect(names.contains("ocr_image") && names.contains("transcribe_audio") && names.contains("read_asset"))
        let ocr = try #require(turn.tools.first { $0.name == "ocr_image" })
        #expect(ocr.parameters.map(\.name).contains("asset"))                          // advertises handles in a host-bound chat
        let host = try #require(turn.toolContext?.modelToolHost)
        #expect(host.assets.all.map(\.handle) == ["a1", "a2", "a3"])
        #expect(host.assets.all.map(\.kind) == ["image", "text", "audio"])
        #expect(turn.toolContext?.modelToolRuntime != nil)

        let conversation = try #require(r.runner.conversations.first { $0.id == r.runner.activeConversationID })
        let sent = conversation.items.compactMap { item -> [ChatAttachment]? in
            if case .message(let m) = item, m.role == .user { return m.attachments } else { return nil }
        }.first ?? []
        #expect(sent.map(\.assetHandle) == ["a1", "a2", "a3"])
        let assets = ConversationStore.folderURL(for: conversation.id, in: r.store).appendingPathComponent("assets")
        #expect(FileManager.default.fileExists(atPath: assets.appendingPathComponent("a1.png").path))
    }

    @Test func modelToolsOffIsExactlyTheCWBehaviour() async throws {
        let r = try rig(toolsOn: false); defer { r.dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        let model = llm()
        #expect(await r.runner.send("what's the total?", attachments: try attachments(in: r.files), for: model))
        await settle(r.runner)

        #expect(r.ocrCalls.get == 1)                                                   // the image was OCR'd eagerly, as before
        let turn = try #require(r.turn.get)
        #expect(!(turn.instructions ?? "").contains("is asset"))
        #expect(!turn.prompt.contains("asset=\""))
        #expect(turn.prompt.contains("TOTAL 42.00"))                                   // the OCR text is in the message
        #expect(turn.prompt.contains("path="))                                         // the by-reference file keeps its path
        #expect(turn.toolContext?.modelToolHost == nil)
        // Exactly today's armed set: every enabled tool, in registry order, none of the model tools.
        let expected = r.runner.availableTools.filter { r.runner.enabledToolNames.contains($0.name) }.map(\.name)
        #expect(turn.tools.map(\.name) == expected)
        #expect(!turn.tools.map(\.name).contains("list_capabilities"))
        let conversation = try #require(r.runner.conversations.first { $0.id == r.runner.activeConversationID })
        let assets = ConversationStore.folderURL(for: conversation.id, in: r.store).appendingPathComponent("assets")
        #expect(!FileManager.default.fileExists(atPath: assets.path))                  // nothing registered
        let sent = conversation.items.compactMap { item -> [ChatAttachment]? in
            if case .message(let m) = item, m.role == .user { return m.attachments } else { return nil }
        }.first ?? []
        #expect(sent.allSatisfy { $0.assetHandle == nil })
    }

    @Test func aTurnWithNoAttachmentsStillOffersListCapabilitiesWhenOn() async throws {
        let r = try rig(toolsOn: true); defer { r.dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        #expect(await r.runner.send("hello", for: llm()))
        await settle(r.runner)
        let turn = try #require(r.turn.get)
        #expect(turn.tools.first?.name == "list_capabilities")
        #expect(!turn.tools.map(\.name).contains("read_asset"))                        // no oversized text asset
        #expect(turn.instructions == nil)                                              // nothing to say: plain chat stays plain
    }

    @Test func theChatPageTurnsModelToolsOnAndTheBenchLeavesThemOff() {
        #expect(ChatController.modelToolsEnabled(for: .page))
        #expect(!ChatController.modelToolsEnabled(for: .bench))
    }

    @Test func theControllerSetsTheRunnerFlagFromTheSurface() async throws {
        let r = try rig(toolsOn: false); defer { r.dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        ChatGate.overrideForTesting = true; defer { ChatGate.overrideForTesting = nil }
        let controller = ChatController(runner: r.runner, conversationDirectory: r.store)
        let model = llm()
        controller.acquire(.page, model: model)
        #expect(await controller.send("hello", from: .page))
        await settle(r.runner)
        #expect(r.runner.modelToolsEnabled)
        #expect(r.turn.get?.toolContext?.modelToolHost != nil)
        controller.acquire(.bench, model: model)
        #expect(await controller.send("hello again", from: .bench))
        await settle(r.runner)
        #expect(!r.runner.modelToolsEnabled)
        #expect(r.turn.get?.toolContext?.modelToolHost == nil)
    }

    @Test func aFileToolStaysOfferedToMLXWhenFoldersAreAttachedAndModelToolsFillTheBudget() throws {
        // The MLX budget is 10 and the user has ~16 tools: with a folder attached the file tools must not be the ones cut.
        let all: [any AgentTool] = ModelRunner.defaultTools()
        let enabled = all.filter { ModelRunner.defaultEnabledToolNames().contains($0.name) }
        let rt = try runtime()
        let sel = ChatToolSet.select(enabled: enabled, ChatToolSet.Request(
            router: .mlx(hasVision: false), modelToolsEnabled: true, gateAmbiguousTools: false, prompt: "look at my files",
            runtime: rt, assets: nil, inlineBudgetTokens: 8_000, foldersAttached: true))
        let names = sel.tools.map(\.name)
        #expect(names.count <= ToolExposurePolicy.mlxToolBudget)
        #expect(names.first == "list_capabilities")
        #expect(names.contains("read_file") && names.contains("list_directory") && names.contains("search_files"))
    }
}
