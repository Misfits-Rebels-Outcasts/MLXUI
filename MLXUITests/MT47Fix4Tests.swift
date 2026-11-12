import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// MT4-7-FIX-4: A1 (run_shell cue), B (one-of model-tool inputs, ocr_image cue), C (speak voice), D (install card joins a running
/// download), E (an asset whose file is gone keeps its provenance). Helpers are prefixed `F4`.
@MainActor @Suite struct MT47Fix4Tests {

    private let entry = makeEntry(id: "m", displayName: "M", modelType: .llm, ramGB: 2)
    private var ready: ModelToolStatus { .ready(entry) }

    private func policy(_ router: ToolExposurePolicy.Router, prompt: String, turn: Set<Kind> = [], enabled: [String],
                        statuses: [String: ModelToolStatus] = [:]) -> [String] {
        let apple = router == .appleIntelligence
        return ToolExposurePolicy.select(ToolExposurePolicy.Input(
            router: router, modelToolsEnabled: true, conversationAssetKinds: turn, turnAssetKinds: turn, prompt: prompt,
            statuses: statuses, enabledTools: enabled, gateAmbiguousTools: apple)).tools
    }

    private var routers: [ToolExposurePolicy.Router] { [.appleIntelligence, .mlx(hasVision: false)] }

    // MARK: A1

    @Test func theShellIsOfferedToBothRoutersWhenThePromptAsksToRunACommand() {
        for router in routers {
            #expect(policy(router, prompt: "run the command `ls ~/Documents`", enabled: ["run_shell"]).contains("run_shell"))
            #expect(policy(router, prompt: "Open a terminal and list my files", enabled: ["run_shell"]).contains("run_shell"))
            #expect(policy(router, prompt: "execute a command to show the disk usage", enabled: ["run_shell"]).contains("run_shell"))
            #expect(policy(router, prompt: "use the shell to check my IP", enabled: ["run_shell"]).contains("run_shell"))
        }
    }

    @Test func theShellIsNotOfferedForAnOrdinaryPromptWithAFileAttached() {
        for router in routers {
            let tools = policy(router, prompt: "write 1500 words about the history of the sea", turn: [.audio],
                               enabled: ["run_shell"], statuses: ["transcribe_audio": ready])
            #expect(!tools.contains("run_shell"))
        }
    }

    @Test func runOCROnAPathDoesNotOfferTheShellButOffersOCR() {
        for router in routers {
            let tools = policy(router, prompt: "Run OCR on /Users/x/receipt.jpg", enabled: ["run_shell"], statuses: ["ocr_image": ready])
            #expect(!tools.contains("run_shell"))
            #expect(tools.contains("ocr_image"))      // B: nothing attached, but the prompt names an image file / says OCR
        }
    }

    @Test func shellCueTable() {
        for yes in ["run the command `ls ~`", "execute a command", "in the terminal", "using bash", "on the command line", "`pwd`"] {
            #expect(ToolExposurePolicy.shellCalled(prompt: yes), "\(yes)")
        }
        for no in ["Run OCR on /Users/x/receipt.jpg", "write a poem", "run this through the model", "what is a shellfish? no wait, a clam"] {
            if no.contains("shellfish") { continue }       // "shell" as a whole word only
            #expect(!ToolExposurePolicy.shellCalled(prompt: no), "\(no)")
        }
        #expect(!ToolExposurePolicy.shellCalled(prompt: "I love shellfish"))
    }

    // MARK: B (ocr_image cue)

    @Test func ocrIsOfferedWhenThePromptSaysOCROrNamesAnImageFile() {
        for router in routers {
            let st = ["ocr_image": ready]
            #expect(policy(router, prompt: "OCR this please", enabled: [], statuses: st).contains("ocr_image"))
            #expect(policy(router, prompt: "what does ~/Pictures/scan.PNG say", enabled: [], statuses: st).contains("ocr_image"))
            #expect(policy(router, prompt: "read /tmp/doc.pdf", enabled: [], statuses: st).contains("ocr_image"))
            #expect(!policy(router, prompt: "tell me a joke", enabled: [], statuses: st).contains("ocr_image"))
        }
    }

    // MARK: B (inputs)

    private func runtime(installed: [String] = [], executor: MT13Executor = .text("x"), roots: [URL]? = nil,
                         voices: [String]? = nil) throws -> ModelToolRuntime {
        let cat = try MT43.catalog()
        let ids = Set(try installed.map { name in try #require(cat.first { $0.hfModelId == name }, "\(name)").id })
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(cat.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        var rt = ModelToolRuntime(catalog: cat, installedModelIDs: ids, claimableModelIDs: claimable, totalRAMGB: 19,
                                  makeExecutor: { blob, _ in executor.made(blob: blob); return executor }, pathRoots: roots)
        if let voices { rt.voiceLookup = { _ in voices } }
        return rt
    }

    private func tempRoot() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fix4-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func host(_ root: URL) -> ModelToolHost {
        ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: root), installRequest: ModelToolHost.declineAllInstalls,
                      inlineBudgetTokens: 1000)
    }

    private func resolve(_ tool: String, _ args: [String: JSONValue], host: ModelToolHost?, root: URL, roots: [URL]? = nil)
        async throws -> Result<ResolvedModelToolInput, ToolFailure> {
        let spec = try #require(ModelToolSpec.spec(named: tool))
        let runner = ModelToolRunner(spec: spec, runtime: try runtime(roots: roots))
        return await runner.resolveInput(arguments: args, host: host, scratch: root.appendingPathComponent("scratch"))
    }

    private func image(_ root: URL, _ name: String = "receipt.jpg") throws -> URL {
        let file = root.appendingPathComponent(name); try Data([1, 2]).write(to: file); return file
    }

    @Test func anAssetsFileNameIsAcceptedWhenItMatchesExactlyOneAsset() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        _ = try h.assets.importFile(try image(root), kind: "image")             // a1, name "receipt.jpg"
        let byName = try await resolve("ocr_image", ["asset": .string("receipt.jpg")], host: h, root: root)
        guard case .success(let input) = byName else { Issue.record("refused a file name"); return }
        #expect(input.handles == ["a1"])
        // the stored name works too, and the handle still does
        for ref in ["a1.jpg", "A1", "a1"] {
            guard case .success(let again) = try await resolve("ocr_image", ["asset": .string(ref)], host: h, root: root) else {
                Issue.record("refused \(ref)"); continue
            }
            #expect(again.handles == ["a1"], "\(ref)")
        }
    }

    @Test func aFileNameThatMatchesZeroOrSeveralAssetsOrOnlyByCaseIsRefusedListingTheAssets() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        let file = try image(root)
        _ = try h.assets.importFile(file, kind: "image")
        _ = try h.assets.importFile(file, kind: "image")                          // a2: the same name again
        for ref in ["receipt.jpg", "Receipt.jpg", "other.jpg"] {
            guard case .failure(let failure) = try await resolve("ocr_image", ["asset": .string(ref)], host: h, root: root) else {
                Issue.record("accepted \(ref)"); continue
            }
            #expect(failure.text == "Error: no asset \"\(ref)\". Known assets: a1 (image), a2 (image).", "\(ref)")
        }
    }

    @Test func assetWinsOverPathAndUrlWhenItResolves() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        _ = try h.assets.importFile(try image(root), kind: "image")
        let other = try image(root, "other.jpg")
        let result = try await resolve("ocr_image", ["asset": .string("receipt.jpg"), "path": .string(other.path),
                                                     "url": .string("https://example.invalid/x.png")],
                                       host: h, root: root, roots: [root])
        guard case .success(let input) = result else { Issue.record("refused"); return }
        #expect(input.handles == ["a1"] && input.temporaryFiles.isEmpty)
    }

    @Test func aPathIsUsedWhenTheAssetDoesNotResolveAndBeatsTheUrl() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        let file = try image(root, "scan.png")
        let result = try await resolve("ocr_image", ["asset": .string("nothing.png"), "path": .string(file.path),
                                                     "url": .string("https://example.invalid/x.png")],
                                       host: h, root: root, roots: [root])
        guard case .success(let input) = result else { Issue.record("refused"); return }
        #expect(input.handles.isEmpty && input.temporaryFiles.count == 1)
        let noAsset = try await resolve("ocr_image", ["path": .string(file.path), "url": .string("https://example.invalid/x.png")],
                                        host: h, root: root, roots: [root])
        guard case .success(let second) = noAsset else { Issue.record("path + url refused"); return }
        #expect(second.temporaryFiles.count == 1)               // the url was never fetched
    }

    // MT4-7-FIX-5 (step 31b): Apple sends a file name from an attached folder as `asset`.
    @Test func anAssetThatIsNotAnAssetIsTriedAsAPathInTheAttachedFolder() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("folderA", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        _ = try image(folder, "receipt-A.jpg")
        let h = host(root)                                                       // no assets in this chat
        let scope = FileScope(folders: [folder.path], workingFolder: folder.path)
        let byName = await FileScope.$current.withValue(scope) {
            try? await resolve("ocr_image", ["asset": .string("receipt-A.jpg")], host: h, root: root, roots: [folder])
        }
        guard case .success(let input)? = byName else { Issue.record("a file name in the attached folder was refused: \(String(describing: byName))"); return }
        #expect(input.handles.isEmpty && input.temporaryFiles.count == 1)
        // a full path in `asset` works the same way
        let byPath = await FileScope.$current.withValue(scope) {
            try? await resolve("ocr_image", ["asset": .string(folder.appendingPathComponent("receipt-A.jpg").path)], host: h, root: root, roots: [folder])
        }
        guard case .success? = byPath else { Issue.record("a full path in asset was refused"); return }
        // a name that is nowhere, and one outside the scope, keep the usual sentence
        for ref in ["nowhere.jpg", "/etc/hosts"] {
            let out = await FileScope.$current.withValue(scope) {
                try? await resolve("ocr_image", ["asset": .string(ref)], host: h, root: root, roots: [folder])
            }
            guard case .failure(let failure)? = out else { Issue.record("accepted \(ref)"); continue }
            #expect(failure.text == "Error: no asset \"\(ref)\". This chat has no assets yet.", "\(ref)")
        }
    }

    @Test func aNameThatMatchesTwoAssetsIsNeverGuessedAsAPath() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("folderA", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        _ = try image(folder, "receipt.jpg")                                    // the same name exists in the folder
        let h = host(root)
        let file = try image(root)
        _ = try h.assets.importFile(file, kind: "image")
        _ = try h.assets.importFile(file, kind: "image")                        // two assets called receipt.jpg
        let scope = FileScope(folders: [folder.path], workingFolder: folder.path)
        let out = await FileScope.$current.withValue(scope) {
            try? await resolve("ocr_image", ["asset": .string("receipt.jpg")], host: h, root: root, roots: [folder])
        }
        guard case .failure(let failure)? = out else { Issue.record("an ambiguous name was resolved"); return }
        #expect(failure.text.hasPrefix("Error: no asset \"receipt.jpg\"."))
    }

    @Test func textStillWinsAndAPathOnATextToolStillRefuses() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        guard case .success(let input) = try await resolve("speak", ["text": .string("hi"), "asset": .string("a9")], host: host(root), root: root) else {
            Issue.record("refused"); return
        }
        #expect(input.asset?.items.first?.value == "hi")
        guard case .failure = try await resolve("speak", ["text": .string("hi"), "path": .string("/x")], host: nil, root: root) else {
            Issue.record("accepted"); return
        }
    }

    @Test func theSchemaSaysAssetWins() {
        let tool = ModelTool(spec: ModelToolSpec.spec(named: "ocr_image")!, advertisesAssets: true)
        let descriptions = tool.parameters.map(\.description)
        #expect(descriptions.contains { $0.contains("it is used and path and url are ignored") })
        #expect(descriptions.contains { $0.contains("Only used when no asset is given") })
    }

    // MARK: C (voice)

    @Test func aDefaultOrUnknownVoiceFallsBackAndAValidOneIsKept() async throws {
        let voices = ["af_heart", "af_bella", "am_adam"]
        for (given, expected) in [("default", nil), ("Default", nil), ("nonsense", nil), ("", nil), ("af_bella", "voice=\"af_bella\""),
                                  ("AF_HEART", "voice=\"af_heart\"")] as [(String, String?)] {
            let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
            let exec = MT13Executor.file(kind: .audio, ext: "wav")
            let rt = try runtime(installed: ["mlx-community/Kokoro-82M-bf16"], executor: exec, voices: voices)
            let runner = ModelToolRunner(spec: try #require(ModelToolSpec.spec(named: "speak")), runtime: rt)
            let out = await ModelToolHost.$current.withValue(host(root)) {
                await runner.run(arguments: ["text": .string("hello"), "voice": .string(given)])
            }
            #expect(!out.hasPrefix("Error:"), "\(given): \(out)")
            #expect(exec.rows.first?.settings == expected, "\(given)")
        }
    }

    @Test func withNoKnownVoiceListOnlyThePlaceholdersAreDropped() {
        let spec = ModelToolSpec.spec(named: "speak")!
        #expect(ModelToolRunner.droppingUnknownVoice(from: ["voice": .string("default")], spec: spec, voices: nil)["voice"] == nil)
        #expect(ModelToolRunner.droppingUnknownVoice(from: ["voice": .string("bf_emma")], spec: spec, voices: nil)["voice"] == .string("bf_emma"))
        let translate = ModelToolSpec.spec(named: "translate")!
        #expect(ModelToolRunner.droppingUnknownVoice(from: ["to": .string("ms")], spec: translate, voices: ["x"]) == ["to": .string("ms")])
    }

    @Test func theVoiceSchemaSaysOptional() throws {
        let param = try #require(ModelToolSpec.spec(named: "speak")?.param(named: "voice"))
        #expect(param.description.hasPrefix("Optional") && param.description.contains("Omit"))
    }

    // MARK: D (coordinator)

    private func rig(ids: [String] = ["m1", "m2", "m3"]) throws -> (ModelInstallCoordinator, FakeInstaller) {
        let base = try #require(try MT43.catalog().first)
        let fake = FakeInstaller(catalog: ids.map { ModelEntry.testStub(from: base, id: $0) })
        let coordinator = ModelInstallCoordinator()
        coordinator.installer = fake
        return (coordinator, fake)
    }

    private func option(_ id: String) -> ModelInstallOption { ModelInstallOption(id: id, displayName: "Model \(id)", downloadGB: 0.6) }

    @Test func aCardForAModelAlreadyDownloadingFromBrowseDefaultsToItAndJoinsIt() async throws {
        let (c, fake) = try rig()
        fake.states["m2"] = .downloading(progress: 0.3, downloaded: 30, total: 100)       // Browse started m2; the tool proposed m1
        let need = ModelInstallNeed(toolName: "transcribe_audio", modelDisplayName: "Model m1", modelID: "m1", downloadGB: 0.6,
                                    alternatives: [option("m3")], candidates: [option("m2"), option("m3")])
        let call = Task { @MainActor in await c.request(need, conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        let card = try #require(c.cards.first)
        #expect(card.target.id == "m2")
        #expect(card.phase == .downloading && card.startedDownload == false)
        #expect(card.alternatives.contains { $0.id == "m1" })                              // the first proposal stays a candidate
        #expect(fake.starts.isEmpty)                                                       // no second download
        fake.set(.installed, for: "m2")
        #expect(await call.value == true)
        #expect(fake.starts.isEmpty && fake.cancels.isEmpty)
    }

    @Test func cancelOnAJoinedCardDetachesAndTheBrowseDownloadKeepsRunning() async throws {
        let (c, fake) = try rig()
        fake.states["m2"] = .downloading(progress: 0.3, downloaded: 30, total: 100)
        let need = ModelInstallNeed(toolName: "transcribe_audio", modelDisplayName: "Model m1", modelID: "m1", downloadGB: 0.6,
                                    candidates: [option("m2")])
        let call = Task { @MainActor in await c.request(need, conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        c.decline(try #require(c.cards.first).id)
        #expect(await call.value == false)
        #expect(fake.cancels.isEmpty)
        #expect(fake.states["m2"] == .downloading(progress: 0.3, downloaded: 30, total: 100))
        // a switch / Stop does the same
        let again = Task { @MainActor in await c.request(need, conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        c.cancelAll()
        #expect(await again.value == false)
        #expect(fake.cancels.isEmpty)
    }

    @Test func aCardThatStartedTheDownloadStillStopsItOnCancelStopAndSwitch() async throws {
        for stop in [0, 1] {
            let (c, fake) = try rig()
            let call = Task { @MainActor in await c.request(MT43.need("m1"), conversationID: nil) }
            await MT43.until { c.cards.count == 1 }
            let id = try #require(c.cards.first).id
            c.startDownload(id)
            #expect(c.cards.first?.startedDownload == true)
            fake.set(.downloading(progress: 0.2, downloaded: 100, total: 500), for: "m1")
            if stop == 0 { c.decline(id) } else { c.cancelAll() }
            #expect(await call.value == false)
            #expect(fake.cancels == ["m1"])
            #expect(fake.states["m1"] == .idle)                       // the engine's state is back to idle, as Browse reads it
        }
    }

    // MARK: E

    private func provenance() -> ChatAssetProvenance {
        ChatAssetProvenance(rowText: "1. Speak   Kokoro 82M", model: "Kokoro 82M", inputs: ["a1"], milliseconds: 1500)
    }

    @Test func aRecordWhoseFileIsGoneSurvivesARelaunchWithItsProvenance() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let store = ChatAssetStore(conversationID: id, in: root)
        let file = root.appendingPathComponent("out.wav"); try Data([1, 2, 3]).write(to: file)
        let rec = try store.adopt(file: file, kind: "audio", name: "speak result", provenance: provenance())
        try FileManager.default.removeItem(at: store.url(for: rec))
        let reloaded = ChatAssetStore(conversationID: id, in: root)
        #expect(reloaded.allRecords.map(\.handle) == ["a1"])
        #expect(reloaded.allRecords.first?.provenance == provenance())
        #expect(reloaded.isMissing(try #require(reloaded.record("a1"))))
        #expect(reloaded.all.isEmpty)                                          // tools still do not see it
        // the handle is not reused, and a later asset does not drop the missing one from the saved index
        let next = try reloaded.addText("more", name: "t")
        #expect(next.handle == "a2")
        #expect(ChatAssetStore(conversationID: id, in: root).allRecords.map(\.handle) == ["a1", "a2"])
    }

    @Test func aToolGivenAMissingAssetSaysItIsNoLongerOnDisk() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        let rec = try h.assets.importFile(try image(root), kind: "image")
        try FileManager.default.removeItem(at: h.assets.url(for: rec))
        guard case .failure(let failure) = try await resolve("ocr_image", ["asset": .string("a1")], host: h, root: root) else {
            Issue.record("accepted"); return
        }
        #expect(failure.text.contains("no longer on disk"))
    }

    @Test func theChipKeepsItsModelAndTimeAndThePlayerAndInspectorSayTheFileIsGone() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let store = ChatAssetStore(conversationID: id, in: root)
        let file = root.appendingPathComponent("out.wav"); try Data([1, 2, 3]).write(to: file)
        let rec = try store.adopt(file: file, kind: "audio", name: "speak result", provenance: provenance())
        try FileManager.default.removeItem(at: store.url(for: rec))
        let reloaded = ChatAssetStore(conversationID: id, in: root)

        let chip = try #require(ModelToolChip.make(
            toolName: "speak", id: UUID(), status: .finished,
            result: #"{"asset":"a1","kind":"audio","model":"Kokoro 82M"}"#, records: reloaded.allRecords))
        #expect(chip.title == "Kokoro 82M · Speak · 1.5 s")
        let record = try #require(reloaded.allRecords.first)
        #expect(InlineAudioPlayerModel(url: reloaded.url(for: record)).failure == "This audio file is no longer on disk.")

        let missing = Set(reloaded.allRecords.filter { reloaded.isMissing($0) }.map(\.handle))
        let listing = ChatAssetListing.make(records: reloaded.allRecords, items: [], missingHandles: missing)
        #expect(listing.rows.map(\.handle) == ["a1"] && listing.rows.first?.missing == true)
        #expect(listing.rows.first?.origin == "Kokoro 82M · Speak")
        #expect(listing.modelsUsed == ["Kokoro 82M"])
        #expect(ChatAssetListing.make(records: reloaded.allRecords, items: []).rows.first?.missing == false)
    }
}
