import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

// MT4-3: install and continue (plan §3.8). Three layers: the card's state machine against a fake installer, the runner's
// resume / decline logic through a host, and the real `ModelRunner.send` path (the host the app builds each turn).

// MARK: - Fixtures

/// An install engine that does what the test says. Nothing downloads.
@MainActor
final class FakeInstaller: ModelInstalling {
    let catalog: [ModelEntry]
    var states: [String: InstallState] = [:]
    private(set) var starts: [String] = []
    private(set) var cancels: [String] = []
    private(set) var finished: [String] = []
    var disk: Double = 500
    private var waiters: [String: [CheckedContinuation<Void, Never>]] = [:]

    init(catalog: [ModelEntry]) { self.catalog = catalog }

    func entry(forModelID id: String) -> ModelEntry? { catalog.first { $0.id == id } }
    func state(of entry: ModelEntry) -> InstallState { states[entry.id] ?? .idle }
    func start(_ entry: ModelEntry) { starts.append(entry.id); set(.resolving, for: entry.id) }
    func cancel(_ entry: ModelEntry) { cancels.append(entry.id); set(.idle, for: entry.id) }
    func availableDiskGB() -> Double { disk }
    func installFinished(_ entry: ModelEntry) { finished.append(entry.id) }
    func changed(_ entry: ModelEntry) async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in waiters[entry.id, default: []].append(c) }
    }
    func set(_ state: InstallState, for id: String) {
        states[id] = state
        for c in waiters.removeValue(forKey: id) ?? [] { c.resume() }
    }
}

nonisolated final class MT43Box<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: T
    init(_ value: T) { self.value = value }
    var get: T { lock.withLock { value } }
    func mutate(_ body: (inout T) -> Void) { lock.withLock { body(&value) } }
}

@MainActor
enum MT43 {
    static func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    static func runtime(installed hf: [String], ram: Double = 19, executor: MT13Executor = .text("x")) throws -> ModelToolRuntime {
        let cat = try catalog()
        let ids = Set(try hf.map { name in try #require(cat.first { $0.hfModelId == name }, "\(name)").id })
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(cat.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        return ModelToolRuntime(catalog: cat, installedModelIDs: ids, claimableModelIDs: claimable, totalRAMGB: ram,
                                makeExecutor: { blob, _ in executor.made(blob: blob); return executor })
    }

    static let parakeet = "mlx-community/parakeet-tdt-0.6b-v3"

    static func need(_ id: String, gb: Double = 0.6, tool: String = "transcribe_audio",
                     alternatives: [ModelInstallOption] = []) -> ModelInstallNeed {
        ModelInstallNeed(toolName: tool, modelDisplayName: "Model \(id)", modelID: id, downloadGB: gb, alternatives: alternatives)
    }

    /// Lets the main actor run its queued work until `condition` holds.
    static func until(_ condition: () -> Bool) async {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !condition(), ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(5)) }
    }
}

// MARK: - The card's state machine

@MainActor @Suite struct ModelInstallCoordinatorTests {

    private func rig(ids: [String] = ["m1", "m2", "m3"]) throws -> (ModelInstallCoordinator, FakeInstaller) {
        let base = try #require(try MT43.catalog().first)
        let entries = ids.map { ModelEntry.testStub(from: base, id: $0) }
        let fake = FakeInstaller(catalog: entries)
        let coordinator = ModelInstallCoordinator()
        coordinator.installer = fake
        return (coordinator, fake)
    }

    @Test func downloadAndContinueResumesTheCallWhenTheMarkerIsThere() async throws {
        let (c, fake) = try rig()
        let result = MT43Box<Bool?>(nil)
        let call = Task { @MainActor in result.mutate { $0 = nil }; let r = await c.request(MT43.need("m1"), conversationID: nil); result.mutate { $0 = r } }
        await MT43.until { c.cards.count == 1 }
        let card = try #require(c.cards.first)
        #expect(card.phase == .offering)
        #expect(fake.starts.isEmpty)                         // a card alone downloads nothing
        c.startDownload(card.id)
        #expect(c.cards.first?.phase == .downloading)
        #expect(fake.starts == ["m1"])
        fake.set(.downloading(progress: 0.5, downloaded: 300, total: 600), for: "m1")
        await Task.yield()
        #expect(result.get == nil && c.cards.first?.phase == .downloading)
        fake.set(.installed, for: "m1")
        await call.value
        #expect(result.get == true)
        #expect(fake.finished == ["m1"])
        #expect(c.cards.first?.phase == .installed)
        c.toolResumed("transcribe_audio")
        #expect(c.cards.isEmpty)
    }

    @Test func cancelAnswersFalseAndStartsNothing() async throws {
        let (c, fake) = try rig()
        let call = Task { @MainActor in await c.request(MT43.need("m1"), conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        c.decline(try #require(c.cards.first).id)
        #expect(await call.value == false)
        #expect(c.cards.isEmpty && fake.starts.isEmpty && fake.cancels.isEmpty)
    }

    @Test func cancelMidDownloadStopsTheDownloadAndAnswersFalse() async throws {
        let (c, fake) = try rig()
        let call = Task { @MainActor in await c.request(MT43.need("m1"), conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        let id = try #require(c.cards.first).id
        c.startDownload(id)
        fake.set(.downloading(progress: 0.2, downloaded: 100, total: 500), for: "m1")
        c.decline(id)
        #expect(await call.value == false)
        #expect(fake.cancels == ["m1"])
        #expect(c.cards.isEmpty)
        await Task.yield()
        #expect(fake.states["m1"] == .idle)                  // the watcher did not turn the cancel into a failure card
        #expect(c.cards.isEmpty)
    }

    @Test func aFailedDownloadOffersRetryAndTheSecondAttemptContinues() async throws {
        let (c, fake) = try rig()
        let call = Task { @MainActor in await c.request(MT43.need("m1"), conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        let id = try #require(c.cards.first).id
        c.startDownload(id)
        fake.set(.error("The network connection was lost.", canRetry: true), for: "m1")
        await MT43.until { c.cards.first?.phase == .failed("The network connection was lost.") }
        #expect(c.cards.first?.phase == .failed("The network connection was lost."))
        c.startDownload(id)                                  // Retry
        #expect(c.cards.first?.phase == .downloading)
        #expect(fake.starts == ["m1", "m1"])
        fake.set(.installed, for: "m1")
        #expect(await call.value == true)
    }

    @Test func aGatedModelShowsTheTokenCardAndWaitsForTheUser() async throws {
        let (c, fake) = try rig()
        let call = Task { @MainActor in await c.request(MT43.need("m1"), conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        let id = try #require(c.cards.first).id
        c.startDownload(id)
        fake.set(.needsAuth("This model is gated. Add a HuggingFace token in Settings, then retry."), for: "m1")
        await MT43.until { if case .gated = c.cards.first?.phase { return true } else { return false } }
        guard case .gated = c.cards.first?.phase else { Issue.record("not gated: \(String(describing: c.cards.first?.phase))"); return }
        #expect(fake.starts == ["m1"])                       // no second attempt on its own
        c.decline(id)
        #expect(await call.value == false)
    }

    @Test func twoCallsForTheSameModelShareOneCardAndOneInstall() async throws {
        let (c, fake) = try rig()
        let a = Task { @MainActor in await c.request(MT43.need("m1"), conversationID: nil) }
        let b = Task { @MainActor in await c.request(MT43.need("m1"), conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        await Task.yield(); await Task.yield()
        #expect(c.cards.count == 1)
        c.startDownload(try #require(c.cards.first).id)
        fake.set(.installed, for: "m1")
        #expect(await a.value == true)
        #expect(await b.value == true)
        #expect(fake.starts == ["m1"])
        // A third call after the install asks nothing.
        #expect(await c.request(MT43.need("m1"), conversationID: nil) == true)
        #expect(fake.starts == ["m1"])
    }

    @Test func aDownloadAlreadyRunningElsewhereIsJoinedNotRestartedOrCancelledByUs() async throws {
        let (c, fake) = try rig()
        fake.states["m1"] = .downloading(progress: 0.1, downloaded: 10, total: 100)      // started from Browse
        let call = Task { @MainActor in await c.request(MT43.need("m1"), conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        let id = try #require(c.cards.first).id
        c.startDownload(id)
        #expect(c.cards.first?.startedDownload == false)
        c.decline(id)                                        // our Cancel does not kill the user's own download
        #expect(await call.value == false)
        #expect(fake.cancels.isEmpty)
    }

    @Test func stopAnswersEveryOpenCardAsCancelAndStopsOurDownloads() async throws {
        let (c, fake) = try rig()
        let a = Task { @MainActor in await c.request(MT43.need("m1"), conversationID: nil) }
        let b = Task { @MainActor in await c.request(MT43.need("m2", tool: "ocr_image"), conversationID: nil) }
        await MT43.until { c.cards.count == 2 }
        c.startDownload(try #require(c.cards.first { $0.target.id == "m2" }).id)
        c.cancelAll()
        #expect(await a.value == false)
        #expect(await b.value == false)
        #expect(c.cards.isEmpty)
        #expect(fake.cancels == ["m2"])                      // only the one that was downloading
    }

    @Test func cancellingTheCallsTaskLeavesNoCardAndNoHang() async throws {
        let (c, _) = try rig()
        let call = Task { @MainActor in await c.request(MT43.need("m1"), conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        call.cancel()
        #expect(await call.value == false)
        await MT43.until { c.cards.isEmpty }
        #expect(c.cards.isEmpty)
    }

    @Test func noEngineWiredMeansDeclined() async throws {
        let c = ModelInstallCoordinator()
        #expect(await c.request(MT43.need("m1"), conversationID: nil) == false)
        #expect(c.cards.isEmpty)
    }

    @Test func aModelAlreadyOnDiskNeedsNoCard() async throws {
        let (c, fake) = try rig()
        fake.states["m1"] = .installed
        #expect(await c.request(MT43.need("m1"), conversationID: nil) == true)
        #expect(c.cards.isEmpty && fake.starts.isEmpty)
    }

    @Test func noRoomOffersNoDownloadButAFittingAlternative() async throws {
        let (c, fake) = try rig()
        fake.disk = 4                                        // 4 GB free: an 8 GB model (18 GB with headroom) cannot go, 0.5 GB (3 GB) can
        let big = MT43.need("m1", gb: 8, alternatives: [
            ModelInstallOption(id: "m2", displayName: "Small", downloadGB: 0.5),
            ModelInstallOption(id: "m3", displayName: "Also big", downloadGB: 9)])
        let call = Task { @MainActor in await c.request(big, conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        let card = try #require(c.cards.first)
        guard case .wontFit(let reason) = card.phase else { Issue.record("phase \(card.phase)"); return }
        #expect(reason.hasPrefix("Not enough disk space"))
        #expect(card.alternatives.map(\.id) == ["m2"])       // only the one that fits
        c.startDownload(card.id)
        #expect(fake.starts.isEmpty)                         // no download button
        c.showChoices(card.id)
        c.choose(card.id, option: "m2")
        #expect(c.cards.first?.target.id == "m2")
        #expect(c.cards.first?.phase == .offering)
        c.startDownload(card.id)
        #expect(fake.starts == ["m2"])
        fake.set(.installed, for: "m2")
        #expect(await call.value == true)
    }

    @Test func chooseAnotherSwapsTheModelAndKeepsTheFirstAsACandidate() async throws {
        let (c, fake) = try rig()
        let need = MT43.need("m1", alternatives: [ModelInstallOption(id: "m2", displayName: "Other", downloadGB: 1.2)])
        let call = Task { @MainActor in await c.request(need, conversationID: nil) }
        await MT43.until { c.cards.count == 1 }
        let id = try #require(c.cards.first).id
        c.showChoices(id)
        #expect(c.cards.first?.phase == .choosing)
        c.hideChoices(id)
        #expect(c.cards.first?.phase == .offering)
        c.showChoices(id)
        c.choose(id, option: "m2")
        #expect(c.cards.first?.target.id == "m2")
        #expect(c.cards.first?.alternatives.map(\.id) == ["m1"])
        c.startDownload(id)
        #expect(fake.starts == ["m2"])
        fake.set(.installed, for: "m2")
        #expect(await call.value == true)
    }

    @Test func theCardsWordsArePlainFacts() {
        let card = ModelInstallCard(conversationID: nil, toolName: "transcribe_audio",
                                    target: ModelInstallOption(id: "x", displayName: "Parakeet TDT 0.6B v3", downloadGB: 0.6),
                                    alternatives: [], phase: .offering)
        #expect(card.headline == "Transcribing needs Parakeet TDT 0.6B v3")
        #expect(card.detail == "0.6 GB download · runs on this Mac only")
        #expect(ModelInstallCard.progressText(downloaded: 400_000_000, total: 600_000_000) == "67% · 0.4 GB of 0.6 GB")
    }
}

extension ModelEntry {
    /// A copy of `base` under another id (the fake catalog's entries).
    static func testStub(from base: ModelEntry, id: String) -> ModelEntry {
        ModelEntry(
            id: id, family: base.family, displayName: "Model \(id)", paramSize: base.paramSize, paramCountB: base.paramCountB,
            modelType: base.modelType, source: base.source, format: base.format, platforms: base.platforms,
            minMacOSVersion: base.minMacOSVersion, hfRepo: base.hfRepo, hfModelId: id, ramGB: 1, downloadSizeGB: 0.6,
            contextWindow: base.contextWindow, license: nil, licenseUrl: nil, description: nil, summary: nil,
            descriptionSource: nil, architecture: nil, languages: nil, lastUpdated: nil, taskTags: nil, benchmarks: nil,
            speedTokensPerSec: nil, speedHardware: nil, speedEstimated: nil, communityDownloads: nil, communityLikes: nil,
            variants: nil)
    }
}

// MARK: - The runner: ask, resume, decline

@MainActor @Suite struct ModelInstallRunnerTests {

    private func tempRoot() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mt43-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func run(_ tool: String, _ args: [String: JSONValue], runtime rt: ModelToolRuntime, host: ModelToolHost) async throws -> String {
        let runner = ModelToolRunner(spec: try #require(ModelToolSpec.spec(named: tool)), runtime: rt)
        return await ModelToolHost.$current.withValue(host) { await runner.run(arguments: args) }
    }

    private func audio(in root: URL, host: ModelToolHost) throws {
        let file = root.appendingPathComponent("clip.wav"); try Data([0, 1]).write(to: file)
        _ = try host.assets.importFile(file, kind: "audio")
    }

    @Test func anInstalledModelResumesTheSameCallWithTheSameArguments() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let exec = MT13Executor.text("hello world")
        let before = try MT43.runtime(installed: [], executor: exec)
        let after = try MT43.runtime(installed: [MT43.parakeet], executor: exec)
        let asked = MT43Box<[ModelInstallNeed]>([])
        let host = ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: root),
                                 installRequest: { need in asked.mutate { $0.append(need) }; return true },
                                 refreshRuntime: { after })
        try audio(in: root, host: host)
        let out = try await run("transcribe_audio", ["asset": .string("a1"), "language": .string("en")], runtime: before, host: host)
        #expect(asked.get.count == 1)
        #expect(asked.get.first?.toolName == "transcribe_audio")
        #expect(exec.rows.count == 1)
        #expect(exec.rows.first?.model == "Parakeet TDT 0.6B v3")
        #expect(FlowSettings(exec.rows.first?.settings).value(for: "lang") == "en")            // the same arguments
        #expect(out.contains("\"text\":\"hello world\"") && out.contains("\"model\":\"Parakeet TDT 0.6B v3\""))
    }

    @Test func aDeclineDropsThatToolForTheRestOfTheTurnAndOnlyThatTool() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let exec = MT13Executor.text("never")
        let rt = try MT43.runtime(installed: [], executor: exec)
        let asked = MT43Box<[String]>([])
        let host = ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: root),
                                 installRequest: { need in asked.mutate { $0.append(need.toolName) }; return false })
        try audio(in: root, host: host)
        let first = try await run("transcribe_audio", ["asset": .string("a1")], runtime: rt, host: host)
        let second = try await run("transcribe_audio", ["asset": .string("a1")], runtime: rt, host: host)
        #expect(first.hasPrefix("The user chose not to download ") && first.hasSuffix("Answer without it, or say what it would need."))
        #expect(second == first)
        #expect(asked.get == ["transcribe_audio"])                                  // the second call asked nothing
        _ = try await run("speak", ["text": .string("hi")], runtime: rt, host: host)
        #expect(asked.get == ["transcribe_audio", "speak"])                         // another tool still asks
        #expect(exec.rows.isEmpty)
        // A new turn (a new host) asks again.
        let next = ModelToolHost(assets: host.assets, installRequest: { need in asked.mutate { $0.append(need.toolName) }; return false })
        _ = try await run("transcribe_audio", ["asset": .string("a1")], runtime: rt, host: next)
        #expect(asked.get.count == 3)
    }

    @Test func anInstallThatDidNotTakeEndsWithOneSentenceAndNoSecondAsk() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let exec = MT13Executor.text("never")
        let rt = try MT43.runtime(installed: [], executor: exec)
        let asked = MT43Box<Int>(0)
        let host = ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: root),
                                 installRequest: { _ in asked.mutate { $0 += 1 }; return true },
                                 refreshRuntime: { rt })                            // still not installed
        try audio(in: root, host: host)
        let out = try await run("transcribe_audio", ["asset": .string("a1")], runtime: rt, host: host)
        #expect(out.hasPrefix("Error: ") && out.contains("did not finish installing"))
        #expect(asked.get == 1 && exec.rows.isEmpty)
    }

    @Test func theNeedOffersFittingNotInstalledAlternativesOnly() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let rt = try MT43.runtime(installed: [], ram: 19)
        let asked = MT43Box<[ModelInstallNeed]>([])
        let host = ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: root),
                                 installRequest: { need in asked.mutate { $0.append(need) }; return false })
        let png = root.appendingPathComponent("p.png"); try Data([0x89, 0x50]).write(to: png)
        _ = try host.assets.importFile(png, kind: "image")
        _ = try await run("ocr_image", ["asset": .string("a1")], runtime: rt, host: host)
        let need = try #require(asked.get.first)
        #expect(!need.alternatives.isEmpty, "OCR has more than one candidate in the catalog")
        #expect(need.alternatives.count <= 4)
        for option in need.alternatives {
            let entry = try #require(rt.catalog.first { $0.id == option.id })
            #expect(option.id != need.modelID)
            #expect(CapabilityResolver.fits(entry, totalRAMGB: 19))
            #expect(!rt.installedModelIDs.contains(option.id))
        }
    }

    @Test func throughTheCardEndToEndInstallThenContinue() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let exec = MT13Executor.text("six words came out of it")
        let before = try MT43.runtime(installed: [], executor: exec)
        let after = try MT43.runtime(installed: [MT43.parakeet], executor: exec)
        let fake = FakeInstaller(catalog: before.catalog)
        let coordinator = ModelInstallCoordinator()
        coordinator.installer = fake
        let host = ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: root),
                                 installRequest: { need in await coordinator.request(need, conversationID: nil) },
                                 progress: { update in
                                     if update.phase == .running { Task { @MainActor in coordinator.toolResumed(update.toolName) } }
                                 },
                                 refreshRuntime: { after })
        try audio(in: root, host: host)
        let call = Task { @MainActor in
            try await self.run("transcribe_audio", ["asset": .string("a1")], runtime: before, host: host)
        }
        await MT43.until { coordinator.cards.count == 1 }
        let card = try #require(coordinator.cards.first)
        #expect(card.headline.hasPrefix("Transcribing needs "))
        #expect(exec.rows.isEmpty && fake.starts.isEmpty)                           // waiting; nothing ran, nothing downloaded
        coordinator.startDownload(card.id)
        fake.set(.installed, for: card.target.id)
        let out = try await call.value
        #expect(out.contains("six words came out of it"))
        #expect(exec.rows.count == 1)
        await MT43.until { coordinator.cards.isEmpty }
        #expect(coordinator.cards.isEmpty)                                          // "Installed. Continuing..." cleared when the call picked up
    }

    @Test func throughTheCardCancelReturnsTheSentenceAndRunsNothing() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let exec = MT13Executor.text("never")
        let rt = try MT43.runtime(installed: [], executor: exec)
        let fake = FakeInstaller(catalog: rt.catalog)
        let coordinator = ModelInstallCoordinator()
        coordinator.installer = fake
        let host = ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: root),
                                 installRequest: { need in await coordinator.request(need, conversationID: nil) })
        try audio(in: root, host: host)
        let call = Task { @MainActor in try await self.run("transcribe_audio", ["asset": .string("a1")], runtime: rt, host: host) }
        await MT43.until { coordinator.cards.count == 1 }
        coordinator.startDownload(try #require(coordinator.cards.first).id)
        coordinator.cancelAll()                                                     // Stop mid-download
        let out = try await call.value
        #expect(out.hasPrefix("The user chose not to download "))
        #expect(exec.rows.isEmpty)
        #expect(fake.cancels.count == 1)
    }
}

// MARK: - The real send path

/// `ModelRunner.send` builds the host each turn: the card must be reachable through it, Stop must answer it, and a turn must not
/// leave one behind. A scripted session stands in for the model; it asks for an install exactly as a tool call would.
@MainActor @Suite struct ModelInstallSendPathTests {

    final class AskingSession: ChatStreamingSession, @unchecked Sendable {
        let turn: MT43Box<RouterTurn?>
        let answer: MT43Box<Bool?>
        init(turn: MT43Box<RouterTurn?>, answer: MT43Box<Bool?>) { self.turn = turn; self.answer = answer }
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            let turn = self.turn, answer = self.answer
            return AsyncThrowingStream { continuation in
                let task = Task {
                    guard let host = turn.get?.toolContext?.modelToolHost else { continuation.finish(); return }
                    let ok = await host.installRequest(MT43.need("m1"))
                    answer.mutate { $0 = ok }
                    continuation.yield(ok ? "installed" : "declined")
                    continuation.finish()
                }
                continuation.onTermination = { _ in task.cancel() }
            }
        }
    }

    private func llm() -> ModelEntry {
        ModelEntry(
            id: "mlx-community--install", family: "Q", displayName: "Qwen3-8B", paramSize: "", paramCountB: nil, modelType: .llm, source: .mlx,
            format: "mlx", platforms: ["macos"], minMacOSVersion: nil, hfRepo: "", hfModelId: "", ramGB: 4, downloadSizeGB: 1,
            contextWindow: 32_000, license: nil, licenseUrl: nil, description: nil, summary: nil, descriptionSource: nil,
            architecture: nil, languages: nil, lastUpdated: nil, taskTags: nil, benchmarks: nil, speedTokensPerSec: nil,
            speedHardware: nil, speedEstimated: nil, communityDownloads: nil, communityLikes: nil, variants: nil)
    }

    private struct Rig {
        let runner: ModelRunner
        let fake: FakeInstaller
        let answer: MT43Box<Bool?>
        let dirs: [URL]
    }

    private func rig() throws -> Rig {
        let fm = FileManager.default
        let store = fm.temporaryDirectory.appendingPathComponent("mt43s-\(UUID().uuidString)")
        let modelDir = fm.temporaryDirectory.appendingPathComponent("mt43s-m-\(UUID().uuidString)")
        for d in [store, modelDir] { try fm.createDirectory(at: d, withIntermediateDirectories: true) }
        let runner = ModelRunner()
        runner.prepare(for: llm(), in: store)
        runner.modelDirectoryForTesting = { _ in modelDir }
        let turn = MT43Box<RouterTurn?>(nil)
        let answer = MT43Box<Bool?>(nil)
        runner.streamingSessionForTesting = { AskingSession(turn: turn, answer: answer) }
        runner.usableTokensForTesting = { _ in 32_000 }
        runner.routerTurnForTesting = { t in turn.mutate { $0 = t } }
        let runtime = try MT43.runtime(installed: [])
        runner.modelToolRuntimeProvider = { runtime }
        runner.modelToolsEnabled = true
        let base = try #require(runtime.catalog.first)
        let fake = FakeInstaller(catalog: [ModelEntry.testStub(from: base, id: "m1")])
        runner.installCoordinator.installer = fake
        return Rig(runner: runner, fake: fake, answer: answer, dirs: [store, modelDir])
    }

    private func settle(_ r: ModelRunner) async {
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while r.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }

    @Test func aToolCallInARealSendWaitsOnTheCardForTheActiveConversation() async throws {
        let r = try rig(); defer { r.dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        #expect(await r.runner.send("transcribe this", for: llm()))
        await MT43.until { r.runner.installCoordinator.cards.count == 1 }
        let card = try #require(r.runner.installCoordinator.cards.first)
        #expect(card.conversationID == r.runner.activeConversationID && card.conversationID != nil)
        #expect(r.runner.isRunning)                                                   // the turn is waiting on the person
        #expect(r.fake.starts.isEmpty)
        r.runner.stop()                                                               // Stop = Cancel
        #expect(r.runner.installCoordinator.cards.isEmpty)
        await MT43.until { r.answer.get != nil }
        #expect(r.answer.get == false)
    }

    @Test func downloadingFromTheCardLetsTheTurnGoOn() async throws {
        let r = try rig(); defer { r.dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        #expect(await r.runner.send("transcribe this", for: llm()))
        await MT43.until { r.runner.installCoordinator.cards.count == 1 }
        let coordinator = r.runner.installCoordinator
        coordinator.startDownload(try #require(coordinator.cards.first).id)
        r.fake.set(.installed, for: "m1")
        await settle(r.runner)
        #expect(r.answer.get == true)
        let reply = r.runner.transcript.items.compactMap { item -> String? in
            if case .message(let m) = item, m.role == .assistant { return m.text } else { return nil }
        }.last
        #expect(reply == "installed")
        #expect(r.fake.starts == ["m1"] && r.fake.finished == ["m1"])
        #expect(coordinator.cards.filter { $0.phase != .installed }.isEmpty)
        await MT43.until { coordinator.cards.isEmpty }
        #expect(coordinator.cards.isEmpty)                                            // no card outlives its turn
    }

    @Test func leavingTheConversationAnswersTheCard() async throws {
        let r = try rig(); defer { r.dirs.forEach { try? FileManager.default.removeItem(at: $0) } }
        #expect(await r.runner.send("transcribe this", for: llm()))
        await MT43.until { r.runner.installCoordinator.cards.count == 1 }
        r.runner.newConversation(for: llm().id)                                       // leave: stops the turn
        #expect(r.runner.installCoordinator.cards.isEmpty)
        await MT43.until { r.answer.get != nil }
        #expect(r.answer.get == false)
    }
}
