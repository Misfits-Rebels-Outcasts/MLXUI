import Foundation
import Testing
@testable import MLXUI

/// What reached the engine seam, and what the ledger said at that moment.
private nonisolated final class EngineSeam: @unchecked Sendable {
    struct Call: Equatable {
        let transcript: String
        let language: String
        let entries: Int
        let held: Int
        let bytes: Int64
    }
    private let lock = NSLock()
    private var recorded: [Call] = []
    private var built = 0
    func noteBuild() { lock.lock(); built += 1; lock.unlock() }
    func note(_ call: Call) { lock.lock(); recorded.append(call); lock.unlock() }
    var builds: Int { lock.lock(); defer { lock.unlock() }; return built }
    var calls: [Call] { lock.lock(); defer { lock.unlock() }; return recorded }
}

/// A test-owned budget and engine cache, and the real `ForcedAlignerStage` calling the real `ForcedAlignerEngine.align`,
/// with only the load and forward pass replaced by a recorder that echoes the transcript back as the one unit.
private nonisolated struct RunInputRig {
    let budget = MemoryBudget(capacityBytes: 20 * 1_073_741_824,
                              cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    let cache: EngineCache
    let seam = EngineSeam()
    let gate = ForcedAlignerGate()
    let directory = URL(fileURLWithPath: "/nonexistent-fa-input-\(UUID().uuidString)")
    /// Stands in for the files' charge (no files here); with the 5.25 GB working set, a call books 6.649 GB.
    static let filesCharge: Int64 = 1_399_000_000
    static let oneBooking: Int64 = 6_649_000_000

    /// As the app derives it from the SDK (`AppFlowExecutorFactory`, the Run sheet).
    var booksWeights: Bool { !((ForcedAlignerSDK() as any ModelSDK) is StageBooksOwnWeights) }

    init() {
        cache = EngineCache(budget: budget, countsLLMWeights: true,
                            modelDirectory: { _ in URL(fileURLWithPath: "/nonexistent-fa") })
    }

    func stage(language: String) -> ForcedAlignerStage {
        let budget = self.budget, seam = self.seam, gate = self.gate, directory = self.directory
        return ForcedAlignerStage(id: "seam", name: "Qwen3 Aligner 0.6B", language: language) { samples, text, language in
            try await ForcedAlignerEngine.align(
                samples, transcript: text, language: language, modelDirectory: directory,
                catalogFallback: Self.filesCharge, budget: budget, gate: gate,
                step: { _, text, _, _, _ in
                    seam.note(.init(transcript: text, language: language, entries: budget.heldCount,
                                    held: budget.inUseCount, bytes: budget.usedBytes))
                    return [AlignedUnit(text: text, start: 0.1, end: 0.5)]
                })
        }
    }

    /// The flow's builder, as `AppFlowExecutorFactory` calls `ForcedAlignerSDK.makeStage`: from the config alone.
    var builder: EngineCache.Builder {
        { _, config in
            seam.noteBuild()
            return stage(language: config.language ?? "en")
        }
    }
}

/// FA-R16 (owner rulings 17 and 19, 2026-11-27, quoted at `AlignTranscript`): the transcript is run input. One engine
/// per language, the value reaches the engine through the real lease wrappers, and a run without it is refused.
@MainActor
struct ForcedAlignerRunInputTests {

    private static let cardID = "mlx-community--Qwen3-ForcedAligner-0.6B-8bit"

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    private func card() throws -> ModelEntry {
        try #require(try loadCatalog().first { $0.id == Self.cardID })
    }

    private func temporaryWorkspace() throws -> FlowWorkspace {
        let ws = FlowWorkspace(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("fa-input-\(UUID().uuidString)"))
        try FileManager.default.createDirectory(at: ws.root, withIntermediateDirectories: true)
        return ws
    }

    private func wav(in directory: URL) throws -> URL {
        let url = directory.appendingPathComponent("a-\(UUID().uuidString).wav")
        try AudioWriter.writeWAV(AudioBuffer(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000), to: url)
        return url
    }

    /// `RealExecutor` with a stage factory that leases from the rig's cache, as the app's factory does from the shared one.
    private func executor(_ rig: RunInputRig, _ ws: FlowWorkspace) throws -> RealExecutor {
        RealExecutor(workspace: ws, flowID: "f", blobDirectory: ws.root.appendingPathComponent("blobs"),
                     makeModelStage: { model, config in
                         try await rig.cache.leasedStage(for: model, config: config, booksWeights: rig.booksWeights,
                                                         builder: rig.builder)
                     },
                     installedModelIDs: [Self.cardID], catalog: try loadCatalog())
    }

    private func alignRow(_ executor: any FlowExecutor, audio: URL, transcript: String) async throws -> Asset {
        try await executor.execute(
            path: "3", row: Row(task: "Align", model: "Qwen3 Aligner 0.6B", settings: "lang=de"),
            inputs: [Asset(items: [Item(kind: .audio, value: nil, path: audio, sourceText: nil)]),
                     Asset(items: [Item(kind: .text, value: transcript, path: nil, sourceText: nil)])],
            transcript: nil, context: nil, usedFlowContent: nil)
    }

    // MARK: - One entry, one booking (ruling 17)

    @Test func twoTranscriptsInOneLanguageMakeOneCacheEntryAndOneBooking() async throws {
        let rig = RunInputRig()
        let ws = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let flow = try executor(rig, ws)
        let audio = try wav(in: ws.root)
        let one = RunInputRig.oneBooking

        let first = try await alignRow(flow, audio: audio, transcript: "Morgen")
        let second = try await alignRow(flow, audio: audio, transcript: "Abend")

        #expect(rig.cache.count == 1, "one cache entry")
        #expect(rig.seam.builds == 1, "one stage built, reused by the second row")
        #expect(rig.cache.totalCachedBytes == 0, "the cached stage books nothing (FA-R20)")
        #expect(rig.budget.heldCount == 0 && rig.budget.usedBytes == 0, "nothing booked between rows")
        #expect(rig.budget.inUseCount == 0, "no lease outlives its row")
        // Each row's own transcript reached the engine, inside the lease.
        #expect(rig.seam.calls == ["Morgen", "Abend"].map {
            EngineSeam.Call(transcript: $0, language: "de", entries: 1, held: 1, bytes: one)
        })
        #expect(first.items.first?.value == "text,start,end\nMorgen,0.100,0.500\n")
        #expect(second.items.first?.value == "text,start,end\nAbend,0.100,0.500\n")
    }

    /// The row-output cache keys on the inputs, so one transcript's table is never served for another.
    @Test func twoTranscriptsNeverShareACachedRowOutput() async throws {
        let rig = RunInputRig()
        let ws = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let caching = CachingExecutor(inner: try executor(rig, ws),
                                      store: FlowCacheStore(root: ws.root.appendingPathComponent("cache")),
                                      cacheTier: "real", catalog: try loadCatalog(), runSeed: 1,
                                      workspace: ws, flowID: "f")
        let audio = try wav(in: ws.root)
        let first = try await alignRow(caching, audio: audio, transcript: "Morgen")
        #expect(!caching.lastCacheHit)
        let second = try await alignRow(caching, audio: audio, transcript: "Abend")
        #expect(!caching.lastCacheHit, "another transcript is another row output")
        let again = try await alignRow(caching, audio: audio, transcript: "Morgen")
        #expect(caching.lastCacheHit, "the same inputs hit")
        #expect(rig.seam.calls.map(\.transcript) == ["Morgen", "Abend"])
        #expect(first.items.first?.value != second.items.first?.value)
        #expect(again.items.first?.value == first.items.first?.value)
    }

    // MARK: - Guard: no transcript, no run

    @Test func aRunWithoutATranscriptIsRefused() async throws {
        let rig = RunInputRig()
        let buffer = AudioBuffer(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000)
        let leased = try await rig.cache.leasedStage(for: try card(), config: StageConfig(language: "de"),
                                                     booksWeights: rig.booksWeights, builder: rig.builder)
        await #expect(throws: AlignTranscript.Missing.self) { _ = try await leased.run(.audio(buffer)) { _ in } }
        #expect(rig.seam.calls.isEmpty, "the engine is never called")

        // An earlier run's transcript does not linger.
        _ = try await AlignTranscript.$current.withValue("Morgen") { try await leased.run(.audio(buffer)) { _ in } }
        await #expect(throws: AlignTranscript.Missing.self) { _ = try await leased.run(.audio(buffer)) { _ in } }
        #expect(rig.seam.calls.map(\.transcript) == ["Morgen"])
        #expect(AlignTranscript.Missing().localizedDescription
                == "The aligner got no transcript for this run — run Align again with the text of the recording.")
    }

    @Test func aRunWindowRunWithoutATranscriptIsRefusedAndHoldsNothing() async throws {
        let rig = RunInputRig()
        let handed = LeasingStage(model: try card(), config: .default, probe: rig.stage(language: "de"),
                                  booksWeights: rig.booksWeights, cache: rig.cache, maxWait: .milliseconds(300))
        let stage = LeasingStage.perRun(handed, config: StageConfig(language: "de")) { _ in rig.stage(language: "de") }
        let buffer = AudioBuffer(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000)
        await #expect(throws: AlignTranscript.Missing.self) { _ = try await stage.run(.audio(buffer)) { _ in } }
        #expect(rig.seam.calls.isEmpty && rig.budget.usedBytes == 0 && rig.budget.inUseCount == 0)
    }

    // MARK: - The Run screen's path

    /// `AlignRunModel` over `LeasingStage.perRun`, first on an empty cache, then on an entry a flow row left warm with
    /// another transcript.
    @Test func theRunScreensTranscriptReachesTheEngineInsideTheLease() async throws {
        let rig = RunInputRig()
        let card = try card()
        let ws = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let audio = try wav(in: ws.root)
        let one = RunInputRig.oneBooking
        let model = AlignRunModel(modelID: card.id, tracker: AlignStepTracker())

        func runScreen(_ transcript: String) async throws {
            let handed = LeasingStage(model: card, config: .default, probe: rig.stage(language: "de"),
                                      booksWeights: rig.booksWeights, cache: rig.cache, maxWait: .milliseconds(300))
            let stage = LeasingStage.perRun(handed, config: StageConfig(language: "de")) { _ in rig.stage(language: "de") }
            model.start(audioURL: audio, transcript: transcript, language: "de", stage: stage)
            await model.task?.value
            #expect(model.errorText == nil)
            #expect(model.result?.units.map(\.text) == [transcript])
        }

        try await runScreen("Morgen")
        #expect(rig.cache.count == 0 && rig.budget.usedBytes == 0 && rig.budget.inUseCount == 0, "released after the run")

        _ = try await alignRow(try executor(rig, ws), audio: audio, transcript: "Mittag")
        #expect(rig.cache.count == 1 && rig.budget.usedBytes == 0 && rig.budget.inUseCount == 0, "cached, nothing booked")
        try await runScreen("Abend")
        #expect(rig.seam.builds == 1, "the Run screen reused the flow's entry")
        #expect(rig.budget.usedBytes == 0 && rig.budget.inUseCount == 0)

        #expect(rig.seam.calls == ["Morgen", "Mittag", "Abend"].map {
            EngineSeam.Call(transcript: $0, language: "de", entries: 1, held: 1, bytes: one)
        })
    }
}
