import Testing
import Foundation
import MLX
@testable import MLXUI

// BCC-1 (R18, `RSI/DelegateBufferCacheBacklog.md`): the flow and Run-window paths hold the Local Server's 1 GiB MLX buffer-cache cap
// (`ServeMLXMemory`) while they run. BCC-0 measured flow 9 at 12.9 GiB of cache against a 1.9 GiB live peak, with `cacheLimit` at
// 17.5 GiB (`EngineCache.init` sets it to the memory budget). These tests fail without the change.

private final class Observed: @unchecked Sendable {
    private let lock = NSLock()
    private var _limits: [Int] = []
    func note() { lock.withLock { _limits.append(Memory.cacheLimit) } }
    var limits: [Int] { lock.withLock { _limits } }
}

private final class FakeLimit: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Int
    init(_ v: Int) { _value = v }
    var value: Int { lock.withLock { _value } }
    func set(_ v: Int) { lock.withLock { _value = v } }
}

private nonisolated struct ProbeStage: PipelineStage {
    let id = "bcc1.probe"
    let name = "BCC1 probe"
    let accepts: MediaKind = .text
    let produces: MediaKind = .text
    let observed: Observed
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        observed.note()
        return .text("ok")
    }
}

struct BufferCacheCapTests {
    private let seventeenGiB = 17 * 1_073_741_824

    // MARK: the mechanism

    @Test func scopedHoldsTheCapOverALargeLimitAndRestoresIt() async throws {
        let limit = FakeLimit(seventeenGiB)
        let memory = ServeMLXMemory(read: { limit.value }, write: { limit.set($0) })
        let inside: Int = await memory.scoped { limit.value }
        #expect(inside == ServeMLXMemory.generationCacheLimit, "the 17 GiB limit was lowered to the 1 GiB cap")
        #expect(limit.value == seventeenGiB, "and put back")
        #expect(!memory.isCapActive)
    }

    @Test func scopedRestoresWhenTheBodyThrows() async {
        struct Boom: Error {}
        let limit = FakeLimit(seventeenGiB)
        let memory = ServeMLXMemory(read: { limit.value }, write: { limit.set($0) })
        do { try await memory.scoped { throw Boom() } } catch {}
        #expect(limit.value == seventeenGiB)
        #expect(!memory.isCapActive)
    }

    @Test func scopedNeverRaisesALowerLimit() async {
        let limit = FakeLimit(300 * 1_048_576)
        let memory = ServeMLXMemory(read: { limit.value }, write: { limit.set($0) })
        let inside: Int = await memory.scoped { limit.value }
        #expect(inside == 300 * 1_048_576)
    }

    @Test func theCapIsActiveForMemoryBudgetsClearerWhileScoped() async {
        let limit = FakeLimit(seventeenGiB)
        let memory = ServeMLXMemory(read: { limit.value }, write: { limit.set($0) })
        let during: Bool = await memory.scoped { memory.isCapActive }
        #expect(during, "MLXCacheClearer asks isCapActive and stands down while a cap is held")
        #expect(!memory.isCapActive)
    }

    // MARK: the two choke points, against the real MLX limit

    /// A flow row (`RealExecutor.execute`): the model stage runs with the cap, however high the limit was.
    @Test @MainActor func aFlowRowRunsUnderTheCap() async throws {
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }
        _ = EngineCache.shared                                  // so a lazy first build cannot touch the limit under us
        Memory.cacheLimit = seventeenGiB                        // what EngineCache.init leaves in force
        let observed = Observed()
        let ws = try #require(FlowWorkspace(root: FileManager.default.temporaryDirectory.appendingPathComponent("bcc1-\(UUID().uuidString)")))
        let flowID = "9-ReadItAloud"
        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory, bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        defer { try? FileManager.default.removeItem(at: ws.directory(for: flowID)) }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json")))
            .domains.flatMap { $0.allModels }
        let stub = ProbeStageAudio(observed: observed)
        let executor = RealExecutor(workspace: ws, flowID: flowID, blobDirectory: ws.directory(for: flowID).appendingPathComponent("blobs"),
                                    makeModelStage: { _, _ in stub }, installedModelIDs: Set(catalog.map(\.id)), catalog: catalog)
        let events = try await FlowInterpreter.run(try GalleryLoader.loadDocument(flowID: flowID), executor: executor)
        #expect(events.filter { $0.kind == .rowFailed }.isEmpty)
        #expect(!observed.limits.isEmpty, "the model stage ran")
        #expect(observed.limits.allSatisfy { $0 <= ServeMLXMemory.generationCacheLimit }, "limits seen by the stage: \(observed.limits)")
    }

    /// A Run window (`LeasingStage.run`).
    @Test func aRunWindowRunsUnderTheCap() async throws {
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }
        _ = EngineCache.shared
        let cache = EngineCache(budget: MemoryBudget(capacityBytes: 10 * 1_073_741_824), countsLLMWeights: false)   // sets Memory.cacheLimit = 10 GiB
        #expect(Memory.cacheLimit == 10 * 1_073_741_824)
        let observed = Observed()
        let model = makeEntry(id: "bcc1-asr", modelType: .asr, ramGB: 1)
        let stage = LeasingStage(model: model, config: .default, probe: ProbeStage(observed: observed), cache: cache)
        _ = try await stage.run(.text("x")) { _ in }
        #expect(observed.limits.count == 1)
        #expect(observed.limits.allSatisfy { $0 <= ServeMLXMemory.generationCacheLimit }, "limits seen by the stage: \(observed.limits)")
    }
}

private nonisolated struct ProbeStageAudio: PipelineStage {
    let id = "bcc1.probe-audio"
    let name = "BCC1 probe audio"
    let accepts: MediaKind = .text
    let produces: MediaKind = .audio
    let observed: Observed
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        observed.note()
        return .audio(AudioBuffer(samples: [Float](repeating: 0.1, count: 2400), sampleRate: 24000))
    }
}

// BCC-3b (R22): the in-app chat / agent turn (`ModelRunner.send`, one `AgentSession` for both) runs under the same cap.
private final class LimitRecordingSession: ChatStreamingSession, @unchecked Sendable {
    private let lock = NSLock()
    private var _seen: [Int] = []
    var seen: [Int] { lock.withLock { _seen } }
    func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        Task {
            lock.withLock { _seen.append(Memory.cacheLimit) }
            continuation.yield("hello")
            try? await Task.sleep(for: .milliseconds(5))
            lock.withLock { _seen.append(Memory.cacheLimit) }
            continuation.finish()
        }
        return stream
    }
}

@MainActor
struct ModelRunnerCacheCapTests {
    @Test func aChatTurnRunsUnderTheCap() async throws {
        let original = Memory.cacheLimit
        defer { Memory.cacheLimit = original }
        _ = EngineCache.shared
        Memory.cacheLimit = 17 * 1_073_741_824
        let entry = makeEntry(id: "bcc3b-model", displayName: "BCC3b")
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("bcc3b-store-\(UUID().uuidString)")
        let modelDir = FileManager.default.temporaryDirectory.appendingPathComponent("bcc3b-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: store); try? FileManager.default.removeItem(at: modelDir) }
        let session = LimitRecordingSession()
        let runner = ModelRunner()
        runner.prepare(for: entry, in: store)
        runner.modelDirectoryForTesting = { _ in modelDir }
        runner.streamingSessionForTesting = { session }
        _ = await runner.send("hi", for: entry)
        let deadline = ContinuousClock.now + .seconds(20)
        while runner.isRunning, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        #expect(session.seen.count == 2, "the stream ran: \(session.seen)")
        #expect(session.seen.allSatisfy { $0 <= ServeMLXMemory.generationCacheLimit }, "limits seen during the turn: \(session.seen)")
    }
}
