import Testing
import Foundation
import CoreGraphics
@testable import MLXUI

/// SC-2b — every model load is on the budget and holds a lease while in use: Run windows (the registry sheet and the
/// OCR / Whisper rebuilds), the VLM weights, Laya and CLM — and the DEBUG-only launch override that makes a
/// contention check possible on a roomy Mac.
struct RunWindowBudgetTests {

    private static let gb = Int64(1_073_741_824)
    private final class Box: Sendable { let name: String; init(_ n: String) { name = n } }

    private nonisolated struct Stub: PipelineStage {
        let id: String
        var name: String { id }
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
    }

    /// An SDK whose `makeStage` counts builds (stands in for the registry's SDK in the rebuild path).
    private nonisolated final class CountingSDK: ModelSDK, @unchecked Sendable {
        let id = "counting"
        private let lock = NSLock()
        private var _builds = 0
        var builds: Int { lock.lock(); defer { lock.unlock() }; return _builds }
        func claim(_ model: ModelEntry) -> ClaimScore { .exact }
        func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
            lock.lock(); _builds += 1; lock.unlock()
            return Stub(id: model.id)
        }
    }

    private func makeBudget(_ gbs: Int64) -> MemoryBudget {
        MemoryBudget(capacityBytes: gbs * Self.gb,
                     cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    private func leasing(_ model: ModelEntry, config: StageConfig = .default, cache: EngineCache,
                         books: Bool = true, maxWait: Duration = .seconds(2)) -> LeasingStage {
        LeasingStage(model: model, config: config, probe: Stub(id: model.id), booksWeights: books,
                     cache: cache, maxWait: maxWait)
    }

    /// A probe whose `run` is under the test's control: it waits for `open`, or throws, or is cancelled.
    private nonisolated final class Controlled: PipelineStage, @unchecked Sendable {
        let id: String
        var name: String { id }
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        private let lock = NSLock()
        private var _open = false, _started = false
        let failure: Error?
        init(id: String, failure: Error? = nil) { self.id = id; self.failure = failure }
        var started: Bool { lock.lock(); defer { lock.unlock() }; return _started }
        func open() { lock.lock(); _open = true; lock.unlock() }
        private var isOpen: Bool { lock.lock(); defer { lock.unlock() }; return _open }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            lock.lock(); _started = true; lock.unlock()
            if let failure { throw failure }
            while !isOpen { try await Task.sleep(for: .milliseconds(10)) }      // throws when cancelled
            return input
        }
    }

    private func waitUntil(_ what: String = "", _ condition: () -> Bool) async throws {
        for _ in 0..<500 where !condition() { try await Task.sleep(for: .milliseconds(10)) }
        #expect(condition(), "timed out waiting for \(what)")
    }

    // MARK: - (1) a Run window leases per run, not while it is open

    @Test(arguments: [ModelType.asr, .tts, .image, .music, .embedding, .segmentation, .video, .upscale])
    func anOpenIdleRunWindowHoldsNothingARunningOneHoldsItAndAfterwardsItIsEvictable(_ kind: ModelType) async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let model = makeEntry(id: "m-\(kind.rawValue)", modelType: kind, ramGB: 3)
        let probe = Controlled(id: model.id)
        let stage = LeasingStage(model: model, config: .default, probe: probe, cache: cache, maxWait: .seconds(2))
        // open and idle: nothing reserved, nothing held
        #expect(budget.usedBytes == 0 && budget.inUseCount == 0 && !stage.isRunning)

        let run = Task { try await stage.run(.text("x")) { _ in } }
        try await waitUntil("the run to start") { probe.started }
        #expect(budget.usedBytes == 3 * Self.gb && budget.inUseCount == 1 && stage.isRunning)   // running: reserved and in use
        probe.open()
        _ = try await run.value
        #expect(budget.inUseCount == 0 && !stage.isRunning)                    // the lease is returned
        #expect(budget.usedBytes == 3 * Self.gb && cache.count == 1)            // warm: still cached, but evictable now
    }

    @Test func anOpenIdleRunWindowDoesNotBlockAnotherLoad() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let idle = LeasingStage(model: makeEntry(id: "idle", modelType: .tts, ramGB: 6), config: .default,
                                probe: Stub(id: "idle"), cache: cache)
        _ = try await idle.run(.text("warm")) { _ in }                          // used once, window left open
        #expect(budget.inUseCount == 0)
        // a flow row that needs the room takes it from the idle window's cached engine — no wait, no refusal
        let started = ContinuousClock.now
        _ = try await cache.leasedStage(for: makeEntry(id: "row", modelType: .asr, ramGB: 6), config: .default,
                                        maxWait: .seconds(30)) { m, _ in Stub(id: m.id) }
        #expect(ContinuousClock.now - started < .seconds(5))
        #expect(cache.count == 1 && budget.usedBytes == 6 * Self.gb)
        #expect(!idle.isRunning)
    }

    @Test func aRunInProgressBlocksAnotherLoadUntilItEnds() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let probe = Controlled(id: "busy")
        let window = LeasingStage(model: makeEntry(id: "busy", modelType: .tts, ramGB: 6), config: .default,
                                  probe: probe, cache: cache)
        let run = Task { try await window.run(.text("x")) { _ in } }
        try await waitUntil("the run to start") { probe.started }
        // refused while the run is in progress, naming the Run window
        do {
            _ = try await cache.leasedStage(for: makeEntry(id: "row", modelType: .asr, ramGB: 6), config: .default,
                                            maxWait: .milliseconds(150)) { m, _ in Stub(id: m.id) }
            Issue.record("expected a WaitError")
        } catch let error as MemoryBudget.WaitError {
            #expect(error.description == "Waiting for memory: busy is in use by a Run window. Try again when it finishes.")
        }
        // …and a waiter that is already waiting proceeds as soon as the run ends
        async let waiting = cache.leasedStage(for: makeEntry(id: "row", modelType: .asr, ramGB: 6), config: .default,
                                              maxWait: .seconds(10)) { m, _ in Stub(id: m.id) }
        try await Task.sleep(for: .milliseconds(100))
        #expect(cache.count == 1)
        probe.open()
        _ = try await run.value
        _ = try await waiting
        #expect(cache.count == 1 && budget.usedBytes == 6 * Self.gb)           // the window's engine was evicted for it
    }

    @Test func aThrowingRunReturnsItsLease() async throws {
        struct Boom: Error {}
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let window = LeasingStage(model: makeEntry(id: "boom", modelType: .asr, ramGB: 3), config: .default,
                                  probe: Controlled(id: "boom", failure: Boom()), cache: cache)
        await #expect(throws: Boom.self) { _ = try await window.run(.text("x")) { _ in } }
        #expect(budget.inUseCount == 0 && !window.isRunning)
    }

    @Test func aCancelledRunReturnsItsLease() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let probe = Controlled(id: "cancel")
        let window = LeasingStage(model: makeEntry(id: "cancel", modelType: .asr, ramGB: 3), config: .default,
                                  probe: probe, cache: cache)
        let run = Task { try await window.run(.text("x")) { _ in } }
        try await waitUntil("the run to start") { probe.started }
        #expect(budget.inUseCount == 1)
        run.cancel()
        _ = await run.result
        #expect(budget.inUseCount == 0 && !window.isRunning)
    }

    @Test func aRunCancelledWhileWaitingForRoomLeavesNothingHeld() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let held = try await cache.lease(for: makeEntry(id: "other", modelType: .tts, ramGB: 6), config: .default,
                                         holder: .chat) { m, _ in Stub(id: m.id) }
        let window = LeasingStage(model: makeEntry(id: "w", modelType: .asr, ramGB: 6), config: .default,
                                  probe: Stub(id: "w"), cache: cache, maxWait: .seconds(60))
        let run = Task { try await window.run(.text("x")) { _ in } }
        try await Task.sleep(for: .milliseconds(100))
        let started = ContinuousClock.now
        run.cancel()
        _ = await run.result
        #expect(ContinuousClock.now - started < .seconds(5))                    // not the 60 s bound
        #expect(budget.inUseCount == 1)                                         // only `other`
        held.hold.release()
    }

    @Test func aRunWindowAndAFlowRowForTheSameModelShareOneEngine() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let model = makeEntry(id: "shared", modelType: .asr, ramGB: 4)
        let builds = Counter()
        let window = LeasingStage(model: model, config: .default, probe: Stub(id: model.id), cache: cache)
        _ = try await window.run(.text("x")) { _ in }                           // the window's run built the engine
        let row = try await cache.leasedStage(for: model, config: .default) { m, _ in builds.bump(); return Stub(id: m.id) }
        #expect(builds.value == 0)                                              // the row reused it: no second build
        #expect(cache.count == 1 && budget.usedBytes == 4 * Self.gb)
        #expect(budget.inUseCount == 1)                                         // held by the row alone
        _ = row
    }

    @Test func aRunThatHasToWaitShowsTheWaitingSentenceInTheRunViewsOwnError() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let held = try await cache.lease(for: makeEntry(id: "whisper-small", modelType: .asr, ramGB: 6), config: .default,
                                         holder: .chat) { m, _ in Stub(id: m.id) }
        let window = LeasingStage(model: makeEntry(id: "ocr-model", modelType: .ocr, ramGB: 6), config: .default,
                                  probe: Stub(id: "ocr-model"), cache: cache, maxWait: .milliseconds(150))
        // the Run view renders what `run` throws via `errorDescription`
        do {
            _ = try await window.run(.text("x")) { _ in }
            Issue.record("expected the refusal")
        } catch let error as MemoryBudget.WaitError {
            #expect(error.errorDescription == "Waiting for memory: whisper-small is in use by Chat. Try again when it finishes.")
        }
        #expect(budget.inUseCount == 1)                                         // the refused run holds nothing
        held.hold.release()
        let out = try await window.run(.text("x")) { _ in }                     // freed: the next run works
        guard case .text("x") = out else { Issue.record("unexpected output"); return }
    }

    @Test func aRebuildIsItsOwnConfigAndEachRunLeasesAndReturns() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let model = makeEntry(id: "whisper", modelType: .asr, ramGB: 3)
        let sdk = CountingSDK()
        let first = LeasingStage(model: model, config: .default, probe: Stub(id: model.id), cache: cache)
        let second = try #require(LeasingStage.rebuilding(first, model: model, sdk: sdk, config: StageConfig(language: "fr"))
                                  as? LeasingStage)
        let third = try #require(LeasingStage.rebuilding(second, model: model, sdk: sdk, config: StageConfig(language: "de"))
                                 as? LeasingStage)
        #expect(budget.inUseCount == 0)                                         // rebuilding holds nothing
        for stage in [first, second, third] {
            _ = try await stage.run(.text("x")) { _ in }
            #expect(budget.inUseCount == 0 && !stage.isRunning)
        }
        #expect(cache.count == 3 && budget.usedBytes == 9 * Self.gb)            // three configs, three warm engines
        #expect(sdk.builds == 2)
    }

    @Test func aRebuildOfANonLeasingStageIsThePlainSDKBuild() {
        let model = makeEntry(id: "x", modelType: .asr)
        let sdk = CountingSDK()
        let out = LeasingStage.rebuilding(Stub(id: "x"), model: model, sdk: sdk, config: .default)
        #expect(out is Stub)
        #expect(sdk.builds == 1)
        #expect(LeasingStage.rebuilding(Stub(id: "x"), model: model, sdk: nil, config: .default) == nil)
    }

    @Test func aStageWhoseEngineBooksItsOwnWeightsIsBookedAtZeroByTheCache() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let stage = LeasingStage(model: makeEntry(id: "vlm", modelType: .vision, ramGB: 5), config: .default,
                                 probe: Stub(id: "vlm"), booksWeights: false, cache: cache)
        _ = try await stage.run(.text("x")) { _ in }
        #expect(budget.usedBytes == 0 && budget.inUseCount == 0)                // not double-booked
        // the SDKs that own their weights say so; the ones that don't, don't
        #expect(VLMSDK() is StageBooksOwnWeights)
        #expect(OCRSDK() is StageBooksOwnWeights)
        #expect(LayaSDK() is StageBooksOwnWeights)
        #expect(CLMSDK() is StageBooksOwnWeights)
        #expect(!(MLXWhisperSDK() is StageBooksOwnWeights))
    }

    // MARK: - (3) Laya / CLM: real bytes, leases, held entries skipped

    /// SC-2c: what a decision engine (Laya, CLM) books for `onDisk` bytes of safetensors.
    private static func decisionCharge(_ onDisk: Int64) -> Int64 {
        Int64((Double(onDisk) * ModelLoadProfile.entry(for: .decision).factor).rounded())
    }

    private func engineDir(safetensors: [(String, Int)]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sc2b-\(UUID().uuidString)")
        for (path, size) in safetensors {
            let url = dir.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(count: size).write(to: url)
        }
        return dir
    }

    @Test func theChargeIsTheRealSafetensorsBytesUnderTheDirectoryNotZero() throws {
        let dir = try engineDir(safetensors: [("model.safetensors", 1_000_000), ("encoder/w.safetensors", 3_000_000),
                                              ("heads/h.safetensors", 500_000)])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(EngineLeaseCache<Box>.charge(directory: dir) == Self.decisionCharge(4_500_000))   // SC-2c: the decision kind's table factor
        #expect(EngineLeaseCache<Box>.charge(directory: dir) > 4_500_000)
        // no safetensors → the directory's size (never 0 for a non-empty directory)
        let plain = FileManager.default.temporaryDirectory.appendingPathComponent("sc2b-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: plain, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: plain) }
        try Data(count: 2_000).write(to: plain.appendingPathComponent("weights.bin"))
        #expect(EngineLeaseCache<Box>.charge(directory: plain) >= 2_000)
    }

    @Test func aLeaseCacheBooksRealBytesAndSkipsHeldEntries() async throws {
        let a = try engineDir(safetensors: [("encoder/w.safetensors", 6_000_000)])
        let b = try engineDir(safetensors: [("encoder/w.safetensors", 6_000_000)])
        defer { try? FileManager.default.removeItem(at: a); try? FileManager.default.removeItem(at: b) }
        // room for one 6 MB charge and half of another: two can't be resident together
        let tight = MemoryBudget(capacityBytes: Self.decisionCharge(6_000_000) * 3 / 2,
                                 cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
        let small = EngineLeaseCache<Box>(budget: tight) { Box($0.lastPathComponent) }
        let leaseA = try await small.lease(for: a, holder: .runWindow)
        #expect(tight.usedBytes == Self.decisionCharge(6_000_000))      // real, not 0
        #expect(tight.inUseCount == 1)
        do {
            _ = try await small.lease(for: b, holder: .flow, maxWait: .milliseconds(150))
            Issue.record("expected a WaitError: A is held")
        } catch let error as MemoryBudget.WaitError {
            #expect(error.description.contains("is in use by a Run window"))
        }
        #expect(await small.residentKeys == [a.path])
        leaseA.release()
        let leaseB = try await small.lease(for: b, holder: .flow, maxWait: .seconds(1))   // an ordinary eviction now
        #expect(await small.residentKeys == [b.path])
        leaseB.release()
    }

    @Test func aLeaseCacheIsAParticipantOfTheSharedBudgetSoOtherLoadsCanEvictItWhenIdle() async throws {
        let charge = Self.decisionCharge(6_000_000)
        let budget = MemoryBudget(capacityBytes: charge * 3 / 2,
                                  cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
        let cache = EngineLeaseCache<Box>(budget: budget) { Box($0.lastPathComponent) }
        let pool = ResidentPool<Box>(budget: budget)
        let dir = try engineDir(safetensors: [("model.safetensors", 6_000_000)])
        defer { try? FileManager.default.removeItem(at: dir) }
        let l = try await cache.lease(for: dir)
        l.release()
        _ = try await pool.value(for: "llm", bytes: charge) { Box("llm") }           // needs the room: idle Laya/CLM goes
        #expect(await cache.residentKeys.isEmpty)
        #expect(budget.usedBytes == charge)
    }

    // MARK: - (2) VLM weights reserved at their real size

    private struct Sentinel: Error {}

    @Test func theVLMReservationIsTheRealChargeAndIsReleasedAfterAFailedLoad() async throws {
        let dir = try engineDir(safetensors: [("model.safetensors", 4_000_000)])
        defer { try? FileManager.default.removeItem(at: dir) }
        let budget = makeBudget(10)
        let seen = Cell<Int64>(-1)
        let pool = ModelContainerPool(budget: budget) { _, _ in
            seen.value = budget.usedBytes            // what the ledger says while the load runs
            throw Sentinel()
        }
        await #expect(throws: StageError.self) {
            _ = try await VLMEngine.generate(prompt: "p", image: Self.image(), modelDir: dir, maxTokens: 1, pool: pool)
        }
        #expect(seen.value == ModelWeightCharge.charge(onDiskBytes: 4_000_000))         // real charge, not 0 / not ramGB
        #expect(budget.usedBytes == 0 && budget.inUseCount == 0)
    }

    @Test func theVLMPoolIsOnTheSharedBudget() {
        #expect(ModelContainerPool.vlm.budget === MemoryBudget.shared)
        #expect(ModelContainerPool.vlm !== ModelContainerPool.shared)
    }

    private static func image() -> CGImage {
        let ctx = CGContext(data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return ctx.makeImage()!
    }

    // MARK: - (4) the DEBUG-only budget override

    @Test func theLaunchOverrideSetsTheSharedCapacityInDebugAndOnlyThere() throws {
        let suite = "sc2b-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let normal = EngineCache.defaultBudget(totalRAMGB: 19.327)
        #expect(MemoryBudget.sharedCapacity(totalRAMGB: 19.327, defaults: defaults) == normal)   // no argument: unchanged
        #expect(MemoryBudget.debugOverrideIsCompiledIn == Self.isDebug)
        #if DEBUG
        defaults.set(8.0, forKey: MemoryBudget.debugOverrideKey)                                // -MLXUIMemoryBudgetGiB 8
        #expect(MemoryBudget.sharedCapacity(totalRAMGB: 19.327, defaults: defaults) == 8 * Self.gb)
        defaults.set(0.5, forKey: MemoryBudget.debugOverrideKey)
        #expect(MemoryBudget.sharedCapacity(totalRAMGB: 19.327, defaults: defaults) == Self.gb / 2)
        for bad in [0.0, -3.0] {                                                                // nonsense is ignored
            defaults.set(bad, forKey: MemoryBudget.debugOverrideKey)
            #expect(MemoryBudget.sharedCapacity(totalRAMGB: 19.327, defaults: defaults) == normal)
        }
        defaults.set("eight", forKey: MemoryBudget.debugOverrideKey)
        #expect(MemoryBudget.sharedCapacity(totalRAMGB: 19.327, defaults: defaults) == normal)
        #endif
    }

    private static var isDebug: Bool {
        #if DEBUG
        true
        #else
        false
        #endif
    }

    @Test func nothingElseChangesTheCapacityOfTheSharedBudget() {
        // `capacityBytes` is a `let`; the only writer is the initializer `shared` is created with.
        #expect(MemoryBudget.shared.capacityBytes == MemoryBudget.sharedCapacity(totalRAMGB: SystemInfo.detect().totalRAMGB))
    }
}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock(); private var n = 0
    func bump() { lock.lock(); n += 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return n }
}

private final class Cell<T>: @unchecked Sendable {
    private let lock = NSLock(); private var stored: T
    init(_ v: T) { stored = v }
    var value: T {
        get { lock.lock(); defer { lock.unlock() }; return stored }
        set { lock.lock(); stored = newValue; lock.unlock() }
    }
}
