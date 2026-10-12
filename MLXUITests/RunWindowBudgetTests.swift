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

    // MARK: - (1) the Run window leases while open and returns on close

    @Test(arguments: [ModelType.asr, .tts, .image, .music, .embedding, .segmentation, .video, .upscale])
    func openingARunWindowReservesAndLeasesAndClosingReturnsIt(_ kind: ModelType) async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let model = makeEntry(id: "m-\(kind.rawValue)", modelType: kind, ramGB: 3)
        let stage = leasing(model, cache: cache)
        #expect(budget.usedBytes == 0)                         // nothing until the sheet opens
        #expect(await stage.prepare() == nil)
        #expect(budget.usedBytes == 3 * Self.gb)               // reserved at the entry's size
        #expect(budget.inUseCount == 1 && stage.isLeased)      // and in use: eviction skips it
        stage.release()                                         // the sheet closes
        #expect(budget.inUseCount == 0 && !stage.isLeased)
        #expect(budget.usedBytes == 3 * Self.gb)               // still resident (warm), but evictable now
    }

    @Test func aRunWindowsEngineIsNotEvictedWhileTheSheetIsOpen() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let open = leasing(makeEntry(id: "open", modelType: .asr, ramGB: 6), cache: cache)
        #expect(await open.prepare() == nil)
        // a flow row that can only fit by evicting the open sheet's engine waits, then is refused
        await #expect(throws: MemoryBudget.WaitError.self) {
            _ = try await cache.leasedStage(for: makeEntry(id: "row", modelType: .tts, ramGB: 6), config: .default,
                                            maxWait: .milliseconds(150)) { m, _ in Stub(id: m.id) }
        }
        #expect(cache.count == 1)
        open.release()
        _ = try await cache.leasedStage(for: makeEntry(id: "row", modelType: .tts, ramGB: 6), config: .default,
                                        maxWait: .seconds(1)) { m, _ in Stub(id: m.id) }
        #expect(cache.count == 1)                              // an ordinary eviction of the closed sheet's engine
    }

    @Test func aRunWindowAndAFlowRowForTheSameModelShareOneEngine() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let model = makeEntry(id: "shared", modelType: .asr, ramGB: 4)
        let builds = Counter()
        let sheet = leasing(model, cache: cache)
        #expect(await sheet.prepare() == nil)
        let row = try await cache.leasedStage(for: model, config: .default) { m, _ in builds.bump(); return Stub(id: m.id) }
        #expect(builds.value == 0)                              // the sheet's engine served the row: no second build
        #expect(cache.count == 1 && budget.usedBytes == 4 * Self.gb)
        sheet.release()
        #expect(budget.inUseCount == 1)                         // the row still holds it
        _ = row
    }

    @Test func aRunWindowThatWaitsOutTheBoundShowsTheWaitingSentenceInItsOwnError() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let held = try await cache.lease(for: makeEntry(id: "whisper-small", modelType: .asr, ramGB: 6), config: .default,
                                         holder: .chat) { m, _ in Stub(id: m.id) }
        let sheet = leasing(makeEntry(id: "ocr-model", modelType: .ocr, ramGB: 6), cache: cache, maxWait: .milliseconds(150))
        let refusal = await sheet.prepare()
        let wait = try #require(refusal as? MemoryBudget.WaitError)
        #expect(wait.description == "Waiting for memory: whisper-small is in use by Chat. Try again when it finishes.")
        // the Run view's existing error bar renders `localizedDescription`/`errorDescription` of what `run` throws
        do {
            _ = try await sheet.run(.text("x")) { _ in }
            Issue.record("expected the refusal")
        } catch let error as MemoryBudget.WaitError {
            #expect(error.errorDescription == wait.description)
        }
        held.hold.release()
        // freed: the next run leases and works (the refusal wasn't remembered)
        let out = try await sheet.run(.text("x")) { _ in }
        guard case .text("x") = out else { Issue.record("unexpected output"); return }
        #expect(sheet.isLeased)
    }

    @Test func aRebuildReturnsTheOldLeaseAndLeasesTheNewConfig() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let model = makeEntry(id: "whisper", modelType: .asr, ramGB: 3)
        let sdk = CountingSDK()
        let first = leasing(model, cache: cache)
        #expect(await first.prepare() == nil)
        #expect(budget.inUseCount == 1)
        let second = try #require(LeasingStage.rebuilding(first, model: model, sdk: sdk, config: StageConfig(language: "fr"))
                                  as? LeasingStage)
        #expect(!first.isLeased)                                // the old one is returned at once
        #expect(await second.prepare() == nil)
        #expect(second.isLeased && budget.inUseCount == 1 && cache.count == 2)   // its own engine (config is part of the key); the first is idle
        let third = try #require(LeasingStage.rebuilding(second, model: model, sdk: sdk, config: StageConfig(language: "de"))
                                 as? LeasingStage)
        #expect(!second.isLeased)
        _ = await third.prepare()
        third.release()
        #expect(budget.inUseCount == 0)
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

    @Test func aDroppedRunWindowReturnsItsLease() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        do {
            let stage = leasing(makeEntry(id: "gone", modelType: .asr, ramGB: 3), cache: cache)
            #expect(await stage.prepare() == nil)
            #expect(budget.inUseCount == 1)
        }
        #expect(budget.inUseCount == 0)
    }

    @Test func aStageWhoseEngineBooksItsOwnWeightsIsBookedAtZeroByTheCache() async throws {
        let budget = makeBudget(10)
        let cache = EngineCache(budget: budget, countsLLMWeights: false)
        let stage = leasing(makeEntry(id: "vlm", modelType: .vision, ramGB: 5), cache: cache, books: false)
        #expect(await stage.prepare() == nil)
        #expect(budget.usedBytes == 0 && budget.inUseCount == 0)    // not double-booked
        // the SDKs that own their weights say so; the ones that don't, don't
        #expect(VLMSDK() is StageBooksOwnWeights)
        #expect(OCRSDK() is StageBooksOwnWeights)
        #expect(LayaSDK() is StageBooksOwnWeights)
        #expect(CLMSDK() is StageBooksOwnWeights)
        #expect(!(MLXWhisperSDK() is StageBooksOwnWeights))
    }

    // MARK: - (3) Laya / CLM: real bytes, leases, held entries skipped

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
        #expect(EngineLeaseCache<Box>.charge(directory: dir) == ModelWeightCharge.charge(onDiskBytes: 4_500_000))
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
        let tight = MemoryBudget(capacityBytes: ModelWeightCharge.charge(onDiskBytes: 6_000_000) * 3 / 2,
                                 cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
        let small = EngineLeaseCache<Box>(budget: tight) { Box($0.lastPathComponent) }
        let leaseA = try await small.lease(for: a, holder: .runWindow)
        #expect(tight.usedBytes == ModelWeightCharge.charge(onDiskBytes: 6_000_000))      // real, not 0
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
        let charge = ModelWeightCharge.charge(onDiskBytes: 6_000_000)
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
