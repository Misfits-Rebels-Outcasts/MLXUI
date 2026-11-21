import Foundation
import MLX
import Testing
@testable import MLXUI

/// MC5-T-book (owner, 2026-11-21): ACE-Step books its memory from the shared `MemoryBudget` on BOTH paths, with one figure. The figure is not a new constant: it is the unmeasured-kind rule
/// `max(files x 1.10, catalog ramGB)` (`ModelLoadProfile.charge`, via `EngineCache.footprint(of:)`), which gives 10.9 GiB (the catalog's `ramGB`) for the installed no-LM folder (about 6.3 GB of
/// files). The flow path already booked it (`AppFlowExecutorFactory.make` -> `EngineCache.leasedStage`); the Run window did not, because `AceStepRunView` ignored the `LeasingStage` the sheet
/// hands it. Everything here runs the tiny checkpoint; nothing loads real weights.
@Suite(.serialized)
@MainActor
struct AceStepBookingTests {

    // MARK: Fixtures

    private static func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    private static func ace() throws -> ModelEntry { try #require(try catalog().first { $0.id == "ACE-Step--Ace-Step1.5" }) }

    private static let gib = Int64(1_073_741_824)

    /// A model folder with no weights (the catalog's `ramGB` is what books), or with one sparse `.safetensors` of `fileBytes` (no disk used).
    private static func folder(fileBytes: Int64 = 0) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-book-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if fileBytes > 0 {
            let file = dir.appendingPathComponent("model.safetensors")
            FileManager.default.createFile(atPath: file.path, contents: nil)
            let handle = try FileHandle(forWritingTo: file)
            try handle.truncate(atOffset: UInt64(fileBytes))
            try handle.close()
        }
        return dir
    }

    private static func expected(_ entry: ModelEntry, in dir: URL) -> Int64 {
        max(Int64((Double(ModelLoadProfile.safetensorsBytes(underneath: dir)) * 1.10).rounded()), Int64(entry.ramGB * Double(gib)))
    }

    private final class Log: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [String] = []
        func add(_ s: String) { lock.withLock { items.append(s) } }
        var all: [String] { lock.withLock { items } }
        func count(_ s: String) -> Int { lock.withLock { items.filter { $0 == s }.count } }
    }

    /// The real engine on the tiny checkpoint with a logged teardown; `hook` sees every phase.
    private static func tinyGenerator(_ dir: URL, log: Log, hook: @escaping @Sendable (AceStepPhase) -> Void = { _ in }) -> AceStepRunModel.Generator {
        { request, onPhase in
            var engine = AceStepEngineTests.engine(dir, observer: { event in
                if case .phase(let p) = event { onPhase(p); hook(p) }
            })
            engine.teardown = { Stream.gpu.synchronize(); Memory.clearCache(); log.add("teardown") }
            return try await engine.generate(request).audio
        }
    }

    nonisolated private static func probe() -> any PipelineStage { AceStepStage(id: "probe", name: "probe") { _, _ in AceStepAudio(samples: [0, 0]) } }

    private final class ModelBox: @unchecked Sendable { @MainActor var model: AceStepRunModel? }

    // MARK: (c) the figure, single-sourced, equal on both paths

    @Test func theBookingIsTheUnmeasuredKindRuleAndTheFlowPathBooksTheSame() async throws {
        let entry = try Self.ace()
        #expect(entry.runnerKind == .music)
        for fileBytes in [Int64(0), 12_000_000_000] {
            let dir = try Self.folder(fileBytes: fileBytes)
            defer { try? FileManager.default.removeItem(at: dir) }
            let want = Self.expected(entry, in: dir)
            let budget = MemoryBudget(capacityBytes: 20 * Self.gib)
            let cache = EngineCache(budget: budget, modelDirectory: { _ in dir })
            #expect(cache.footprint(of: entry) == want, "files \(fileBytes): \(cache.footprint(of: entry)) vs max(files x 1.10, ramGB) \(want)")
            // The flow path: leased stage, booked through the cache; read what the ledger holds while the row has it.
            let stage = try await cache.leasedStage(for: entry, config: StageConfig(seed: 7, steps: 4), booksWeights: true,
                                                    footprintOverride: Flux2KleinBooking.leaseOverride(for: entry, config: StageConfig(seed: 7, steps: 4))) { _, _ in Self.probe() }
            #expect(budget.usedBytes == want, "the flow row booked \(budget.usedBytes), expected \(want)")
            _ = stage
        }
        // 10.9 GiB for the real install (6.3 GB of files x 1.10 = 6.96 GB is below it).
        #expect(entry.ramGB == 10.9, "the catalog figure the empty-folder case books (10.9 GiB = 11.70 GB)")
    }

    // MARK: (d) MusicGen is untouched

    @Test func musicGenBooksExactlyWhatItBookedBefore() throws {
        let musicgen = try #require(try Self.catalog().first { $0.family == "MusicGen" && $0.runnerKind == .music })
        let dir = try Self.folder(fileBytes: 1_000_000_000)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = EngineCache(budget: MemoryBudget(capacityBytes: 20 * Self.gib), modelDirectory: { _ in dir })
        let before = ModelLoadProfile.charge(kind: .music, directory: dir, catalogFallback: Int64(musicgen.ramGB * Double(Self.gib)))   // the generic rule, as before MC5
        #expect(cache.footprint(of: musicgen) == before)
        #expect(AceStepSDK.claimsAceStep(musicgen) == false)
        print("MC5-T-book MusicGen \(musicgen.id): ramGB \(musicgen.ramGB), books \(before) B before and \(cache.footprint(of: musicgen)) B after")
    }

    // MARK: (a) Run window: held during the run, released after teardown, cancel included

    @Test(arguments: [false, true])
    func theRunWindowHoldsTheReservationDuringTheRunAndReleasesItAfterTeardown(cancel: Bool) async throws {
        let entry = try Self.ace()
        let tiny = try AceStepEngineTests.makeFolder()
        let modelDir = try Self.folder()
        defer { try? FileManager.default.removeItem(at: tiny); try? FileManager.default.removeItem(at: modelDir) }
        let want = Self.expected(entry, in: modelDir)
        let budget = MemoryBudget(capacityBytes: 20 * Self.gib)
        let cache = EngineCache(budget: budget, modelDirectory: { _ in modelDir })
        let handed = LeasingStage(model: entry, config: .default, probe: Self.probe(), cache: cache, maxWait: .seconds(2))
        let log = Log(), seen = Log(), holder = ModelBox()
        let base = Self.tinyGenerator(tiny, log: log) { phase in
            if case .denoise(let step, _) = phase, step == 1 {
                seen.add("\(budget.usedBytes)")
                if cancel { Task { @MainActor in holder.model?.cancel() } }
            }
        }
        Memory.clearCache()
        let model = AceStepRunModel(generator: AceStepRunModel.leasedGenerator(modelID: entry.id, base: base, handedStage: handed))
        holder.model = model
        model.generate()
        while model.isRunning { try await Task.sleep(for: .milliseconds(1)) }
        // The instant the model reads idle, the engine has torn down and the lease is back.
        #expect(seen.all == ["\(want)"], "the run held the booking while it ran: \(seen.all), expected \(want)")
        #expect(log.count("teardown") == 1, "teardown ran before idle: \(log.all)")
        #expect(budget.usedBytes == 0 && budget.inUseCount == 0, "reservation released at idle: used \(budget.usedBytes), in use \(budget.inUseCount)")
        #expect(cancel ? (model.wavData == nil && model.errorText == nil) : (model.wavData != nil), "cancel \(cancel): wav \(model.wavData != nil), error \(String(describing: model.errorText))")
    }

    // MARK: (a) flow row: held while the row runs, no hold left after a cancel

    @Test func aCancelledFlowRowBooksTheSameFigureAndLeavesNoHoldAfterTeardown() async throws {
        let entry = try Self.ace()
        let tiny = try AceStepEngineTests.makeFolder()
        let modelDir = try Self.folder()
        defer { try? FileManager.default.removeItem(at: tiny); try? FileManager.default.removeItem(at: modelDir) }
        let want = Self.expected(entry, in: modelDir)
        let budget = MemoryBudget(capacityBytes: 20 * Self.gib)
        let cache = EngineCache(budget: budget, modelDirectory: { _ in modelDir })
        let log = Log(), seen = Log(), cancelBox = CancelBox()
        let generate = Self.tinyGenerator(tiny, log: log) { phase in
            if case .denoise(let step, _) = phase, step == 1 { seen.add("\(budget.usedBytes)/\(budget.inUseCount)"); cancelBox.fire() }
        }
        let make: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { model, config in
            try await cache.leasedStage(for: model, config: config, booksWeights: true,
                                        footprintOverride: Flux2KleinBooking.leaseOverride(for: model, config: config)) { _, _ in
                AceStepStage(id: "acestep.tiny", name: "tiny", settings: AceStepRunSettings(seed: config.seed, steps: config.steps)) { request, onPhase in try await generate(request, onPhase) }
            }
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-book-flow-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let executor = RealExecutor(workspace: FlowWorkspace(root: root), flowID: "mc5tbook", blobDirectory: root, makeModelStage: make,
                                    installedModelIDs: [entry.id], catalog: [entry])
        let doc = try CatParser.parse("mlxflow 0.8\n1. Generate Sound   ACE-Step 1.5; steps=4; seed=7 \"calm piano\"\n\nmodels:\n  ACE-Step 1.5 = ACE-Step/Ace-Step1.5\n")
        let task = Task { _ = try await FlowInterpreter.run(doc, executor: executor, definitions: doc.definitions, presets: doc.presets) }
        cancelBox.set { task.cancel() }
        _ = try? await task.value
        #expect(seen.all == ["\(want)/1"], "booked and held while the row ran: \(seen.all), expected \(want)/1")
        #expect(log.count("teardown") == 1, "\(log.all)")
        // The row let go of its stage (the lease rode on it): nothing in use. The warm entry stays on the ledger, evictable.
        var held = budget.inUseCount
        for _ in 0 ..< 100 where held != 0 { try await Task.sleep(for: .milliseconds(100)); held = budget.inUseCount }
        #expect(held == 0, "no hold left after the cancelled row: \(held)")
    }

    private final class CancelBox: @unchecked Sendable {
        private let lock = NSLock()
        private var cancel: (@Sendable () -> Void)?
        private var fired = false
        func set(_ f: @escaping @Sendable () -> Void) { lock.withLock { cancel = f; if fired { f() } } }
        func fire() { let f = lock.withLock { () -> (@Sendable () -> Void)? in fired = true; return cancel }; f?() }
    }

    // MARK: (b) a 9.6 GB ledger (the 16 GB Mac): evict the others, proceed over budget, never refuse

    @Test func onA16GBMacsLedgerTheBookingEvictsEveryOtherModelAndProceedsOverBudget() async throws {
        let entry = try Self.ace()
        let tiny = try AceStepEngineTests.makeFolder()
        let modelDir = try Self.folder()
        defer { try? FileManager.default.removeItem(at: tiny); try? FileManager.default.removeItem(at: modelDir) }
        let want = Self.expected(entry, in: modelDir)
        let capacity = EngineCache.defaultBudget(totalRAMGB: 16)                  // max(4 GB, 16 x 0.6) = 9.6 GiB, the real formula
        #expect(capacity == Int64(9.6 * Double(Self.gib)) && want > capacity, "the booking is larger than the whole ledger")
        let budget = MemoryBudget(capacityBytes: capacity)
        let cache = EngineCache(budget: budget, modelDirectory: { _ in modelDir })

        // Two other models resident and idle (leased, then returned), 3 GiB each.
        let others = try Self.catalog().filter { $0.id != entry.id && $0.runnerKind != .llm }.prefix(2)
        #expect(others.count == 2)
        for other in others {
            let (_, hold) = try await cache.lease(for: other, config: .default, footprintOverride: 3 * Self.gib) { _, _ in Self.probe() }
            hold.release()
        }
        #expect(cache.count == 2 && budget.usedBytes == 6 * Self.gib)

        let log = Log(), during = Log()
        let base = Self.tinyGenerator(tiny, log: log) { phase in
            if case .denoise(let step, _) = phase, step == 1 { during.add("\(budget.usedBytes)/\(cache.count)") }
        }
        let handed = LeasingStage(model: entry, config: .default, probe: Self.probe(), cache: cache, maxWait: .seconds(2))
        let model = AceStepRunModel(generator: AceStepRunModel.leasedGenerator(modelID: entry.id, base: base, handedStage: handed))
        model.generate()
        await model.finished()
        #expect(model.errorText == nil, "not refused: \(String(describing: model.errorText))")
        #expect(model.wavData != nil, "the run's stage ran")
        #expect(during.all == ["\(want)/1"], "both other models were evicted and the run is over budget: \(during.all), expected \(want)/1 (capacity \(capacity))")
        #expect(cache.count == 0 && budget.usedBytes == 0, "and everything is back after the run: \(cache.count) entries, \(budget.usedBytes) B")
    }
}
