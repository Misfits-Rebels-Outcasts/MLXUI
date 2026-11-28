import Foundation
import Testing
@testable import MLXUI

/// What the ledger said while a call was inside the step.
private nonisolated struct LedgerReading: Equatable, Sendable {
    let entries: Int
    let held: Int
    let bytes: Int64
    /// What the `EngineCache` itself booked at that moment (`nil` where no cache is involved).
    var cached: Int64? = nil
}

/// Stands in for the load and the forward pass: reads the test-owned ledger, then returns, throws or waits.
private nonisolated final class BookingSeam: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [LedgerReading] = []
    private var holds: [String: AlignRunGate] = [:]

    var readings: [LedgerReading] { lock.lock(); defer { lock.unlock() }; return recorded }

    func hold(_ transcript: String) -> AlignRunGate {
        lock.lock(); defer { lock.unlock() }
        let gate = holds[transcript] ?? AlignRunGate()
        holds[transcript] = gate
        return gate
    }

    private func note(_ reading: LedgerReading, _ transcript: String) -> AlignRunGate? {
        lock.lock(); defer { lock.unlock() }
        recorded.append(reading)
        return holds[transcript]
    }

    func step(_ budget: MemoryBudget, cache: EngineCache? = nil) -> ForcedAlignerEngine.Step {
        { _, text, _, _, _ in
            let reading = LedgerReading(entries: budget.heldCount, held: budget.inUseCount, bytes: budget.usedBytes,
                                        cached: cache?.totalCachedBytes)
            if let held = self.note(reading, text) { await held.wait() }
            return [AlignedUnit(text: text, start: 0.1, end: 0.5)]
        }
    }
}

/// A budget participant that holds nothing it will give back.
private nonisolated final class Bystander: MemoryBudgetParticipant {
    func evictLeastRecentlyUsed() async -> Bool { false }
}

/// FA-R20 — an alignment's booking on the shared ledger, with a test-owned budget and the load and forward pass replaced.
///
/// Owner ruling 7 (Boo, 2026-11-27): "Set it from your own FA-7 run: the highest measured peak rounded up to 0.1 GB.
/// Keep the `.align` booking at or above that peak."
@MainActor
struct ForcedAlignerBookingTests {

    private nonisolated static let cardID = "mlx-community--Qwen3-ForcedAligner-0.6B-8bit"
    /// The `.align` row's per-call working set.
    private nonisolated static let workingSet: Int64 = 5_250_000_000
    /// A 1,000-byte safetensors file: 1,100 booked for the file, plus the working set.
    private nonisolated static let smallBooking: Int64 = 5_250_001_100
    /// The highest measured peak (ruling 7's run).
    private nonisolated static let measuredPeak: Int64 = 6_538_000_000
    /// With no files: the card's `ramGB` 6.6 x 1,073,741,824, plus the working set.
    private nonisolated static let fallback: Int64 = 7_086_696_038
    private nonisolated static let fallbackBooking: Int64 = 12_336_696_038
    private nonisolated static let samples = [Float](repeating: 0, count: 1_600)

    private nonisolated static func newBudget(_ capacity: Int64 = 40 * 1_073_741_824) -> MemoryBudget {
        MemoryBudget(capacityBytes: capacity,
                     cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    private func modelFolder(bytes: Int = 1_000) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("fa-booking-\(UUID().uuidString)/mlx-community--Fake-Aligner", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try Data(count: bytes).write(to: url.appendingPathComponent("model.safetensors"))
        return url
    }

    private func align(_ transcript: String, _ directory: URL, _ budget: MemoryBudget, _ seam: BookingSeam,
                       gate: ForcedAlignerGate, catalogFallback: Int64? = nil,
                       maxWait: Duration = .seconds(5), step: ForcedAlignerEngine.Step? = nil) async throws -> [AlignedUnit] {
        try await ForcedAlignerEngine.align(Self.samples, transcript: transcript, language: "de",
                                            modelDirectory: directory, catalogFallback: catalogFallback,
                                            budget: budget, maxWait: maxWait, gate: gate,
                                            step: step ?? seam.step(budget))
    }

    private func waitUntil(_ what: Comment, _ condition: () async -> Bool) async throws {
        for _ in 0..<400 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record(what)
    }

    /// Another model, booked and in use.
    private func occupy(_ budget: MemoryBudget, bytes: Int64, owner: Bystander) async -> (MemoryBudget.Reservation, MemoryBudget.Hold) {
        let reservation = await budget.reserve(bytes: bytes, for: owner, label: "mlx-community--Other-Model")
        let hold = budget.hold(reservation, label: "mlx-community--Other-Model", holder: .chat)
        budget.commit(reservation)
        return (reservation, hold)
    }

    // MARK: - (a) What a call holds

    @Test func aCallHoldsTheFilesChargeAndTheWorkingSetAndGivesItBack() async throws {
        let directory = try modelFolder()
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let budget = Self.newBudget(), seam = BookingSeam()
        #expect(ModelLoadProfile.workingSetBytes(for: .align) == Self.workingSet)
        #expect(ModelLoadProfile.charge(kind: .align, directory: directory, catalogFallback: Self.fallback)
                + Self.workingSet == Self.smallBooking, "files on disk win over the catalog figure")

        _ = try await align("eins", directory, budget, seam, gate: ForcedAlignerGate(), catalogFallback: Self.fallback)

        #expect(seam.readings == [LedgerReading(entries: 1, held: 1, bytes: Self.smallBooking)])
        #expect(budget.usedBytes == 0 && budget.heldCount == 0 && budget.inUseCount == 0)
    }

    @Test func withNoFilesToMeasureTheCatalogFigureStandsInForThem() async throws {
        let budget = Self.newBudget(), seam = BookingSeam()
        let nowhere = URL(fileURLWithPath: "/nonexistent-fa-booking-\(UUID().uuidString)")
        _ = try await align("eins", nowhere, budget, seam, gate: ForcedAlignerGate(), catalogFallback: Self.fallback)
        #expect(seam.readings == [LedgerReading(entries: 1, held: 1, bytes: Self.fallbackBooking)])
        #expect(budget.usedBytes == 0)
    }

    /// The pinned install's weights (1.272 GB, FA-7) book at or above the measured peak.
    @Test func theRealInstallsSizeBooksAtOrAboveTheMeasuredPeak() {
        let booked = ModelWeightCharge.charge(onDiskBytes: 1_272_000_000) + ModelLoadProfile.workingSetBytes(for: .align)
        #expect(booked >= Self.measuredPeak)
    }

    private nonisolated enum Ending: CaseIterable, Sendable { case engineFailure, noWords, cancellation }

    @Test(arguments: [Ending.engineFailure, .noWords, .cancellation])
    private func theBookingComesBackHoweverTheStepEnds(_ ending: Ending) async throws {
        let directory = try modelFolder()
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let budget = Self.newBudget(), seam = BookingSeam()
        let read = seam.step(budget)
        let step: ForcedAlignerEngine.Step = { samples, text, language, duration, folder in
            _ = try await read(samples, text, language, duration, folder)
            switch ending {
            case .engineFailure: throw CocoaError(.fileNoSuchFile)
            case .noWords: return []
            case .cancellation: throw CancellationError()
            }
        }
        await #expect(throws: (any Error).self) {
            _ = try await align("eins", directory, budget, seam, gate: ForcedAlignerGate(), step: step)
        }
        #expect(seam.readings == [LedgerReading(entries: 1, held: 1, bytes: Self.smallBooking)])
        #expect(budget.usedBytes == 0 && budget.inUseCount == 0)
    }

    @Test func aCancelledCallKeepsItsBookingUntilTheStepReturns() async throws {
        let directory = try modelFolder()
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let budget = Self.newBudget(), seam = BookingSeam(), gate = ForcedAlignerGate()
        let held = seam.hold("eins")
        let call = Task { try await align("eins", directory, budget, seam, gate: gate) }
        try await waitUntil("the step was never reached") { held.isEntered }
        call.cancel()
        try await Task.sleep(for: .milliseconds(100))
        #expect(budget.usedBytes == Self.smallBooking && budget.inUseCount == 1, "the model is still resident")
        held.open()
        _ = try? await call.value
        #expect(budget.usedBytes == 0 && budget.inUseCount == 0)
    }

    @Test func theBookingNamesTheModelAndWhoIsRunningIt() async throws {
        let directory = try modelFolder()
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        for expected in [MemoryBudget.HolderKind.flow, .runWindow] {
            let budget = Self.newBudget(), seam = BookingSeam(), gate = ForcedAlignerGate()
            let held = seam.hold("eins")
            let call = Task {
                if expected == .runWindow {
                    try await MemoryBudget.RunWindowScope.$holder.withValue(.runWindow) {
                        try await align("eins", directory, budget, seam, gate: gate)
                    }
                } else {
                    try await align("eins", directory, budget, seam, gate: gate)
                }
            }
            try await waitUntil("the step was never reached") { held.isEntered }
            var blockers: [MemoryBudget.Blocker] = []
            do {
                let probe = try await budget.reserveBounded(bytes: budget.capacityBytes, for: Bystander(), label: "probe",
                                                            maxWait: .milliseconds(20), pinned: .never)
                budget.release(probe)
            } catch let error as MemoryBudget.WaitError {
                blockers = error.blockers
            }
            #expect(blockers == [MemoryBudget.Blocker(model: "Fake-Aligner", holder: expected)])
            held.open()
            _ = try await call.value
        }
    }

    @Test func anIdleModelIsEvictedToMakeRoom() async throws {
        let directory = try modelFolder()
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let budget = Self.newBudget(6_000_000_000), seam = BookingSeam()
        let pool = ResidentPool<Int>(budget: budget)
        _ = try await pool.value(for: "idle", bytes: 2_000_000_000) { 1 }
        _ = try await align("eins", directory, budget, seam, gate: ForcedAlignerGate())
        #expect(seam.readings == [LedgerReading(entries: 1, held: 1, bytes: Self.smallBooking)], "the idle model went")
        #expect(await pool.residentKeys.isEmpty)
        #expect(budget.usedBytes == 0)
    }

    // MARK: - (c) One reservation at a time

    @Test func twoQueuedCallsNeverHoldTwoReservations() async throws {
        let directory = try modelFolder()
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let budget = Self.newBudget(), seam = BookingSeam(), gate = ForcedAlignerGate()
        let first = seam.hold("eins"), second = seam.hold("zwei")
        let a = Task { try await align("eins", directory, budget, seam, gate: gate) }
        try await waitUntil("the step was never reached") { first.isEntered }
        let b = Task { try await align("zwei", directory, budget, seam, gate: gate) }
        try await waitUntil("zwei never queued") { await gate.queuedCount(ForcedAlignerEngine.gateKey(directory)) == 1 }
        #expect(budget.heldCount == 1 && budget.usedBytes == Self.smallBooking, "a queued call books nothing")
        first.open()
        _ = try await a.value
        try await waitUntil("zwei never ran") { second.isEntered }
        #expect(budget.heldCount == 1 && budget.usedBytes == Self.smallBooking)
        second.open()
        _ = try await b.value
        let one = LedgerReading(entries: 1, held: 1, bytes: Self.smallBooking)
        #expect(seam.readings == [one, one])
        #expect(budget.usedBytes == 0)
    }

    // MARK: - (d) No room

    @Test func noRoomIsAOneSentenceRefusalAndTheStepNeverRuns() async throws {
        let directory = try modelFolder()
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let budget = Self.newBudget(6_000_000_000), seam = BookingSeam(), gate = ForcedAlignerGate(), other = Bystander()
        let (reservation, hold) = await occupy(budget, bytes: 2_000_000_000, owner: other)
        let blocker = MemoryBudget.Blocker(model: "Other-Model", holder: .chat)

        await #expect(throws: ForcedAlignerEngine.EngineError.memoryBusy(blocker)) {
            _ = try await align("eins", directory, budget, seam, gate: gate, maxWait: .milliseconds(100))
        }
        #expect(seam.readings.isEmpty, "the step never runs")
        #expect(budget.heldCount == 1 && budget.usedBytes == 2_000_000_000, "nothing of the aligner's is left booked")
        let error: any Error = ForcedAlignerEngine.EngineError.memoryBusy(blocker)
        #expect(MLXErrorGuard.sentence(for: error)
                == "There isn't room in memory to align yet because Other-Model is in use by Chat, so try again when it finishes.")
        #expect(ForcedAlignerEngine.EngineError.memoryBusy(nil).errorDescription
                == "There isn't room in memory to align yet because another model is still loading or in use, so try again when it finishes.")

        budget.release(reservation)
        hold.release()
        _ = try await align("zwei", directory, budget, seam, gate: gate)   // the turn was freed
        #expect(seam.readings.count == 1)
    }

    @Test func aCallCancelledWhileItWaitsForRoomIsACancellationNotARefusal() async throws {
        let directory = try modelFolder()
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let budget = Self.newBudget(6_000_000_000), seam = BookingSeam(), other = Bystander()
        let (reservation, hold) = await occupy(budget, bytes: 2_000_000_000, owner: other)
        let call = Task { try await align("eins", directory, budget, seam, gate: ForcedAlignerGate(), maxWait: .seconds(30)) }
        try await Task.sleep(for: .milliseconds(150))
        #expect(seam.readings.isEmpty, "waiting for room")
        call.cancel()
        await #expect(throws: CancellationError.self) { _ = try await call.value }
        #expect(seam.readings.isEmpty)
        #expect(budget.usedBytes == 2_000_000_000)
        budget.release(reservation)
        hold.release()
    }

    /// The app-wide rule (S1-1b step 4), unchanged: with nothing else held, one model larger than the whole capacity
    /// proceeds over budget.
    @Test func aloneAndLargerThanTheCapacityItStillRunsAsEveryLoadDoes() async throws {
        let directory = try modelFolder()
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let budget = Self.newBudget(1_000_000_000), seam = BookingSeam()
        _ = try await align("eins", directory, budget, seam, gate: ForcedAlignerGate())
        #expect(seam.readings == [LedgerReading(entries: 1, held: 1, bytes: Self.smallBooking)])
        #expect(budget.usedBytes == 0)
    }

    // MARK: - (b) The flow path and the Run window, with `booksWeights` derived from the SDK

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    private func card() throws -> ModelEntry {
        try #require(try loadCatalog().first { $0.id == Self.cardID })
    }

    /// What `AppFlowExecutorFactory` and the Run sheet derive from the registry.
    private func booksWeights(_ model: ModelEntry) throws -> Bool {
        let registry = ModelRegistry()
        ForcedAlignerModule.register(into: registry)
        let sdk = try #require(registry.bestModule(for: model)?.sdk)
        return !(sdk is StageBooksOwnWeights)
    }

    /// A test-owned budget and engine cache, and the app's stage on a model folder that does not exist (so the booking
    /// is the catalog fallback plus the working set), with only the load and forward pass replaced.
    private struct AppRig {
        let budget = ForcedAlignerBookingTests.newBudget()
        let cache: EngineCache
        let seam = BookingSeam()
        let modelID = "fa-booking-no-such-model-\(UUID().uuidString)"

        init() {
            cache = EngineCache(budget: budget, countsLLMWeights: true,
                                modelDirectory: { _ in URL(fileURLWithPath: "/nonexistent-fa-booking") })
        }

        func stage(_ language: String) -> ForcedAlignerStage {
            ForcedAlignerStage(modelID: modelID, language: language,
                               footprintBytes: ForcedAlignerBookingTests.fallback,
                               budget: budget, step: seam.step(budget, cache: cache))
        }
    }

    private func workspace() throws -> (FlowWorkspace, URL) {
        let ws = FlowWorkspace(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("fa-booking-ws-\(UUID().uuidString)"))
        try FileManager.default.createDirectory(at: ws.root, withIntermediateDirectories: true)
        let wav = ws.root.appendingPathComponent("a.wav")
        try AudioWriter.writeWAV(AudioBuffer(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000), to: wav)
        return (ws, wav)
    }

    private func alignRow(_ rig: AppRig, _ ws: FlowWorkspace, _ audio: URL) async throws -> Asset {
        let books = try booksWeights(try card())
        let executor = RealExecutor(workspace: ws, flowID: "f", blobDirectory: ws.root.appendingPathComponent("blobs"),
                                    makeModelStage: { model, config in
                                        try await rig.cache.leasedStage(for: model, config: config, booksWeights: books) { _, config in
                                            rig.stage(config.language ?? "en")
                                        }
                                    },
                                    installedModelIDs: [Self.cardID], catalog: try loadCatalog())
        return try await executor.execute(
            path: "3", row: Row(task: "Align", model: "Qwen3 Aligner 0.6B", settings: "lang=de"),
            inputs: [Asset(items: [Item(kind: .audio, value: nil, path: audio, sourceText: nil)]),
                     Asset(items: [Item(kind: .text, value: "Morgen", path: nil, sourceText: nil)])],
            transcript: nil, context: nil, usedFlowContent: nil)
    }

    @Test func theSDKSaysItsStageBooksItsOwnWeights() throws {
        #expect((ForcedAlignerSDK() as any ModelSDK) is StageBooksOwnWeights)
        #expect(try booksWeights(try card()) == false)
    }

    @Test func aFlowRowsOnlyAlignerBookingIsTheEnginesReservation() async throws {
        let rig = AppRig()
        let (ws, audio) = try workspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        _ = try await alignRow(rig, ws, audio)
        #expect(rig.seam.readings == [LedgerReading(entries: 1, held: 1, bytes: Self.fallbackBooking, cached: 0)])
        #expect(rig.cache.count == 1 && rig.cache.totalCachedBytes == 0, "the cached stage books nothing")
        #expect(rig.budget.usedBytes == 0 && rig.budget.heldCount == 0)
    }

    @Test func aFlowRowWaitsForRoomInsteadOfLoading() async throws {
        let rig = AppRig()
        let (ws, audio) = try workspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        // The engine's wait is the app's 60 s here, so the row is cancelled rather than left to it.
        let other = Bystander()
        let (reservation, hold) = await occupy(rig.budget, bytes: rig.budget.capacityBytes - 1_000, owner: other)
        let row = Task { try await alignRow(rig, ws, audio) }
        try await Task.sleep(for: .milliseconds(200))
        #expect(rig.seam.readings.isEmpty, "the row waits for room instead of loading")
        row.cancel()
        await #expect(throws: CancellationError.self) { _ = try await row.value }
        #expect(rig.seam.readings.isEmpty)
        rig.budget.release(reservation)
        hold.release()
        #expect(rig.budget.usedBytes == 0)
    }

    /// Ruling 11 with the engine's booking: Stop releases nothing until the step returns.
    @Test(arguments: [false, true])
    func aRunWindowRunHoldsOneBookingUntilTheStepReturns(stop: Bool) async throws {
        let rig = AppRig()
        let card = try card()
        let (ws, audio) = try workspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let handed = LeasingStage(model: card, config: .default, probe: rig.stage("de"),
                                  booksWeights: try booksWeights(card), cache: rig.cache, maxWait: .milliseconds(300))
        let stage = LeasingStage.perRun(handed, config: StageConfig(language: "de")) { _ in rig.stage("de") }
        let held = rig.seam.hold("Abend")
        let model = AlignRunModel(modelID: card.id, tracker: AlignStepTracker())
        model.start(audioURL: audio, transcript: "Abend", language: "de", stage: stage)
        try await waitUntil("the step was never reached") { held.isEntered }
        #expect(rig.seam.readings == [LedgerReading(entries: 1, held: 1, bytes: Self.fallbackBooking, cached: 0)])
        if stop { model.stop() }
        try await Task.sleep(for: .milliseconds(150))
        #expect(rig.budget.usedBytes == Self.fallbackBooking && rig.budget.inUseCount == 1)
        held.open()
        await model.task?.value
        #expect(rig.budget.usedBytes == 0 && rig.budget.inUseCount == 0 && rig.cache.count == 0)
        #expect(model.errorText == nil)
        #expect((model.result == nil) == stop)
    }

    // MARK: - The "make room?" question

    @Test func theRoomQuestionAsksForTheWholeBooking() async throws {
        let directory = try modelFolder()
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let card = try card()
        let cache = EngineCache(budget: Self.newBudget(), countsLLMWeights: true, modelDirectory: { _ in directory })
        let books = try booksWeights(card)
        let sheet = LeasingStage(model: card, config: .default,
                                 probe: ForcedAlignerStage(modelID: card.id, footprintBytes: Self.fallback),
                                 booksWeights: books, cache: cache)
        #expect(await sheet.bytesNeeded() == Self.smallBooking)
        #expect(await FlowServedLoadGate.bytesNeeded(card, cache: cache, booksWeights: books) == Self.smallBooking)
    }
}
