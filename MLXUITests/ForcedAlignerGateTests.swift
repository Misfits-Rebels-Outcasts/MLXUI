import Foundation
import Testing
import Observation
@testable import MLXUI

/// Stands in for the load and the forward pass inside the gate. A call whose transcript has a `hold` stays in flight
/// until the test opens it (deaf to cancellation, as `generate` is); any other suspends briefly. Counts how many are in
/// flight at once, which with the real step is how many copies of the model are resident.
private nonisolated final class AlignSteps: @unchecked Sendable {
    private let lock = NSLock()
    private var order: [String] = []
    private var current = 0
    private var most = 0
    private var holds: [String: AlignRunGate] = [:]

    var started: [String] { lock.lock(); defer { lock.unlock() }; return order }
    var peak: Int { lock.lock(); defer { lock.unlock() }; return most }

    func hold(_ transcript: String) -> AlignRunGate {
        lock.lock(); defer { lock.unlock() }
        let gate = holds[transcript] ?? AlignRunGate()
        holds[transcript] = gate
        return gate
    }

    private func enter(_ transcript: String) -> AlignRunGate? {
        lock.lock(); defer { lock.unlock() }
        order.append(transcript)
        current += 1
        most = max(most, current)
        return holds[transcript]
    }

    private func leave() { lock.lock(); current -= 1; lock.unlock() }

    func run(_ transcript: String) async throws -> [AlignedUnit] {
        let held = enter(transcript)
        defer { leave() }
        if let held { await held.wait() } else { try await Task.sleep(for: .milliseconds(50)) }
        return [AlignedUnit(text: transcript, start: 0.1, end: 0.5)]
    }
}

/// A budget participant that holds nothing, for asking the ledger who is in the way.
private nonisolated final class NothingToEvict: MemoryBudgetParticipant {
    func evictLeastRecentlyUsed() async -> Bool { false }
}

/// A test-owned gate, budget and engine cache, and the real `ForcedAlignerStage` calling the real
/// `ForcedAlignerEngine.align`, with only the load and forward pass replaced.
private nonisolated struct GateRig {
    let gate = ForcedAlignerGate()
    let steps = AlignSteps()
    let directory = URL(fileURLWithPath: "/nonexistent-fa-gate-\(UUID().uuidString)")
    let budget = MemoryBudget(capacityBytes: 20 * 1_073_741_824,
                              cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    let cache: EngineCache

    init() {
        cache = EngineCache(budget: budget, countsLLMWeights: true,
                            modelDirectory: { _ in URL(fileURLWithPath: "/nonexistent-fa") })
    }

    var key: String { ForcedAlignerEngine.gateKey(directory) }
    var queued: Int { get async { await gate.queuedCount(key) } }

    /// One engine call.
    func align(_ transcript: String, language: String = "de", directory other: URL? = nil) async throws -> [AlignedUnit] {
        let steps = self.steps
        return try await ForcedAlignerEngine.align(
            [Float](repeating: 0, count: 1_600), transcript: transcript, language: language,
            modelDirectory: other ?? directory, gate: gate,
            step: { _, text, _, _, _ in try await steps.run(text) })
    }

    func stage(language: String) -> ForcedAlignerStage {
        let rig = self
        return ForcedAlignerStage(id: "gated", name: "Qwen3 Aligner 0.6B", language: language) { _, text, language in
            try await rig.align(text, language: language)
        }
    }

    var builder: EngineCache.Builder {
        { _, config in stage(language: config.language ?? "en") }
    }

    /// The stage the Run view builds: `LeasingStage.perRun` over the sheet's `LeasingStage`.
    func sheetStage(_ card: ModelEntry, language: String) -> any PipelineStage {
        let handed = LeasingStage(model: card, config: .default, probe: stage(language: language),
                                  cache: cache, maxWait: .milliseconds(300))
        return LeasingStage.perRun(handed, config: StageConfig(language: language)) { _ in stage(language: language) }
    }

    /// Who holds the booked entries, one per hold: what a reservation that cannot fit is told. Only asked while
    /// something is held.
    func holders() async -> [MemoryBudget.HolderKind?] {
        let probe = NothingToEvict()
        do {
            let reservation = try await budget.reserveBounded(bytes: budget.capacityBytes, for: probe, label: "probe",
                                                              maxWait: .milliseconds(20), pinned: .never)
            budget.release(reservation)
            return []
        } catch let error as MemoryBudget.WaitError {
            return error.blockers.map(\.holder)
        } catch {
            return []
        }
    }
}

/// FA-R17 (owner rulings 20 and 21, 2026-11-27, quoted at `ForcedAlignerGate` and `AlignRunModel.blockedText`): one load
/// and forward pass per model at a time, a wait that can be cancelled, what the Run screen says meanwhile, and what the
/// memory ledger holds while a flow row and a Run sheet are both in flight.
@MainActor
struct ForcedAlignerGateTests {

    private static let cardID = "mlx-community--Qwen3-ForcedAligner-0.6B-8bit"
    private static let sentence = "Waiting for the last alignment to finish…"
    /// The card's booking with no files to measure: `ramGB` 6.6 x 1,073,741,824.
    private static let oneBooking: Int64 = 7_086_696_038

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
            .appendingPathComponent("fa-gate-\(UUID().uuidString)"))
        try FileManager.default.createDirectory(at: ws.root, withIntermediateDirectories: true)
        return ws
    }

    private func wav(in directory: URL) throws -> URL {
        let url = directory.appendingPathComponent("a-\(UUID().uuidString).wav")
        try AudioWriter.writeWAV(AudioBuffer(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000), to: url)
        return url
    }

    private func executor(_ rig: GateRig, _ ws: FlowWorkspace, flowID: String = "f") throws -> RealExecutor {
        RealExecutor(workspace: ws, flowID: flowID, blobDirectory: ws.root.appendingPathComponent("blobs"),
                     makeModelStage: { model, config in
                         try await rig.cache.leasedStage(for: model, config: config, booksWeights: true,
                                                         builder: rig.builder)
                     },
                     installedModelIDs: [Self.cardID], catalog: try loadCatalog())
    }

    private func waitUntil(_ what: Comment, _ condition: () async -> Bool) async throws {
        for _ in 0..<400 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record(what)
    }

    private func entered(_ gate: AlignRunGate) async throws {
        try await waitUntil("the step was never reached") { gate.isEntered }
    }

    // MARK: - One at a time

    /// The step suspends, as the real load does: the turn is held across it.
    @Test func fourConcurrentAlignsOfOneModelRunOneAtATime() async throws {
        let rig = GateRig()
        let words = ["eins", "zwei", "drei", "vier"]
        let results = try await withThrowingTaskGroup(of: String.self) { group in
            for word in words {
                group.addTask { try await rig.align(word).map(\.text).joined() }
            }
            return try await group.reduce(into: [String]()) { $0.append($1) }
        }
        #expect(results.sorted() == words.sorted())
        #expect(rig.steps.started.count == 4)
        #expect(rig.steps.peak == 1, "never two in flight")
    }

    @Test func waitersRunInTheOrderTheyArrivedAndAnotherModelDoesNotWait() async throws {
        let rig = GateRig()
        let first = rig.steps.hold("eins")
        let a = Task { try await rig.align("eins") }
        try await entered(first)
        let b = Task { try await rig.align("zwei") }
        try await waitUntil("zwei never queued") { await rig.queued == 1 }
        let c = Task { try await rig.align("drei") }
        try await waitUntil("drei never queued") { await rig.queued == 2 }

        let other = try await rig.align("autre", directory: URL(fileURLWithPath: "/nonexistent-fa-other"))
        #expect(other.map(\.text) == ["autre"], "another model folder has its own turn")
        #expect(rig.steps.started == ["eins", "autre"])

        first.open()
        _ = try await (a.value, b.value, c.value)
        #expect(rig.steps.started == ["eins", "autre", "zwei", "drei"])
    }

    @Test func aFailedStepFreesTheTurn() async throws {
        let rig = GateRig()
        let samples = [Float](repeating: 0, count: 1_600)
        await #expect(throws: StageError.self) {
            _ = try await ForcedAlignerEngine.align(samples, transcript: "eins", language: "de",
                                                    modelDirectory: rig.directory, gate: rig.gate,
                                                    step: { _, _, _, _, _ in throw CocoaError(.fileNoSuchFile) })
        }
        await #expect(throws: ForcedAlignerEngine.EngineError.noWords) {
            _ = try await ForcedAlignerEngine.align(samples, transcript: "…", language: "de",
                                                    modelDirectory: rig.directory, gate: rig.gate,
                                                    step: { _, _, _, _, _ in [] })
        }
        #expect(try await rig.align("zwei").map(\.text) == ["zwei"])
    }

    // MARK: - A cancelled wait (ruling 20)

    @Test func aCancelledWaiterThrowsAtOnceAndNeverRuns() async throws {
        let rig = GateRig()
        let first = rig.steps.hold("eins")
        let a = Task { try await rig.align("eins") }
        try await entered(first)
        let b = Task { try await rig.align("zwei") }
        try await waitUntil("zwei never queued") { await rig.queued == 1 }
        let c = Task { try await rig.align("drei") }
        try await waitUntil("drei never queued") { await rig.queued == 2 }

        b.cancel()
        // B returns while A is still in flight.
        await #expect(throws: CancellationError.self) { _ = try await b.value }
        #expect(await rig.queued == 1, "B left the queue")
        #expect(rig.steps.started == ["eins"])

        first.open()
        _ = try await (a.value, c.value)
        #expect(rig.steps.started == ["eins", "drei"], "B never ran")
    }

    /// The grant wins the race with the cancellation: the call hears its turn has come, is cancelled there, and still
    /// does not run.
    @Test func aWaiterCancelledJustAsItsTurnComesNeverRuns() async throws {
        let rig = GateRig()
        let first = rig.steps.hold("eins")
        let a = Task { try await rig.align("eins") }
        try await entered(first)
        let heard = AlignRunCallLog()
        let b = Task {
            try await AlignWaiting.$report.withValue({ waiting in
                heard.enter()
                if !waiting { withUnsafeCurrentTask { $0?.cancel() } }
            }) { try await rig.align("zwei") }
        }
        try await waitUntil("zwei never queued") { await rig.queued == 1 }
        first.open()
        _ = try await a.value
        await #expect(throws: CancellationError.self) { _ = try await b.value }
        #expect(heard.count == 2, "queued, then its turn")
        #expect(rig.steps.started == ["eins"])
        #expect(try await rig.align("drei").map(\.text) == ["drei"], "the turn was passed on")
    }

    @Test func aCallAlreadyCancelledDoesNotRunEvenWithTheGateFree() async throws {
        let rig = GateRig()
        let call = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await rig.align("eins")
        }
        await #expect(throws: CancellationError.self) { _ = try await call.value }
        #expect(rig.steps.started.isEmpty)
        #expect(try await rig.align("zwei").map(\.text) == ["zwei"])
    }

    @Test(arguments: [false, true])
    func stopOrDoneOnAWaitingRunScreenRunEndsItWithoutRunning(done: Bool) async throws {
        let rig = GateRig()
        let ws = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let audio = try wav(in: ws.root)
        let tracker = AlignStepTracker()
        let first = rig.steps.hold("eins")
        let a = Task { try await rig.align("eins") }   // another alignment of the model, a flow row's say
        try await entered(first)
        let model = AlignRunModel(modelID: Self.cardID, tracker: tracker)
        model.start(audioURL: audio, transcript: "zwei", language: "de",
                    stage: rig.sheetStage(try card(), language: "de"))
        try await waitUntil("the screen never said it was waiting") { model.progressText == Self.sentence }

        model.stop()                  // Stop, or the Done button
        if done { model.stop() }      // and the sheet disappearing
        await model.task?.value       // returns while the first alignment is still in flight
        #expect(rig.steps.started == ["eins"])
        #expect(model.result == nil && model.errorText == nil)
        #expect(!model.isRunning && !model.isStopping && !model.isWaiting)
        #expect(!tracker.isBusy(Self.cardID))
        #expect(rig.budget.usedBytes == 0 && rig.budget.inUseCount == 0, "its lease went back")
        #expect(await rig.queued == 0)

        first.open()
        _ = try await a.value
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.steps.started == ["eins"], "it never runs")
        #expect(model.result == nil && model.errorText == nil)
    }

    @Test func aCancelledFlowRowThatIsWaitingNeverRuns() async throws {
        let rig = GateRig()
        let ws = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let audio = try wav(in: ws.root)
        let flow = try executor(rig, ws)
        let first = rig.steps.hold("eins")
        let a = Task { try await rig.align("eins") }
        try await entered(first)
        let row = Task {
            try await flow.execute(
                path: "3", row: Row(task: "Align", model: "Qwen3 Aligner 0.6B", settings: "lang=de"),
                inputs: [Asset(items: [Item(kind: .audio, value: nil, path: audio, sourceText: nil)]),
                         Asset(items: [Item(kind: .text, value: "zwei", path: nil, sourceText: nil)])],
                transcript: nil, context: nil, usedFlowContent: nil)
        }
        try await waitUntil("the row never queued") { await rig.queued == 1 }
        #expect(rig.budget.inUseCount == 1, "leased while it waits")

        row.cancel()
        await #expect(throws: CancellationError.self) { _ = try await row.value }
        #expect(rig.budget.inUseCount == 0)
        first.open()
        _ = try await a.value
        #expect(rig.steps.started == ["eins"])
    }

    /// The same through `FlowRunner`, cancelled the way a flow is: the task reading its events is cancelled.
    @Test func cancellingAFlowRunDropsItsWaitingAlignRow() async throws {
        let rig = GateRig()
        let ws = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let flowID = "align-wait"
        let directory = ws.directory(for: flowID)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: try wav(in: ws.root), to: directory.appendingPathComponent("memo.wav"))
        try "zwei".write(to: directory.appendingPathComponent("memo.txt"), atomically: true, encoding: .utf8)
        let document = try CatParser.parse("""
        mlxflow 0.8
        1. Read Audio   memo.wav
        2. Read Text    memo.txt
        3. Align (1,2)  Qwen3 Aligner 0.6B; lang=de
        """)
        let exec = try executor(rig, ws, flowID: flowID)
        let context = FlowRunner.RunContext(flowID: flowID, workspace: ws,
                                            blobDirectory: exec.blobDirectory, executor: exec)
        let first = rig.steps.hold("eins")
        let a = Task { try await rig.align("eins") }
        try await entered(first)
        let run = Task {
            for await _ in FlowRunner().run(document, context: context) {}
        }
        try await waitUntil("the Align row never queued") { await rig.queued == 1 }

        run.cancel()
        try await waitUntil("the row stayed queued") { await rig.queued == 0 }
        try await waitUntil("the row kept its lease") { rig.budget.inUseCount == 0 }
        first.open()
        _ = try await a.value
        try await Task.sleep(for: .milliseconds(100))
        #expect(rig.steps.started == ["eins"])
    }

    // MARK: - What the screens say (rulings 20 and 21)

    @Test func theRunScreenSaysItIsWaitingWhileQueued() async throws {
        let rig = GateRig()
        let ws = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let audio = try wav(in: ws.root)
        let first = rig.steps.hold("eins")
        let second = rig.steps.hold("zwei")
        let a = Task { try await rig.align("eins") }
        try await entered(first)
        let model = AlignRunModel(modelID: Self.cardID, tracker: AlignStepTracker())
        model.start(audioURL: audio, transcript: "zwei", language: "de",
                    stage: rig.sheetStage(try card(), language: "de"))
        try await waitUntil("the screen never said it was waiting") { model.progressText == Self.sentence }
        #expect(model.isRunning && model.isWaiting && !model.canChangeInputs)
        #expect(model.blockedText == nil, "the running sheet says it once, on its running row")
        #expect(!second.isEntered)

        first.open()
        _ = try await a.value
        try await entered(second)
        try await waitUntil("still saying it was waiting") { model.progressText == "Timing the transcript…" }
        #expect(!model.isWaiting)

        second.open()
        await model.task?.value
        #expect(model.result?.units.map(\.text) == ["zwei"])

        // A run with nothing ahead of it never waits.
        let third = rig.steps.hold("drei")
        model.start(audioURL: audio, transcript: "drei", language: "de",
                    stage: rig.sheetStage(try card(), language: "de"))
        try await entered(third)
        try await Task.sleep(for: .milliseconds(100))
        #expect(model.progressText == "Timing the transcript…" && !model.isWaiting)
        third.open()
        await model.task?.value
    }

    @Test func aReopenedDisabledSheetSaysWhy() async throws {
        let rig = GateRig()
        let ws = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let audio = try wav(in: ws.root)
        let tracker = AlignStepTracker()
        let step = rig.steps.hold("eins")
        let first = AlignRunModel(modelID: Self.cardID, tracker: tracker)
        #expect(first.blockedText == nil)
        first.start(audioURL: audio, transcript: "eins", language: "de",
                    stage: rig.sheetStage(try card(), language: "de"))
        try await entered(step)
        #expect(first.blockedText == nil)
        first.stop()   // Done

        let reopened = AlignRunModel(modelID: Self.cardID, tracker: tracker)
        #expect(!reopened.canChangeInputs)
        let changed = AlignRunCallLog()
        withObservationTracking {
            #expect(reopened.blockedText == Self.sentence)
        } onChange: { changed.enter() }
        #expect(AlignRunModel(modelID: "another-aligner", tracker: tracker).blockedText == nil)

        step.open()
        await first.task?.value
        #expect(changed.count == 1, "an open sheet is told")
        #expect(reopened.blockedText == nil && reopened.canChangeInputs)

        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("MLXUI/Modules/ForcedAligner/ForcedAlignerUI.swift"),
                                encoding: .utf8)
        #expect(source.contains("if let blocked = model.blockedText { blockedRow(blocked) }"), "the view shows it")
    }

    // MARK: - The booking, with a flow row and a Run sheet both in flight

    /// A flow-style row, as `AppFlowExecutorFactory` leases it.
    private func flowRow(_ rig: GateRig, _ card: ModelEntry, transcript: String, language: String) -> Task<Media, any Error> {
        Task {
            let leased = try await rig.cache.leasedStage(for: card, config: StageConfig(language: language),
                                                         booksWeights: true, builder: rig.builder)
            let buffer = AudioBuffer(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000)
            return try await AlignTranscript.$current.withValue(transcript) {
                try await leased.run(.audio(buffer)) { _ in }
            }
        }
    }

    /// Observed, same language ("en"), 1 booking of 7,086,696,038 bytes throughout both runs:
    /// - both in flight: 1 cache entry, 1 booking, 2 holds (flow + Run window), 1 resident load
    /// - flow first, flow done: 1 cache entry, 1 booking, 1 hold (Run window)
    /// - sheet first, sheet done: 0 cache entries (the sheet drops its entry), the booking stays for the flow's 1 hold
    /// - both done: 0 entries, 0 bytes
    @Test(arguments: [true, false])
    func oneLanguageIsOneEntryOneBookingTwoHoldsAndOneResidentLoad(flowFirst: Bool) async throws {
        let rig = GateRig()
        let card = try card()
        let ws = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let audio = try wav(in: ws.root)
        #expect(rig.cache.footprint(of: card) == Self.oneBooking)
        let flowStep = rig.steps.hold("Morgen")
        let sheetStep = rig.steps.hold("Abend")
        let model = AlignRunModel(modelID: card.id, tracker: AlignStepTracker())
        func startSheet() {
            model.start(audioURL: audio, transcript: "Abend", language: "en",
                        stage: rig.sheetStage(card, language: "en"))
        }

        let flow: Task<Media, any Error>
        if flowFirst {
            flow = flowRow(rig, card, transcript: "Morgen", language: "en")
            try await entered(flowStep)
            startSheet()
        } else {
            startSheet()
            try await entered(sheetStep)
            flow = flowRow(rig, card, transcript: "Morgen", language: "en")
        }
        try await waitUntil("the second call never queued") { await rig.queued == 1 }

        #expect(rig.cache.count == 1)
        #expect(rig.budget.heldCount == 1 && rig.budget.usedBytes == Self.oneBooking)
        #expect(rig.budget.inUseCount == 1)
        #expect(await rig.holders() == (flowFirst ? [.flow, .runWindow] : [.runWindow, .flow]))
        #expect(rig.steps.started.count == 1)

        if flowFirst {
            flowStep.open()
            _ = try await flow.value
            try await entered(sheetStep)
            #expect(rig.cache.count == 1)
            #expect(await rig.holders() == [.runWindow])
        } else {
            sheetStep.open()
            await model.task?.value
            try await entered(flowStep)
            #expect(rig.cache.count == 0)
            #expect(await rig.holders() == [.flow])
        }
        #expect(rig.budget.heldCount == 1 && rig.budget.usedBytes == Self.oneBooking)

        flowStep.open()
        sheetStep.open()
        _ = try await flow.value
        await model.task?.value
        #expect(model.result?.units.map(\.text) == ["Abend"])
        #expect(rig.cache.count == 0 && rig.budget.heldCount == 0 && rig.budget.usedBytes == 0)
        #expect(rig.steps.peak == 1, "one resident load at a time")
    }

    /// Observed, "en" flow row + "de" Run sheet:
    /// - both in flight: 2 cache entries, 2 bookings (14,173,392,076 bytes), 1 hold each, 1 resident load (one model
    ///   folder, one turn)
    /// - flow done: 2 entries, 2 bookings, 1 hold (Run window)
    /// - both done: 1 entry, 1 booking of 7,086,696,038 bytes, 0 holds (the flow's entry stays warm and evictable; the
    ///   sheet drops its own)
    @Test func twoLanguagesAreTwoEntriesTwoBookingsAndStillOneResidentLoad() async throws {
        let rig = GateRig()
        let card = try card()
        let ws = try temporaryWorkspace()
        defer { try? FileManager.default.removeItem(at: ws.root) }
        let audio = try wav(in: ws.root)
        let flowStep = rig.steps.hold("morning")
        let sheetStep = rig.steps.hold("Abend")
        let model = AlignRunModel(modelID: card.id, tracker: AlignStepTracker())

        let flow = flowRow(rig, card, transcript: "morning", language: "en")
        try await entered(flowStep)
        model.start(audioURL: audio, transcript: "Abend", language: "de", stage: rig.sheetStage(card, language: "de"))
        try await waitUntil("the sheet never queued") { await rig.queued == 1 }

        #expect(rig.cache.count == 2)
        #expect(rig.budget.heldCount == 2 && rig.budget.usedBytes == 2 * Self.oneBooking)
        #expect(rig.budget.inUseCount == 2)
        let holders = await rig.holders()
        #expect(holders.count == 2 && holders.contains(.flow) && holders.contains(.runWindow))
        #expect(!sheetStep.isEntered)

        flowStep.open()
        _ = try await flow.value
        try await entered(sheetStep)
        #expect(rig.cache.count == 2, "the flow's entry stays warm")
        #expect(rig.budget.usedBytes == 2 * Self.oneBooking && rig.budget.inUseCount == 1)

        sheetStep.open()
        await model.task?.value
        #expect(rig.cache.count == 1 && rig.budget.usedBytes == Self.oneBooking && rig.budget.inUseCount == 0,
                "the sheet's entry is dropped; the flow's is warm and evictable")
        #expect(rig.steps.peak == 1, "one resident load at a time")
    }
}
