import Testing
import Foundation
@testable import MLXUI

/// SC-5b (gate D) — a flow row asks before its model load would stop keeping a served model ready. The executor's gate
/// throws a `ServedLoadQuestion`, the interpreter parks the run on it like a human row, the session shows it (and the
/// Local Server screen lists it) until a person answers: "Load anyway" pushes the models out for the row, "Cancel" fails
/// the row with the question's sentence. Nothing answers it for them — an unattended firing included.
private let gb = Int64(1_073_741_824)

private final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _pushed: [[String]] = []
    private var _checks = 0
    func push(_ ids: [String]) { lock.lock(); _pushed.append(ids); lock.unlock() }
    func check() { lock.lock(); _checks += 1; lock.unlock() }
    var pushed: [[String]] { lock.lock(); defer { lock.unlock() }; return _pushed }
    var checks: Int { lock.lock(); defer { lock.unlock() }; return _checks }
}

private func gate(flow: String = "flow-a", ids: [String] = ["mlx-community/Served-4bit"], bytes: Int64 = 4 * gb,
                  recorder: Recorder = Recorder(), decisions: ServedLoadDecisions = .shared,
                  enabled: Bool = true) -> FlowServedLoadGate {
    FlowServedLoadGate(flowID: flow, bytesNeeded: { _ in bytes },
                       check: { _ in recorder.check(); return ids },
                       pushOut: { recorder.push($0) },
                       reloadSeconds: { _ in 9 },
                       decisions: decisions, isEnabled: { enabled })
}

struct ServeAskFlowGateTests {
    private let model = makeEntry(id: "row-model", displayName: "Row Model")

    @Test func nothingInTheWayRunsStraightThrough() async throws {
        let g = gate(ids: [])
        try await g(path: "2", model: model)
    }

    @Test func aDisplacementWithNoAnswerThrowsTheQuestion() async throws {
        let g = gate(decisions: ServedLoadDecisions())
        do { try await g(path: "2", model: model); Issue.record("expected the question") } catch let q as ServedLoadQuestion {
            #expect(q.servedIDs == ["mlx-community/Served-4bit"])
            #expect(q.sentence == "Loading Row Model will stop keeping Served-4bit ready for connected apps. "
                    + "It stays served and reloads on the next request (about 9 s).")
        }
    }

    @Test func loadAnywayPushesOutAndIsConsumed() async throws {
        let rec = Recorder(), decisions = ServedLoadDecisions()
        let g = gate(recorder: rec, decisions: decisions)
        decisions.set(.loadAnyway, flowID: "flow-a", path: "2")
        try await g(path: "2", model: model)
        #expect(rec.pushed == [["mlx-community/Served-4bit"]])
        await #expect(throws: ServedLoadQuestion.self) { try await g(path: "2", model: model) }      // one-shot
    }

    @Test func cancelFailsTheRowWithTheSentenceAndPushesNothing() async throws {
        let rec = Recorder(), decisions = ServedLoadDecisions()
        let g = gate(recorder: rec, decisions: decisions)
        decisions.set(.cancel, flowID: "flow-a", path: "2")
        do { try await g(path: "2", model: model); Issue.record("expected a failure") } catch let FlowError.stageFailure(row, message) {
            #expect(row == "2" && message.hasPrefix("Loading Row Model will stop keeping Served-4bit ready"))
        }
        #expect(rec.pushed.isEmpty)
    }

    @Test func anAnswerIsForThatFlowAndThatRowOnly() async throws {
        let decisions = ServedLoadDecisions()
        decisions.set(.loadAnyway, flowID: "other-flow", path: "2")
        decisions.set(.loadAnyway, flowID: "flow-a", path: "3")
        let g = gate(decisions: decisions)
        await #expect(throws: ServedLoadQuestion.self) { try await g(path: "2", model: model) }
    }

    @Test func gateClosedMeansNoQuestionAndNoCheck() async throws {
        let rec = Recorder()
        let g = gate(recorder: rec, enabled: false)
        try await g(path: "2", model: model)
        #expect(rec.checks == 0)
    }

    @Test func aModelAlreadyResidentAsksNothing() async throws {
        let rec = Recorder()
        let g = gate(bytes: 0, recorder: rec)
        try await g(path: "2", model: model)
        #expect(rec.checks == 0)
    }

    @Test func realExecutorGatesAModelRowBeforeAnyStageIsBuilt() async throws {
        let entry = makeEntry(id: "gated", modelType: .llm, source: .mlx, hfModelId: "mlx-community/Test-Gated-4bit")
        let built = Recorder()
        var executor = RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory.appendingPathComponent("sc5b-\(UUID().uuidString)")),
            flowID: "flow-a",
            blobDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("sc5b-blob-\(UUID().uuidString)"),
            makeModelStage: { _, _ in built.check(); throw StageError.modelNotInstalled(id: "none") },
            installedModelIDs: [entry.id], catalog: [entry])
        let g = gate(decisions: ServedLoadDecisions())
        executor.servedLoadGate = { path, model in try await g(path: path, model: model) }
        let row = Row(task: "Extract Structured", model: entry.hfModelId, settings: "name")
        await #expect(throws: ServedLoadQuestion.self) {
            _ = try await executor.execute(path: "1", row: row, inputs: [Asset(items: [Item(kind: .text, value: "x", path: nil, sourceText: nil)])],
                                           transcript: nil, context: nil, usedFlowContent: nil)
        }
        #expect(built.checks == 0)                                   // asked before anything was loaded
    }
}

/// A `MockExecutor` whose row 2 goes through a served-load gate first, as `RealExecutor` does.
private final class GatedExecutor: FlowExecutor, @unchecked Sendable {
    let mock: MockExecutor
    let gate: FlowServedLoadGate
    let model = makeEntry(id: "row-model", displayName: "Row Model")
    init(blobs: URL, gate: FlowServedLoadGate) { mock = MockExecutor(blobDirectory: blobs); self.gate = gate }
    func execute(path: String, row: Row, inputs: [Asset], transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?, usedFlowContent: String?) async throws -> Asset {
        if path == "2" { try await gate(path: path, model: model) }
        return try await mock.execute(path: path, row: row, inputs: inputs, transcript: transcript,
                                      context: context, usedFlowContent: usedFlowContent)
    }
    var lastCacheHit: Bool { mock.lastCacheHit }
    var lastTag: String? { mock.lastTag }
    var lastTimeoutFlag: (code: String, message: String)? { mock.lastTimeoutFlag }
    var lastProviderDeciderFlag: (code: String, message: String)? { mock.lastProviderDeciderFlag }
    var lastStaged: (id: String, kind: String, summary: String)? { mock.lastStaged }
    var lastDeciderDetail: (tag: String, confidence: Double, probabilities: [(label: String, probability: Double)],
                            expectedLevel: Double?, stateTruncated: Bool)? { mock.lastDeciderDetail }
}

@MainActor
@Suite(.serialized)
struct ServeAskFlowSessionTests {
    private let flow = """
    mlxflow 0.8
    1. Read Text    memo.txt
    2. Save Text    out.md
    """

    private struct Harness {
        let doc: FlowDocument
        let session: FlowRunSession
        let context: FlowRunner.RunContext
        let recorder: Recorder
        let flowID: String
    }

    private func harness(enabled: Bool = true) throws -> Harness {
        let doc = try CatParser.parse(flow)
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("sc5b-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let flowID = "flow-\(UUID().uuidString)"
        let rec = Recorder()
        let executor = GatedExecutor(blobs: base.appendingPathComponent("blobs"),
                                     gate: gate(flow: flowID, recorder: rec, enabled: enabled))
        let context = FlowRunner.RunContext(flowID: flowID, workspace: FlowWorkspace(root: base), blobDirectory: base, executor: executor)
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.run(doc, catalog: [], installedModelIDs: [], totalRAMGB: 128, claimableModelIDs: []), doc: doc)
        return Harness(doc: doc, session: session, context: context, recorder: rec, flowID: flowID)
    }

    private func until(_ timeout: Duration = .seconds(5), _ condition: () -> Bool) async -> Bool {
        let end = ContinuousClock.now.advanced(by: timeout)
        while ContinuousClock.now < end { if condition() { return true }; try? await Task.sleep(for: .milliseconds(20)) }
        return condition()
    }

    @Test func aManualRunParksOnTheQuestionAndLoadAnywayProceeds() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let h = try harness()
        h.session.start(doc: h.doc, runner: FlowRunner(), context: h.context)
        #expect(await until { h.session.servedLoadSentence != nil })
        let sentence = try #require(h.session.servedLoadSentence)
        #expect(sentence.hasPrefix("Loading Row Model will stop keeping Served-4bit ready for connected apps."))
        #expect(h.session.parked?.isServedLoad == true && !h.session.isRunning && h.session.completedRunToken == 0)
        #expect(ServedLoadWaiting.shared.items.contains { $0.flow == h.flowID })      // the Local Server screen lists it

        h.session.answerServedLoad(loadAnyway: true)
        #expect(await until { h.session.status(for: h.doc.rows[1].id) == .succeeded && !h.session.isRunning })
        #expect(h.recorder.pushed == [["mlx-community/Served-4bit"]])
        #expect(h.session.parked == nil && h.session.completedRunToken == 1)
        #expect(!ServedLoadWaiting.shared.items.contains { $0.flow == h.flowID })
    }

    @Test func cancelFailsTheRowWithTheSentenceAndStopsTheRun() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let h = try harness()
        h.session.start(doc: h.doc, runner: FlowRunner(), context: h.context)
        #expect(await until { h.session.servedLoadSentence != nil })
        let sentence = try #require(h.session.servedLoadSentence)

        h.session.answerServedLoad(loadAnyway: false)
        #expect(await until { h.session.status(for: h.doc.rows[1].id) == .failed })
        #expect(h.session.statusNote(for: h.doc.rows[1].id)?.text.contains("Loading Row Model will stop keeping Served-4bit") == true)
        #expect(h.session.errorSentence?.contains(String(sentence.prefix(40))) == true)
        #expect(h.recorder.pushed.isEmpty && h.session.parked == nil && h.session.completedRunToken == 0)
    }

    @Test func anUnattendedFiringParksAndIsNeverAutoAnswered() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let h = try harness()
        h.session.start(doc: h.doc, runner: FlowRunner(), context: h.context,
                        occurrence: FlowInterpreter.Occurrence(path: nil, tick: "06:30", payload: nil))
        #expect(await until { h.session.servedLoadSentence != nil })
        try await Task.sleep(for: .seconds(1))                                     // nobody answers
        #expect(h.session.servedLoadSentence != nil && !h.session.isRunning)
        #expect(h.recorder.pushed.isEmpty && h.session.status(for: h.doc.rows[1].id) != .failed)

        // answered from the Local Server screen's list: the flow resumes
        let item = try #require(ServedLoadWaiting.shared.items.first { $0.flow == h.flowID })
        item.answer(true)
        #expect(await until { h.session.status(for: h.doc.rows[1].id) == .succeeded && !h.session.isRunning })
        #expect(h.recorder.pushed.count == 1)
    }

    @Test func anUnattendedFiringCancelledFromTheServerScreenFailsTheRow() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let h = try harness()
        h.session.start(doc: h.doc, runner: FlowRunner(), context: h.context,
                        occurrence: FlowInterpreter.Occurrence(path: nil, tick: "06:30", payload: nil))
        #expect(await until { h.session.servedLoadSentence != nil })
        let item = try #require(ServedLoadWaiting.shared.items.first { $0.flow == h.flowID })
        item.answer(false)
        #expect(await until { h.session.status(for: h.doc.rows[1].id) == .failed })
        #expect(h.recorder.pushed.isEmpty)
    }

    @Test func gateClosedMeansTheFlowJustRuns() async throws {
        LocalServerGate.overrideForTesting = false
        defer { LocalServerGate.overrideForTesting = nil }
        let h = try harness(enabled: false)
        h.session.start(doc: h.doc, runner: FlowRunner(), context: h.context)
        #expect(await until { h.session.completedRunToken == 1 })
        #expect(h.session.servedLoadSentence == nil && h.recorder.checks == 0)
    }

    @Test func stoppingTheRunWithdrawsTheQuestionFromTheServerScreen() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let h = try harness()
        h.session.start(doc: h.doc, runner: FlowRunner(), context: h.context)
        #expect(await until { h.session.servedLoadSentence != nil })
        let sentence = try #require(h.session.servedLoadSentence)
        h.session.cancel()
        #expect(h.session.servedLoadSentence == nil)
        #expect(!ServedLoadWaiting.shared.items.contains { $0.flow == h.flowID })
    }
}
