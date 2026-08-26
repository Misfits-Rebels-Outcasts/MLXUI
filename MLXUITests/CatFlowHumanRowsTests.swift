import Testing
import Foundation
@testable import MLXUI

/// CFM-R10-Human — Ask Human / Human Input run for real: `wait=forever` rows park and resume
/// on an answer, timeout rows resolve to their default with the F002 disclosure, and `canRun`
/// no longer refuses the human class.
struct CatFlowHumanRowsTests {

    private func parse(_ text: String) throws -> FlowDocument {
        try CatParser.parse(text)
    }

    private func editor(_ rows: [Row]) throws -> FlowEditorModel {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return FlowEditorModel(name: "Human", document: FlowDocument(version: "0.8", rows: rows),
                               workspace: FlowWorkspace(root: root))
    }

    private func row(_ task: String, settings: String? = nil) -> Row {
        Row(id: UUID(), task: task, settings: settings)
    }

    private let humanFlow = """
    catflow 0.8
    1. Read Text    memo.txt
    2. Ask Human    (1)  "Send this reply?" ; wait=forever ; tags: approve, edit
       -> { approve: 3 | edit: 3 }
    3. Save Text    out.md
    """

    // MARK: - canRun

    @Test func canRunAllowsHumanRows() throws {
        let doc = try parse(humanFlow)
        #expect(FlowRunner.canRun(doc) == .runnable)
    }

    // MARK: - Interpreter park / resume

    @Test func waitForeverParksWithoutAnAnswer() async throws {
        let doc = try parse(humanFlow)
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let events = try await FlowInterpreter.run(doc, executor: mock)
        guard let parked = events.first(where: { $0.kind == .runParked }) else {
            Issue.record("expected a parked run")
            return
        }
        #expect(parked.path == "2")
        #expect(parked.parkPrompt == "Send this reply?")
        #expect(parked.parkPolicy == "wait=forever")
    }

    @Test func answeringResumesAndCompletesTheRun() async throws {
        let doc = try parse(humanFlow)
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let events = try await FlowInterpreter.run(
            doc, executor: mock,
            answers: ["2": FlowInterpreter.HumanAnswer(tag: "approve", text: nil)])
        #expect(events.contains { $0.kind == .runResumed })
        #expect(events.last?.kind == .runCompleted)
        // The answer's tag routed the decide clause.
        #expect(events.contains { $0.kind == .rowCompleted && $0.path == "3" })
    }

    // MARK: - Timeout rows park and fall back (CFM-R10-Human timeout)

    private let timeoutFlow = """
    catflow 0.8
    1. Read Text    memo.txt
    2. Ask Human    (1)  "Send this reply?" ; timeout=2h; default=edit
       -> { edit: 3 }
    3. Save Text    out.md
    """

    @Test func timeoutRowParksLikeWaitForever() async throws {
        let doc = try parse(timeoutFlow)
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        // The GUI path (parkOnTimeout) parks a timeout row; the conformance default doesn't.
        let events = try await FlowInterpreter.run(doc, executor: mock, parkOnTimeout: true)
        guard let parked = events.first(where: { $0.kind == .runParked }) else {
            Issue.record("expected a parked run for a timeout row")
            return
        }
        #expect(parked.path == "2")
        #expect(parked.parkPrompt == "Send this reply?")
        #expect(parked.parkPolicy == "timeout=2h")
    }

    @Test func timeoutDefaultAnswerResolvesWithF002AndCompletes() async throws {
        let doc = try parse(timeoutFlow)
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let events = try await FlowInterpreter.run(
            doc, executor: mock,
            answers: ["2": FlowInterpreter.HumanAnswer(tag: nil, text: nil, isTimeoutDefault: true)],
            parkOnTimeout: true)
        // The declared default tag fires the clause and the F002 disclosure names it.
        guard let f002 = events.first(where: { $0.kind == .flagRaised && $0.code == "F002" }) else {
            Issue.record("expected the F002 timeout disclosure")
            return
        }
        #expect(f002.message == "Nobody answered by 2h — proceeded as `edit`, unreviewed.")
        #expect(events.contains { $0.kind == .runResumed })
        #expect(events.contains { $0.kind == .rowCompleted && $0.path == "3" })
        #expect(events.last?.kind == .runCompleted)
    }

    // MARK: - RealExecutor timeout defaults (the F002 path)

    private func realExecutor() -> RealExecutor {
        RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "test",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "unused", kind: .llm) },
            installedModelIDs: [],
            catalog: [])
    }

    @Test func askHumanTimeoutResolvesToDefaultWithF002() async throws {
        let r = row("Ask Human", settings: #"timeout=4h; default=edit"#)
        let executor = realExecutor()
        let input = Asset(items: [Item(kind: .text, value: "draft", path: nil, sourceText: nil)])
        let output = try await executor.execute(path: "2", row: r, inputs: [input])
        // The default tag fires and the F002 disclosure names the fallback in plain words.
        #expect(executor.lastTag == "edit")
        #expect(executor.lastTimeoutFlag?.code == "F002")
        #expect(executor.lastTimeoutFlag?.message == "Nobody answered by 4h — proceeded as `edit`, unreviewed.")
        #expect(output.items.first?.value == "draft")   // passthrough, unchanged
    }

    @Test func humanInputTimeoutResolvesUnchangedWithF002() async throws {
        let r = row("Human Input", settings: "timeout=1h")
        let executor = realExecutor()
        let input = Asset(items: [Item(kind: .text, value: "hello", path: nil, sourceText: nil)])
        let output = try await executor.execute(path: "3", row: r, inputs: [input])
        #expect(executor.lastTimeoutFlag?.code == "F002")
        #expect(executor.lastTimeoutFlag?.message == "Nobody answered by 1h — proceeded as `unchanged`, unreviewed.")
        #expect(output.items.first?.value == "hello")
    }

    // MARK: - Session-level park → answer (the g3 flow)

    @Test func sessionParksAndAnsweringDrivesTheRunner() async throws {
        let doc = try parse(humanFlow)
        let session = FlowRunSession()
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let context = FlowRunner.RunContext(flowID: "t", workspace: FlowWorkspace(root: base),
                                            blobDirectory: base, executor: mock)
        let runner = FlowRunner()
        session.prepareInstall(FlowPreflight.run(doc, catalog: [], installedModelIDs: [],
                                                 totalRAMGB: 128),
                               doc: doc)

        session.start(doc: doc, runner: runner, context: context)
        // Poll for the parked state (the run task applies events asynchronously).
        var parkedInfo: FlowRunSession.ParkedInfo?
        for _ in 0..<40 {
            if let parked = session.parked { parkedInfo = parked; break }
            try await Task.sleep(for: .milliseconds(50))
        }
        let parked = try #require(parkedInfo)
        #expect(parked.prompt == "Send this reply?")
        // A wait=forever row parks without a deadline.
        #expect(parked.deadline == nil)

        session.answer(tag: "approve", for: parked, doc: doc, runner: runner, context: context)
        for _ in 0..<40 {
            if session.parked == nil, session.status(for: doc.rows[2].id) == .succeeded { break }
            try await Task.sleep(for: .milliseconds(50))
        }
        #expect(session.parked == nil)
        #expect(session.status(for: doc.rows[2].id) == .succeeded)
    }

    @Test func sessionFallbackTimeoutRunsToTheDefault() async throws {
        // A one-second timeout so the deadline is reachable, not a claim about the clock.
        let doc = try parse("""
        catflow 0.8
        1. Read Text    memo.txt
        2. Ask Human    (1)  "Send this reply?" ; timeout=1s; default=edit
           -> { edit: 3 }
        3. Save Text    out.md
        """)
        let session = FlowRunSession()
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let context = FlowRunner.RunContext(flowID: "t", workspace: FlowWorkspace(root: base),
                                            blobDirectory: base, executor: mock)
        let runner = FlowRunner()
        session.prepareInstall(FlowPreflight.run(doc, catalog: [], installedModelIDs: [],
                                                 totalRAMGB: 128),
                               doc: doc)

        session.start(doc: doc, runner: runner, context: context)
        var parkedInfo: FlowRunSession.ParkedInfo?
        for _ in 0..<40 {
            if let parked = session.parked { parkedInfo = parked; break }
            try await Task.sleep(for: .milliseconds(50))
        }
        let parked = try #require(parkedInfo)
        #expect(parked.policy == "timeout=1s")
        #expect(parked.deadline != nil)

        // No live answer — the fallback re-runs with the row's default.
        session.fallbackTimeout(for: parked, doc: doc, runner: runner, context: context)
        for _ in 0..<40 {
            if session.parked == nil, session.status(for: doc.rows[2].id) == .succeeded { break }
            try await Task.sleep(for: .milliseconds(50))
        }
        #expect(session.parked == nil)
        #expect(session.status(for: doc.rows[2].id) == .succeeded)
    }

    // MARK: - The friendly wording's data source (g3): a gallery human flow is runnable

    @Test func minutesNameFixIsRunnableNow() throws {
        // 39-MinutesNameFix is "Whisper fumbles names; people don't. One human row …" — it was
        // refused for its human row; now it runs (parked until answered).
        let doc = try GalleryLoader.loadDocument(flowID: "39-MinutesNameFix")
        #expect(FlowRunner.canRun(doc) == .runnable)
    }
}
