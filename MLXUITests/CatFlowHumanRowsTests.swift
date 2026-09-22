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
    mlxflow 0.8
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
    mlxflow 0.8
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
                                                 totalRAMGB: 128, claimableModelIDs: []),
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

    /// WR-6 (`RSI/DelegateWorkspaceRunBacklog.md`) — the missing neighbour of cancel-clears-
    /// parked: `answer(text:)`, "Send" on a **Human Input** row (a different function, a
    /// different row class, from `answer(tag:)`'s Ask Human path above — that test alone
    /// doesn't cover this one). Send is the only one of the three parked exits — Send,
    /// timeout fallback, Stop — a user hits on the happy path, so a future change to `start()`
    /// that silently broke it would ship unnoticed without this. Proves both that `parked`
    /// clears and that the run resumes carrying the **typed** text, not the row's own default.
    @Test func sessionParksOnHumanInputAndSendingTextDrivesTheRunnerWithTheTypedText() async throws {
        let doc = try parse("""
        mlxflow 0.8
        1. Template      ""
        2. Human Input   "Type something:"; wait=forever
        3. Save Text     out.md
        """)
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        // `Save Text` writes into the flow's own directory (`root/<flowID>/`) — create it
        // first, the way `FlowWorkspace.prepare` normally would on a real open.
        try FileManager.default.createDirectory(at: base.appendingPathComponent("test"),
                                                withIntermediateDirectories: true)
        // The real executor, rooted at `base` (not the shared `realExecutor()` helper, which
        // hardcodes the raw system temp directory as its workspace root — fine for the other
        // tests here, which never check saved file content, but wrong for this one, which
        // needs `Save Text` to actually land under `base`).
        let executor = RealExecutor(
            workspace: FlowWorkspace(root: base), flowID: "test", blobDirectory: base,
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "unused", kind: .llm) },
            installedModelIDs: [], catalog: [])
        let context = FlowRunner.RunContext(flowID: "test", workspace: FlowWorkspace(root: base),
                                            blobDirectory: base, executor: executor)
        let runner = FlowRunner()
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.run(doc, catalog: [], installedModelIDs: [],
                                                 totalRAMGB: 128, claimableModelIDs: []),
                               doc: doc)

        session.start(doc: doc, runner: runner, context: context)
        var parkedInfo: FlowRunSession.ParkedInfo?
        for _ in 0..<40 {
            if let parked = session.parked { parkedInfo = parked; break }
            try await Task.sleep(for: .milliseconds(50))
        }
        let parked = try #require(parkedInfo)
        #expect(parked.prompt == "Type something:")
        #expect(parked.deadline == nil)

        session.answer(text: "hello world", for: parked, doc: doc, runner: runner, context: context)
        for _ in 0..<40 {
            if session.parked == nil, session.status(for: doc.rows[2].id) == .succeeded { break }
            try await Task.sleep(for: .milliseconds(50))
        }
        #expect(session.parked == nil)
        #expect(session.status(for: doc.rows[2].id) == .succeeded)
        let saved = try String(contentsOf: base.appendingPathComponent("test/out.md"), encoding: .utf8)
        #expect(saved.contains("hello world"))
    }

    @Test func sessionParkedInfoCarriesTheDefaultTextThroughToTheView() async throws {
        // HR-4's whole point: the value has to survive all five hops — ParkRun →
        // PathEvent.runParked → FlowRunner.RunEvent.parked → FlowRunSession.ParkedInfo — so
        // `FlowHumanPromptView` has something to prefill from. Checked at the session layer,
        // the last hop before the view itself.
        let doc = try parse("""
        mlxflow 0.8
        1. Template      "https://news.ycombinator.com"
        2. Human Input   "Enter URL:" ; timeout=30s; default=unchanged
        """)
        let session = FlowRunSession()
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        // The real executor, not the mock: `Template`'s literal pattern passthrough is what
        // makes the incoming value predictable end to end (the mock fingerprints every task's
        // output generically, `Template` included — that's exercised by the mock's own tests).
        let context = FlowRunner.RunContext(flowID: "t", workspace: FlowWorkspace(root: base),
                                            blobDirectory: base, executor: realExecutor())
        let runner = FlowRunner()
        session.prepareInstall(FlowPreflight.run(doc, catalog: [], installedModelIDs: [],
                                                 totalRAMGB: 128, claimableModelIDs: []),
                               doc: doc)

        session.start(doc: doc, runner: runner, context: context)
        var parkedInfo: FlowRunSession.ParkedInfo?
        for _ in 0..<40 {
            if let parked = session.parked { parkedInfo = parked; break }
            try await Task.sleep(for: .milliseconds(50))
        }
        let parked = try #require(parkedInfo)
        #expect(parked.defaultText == "https://news.ycombinator.com")
    }

    @Test func sessionFallbackTimeoutRunsToTheDefault() async throws {
        // A one-second timeout so the deadline is reachable, not a claim about the clock.
        let doc = try parse("""
        mlxflow 0.8
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
                                                 totalRAMGB: 128, claimableModelIDs: []),
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

    // MARK: - HR-4: the parked event carries the row's incoming default (SPEC-Q233)

    private let templateDefaultFlow = """
    mlxflow 0.8
    1. Template      "https://news.ycombinator.com"
    2. Human Input   "Enter URL:" ; timeout=30s; default=unchanged
    """

    @Test func parkedHumanInputCarriesTheTemplateRowsOutputAsItsDefaultText() async throws {
        let doc = try parse(templateDefaultFlow)
        // The real executor: `Template`'s literal pattern passthrough is what makes the
        // incoming value the exact URL, not a mock fingerprint of it.
        // The GUI path (parkOnTimeout) parks a timeout row, same as timeoutRowParksLikeWaitForever.
        let events = try await FlowInterpreter.run(doc, executor: realExecutor(), parkOnTimeout: true)
        guard let parked = events.first(where: { $0.kind == .runParked }) else {
            Issue.record("expected a parked run")
            return
        }
        #expect(parked.path == "2")
        #expect(parked.parkDefaultText == "https://news.ycombinator.com")
    }

    @Test func parkedRowOneHumanInputHasNoDefaultText() async throws {
        // Row 1 has nothing upstream — `inputs` is empty at the throw site, so there is
        // nothing to show as a default. `default_text` is absent, not an empty string
        // (SPEC-Q233's own distinction).
        let doc = try parse("""
        mlxflow 0.8
        1. Human Input   "Enter URL:" ; timeout=30s; default=unchanged
        """)
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let events = try await FlowInterpreter.run(doc, executor: mock, parkOnTimeout: true)
        guard let parked = events.first(where: { $0.kind == .runParked }) else {
            Issue.record("expected a parked run")
            return
        }
        #expect(parked.parkDefaultText == nil)
    }

    @Test func lettingTheTimerRunOutStillProducesTheSameDefaultTextRegardless() async throws {
        // HR-4 is display-only: prefilling the sheet must not change what an unanswered
        // timeout row resolves to. Same flow as above, but driven all the way through — the
        // resolved row's own `rowCompleted` output is checked directly, not just the F002
        // disclosure's wording.
        let doc = try parse(templateDefaultFlow)
        let events = try await FlowInterpreter.run(
            doc, executor: realExecutor(),
            answers: ["2": FlowInterpreter.HumanAnswer(tag: nil, text: nil, isTimeoutDefault: true)],
            parkOnTimeout: true)
        #expect(events.contains { $0.kind == .runResumed })
        #expect(events.last?.kind == .runCompleted)
        let f002 = try #require(events.first(where: { $0.kind == .flagRaised && $0.code == "F002" }))
        #expect(f002.message == "Nobody answered by 30s — proceeded as `unchanged`, unreviewed.")
        let resolved = try #require(events.first(where: { $0.kind == .rowCompleted && $0.path == "2" }))
        #expect(resolved.output?.items.first?.value == "https://news.ycombinator.com")
    }

    @Test func parkDefaultTextIsAbsentForAMultiItemOrNonTextInput() async throws {
        // parkDefaultText only ever names a single text item — a list or a non-text kind has
        // no sensible one-string representation, so it's nil rather than a guessed join.
        let doc = try parse("""
        mlxflow 0.8
        1. Read Images   images/
        2. Human Input   (1)  "Describe these:" ; timeout=30s; default=unchanged
        """)
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-human-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let events = try await FlowInterpreter.run(doc, executor: mock, parkOnTimeout: true)
        guard let parked = events.first(where: { $0.kind == .runParked }) else {
            Issue.record("expected a parked run")
            return
        }
        #expect(parked.parkDefaultText == nil)
    }
}
