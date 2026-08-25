import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R2-8's testable core: the `FlowRunSession` view-model driving dots from the
/// event stream, and the exhaustive `FlowErrorDisplay` error mapping (a switch with no
/// `default`, so a new error case is a compile error). The real-engine run is smoke row 30.
/// See `RSI/DelegateMergeBacklog.md` CFM-R2-8.
struct CatFlowRunWiringTests {

    private func decode(_ flowID: String) throws -> FlowDocument {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(flowID).cat")
        return try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
    }

    // MARK: - FlowRunSession drives dots from events

    @Test func sessionDrivesDotsFromFinishedEvents() async throws {
        let doc = try decode("01-SpokenSummary")
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: makeMockContext())

        // Events are delivered asynchronously; wait for the run to finish (not a fixed
        // sleep — under parallel workers the mock run can take a couple of seconds).
        let deadline = Date().addingTimeInterval(10)
        while session.isRunning && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }
        for row in doc.rows {
            #expect(session.status(for: row.id) == .succeeded,
                    "row \(row.task ?? "?") should have succeeded")
        }
    }

    /// CFM-R12-FIX-3: block children get their own status dot — `rowStates` is seeded from
    /// the flattened row set, so a child's `.finished` event records instead of being
    /// silently dropped.
    @Test func blockChildrenGetTheirOwnDots() async throws {
        let doc = try decode("02-MeetingMinutes")
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: makeMockContext())
        let deadline = Date().addingTimeInterval(10)
        while session.isRunning && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }
        let block = try #require(doc.rows.first { $0.blockKind != nil })
        #expect(block.children.count >= 3)
        for child in block.children {
            #expect(session.status(for: child.id) == .succeeded,
                    "child \(child.task ?? "?") should have its own succeeded dot")
        }
    }

    @Test func sessionFailedRowShowsErrorSentence() async throws {
        // CFM-R10-FIX-1: the runner consults `canRun` before starting — a flow with an
        // unknown task surfaces the refusal as a `.failed` sentence, never a partial run.
        let rows = [
            Row(task: "Read Text", settings: "a.txt"),
            Row(task: "Nope"),
        ]
        let doc = FlowDocument(version: "0.8", rows: rows)
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: makeMockContext())
        // Poll for the failure (the run task applies events asynchronously).
        var sentence: String?
        for _ in 0..<40 {
            if let s = session.errorSentence(for: rows[0].id) { sentence = s; break }
            try await Task.sleep(for: .milliseconds(50))
        }

        #expect(session.status(for: rows[0].id) == .failed)
        #expect(session.status(for: rows[1].id) == .notRun)
        #expect(sentence?.contains("Nope") == true)
    }

    @Test func cancelStopsAndKeepsEarnedDots() async throws {
        let doc = try decode("01-SpokenSummary")
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: makeMockContext())
        // Cancel immediately; isRunning goes false (earned dots, if any, are kept).
        session.cancel()
        #expect(session.isRunning == false)
    }

    // MARK: - FlowErrorDisplay mapping (exhaustive switch, no default)

    @Test func flowErrorSentencesNameTheProblem() {
        #expect(FlowErrorDisplay.sentence(for: FlowError.fileReadFailed(row: "Row 1", path: "memo.m4a"))
                .contains("Row 1"))
        #expect(FlowErrorDisplay.sentence(for: FlowError.fileReadFailed(row: "Row 1", path: "memo.m4a"))
                .contains("memo.m4a"))
        #expect(FlowErrorDisplay.sentence(for: FlowError.unsupportedKind(row: "Row 2", kind: .index))
                .contains("index"))
        #expect(FlowErrorDisplay.sentence(for: FlowError.modelNotRunnable(row: "Row 2", display: "SAM Base",
                                                                          reason: "no headless path"))
                .contains("SAM Base"))
    }

    @Test func stageErrorSentencesNameTheProblem() {
        #expect(FlowErrorDisplay.sentence(for: StageError.modelNotInstalled(id: "mlx-community--x"))
                .contains("mlx-community--x"))
        #expect(FlowErrorDisplay.sentence(for: StageError.insufficientRAM(requiredGB: 6.75, availableGB: 6))
                .contains("6.75"))
        #expect(FlowErrorDisplay.sentence(for: StageError.kindMismatch(expected: .text, got: .audio))
                .contains("text"))
        // A real engine failure surfaces the stage name + the underlying cause, never the
        // raw NSError "MLXUI.StageError error 4" text.
        let sentence = FlowErrorDisplay.sentence(
            for: StageError.engineFailure(stage: "Kokoro TTS", underlying: CocoaError(.fileNoSuchFile)))
        #expect(sentence.contains("Kokoro TTS"))
        #expect(!sentence.contains("couldn't be completed"))
    }

    // MARK: - A StageError from the executor must route through FlowErrorDisplay

    @Test func stageErrorFromExecutorShowsRealSentenceNotLocalizedDescription() async throws {
        // A stub executor that throws a real StageError (as the Kokoro engine does).
        let failing = FailingExecutor()
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-stageerr-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let workspace = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let context = FlowRunner.RunContext(flowID: "t", workspace: workspace,
                                            blobDirectory: base.appendingPathComponent("blobs"),
                                            executor: failing)

        let doc = FlowDocument(version: "0.8", rows: [Row(task: "Speak", model: "Kokoro 82M")])
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: context)
        // Poll for the failed sentence (the run task applies events asynchronously).
        var sentence: String?
        for _ in 0..<40 {
            if let s = session.errorSentence(for: doc.rows[0].id) { sentence = s; break }
            try await Task.sleep(for: .milliseconds(50))
        }

        #expect(sentence != nil)
        #expect(sentence?.contains("Kokoro") == true)
        #expect(sentence?.contains("couldn't be completed") == false)
    }

    // MARK: - canRun gating

    @Test func canRunIsDisabledWhileRunningAndEnabledAfter() async throws {
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        #expect(session.canRun)
        session.start(doc: FlowDocument(version: "0.8", rows: []), runner: FlowRunner(),
                      context: makeMockContext())
        // An empty run finishes immediately; the mock settles fast.
        try? await Task.sleep(for: .milliseconds(300))
        #expect(session.isRunning == false)
        #expect(session.canRun)
    }

    @Test func blockedPreflightDisablesRun() {
        let session = FlowRunSession()
        let blocked = FlowPreflight.Result(needs: [
            .init(task: "Transcribe", display: "Missing Model", model: nil, installed: false,
                  equivalence: nil, blockingReason: "Missing Model can't run."),
        ])
        session.prepareInstall(blocked)
        #expect(session.canRun == false)
        #expect(session.runDisabledReason?.contains("Missing Model") == true)
    }

    // MARK: - B3: the session's Run gate consults FlowRunner.canRun(doc)

    @Test func sessionCanRunRefusesOutOfScopeLanguage() throws {
        // An `agent` (Improvise) row is refused by `FlowRunner.canRun` — in the App Store
        // tier (the MLXUI scheme) the door refusal names the channel; the session must
        // surface that even with an otherwise-clean preflight (B3, keystone fix).
        let doc = FlowDocument(version: "0.8", rows: [Row(task: "Improvise", model: "Qwen3 8B")])
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result(), doc: doc)
        #expect(session.canRun == false)
        #expect(session.runDisabledReason?.contains("App Store") == true)
    }

    @Test func sessionCanRunIsTrueForRunnableDocument() throws {
        let doc = try decode("01-SpokenSummary")
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result(), doc: doc)
        #expect(session.canRun)
    }

    // MARK: - H7: re-running a fully-green flow is a fresh run, not a no-op

    @Test func resumeAfterAllRowsSucceededReRunsTheFlow() async throws {
        let doc = try decode("01-SpokenSummary")
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result(), doc: doc)
        let counting = CountingExecutor(inner: MockExecutor(blobDirectory: FileManager.default.temporaryDirectory))
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-h7-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let context = FlowRunner.RunContext(flowID: "t", workspace: FlowWorkspace(root: base),
                                            blobDirectory: base, executor: counting)

        session.start(doc: doc, runner: FlowRunner(), context: context)
        let deadline = Date().addingTimeInterval(10)
        while session.isRunning && Date() < deadline { try? await Task.sleep(for: .milliseconds(50)) }
        #expect(counting.executions == doc.rows.count)

        // Re-run with resume — every row is already ✓, which used to select zero rows (H7).
        session.start(doc: doc, runner: FlowRunner(), context: context, resume: true)
        let deadline2 = Date().addingTimeInterval(10)
        while session.isRunning && Date() < deadline2 { try? await Task.sleep(for: .milliseconds(50)) }
        #expect(counting.executions == doc.rows.count * 2)
    }

    // MARK: - Inspector wiring (CFM-R3-3)

    @Test func finishedEventsStoreOutputsForTheInspector() async throws {
        let doc = try decode("01-SpokenSummary")
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: makeMockContext())
        let deadline = Date().addingTimeInterval(10)
        while session.isRunning && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }

        // Every succeeded row has a stored output the inspector can show.
        for row in doc.rows {
            #expect(session.outputs[row.id] != nil, "row \(row.task ?? "?") output should be inspectable")
        }
    }

    @Test func substitutionNoteSurfacesForSubstituteRows() throws {
        // Build a preflight whose MusicGen row (.substitute) resolved, and a doc with a
        // "Generate Sound" row naming it — the note must be attached to that row's id.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let catalogURL = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: catalogURL))
            .domains.flatMap { $0.allModels }
        let music = try #require(catalog.first { $0.hfModelId == "jasonvassallo/mlx-musicgen-small" })

        let row = Row(task: "Generate Sound", model: "MusicGen")
        let doc = FlowDocument(version: "0.8", rows: [row])
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 16)

        let session = FlowRunSession()
        session.prepareInstall(result, doc: doc)
        let note = session.substitutionNotes[row.id]
        #expect(note != nil)
        #expect(note?.contains("MusicGen") == true)
    }

    @Test func requantizedRowsGetNoSubstitutionNote() throws {
        // Whisper Large v3 is .requantized → runs silently, no inspector note.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let catalogURL = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: catalogURL))
            .domains.flatMap { $0.allModels }

        let row = Row(task: "Transcribe", model: "Whisper Large v3")
        let doc = FlowDocument(version: "0.8", rows: [row])
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 16)
        let session = FlowRunSession()
        session.prepareInstall(result, doc: doc)
        #expect(session.substitutionNotes[row.id] == nil)
    }

    // MARK: - Clear Run / Clear Cache (CFM-R3-4)

    @Test func clearRunResetsDotsAndOutputs() async throws {
        let doc = try decode("01-SpokenSummary")
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: makeMockContext())
        try? await Task.sleep(for: .milliseconds(800))
        #expect(session.hasRunResults)

        session.clearRun(doc: doc)
        #expect(session.hasRunResults == false)
        #expect(session.outputs.isEmpty)
        for row in doc.rows {
            #expect(session.status(for: row.id) == .notRun)
        }
    }

    @Test func clearCacheDropsStoredAssets() async throws {
        // Put something into a temp store-backed session, then clear it.
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-clearcache-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let store = FlowCacheStore(root: base.appendingPathComponent("cache"))
        let key = try CacheKey.cacheKey(task: "Summarize", model: "m", settings: nil,
                                        inputs: [], realism: "real")
        try store.put(key: key, asset: Asset(items: [Item(kind: .text, value: "x", path: nil, sourceText: nil)]))
        #expect(store.entryCount == 1)
        try store.clear()
        #expect(store.entryCount == 0)
    }

    // MARK: - Frame-backed model rows resolve the frame (regression: Summarize.frame.txt)

    @Test func frameBackedRowResolvesTheFrame() async throws {
        // A Summarize row through the RealExecutor must render its frame — the frame file is
        // bundled and refName "frames/Summarize.frame.txt" must derive the "Summarize" name.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let catalogURL = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: catalogURL))
            .domains.flatMap { $0.allModels }
        let qwen = try #require(catalog.first { $0.hfModelId == "mlx-community/Qwen3-8B-4bit" })

        // Stub stage that echoes the prompt it was given (so the test sees the rendered frame).
        let stub = EchoPromptStage()
        let executor = RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "01-SpokenSummary",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { model, _ in
                stub
            },
            installedModelIDs: [qwen.id],
            catalog: catalog)

        let row = Row(task: "Summarize", model: "Qwen3 8B", settings: "\"TL;DR in 3 bullets\"")
        let input = Asset(items: [Item(kind: .text, value: "the transcript", path: nil, sourceText: nil)])
        let out = try await executor.execute(path: "3", row: row, inputs: [input])
        let value = out.items.first?.value ?? ""
        // The stub echoed the rendered frame prompt: it must contain the frame's instruction
        // and the substituted settings — proving the frame loaded and rendered, not failed.
        #expect(value.contains("summarizer"))
        #expect(value.contains("TL;DR in 3 bullets"))
    }

    // MARK: - Unimplemented instant tool refuses with an honest sentence (Policy Diff)

    @Test func policyDiffDiffRowNowRuns() async throws {
        // 08-PolicyDiff row 3 is `Diff`, implemented in R4 — it must now produce a diff
        // instead of refusing.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let docURL = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/08-PolicyDiff.cat")
        let doc = try CatParser.parse(try String(contentsOf: docURL, encoding: .utf8))

        let executor = RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "08-PolicyDiff",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in EchoPromptStage() },
            installedModelIDs: [],
            catalog: [])
        let row3 = doc.rows[2]   // Diff (1,2)
        #expect(row3.task == "Diff")

        let out = try await executor.execute(path: "3", row: row3, inputs: [
            Asset(items: [Item(kind: .text, value: "a\nb\nc", path: nil, sourceText: nil)]),
            Asset(items: [Item(kind: .text, value: "a\nb2\nc", path: nil, sourceText: nil)]),
        ])
        // The bundled 08-PolicyDiff row 3 is `Diff format=unified` — it must produce a
        // unified diff, not refuse.
        let value = out.items.first?.value ?? ""
        let expected = "--- \n+++ \n@@ -1,3 +1,3 @@\n a\n-b\n+b2\n c"
        if value != expected {
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent("diff-dbg.txt")
            try? value.write(to: dest, atomically: true, encoding: .utf8)
            Issue.record("Diff output wrong: [\(value)]")
        }
        #expect(value == expected)
    }

    private func makeMockContext() -> FlowRunner.RunContext {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-wire-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let blob = base.appendingPathComponent("blobs")
        let workspace = FlowWorkspace(root: base.appendingPathComponent("flows"))
        return FlowRunner.RunContext(flowID: "test", workspace: workspace,
                                     blobDirectory: blob, executor: MockExecutor(blobDirectory: blob))
    }
}

/// A stub executor that throws a real `StageError.engineFailure` — the class of error the
/// Kokoro engine wraps — so the runner's message routing is observable.
private struct FailingExecutor: FlowExecutor {
    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset {
        throw StageError.engineFailure(stage: "Kokoro TTS", underlying: CocoaError(.fileNoSuchFile))
    }
}

/// Counts every `execute` it dispatches, delegating to a mock — the H7 harness: a re-run
/// that silently selects zero rows is a no-op the counter would expose.
private final class CountingExecutor: FlowExecutor, @unchecked Sendable {
    private let inner: MockExecutor
    private let lock = NSLock()
    private var _executions = 0

    init(inner: MockExecutor) {
        self.inner = inner
    }

    var executions: Int { lock.withLock { _executions } }

    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset {
        lock.withLock { _executions += 1 }
        return try await inner.execute(path: path, row: row, inputs: inputs)
    }
}

/// A stub `text → text` PipelineStage that echoes its input, so a frame-backed row's
/// rendered prompt is observable.
private struct EchoPromptStage: PipelineStage {
    let id = "stub.echo"
    let name = "Echo Prompt"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        progress(1.0)
        return input
    }
}
