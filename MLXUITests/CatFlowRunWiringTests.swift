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

    /// DA-10 repro: a body row skipped by `<each on_error=skip>` must, in the session, end at
    /// `.needsAttention` (△) AND carry the skip reason in `errorSentence`, with `wasSkipped`
    /// set so the run-log line renders it as a skip. (30-ReceiptsExpense: the owner saw △ but
    /// no sentence.)
    @Test func skippedEachChildCarriesItsReasonInTheSession() async throws {
        let doc = try CatParser.parse("""
        mlxflow 0.8
        1. Template   "a; b"
        2. Split   (1)   by=lines
        3. <each go; on_error=skip>   (2)
            1. Summarize   {item}
        4. Join Text   (3)
        """)
        let child = try #require(doc.rows[2].children.first)

        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-skip-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let blob = base.appendingPathComponent("blobs")
        let ctx = FlowRunner.RunContext(
            flowID: "t",
            workspace: FlowWorkspace(root: base.appendingPathComponent("flows")),
            blobDirectory: blob,
            executor: RowFailingExecutor(failTask: "Summarize",
                                         inner: MockExecutor(blobDirectory: blob)))

        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: ctx)
        let deadline = Date().addingTimeInterval(10)
        while session.isRunning && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }

        #expect(session.status(for: child.id) == .needsAttention)
        #expect(session.wasSkipped(child.id))
        let sentence = session.errorSentence(for: child.id)
        #expect(sentence != nil && !(sentence ?? "").isEmpty,
                "the skipped child must carry a non-empty reason; got \(String(describing: sentence))")
        #expect(sentence?.contains("receipts are on fire") == true)

        // DA-10-FIX-1: the reason is reachable without the inline caption — a run summary and
        // the inspector's status note both carry it.
        #expect(session.skipSummary?.contains("receipts are on fire") == true)
        #expect(session.skipSummary?.contains("skipped") == true)
        let note = session.statusNote(for: child.id)
        #expect(note?.isSkip == true)
        #expect(note?.text.contains("receipts are on fire") == true)
        // A row that simply succeeded has no status note.
        #expect(session.statusNote(for: doc.rows[0].id) == nil)
    }

    /// DA-10-FIX-3 repro: an `<each on_error=skip>` body row reuses the same row id across
    /// items. When one item skips (△) but a *later* item of the same row then succeeds, that
    /// later item's own `.started` event must not repaint the dot ● — and since a skipped
    /// row's `.finished` never promotes it to ✓ either (the sticky-△ rule
    /// `skippedEachChildCarriesItsReasonInTheSession` covers), the dot was left stuck on ●
    /// with no later event to move it anywhere. (Owner repro: `31-ResearchBrief.cat` row 3.1
    /// stuck ● while rows 4-6 had already finished ✓.)
    @Test func laterItemSucceedingAfterAnEarlierSkipLeavesTheDotAtNeedsAttention() async throws {
        let doc = try CatParser.parse("""
        mlxflow 0.8
        1. Template   "a\\nb\\nc"
        2. Split   (1)   by=lines
        3. <each go; on_error=skip>   (2)
            1. Summarize   {item}
        4. Join Text   (3)
        """)
        let child = try #require(doc.rows[2].children.first)

        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-skip-later-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let blob = base.appendingPathComponent("blobs")
        let ctx = FlowRunner.RunContext(
            flowID: "t",
            workspace: FlowWorkspace(root: base.appendingPathComponent("flows")),
            blobDirectory: blob,
            executor: NthCallFailingExecutor(failTask: "Summarize", failOnCallIndex: 1,
                                             inner: MockExecutor(blobDirectory: blob)))

        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: ctx)
        let deadline = Date().addingTimeInterval(10)
        while session.isRunning && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }

        // The run continued past the skip (`on_error=skip`) and completed — row 4 (which
        // consumes row 3's output) must have finished, not stalled.
        #expect(session.status(for: doc.rows[3].id) == .succeeded)
        // The regression itself: item "c" (the third, succeeding item) running after item
        // "b" skipped must leave the dot at △, not stuck on ● or falsely promoted to ✓.
        #expect(session.status(for: child.id) == .needsAttention)
        #expect(session.wasSkipped(child.id))
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

    /// FIP-3: `.missingInput` carries the fully-rendered `ErrorCatalog` R903 sentence already
    /// — `FlowErrorDisplay` must show it verbatim, with no further "Row N" prefixing (unlike
    /// every other `FlowError` case, whose sentence `FlowErrorDisplay` builds itself).
    @Test func missingInputShowsTheR903SentenceVerbatim() throws {
        let message = try ErrorCatalog.fill(code: "R903", values: [
            "n": "3", "path": "notes.txt", "reason": "no such file", "n-1": "2",
        ], isV08: true)
        #expect(message == "Row 3 couldn't read notes.txt: no such file. Fix or re-point the row; rows 1–2 are cached.")
        #expect(FlowErrorDisplay.sentence(for: FlowError.missingInput(row: "3", message: message)) == message)
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

    // MARK: - CFM-R17-FIX-7: an auto-run blocked only on a download opens the install path

    @Test func autoRunBlockedOnlyOnAMissingModelIsNotAHardRefusal() throws {
        // A runnable flow whose one model isn't installed: `canRun` is false, but the block
        // is a download, not a door/RAM/scope refusal — so an auto-run raises the install
        // sheet (via `run(doc)`) instead of doing nothing, and `didAutoRun` is left unspent.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
        let catalog = try JSONDecoder().decode(
            BrowserData.self,
            from: Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")))
            .domains.flatMap { $0.allModels }

        let doc = FlowDocument(version: "0.8", rows: [Row(task: "Transcribe", model: "Whisper Large v3")])
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 16,
                                       claimableModelIDs: [])
        #expect(!result.toDownload.isEmpty, "the model must be uninstalled for this to test anything")
        #expect(!result.isBlocked)

        let session = FlowRunSession()
        session.prepareInstall(result, doc: doc)
        #expect(session.canRun == false)
        #expect(session.blockedOnlyOnDownloads == true)
        #expect(session.runDisabledReason?.contains("Install") == true)
    }

    @Test func aHardRefusalIsNotBlockedOnlyOnDownloads() throws {
        // An out-of-subset row (Improvise) — `blockedOnlyOnDownloads` must stay false so the
        // auto-run leaves it to the disabled Run + reason, not the install sheet.
        let doc = FlowDocument(version: "0.8", rows: [Row(task: "Improvise", model: "Qwen3 8B")])
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result(), doc: doc)
        #expect(session.canRun == false)
        #expect(session.blockedOnlyOnDownloads == false)
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
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 16,
                                       claimableModelIDs: [])

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
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 16,
                                       claimableModelIDs: [])
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

    // CACHE-Q: the ⋯ menu is now `FlowMaintenanceMenu`, shared by `FlowListView` and
    // `FlowEditorView` (a working My Workflows flow opens in the editor, which had no such
    // control). Its result sentence is the one string both views depend on.
    @Test func maintenanceMenuClearNoticeReadsForEveryCount() {
        #expect(FlowMaintenance.clearNotice(clearedCount: nil)
            == "Dots reset. The cache was already empty.")
        #expect(FlowMaintenance.clearNotice(clearedCount: 0)
            == "Dots reset. The cache was already empty.")
        #expect(FlowMaintenance.clearNotice(clearedCount: 1)
            == "Cleared 1 cached output; dots reset.")
        #expect(FlowMaintenance.clearNotice(clearedCount: 4)
            == "Cleared 4 cached outputs; dots reset.")
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

/// Throws for one named task, delegates everything else to a real mock — so an
/// `<each on_error=skip>` body can be made to fail deterministically.
private struct RowFailingExecutor: FlowExecutor {
    let failTask: String
    let inner: MockExecutor
    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset {
        if row.task == failTask {
            throw FlowError.stageFailure(row: path, message: "the receipts are on fire")
        }
        return try await inner.execute(path: path, row: row, inputs: inputs,
                                       transcript: transcript, context: context,
                                       usedFlowContent: usedFlowContent)
    }
}

/// Fails only the Nth (0-indexed) call to a named task, delegating every other call —
/// including other items of the same task — to a real mock. Lets a DA-10-FIX-3 repro fail
/// exactly one item of a multi-item `<each on_error=skip>` while its siblings succeed, unlike
/// `RowFailingExecutor` (which fails a named task unconditionally, for every item). Counts
/// calls rather than matching settings content: `MockExecutor` doesn't echo an `<each>` item's
/// real text, only a synthetic per-item trace string, so content isn't a reliable selector
/// here. `runEach` calls sequentially (`for item in source.items`, `await`ed in order), so a
/// plain counter under a lock is enough — no risk of two items racing for the same index.
private final class NthCallFailingExecutor: FlowExecutor, @unchecked Sendable {
    let failTask: String
    let failOnCallIndex: Int
    let inner: MockExecutor
    private let lock = NSLock()
    private var callIndex = 0

    init(failTask: String, failOnCallIndex: Int, inner: MockExecutor) {
        self.failTask = failTask
        self.failOnCallIndex = failOnCallIndex
        self.inner = inner
    }

    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset {
        if row.task == failTask {
            let index = lock.withLock { () -> Int in
                defer { callIndex += 1 }
                return callIndex
            }
            if index == failOnCallIndex {
                throw FlowError.stageFailure(row: path, message: "the receipts are on fire")
            }
        }
        return try await inner.execute(path: path, row: row, inputs: inputs,
                                       transcript: transcript, context: context,
                                       usedFlowContent: usedFlowContent)
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
private nonisolated struct EchoPromptStage: PipelineStage {
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
