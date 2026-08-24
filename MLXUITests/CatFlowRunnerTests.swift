import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R2-6: `FlowRunner.linear` + `FlowEvent` + `MockExecutor`. `01-SpokenSummary`
/// runs end-to-end under the mock executor (no models, no downloads) and emits the expected
/// event sequence; out-of-scope features refuse with a named reason. See
/// `RSI/DelegateMergeBacklog.md` CFM-R2-6.
struct CatFlowRunnerTests {

    private func decode(_ flowID: String) throws -> FlowDocument {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(flowID).cat")
        return try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
    }

    private func makeContext(_ flowID: String = "01-SpokenSummary") throws -> FlowRunner.RunContext {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-run-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let blob = base.appendingPathComponent("blobs")
        let workspace = FlowWorkspace(root: base.appendingPathComponent("flows"))
        return FlowRunner.RunContext(flowID: flowID, workspace: workspace,
                                     blobDirectory: blob, executor: MockExecutor(blobDirectory: blob))
    }

    private func events(_ doc: FlowDocument, context: FlowRunner.RunContext) async throws -> [FlowEvent] {
        let runner = FlowRunner()
        var result: [FlowEvent] = []
        for await event in runner.run(doc, context: context) {
            result.append(event)
        }
        return result
    }

    private func events(_ doc: FlowDocument, context: FlowRunner.RunContext,
                        startIndex: Int, resumeOutputs: [UUID: Asset]) async throws -> [FlowEvent] {
        let runner = FlowRunner()
        var result: [FlowEvent] = []
        for await event in runner.run(doc, context: context, startIndex: startIndex,
                                      resumeOutputs: resumeOutputs) {
            result.append(event)
        }
        return result
    }

    // MARK: - Caching (CFM-R3-4)

    @Test func secondRunThroughCacheEmitsCacheHit() async throws {
        let doc = try decode("01-SpokenSummary")
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-cacherun-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let store = FlowCacheStore(root: base.appendingPathComponent("cache"))

        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let workspace = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let caching = CachingExecutor(inner: mock, store: store, cacheTier: "real",
                                      catalog: [], runSeed: 0, workspace: workspace, flowID: "t")
        let context = FlowRunner.RunContext(flowID: "t", workspace: workspace,
                                            blobDirectory: base.appendingPathComponent("blobs"),
                                            executor: caching)

        // First run: real execution, no hits.
        let first = try await events(doc, context: context)
        #expect(first.contains { if case .cacheHit = $0 { return true }; return false } == false)

        // Second run: every cacheable row is a hit (Save rows are NEVER_CACHE).
        let second = try await events(doc, context: context)
        let hits = second.filter { if case .cacheHit = $0 { return true }; return false }.count
        let started = second.filter { if case .started = $0 { return true }; return false }.count
        #expect(hits > 0)
        #expect(hits <= started)   // some rows may be NEVER_CACHE (Save*)
    }

    @Test func startIndexIsIgnoredByTheFullInterpreter() async throws {
        // CFM-R7: the full interpreter (which now backs `FlowRunner.run`) always runs the
        // whole flow from the top — `startIndex`/`resumeOutputs` are retained only for the
        // session's resume bookkeeping. Unchanged rows replay as cache hits, which is fast.
        let doc = try decode("01-SpokenSummary")
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-resume-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let blob = base.appendingPathComponent("blobs")
        let workspace = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let context = FlowRunner.RunContext(flowID: "t", workspace: workspace,
                                            blobDirectory: blob,
                                            executor: MockExecutor(blobDirectory: blob))

        // Even a "resume from row 3" call runs every row.
        let resumeOutputs: [UUID: Asset] = [
            doc.rows[1].id: Asset(items: [Item(kind: .text, value: "the transcript",
                                               path: nil, sourceText: nil)]),
        ]
        let events = try await events(doc, context: context, startIndex: 3,
                                      resumeOutputs: resumeOutputs)
        let started = events.compactMap { event -> UUID? in
            if case .started(let id) = event { return id }
            return nil
        }
        #expect(started.map(\.uuidString) == doc.rows.map(\.id.uuidString))
    }

    // MARK: - canRun

    @Test func spokenSummaryIsRunnable() throws {
        #expect(FlowRunner.canRun(try decode("01-SpokenSummary")) == .runnable)
    }

    @Test func photoWebPrepIsBlockedByUnportedToolsUntilR12_5() throws {
        // CFM-R7-FIX-2: blocks are no longer refused per se. R12-4 added the unported-tool
        // gate, so `21-PhotoWebPrep` is now honestly refused *before* running — it uses
        // `Resize`/`Watermark`/`Save Images`, which aren't ported yet (R12-5/7). A block
        // flow whose tools all exist (02-MeetingMinutes) stays runnable.
        guard case .notRunnable(let reason) = FlowRunner.canRun(try decode("21-PhotoWebPrep")) else {
            Issue.record("expected notRunnable (unported tools)")
            return
        }
        #expect(reason.contains("doesn't run yet"))
        #expect(FlowRunner.canRun(try decode("02-MeetingMinutes")) == .runnable)
    }

    // MARK: - B4: auto-chain is shape-gated

    @Test func incompatibleAutoChainFeedsNoInput() async throws {
        // `1. Read Text / 2. Summarize / 3. Save Text summary.txt / 4. Title`. Row 3 gives
        // `Single(status)`; row 4 (Title) auto-chains it — shape-incompatible, so it must
        // receive NO input (the Python's `single_compatible` gate), not the status string.
        let rows = [
            Row(task: "Read Text", settings: "a.txt"),
            Row(task: "Summarize", model: "Qwen3 8B"),
            Row(task: "Save Text", settings: "summary.txt"),
            Row(task: "Title", model: "Qwen3 8B"),
        ]
        let doc = FlowDocument(version: "0.8", rows: rows)
        let recording = RecordingExecutor()
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-b4-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let context = FlowRunner.RunContext(flowID: "t", workspace: FlowWorkspace(root: base),
                                            blobDirectory: base, executor: recording)
        for await _ in FlowRunner().run(doc, context: context) {}

        // Row 2 (Summarize) chains row 1's text — compatible, fed.
        #expect(recording.received["2"]?.isEmpty == false)
        // Row 4 (Title) chains row 3's status — incompatible, fed nothing (B4).
        #expect(recording.received["4"]?.isEmpty == true)
    }

    // MARK: - Full mock run of Spoken Summary

    @Test func spokenSummaryRunsEndToEndWithExpectedEventSequence() async throws {
        let doc = try decode("01-SpokenSummary")
        let context = try makeContext()
        let events = try await events(doc, context: context)

        // One started + one finished per row, in order; no failures.
        let started = events.compactMap { event -> UUID? in
            if case .started(let id) = event { return id }
            return nil
        }
        let finished = events.compactMap { event -> UUID? in
            if case .finished(let id, _) = event { return id }
            return nil
        }
        #expect(started.map(\.uuidString) == doc.rows.map(\.id.uuidString))
        #expect(finished.map(\.uuidString) == doc.rows.map(\.id.uuidString))
        #expect(events.contains { if case .failed = $0 { return true }; return false } == false)
    }

    @Test func row6InputIsRow2OutputNotRow5() async throws {
        let doc = try decode("01-SpokenSummary")
        let context = try makeContext()
        let events = try await events(doc, context: context)

        // Row 6 (Save Text, ref (2)) is fed row 2's output — a text asset whose content
        // fingerprint embeds the Transcribe path. Row 5's output is the WAV path for Save
        // Audio — different content entirely.
        guard let row2Event = events.last(where: { if case .finished(let id, _) = $0, id == doc.rows[1].id { return true }; return false }),
              case .finished(_, let row2Asset) = row2Event else {
            Issue.record("row 2 never finished")
            return
        }
        guard let row5Event = events.last(where: { if case .finished(let id, _) = $0, id == doc.rows[4].id { return true }; return false }),
              case .finished(_, let row5Asset) = row5Event else {
            Issue.record("row 5 never finished")
            return
        }
        guard let row6Event = events.last(where: { if case .finished(let id, _) = $0, id == doc.rows[5].id { return true }; return false }),
              case .finished(_, let row6Asset) = row6Event else {
            Issue.record("row 6 never finished")
            return
        }
        // Row 6's status value embeds its input's fingerprint — row 2's text content, not
        // row 5's audio content.
        let row6Text = row6Asset.items.first?.value ?? ""
        let row2Marker = row2Asset.items.first?.value ?? ""
        let row5Marker = row5Asset.items.first?.value ?? ""
        #expect(row2Marker.hasPrefix("[mock:text]"))
        #expect(row6Text.contains(row2Marker))
        #expect(!row6Text.contains(row5Marker))
    }

    @Test func outputsAreStoredPerRowID() async throws {
        let doc = try decode("01-SpokenSummary")
        let context = try makeContext()
        let events = try await events(doc, context: context)

        let finishedIDs = events.compactMap { event -> UUID? in
            if case .finished(let id, _) = event { return id }
            return nil
        }
        // All six distinct row ids finished.
        #expect(Set(finishedIDs).count == 6)
    }

    // MARK: - Stage failure halts

    @Test func stageFailureEmitsFailedAndHalts() async throws {
        // CFM-R10-FIX-1: `FlowRunner.run` consults `canRun` before starting — a flow with an
        // unknown task is refused upfront (a `.failed` on row 1), never partially started.
        let rows = [
            Row(task: "Read Text", settings: "a.txt"),
            Row(task: "Read Text", settings: "b.txt"),
            Row(task: "TotallyMissingTask"),
            Row(task: "Save Text", settings: "out.txt"),
        ]
        let doc = FlowDocument(version: "0.8", rows: rows)
        let context = try makeContext()
        let events = try await events(doc, context: context)

        let startedCount = events.filter { if case .started = $0 { return true }; return false }.count
        let failed = events.filter { if case .failed = $0 { return true }; return false }.count
        #expect(startedCount == 0)
        #expect(failed == 1)
    }

    // MARK: - MockExecutor determinism

    @Test func mockExecutorIsDeterministic() async throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-mock-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)

        let a = MockExecutor(blobDirectory: base.appendingPathComponent("a"))
        let b = MockExecutor(blobDirectory: base.appendingPathComponent("b"))
        let row = Row(task: "Summarize", model: "Qwen3 8B", settings: "\"TL;DR\"")
        let input = Asset(items: [Item(kind: .text, value: "the transcript", path: nil, sourceText: nil)])

        let outA = try await a.execute(path: "3", row: row, inputs: [input])
        let outB = try await b.execute(path: "3", row: row, inputs: [input])
        #expect(outA.items.first?.value == outB.items.first?.value)
    }

    @Test func mockTextItemHasKindDrivenPrefix() async throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-mock2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))

        let row = Row(task: "Transcribe")
        let input = Asset(items: [Item(kind: .audio, value: nil,
                                       path: base.appendingPathComponent("blob.bin"), sourceText: nil)])
        let out = try await mock.execute(path: "2", row: row, inputs: [input])
        let value = out.items.first?.value ?? ""
        #expect(value.hasPrefix("[mock:text]"))
    }
}

/// An executor that records the inputs each row received (by path) — the B4 harness: it
/// proves an incompatible auto-chain row is fed nothing rather than the wrong output. Outputs
/// follow the task's real kind (Save Text → status), so the interpreter's runtime shape gate
/// sees what a real flow produces.
private final class RecordingExecutor: FlowExecutor, @unchecked Sendable {
    private let lock = NSLock()
    private var _received: [String: [Asset]] = [:]
    var received: [String: [Asset]] { lock.withLock { _received } }

    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset {
        lock.withLock { _received[path] = inputs }
        let kind: Kind = row.task == "Save Text" ? .status : .text
        return Asset(items: [Item(kind: kind, value: "ok", path: nil, sourceText: nil)])
    }
}
