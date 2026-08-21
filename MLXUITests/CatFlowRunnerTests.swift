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
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(flowID).parse.json")
        return try JSONDecoder().decode(FlowDocument.self, from: Data(contentsOf: url))
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

    // MARK: - canRun

    @Test func spokenSummaryIsRunnable() throws {
        #expect(FlowRunner.canRun(try decode("01-SpokenSummary")) == .runnable)
    }

    @Test func photoWebPrepIsNotRunnableNamingEach() throws {
        let doc = try decode("21-PhotoWebPrep")
        let runnability = FlowRunner.canRun(doc)
        guard case .notRunnable(let reason) = runnability else {
            Issue.record("expected not runnable")
            return
        }
        #expect(reason.contains("each") || reason.contains("block"))
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
        // A flow whose row 3 is an unknown task → the executor throws → .failed, no later
        // .started. Use a minimal hand-built document so the failure row is deterministic.
        let rows = [
            Row(task: "Read Text", settings: "a.txt"),
            Row(task: "Read Text", settings: "b.txt"),
            Row(task: "TotallyMissingTask"),
            Row(task: "Save Text", settings: "out.txt"),
        ]
        let doc = FlowDocument(version: "0.8", rows: rows)
        let context = try makeContext()
        let events = try await events(doc, context: context)

        // The run halts at row 3: exactly two rows start, then a failure.
        let startedCount = events.filter { if case .started = $0 { return true }; return false }.count
        let failed = events.filter { if case .failed = $0 { return true }; return false }.count
        #expect(startedCount == 2)
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
