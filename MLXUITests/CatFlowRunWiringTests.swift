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
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(flowID).parse.json")
        return try JSONDecoder().decode(FlowDocument.self, from: Data(contentsOf: url))
    }

    // MARK: - FlowRunSession drives dots from events

    @Test func sessionDrivesDotsFromFinishedEvents() async throws {
        let doc = try decode("01-SpokenSummary")
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: makeMockContext())

        // Events are delivered asynchronously; wait for the run to settle.
        try? await Task.sleep(for: .milliseconds(800))
        for row in doc.rows {
            #expect(session.status(for: row.id) == .succeeded,
                    "row \(row.task ?? "?") should have succeeded")
        }
    }

    @Test func sessionFailedRowShowsErrorSentence() async throws {
        // A flow whose row 2 is an unknown task → .failed with a row-naming sentence.
        let rows = [
            Row(task: "Read Text", settings: "a.txt"),
            Row(task: "Nope"),
        ]
        let doc = FlowDocument(version: "0.8", rows: rows)
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: makeMockContext())
        try? await Task.sleep(for: .milliseconds(500))

        #expect(session.status(for: rows[0].id) == .succeeded)
        let row2 = rows[1].id
        #expect(session.status(for: row2) == .failed)
        let sentence = session.errorSentence(for: row2)
        #expect(sentence != nil)
        #expect(sentence?.contains("Nope") == true || sentence?.lowercased().contains("unknown") == true)
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
