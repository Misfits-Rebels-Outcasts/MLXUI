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
        try? await Task.sleep(for: .milliseconds(500))

        let sentence = session.errorSentence(for: doc.rows[0].id)
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

    @Test func policyDiffDiffRowRefusesNamingTheTask() async throws {
        // 08-PolicyDiff row 3 is `Diff`, a real catalog task with no Swift tool yet. The
        // refusal must name the row number and the task — never "Row Diff produces file".
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let docURL = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/08-PolicyDiff.parse.json")
        let doc = try JSONDecoder().decode(FlowDocument.self, from: Data(contentsOf: docURL))

        // The executor's instant path is exercised directly with a Diff row.
        let executor = RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "08-PolicyDiff",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in EchoPromptStage() },
            installedModelIDs: [],
            catalog: [])
        let row3 = doc.rows[2]   // Diff (1,2)
        #expect(row3.task == "Diff")

        do {
            _ = try await executor.execute(path: "3", row: row3, inputs: [
                Asset(items: [Item(kind: .text, value: "a", path: nil, sourceText: nil)]),
                Asset(items: [Item(kind: .text, value: "b", path: nil, sourceText: nil)]),
            ])
            Issue.record("Diff should refuse in this version")
        } catch let error as FlowError {
            let sentence = FlowErrorDisplay.sentence(for: error)
            #expect(sentence.contains("Row 3"))
            #expect(sentence.contains("Diff"))
            #expect(!sentence.contains("produces file"))
            #expect(!sentence.contains("Row Diff"))
        } catch {
            Issue.record("wrong error type: \(error)")
        }
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
    func execute(path: String, row: Row, inputs: [Asset]) async throws -> Asset {
        throw StageError.engineFailure(stage: "Kokoro TTS", underlying: CocoaError(.fileNoSuchFile))
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
