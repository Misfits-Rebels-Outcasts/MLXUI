import Testing
import Foundation
@testable import MLXUI

/// READ-UPSTREAM-1 — end-to-end coverage for the fix: `Read Text`/`Read Audio`/`Read Video`/
/// `Read Index` now check an upstream item of the matching kind before falling back to
/// settings (`tools/files.py::_resolve_path` is unconditional about this in the reference;
/// the Swift port used to special-case these four tools to skip the check entirely). These
/// tests run real gallery flows end to end through `FlowInterpreter` + `RealExecutor`, with
/// only the model-calling closures faked out, to prove the `<each>`-fanned `.file` item now
/// actually reaches `Read Text`/`Read Audio` — not just that the unit-level tool call works
/// (`CatFlowToolTests`/`CatFlowVideoToolsTests`/`CatFlowIndexStoreTests` cover that already).
struct ReadUpstreamIntegrationTests {

    private func makeBase() throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("read-upstream-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    /// A reference box so a fake closure can report back through a `let` capture — same
    /// pattern as `LayaDeciderTests.StageFlag`. `<each>` runs strictly sequentially, so no
    /// synchronization is needed for the append.
    private final class Box<T>: @unchecked Sendable {
        var values: [T] = []
    }

    private func fakeChoiceAnswer(tag: String) -> LayaAnswer {
        var answer = LayaAnswer(type: .choice, confidence: 0.9, probabilities: [0.9, 0.1],
                                optionLabels: [], stateTruncated: false)
        answer.choiceLabel = tag
        return answer
    }

    // MARK: - 77-TicketRouter: every ticket's own text reaches Laya, none fall back/collide

    @Test func flow77RoutesEveryTicketUsingItsOwnFileNotAFallback() async throws {
        let base = try makeBase()
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let flowID = "77-TicketRouter"

        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let doc = try GalleryLoader.loadDocument(flowID: flowID)

        let laya = makeEntry(id: "aac6fef--laya-mlx", family: "Laya", displayName: "Laya 0.4B",
                             modelType: .decision, hfRepo: "aac6fef", hfModelId: "aac6fef/laya-mlx")
        let seenStates = Box<String>()
        var executor = RealExecutor(
            workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in
                throw StageError.unsupportedModel(id: "probe", kind: .llm)
            },
            installedModelIDs: [laya.id], catalog: [laya])
        executor.askLaya = { _, state, _ in
            seenStates.values.append(state)
            return [self.fakeChoiceAnswer(tag: "other")]
        }

        let events = try await FlowInterpreter.run(doc, executor: executor)

        let f003 = events.filter { $0.kind == .flagRaised && $0.code == "F003" }
        #expect(f003.isEmpty, "no ticket should fail: \(f003.map(\.message))")

        // Each of the 5 real ticket bodies must have reached Laya exactly once, as its own
        // distinct text -- proving Read Text pulled the upstream `.file` item's own content
        // rather than falling back to (empty) settings or repeating one file five times.
        let subjects = ["Charged twice", "Export keeps crashing", "changing my email",
                        "VAT number", "Interview request"]
        for subject in subjects {
            #expect(seenStates.values.contains { $0.contains(subject) },
                    "no Laya call carried a ticket mentioning '\(subject)'")
        }
        #expect(Set(seenStates.values).count == 5, "each ticket's text must be distinct")

        let saved = try String(contentsOf: ws.directory(for: flowID).appendingPathComponent("routed-tickets.md"),
                               encoding: .utf8)
        for subject in subjects {
            #expect(saved.contains(subject), "routed-tickets.md is missing the '\(subject)' ticket")
        }
    }

    // MARK: - 37-InboxTriage: same proof, through the LLM (Summarize/Classify) dispatch path

    private nonisolated final class RecordingStage: PipelineStage, @unchecked Sendable {
        let id = "test.recording-stage"
        let name = "Recording Stage"
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        let seen: Box<String>

        init(seen: Box<String>) { self.seen = seen }

        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            guard case .text(let prompt) = input else {
                throw FlowError.stageFailure(row: "test", message: "expected a text prompt")
            }
            seen.values.append(prompt)
            progress(1.0)
            if prompt.contains("Answer with exactly one of these words") {
                return .text("read")   // any declared tag; grouping accuracy isn't under test
            }
            return .text("(summary)")
        }
    }

    @Test func flow37TriagesEveryMailUsingItsOwnFileNotAFallback() async throws {
        let base = try makeBase()
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let flowID = "37-InboxTriage"

        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let doc = try GalleryLoader.loadDocument(flowID: flowID)

        let ministral = makeEntry(id: "mlx-community--Ministral-3-3B-Instruct-2512-4bit",
                                  family: "Ministral", displayName: "Ministral 3B", modelType: .llm,
                                  hfRepo: "mlx-community/Ministral-3-3B-Instruct-2512",
                                  hfModelId: "mlx-community/Ministral-3-3B-Instruct-2512-4bit")
        let seenPrompts = Box<String>()
        let stage = RecordingStage(seen: seenPrompts)
        let executor = RealExecutor(
            workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in stage },
            installedModelIDs: [ministral.id], catalog: [ministral])

        let events = try await FlowInterpreter.run(doc, executor: executor)

        let f003 = events.filter { $0.kind == .flagRaised && $0.code == "F003" }
        #expect(f003.isEmpty, "no mail should fail: \(f003.map(\.message))")

        // Each of the 3 real mail bodies must have reached the model at least once, as its own
        // distinct text -- the Summarize/Classify prompts both render the upstream `.file`
        // item's own content, which never worked before this fix.
        let subjects = ["Contract renewal", "weekly digest", "production down"]
        for subject in subjects {
            #expect(seenPrompts.values.contains { $0.contains(subject) },
                    "no model call carried a mail mentioning '\(subject)'")
        }

        let saved = try String(contentsOf: ws.directory(for: flowID).appendingPathComponent("inbox-brief.md"),
                               encoding: .utf8)
        #expect(saved.contains("(summary)"), "inbox-brief.md should carry the (faked) summaries")
    }

    // MARK: - on_error=skip: the real count, given exactly one genuinely unreadable file

    @Test func oneUnreadableFileInAThreeItemEachGivesOneFailedTwoDelivered() async throws {
        let base = try makeBase()
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let flowID = "probe"
        let ticketsDir = ws.directory(for: flowID).appendingPathComponent("tickets")
        try FileManager.default.createDirectory(at: ticketsDir, withIntermediateDirectories: true)
        try "A".write(to: ticketsDir.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
        let unreadable = ticketsDir.appendingPathComponent("b.txt")
        try "B".write(to: unreadable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: unreadable.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: unreadable.path) }
        try "C".write(to: ticketsDir.appendingPathComponent("c.txt"), atomically: true, encoding: .utf8)

        let text = """
        mlxflow 0.8
        1. Read Files   tickets/; pattern=*.txt
        2. <each one; on_error=skip>   [file] -> [text]
             1. Read Text   (input:1)
             2. Template                "{1}"
        3. Join Text    separator="\\n"
        4. Save Text    out.md
        """
        let doc = try CatParser.parse(text)
        let executor = RealExecutor(
            workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw FlowError.unknownTask(row: "no model in this flow") },
            installedModelIDs: [], catalog: [])

        let events = try await FlowInterpreter.run(doc, executor: executor)

        let f003 = events.filter { $0.kind == .flagRaised && $0.code == "F003" }
        #expect(f003.count == 1, "exactly one item should have failed: \(f003.map(\.message))")
        #expect(f003.first?.message?.contains("Item 2 of 3 failed") == true, "\(f003.first?.message ?? "nil")")
        #expect(f003.first?.message?.contains("was skipped") == true)
        #expect(f003.first?.message?.contains("2 delivered") == true, "\(f003.first?.message ?? "nil")")

        let saved = try String(contentsOf: ws.directory(for: flowID).appendingPathComponent("out.md"), encoding: .utf8)
        #expect(saved.contains("A"))
        #expect(saved.contains("C"))
    }
}
