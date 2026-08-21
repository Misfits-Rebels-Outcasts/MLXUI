import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R4-1: the six new §2 text instant tools (Split, Filter, Dedupe, Count,
/// Join Text, Template) + Diff (CFM-R4-2) ported from `tools/text.py`, and the full
/// `29-LogTriage` flow running under the mock executor. Expected values are the Python
/// runtime's own outputs (verified by running `tools/text.py` directly). See
/// `RSI/DelegateMergeBacklog.md` CFM-R4-1.
struct CatFlowTextToolsTests {

    private func text(_ value: String) -> Item {
        Item(kind: .text, value: value, path: nil, sourceText: nil)
    }

    private func asset(_ texts: [String]) -> Asset {
        Asset(items: texts.map(text))
    }

    // MARK: - Split

    @Test func splitByLines() async throws {
        let out = try await SplitTool(settings: "by=lines").run(asset(["a\nb\n\nc"])) { _ in }
        #expect(out.items.map(\.value) == ["a", "b", "c"])
    }

    @Test func splitTokensChunked() async throws {
        let out = try await SplitTool(settings: "chunk_size=3").run(asset(["one two three four five six"])) { _ in }
        #expect(out.items.map(\.value) == ["one two three", "four five six"])
    }

    // MARK: - Filter

    @Test func filterByRegex() async throws {
        let out = try await FilterTool(settings: "regex=\"ERROR|FATAL\"")
            .run(asset(["ERROR x", "INFO y", "FATAL z"])) { _ in }
        #expect(out.items.map(\.value) == ["ERROR x", "FATAL z"])
    }

    @Test func filterByLength() async throws {
        let out = try await FilterTool(settings: "length>2").run(asset(["ab", "abc", "abcd"])) { _ in }
        #expect(out.items.map(\.value) == ["abc", "abcd"])
    }

    // MARK: - Dedupe

    @Test func dedupeRemovesExactDuplicates() async throws {
        let out = try await DedupeTool(settings: nil).run(asset(["a", "b", "a", "c", "b"])) { _ in }
        #expect(out.items.map(\.value) == ["a", "b", "c"])
    }

    // MARK: - Count

    @Test func countItems() async throws {
        let out = try await CountTool(settings: "items").run(asset(["a", "b", "c"])) { _ in }
        #expect(out.items.first?.value == "3")
    }

    // MARK: - Join Text

    @Test func joinTextWithSeparator() async throws {
        let out = try await JoinTextTool(settings: "separator=\"\\n\"")
            .run(asset(["a", "b", "c"])) { _ in }
        #expect(out.items.first?.value == "a\nb\nc")
    }

    @Test func joinTextDefaultIsNewline() async throws {
        let out = try await JoinTextTool(settings: nil).run(asset(["a", "b"])) { _ in }
        #expect(out.items.first?.value == "a\nb")
    }

    // MARK: - Template

    @Test func templateSubstitutesNumberedRefs() async throws {
        // The Python unquotes the whole settings string (escaped \n becomes a newline).
        let out = try await TemplateTool(settings: "\"unique errors: {1}\\n\\n{2}\"")
            .run(asset(["X", "Y"])) { _ in }
        #expect(out.items.first?.value == "unique errors: X\n\nY")
    }

    @Test func templateMissingRefIsEmpty() async throws {
        let out = try await TemplateTool(settings: "\"[{5}] {1}\"").run(asset(["X"])) { _ in }
        #expect(out.items.first?.value == "[] X")
    }

    // MARK: - Diff (CFM-R4-2)

    @Test func diffInlineMatchesPython() async throws {
        let out = try await DiffTool(settings: nil)
            .run(asset(["a\nb\nc", "a\nb2\nc"])) { _ in }
        #expect(out.items.first?.value == "  a\n- b\n+ b2\n  c")
    }

    @Test func diffUnifiedMatchesPython() async throws {
        let out = try await DiffTool(settings: "format=unified")
            .run(asset(["a\nb\nc", "a\nb2\nc"])) { _ in }
        let actual = out.items.first?.value ?? ""
        let expected = "--- \n+++ \n@@ -1,3 +1,3 @@\n a\n-b\n+b2\n c"
        if actual != expected {
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent("unified-dbg.txt")
            try? ("actual: [\(actual)]\n\nexpected: [\(expected)]").write(to: dest, atomically: true, encoding: .utf8)
        }
        #expect(actual == expected)
    }

    // MARK: - Full Log Triage flow under mock

    @Test func logTriageRunsEndToEndUnderMock() async throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/29-LogTriage.cat")
        let doc = try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
        #expect(doc.rows.count == 9)

        // Runnable: all instant text tools + one model row, no blocks/clauses.
        #expect(FlowRunner.canRun(doc) == .runnable)

        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-logtriage-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let blob = base.appendingPathComponent("blobs")
        let workspace = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let context = FlowRunner.RunContext(flowID: "29-LogTriage", workspace: workspace,
                                            blobDirectory: blob,
                                            executor: MockExecutor(blobDirectory: blob))

        let runner = FlowRunner()
        var started: [UUID] = []
        var finished: [UUID] = []
        for await event in runner.run(doc, context: context) {
            if case .started(let id) = event { started.append(id) }
            if case .finished(let id, _) = event { finished.append(id) }
        }
        #expect(started.map(\.uuidString) == doc.rows.map(\.id.uuidString))
        #expect(finished.map(\.uuidString) == doc.rows.map(\.id.uuidString))
    }

    @Test func logTriageChainBreakAndRefs() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/29-LogTriage.cat")
        let doc = try CatParser.parse(try String(contentsOf: url, encoding: .utf8))

        // Row 6 (Join Text) starts a new chain.
        #expect(doc.rows[5].chainBreak == true)
        // Row 8 (Template) bundles (5,7) — Count + Summarize.
        let row8 = doc.rows[7]
        #expect(row8.task == "Template")
        let refIDs = row8.refs.compactMap { ref -> UUID? in
            if case .rowRef(let id) = ref { return id }
            return nil
        }
        #expect(refIDs.count == 2)
        #expect(refIDs.contains(doc.rows[4].id))   // row 5 = Count
        #expect(refIDs.contains(doc.rows[6].id))   // row 7 = Summarize
    }
}
