import Testing
import Foundation
@testable import MLXUI

/// FIP-2 — `FlowInputAdvisory` renders `FlowInputFile` (FIP-1) as the same
/// `FlowPreflight.RowAdvisory` shape the UI already treats as a non-blocking banner. Owner
/// ruling, 2026-09-16 (`RSI/DelegateFlowInputPreflightBacklog.md`, Q1): "Warn only
/// (Recommended)" — there is no blocking path to test here at all, only that the right row
/// is named (or nothing is, for a correct flow) in the right words.
struct CatFlowInputAdvisoryTests {

    private func makeScope(root: URL, flowID: String = "flow-1") -> FlowScope {
        FlowScope(identity: flowID, workspace: FlowWorkspace(root: root), locationID: flowID)
    }

    private func row(_ task: String?, settings: String? = nil, refs: [Ref] = [],
                     children: [Row] = []) -> Row {
        Row(id: UUID(), task: task, settings: settings, refs: refs, children: children)
    }

    private func tempRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("catflow-input-advisory-\(UUID().uuidString)")
    }

    @Test func aPresentFileAdvisesNothing() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        let dir = scope.workspace.directory(for: "flow-1")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try "hi".write(to: dir.appendingPathComponent("notes.txt"), atomically: true, encoding: .utf8)

        let r = row("Read Text", settings: "notes.txt")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [r])
        #expect(FlowInputAdvisory.advisory(for: doc, scope: scope) == nil)
    }

    @Test func aMissingFileNamesTheRowAndTheFile() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        try FileManager.default.createDirectory(at: scope.workspace.directory(for: "flow-1"),
                                                withIntermediateDirectories: true)

        let r = row("Read Text", settings: "notes.txt")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [r])
        let advisory = FlowInputAdvisory.advisory(for: doc, scope: scope)
        #expect(advisory?.task == "Read Text")
        #expect(advisory?.reason.contains("notes.txt") == true)
        // Warn only: no fix-it action exists for "the file isn't there" — there is no
        // settings pane that creates one.
        #expect(advisory?.action == nil)
    }

    @Test func readIndexGetsItsOwnBuildFirstWording() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        try FileManager.default.createDirectory(at: scope.workspace.directory(for: "flow-1"),
                                                withIntermediateDirectories: true)

        let r = row("Read Index", settings: "library.index")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [r])
        let advisory = FlowInputAdvisory.advisory(for: doc, scope: scope)
        #expect(advisory?.task == "Read Index")
        #expect(advisory?.reason.contains("built") == true)
    }

    /// The precedence check this whole stream exists for: a row fed by a matching upstream
    /// reads no file at all, even though its settings still carry a path that doesn't exist —
    /// flagging it would put a false warning on a correct flow.
    @Test func aRowFedByUpstreamIsNeverFlaggedEvenWithAStalePath() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        let dir = scope.workspace.directory(for: "flow-1")
        // The upstream row's own path must itself be present, or its *own* missing-input
        // advisory would fire first and mask the precedence check this test is about.
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("docs"),
                                                withIntermediateDirectories: true)

        let upstream = row("Read Files", settings: "docs/; pattern=*.pdf")
        let downstream = row("Read PDF", settings: "stale-does-not-exist.pdf")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [upstream, downstream])
        #expect(FlowInputAdvisory.advisory(for: doc, scope: scope) == nil)
    }

    @Test func reportsOnlyTheFirstMissingRowNotEvery() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        try FileManager.default.createDirectory(at: scope.workspace.directory(for: "flow-1"),
                                                withIntermediateDirectories: true)

        let first = row("Read Text", settings: "one.txt")
        let second = row("Read Audio", settings: "two.wav")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [first, second])
        let advisory = FlowInputAdvisory.advisory(for: doc, scope: scope)
        #expect(advisory?.task == "Read Text")
    }

    @Test func walksIntoBlockChildrenNotJustTopLevelRows() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        try FileManager.default.createDirectory(at: scope.workspace.directory(for: "flow-1"),
                                                withIntermediateDirectories: true)

        let child = row("Read Text", settings: "missing.txt")
        let block = row(nil, children: [child])
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [block])
        let advisory = FlowInputAdvisory.advisory(for: doc, scope: scope)
        #expect(advisory?.task == "Read Text")
    }
}
