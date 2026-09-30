import Testing
import Foundation
@testable import MLXUI

/// FIP-1 — the input counterpart to `FlowSavedFile`: which file (or index directory) a
/// `Read *` row will actually read when the flow runs, honoring the upstream-wins-else
/// -settings-path precedence documented at `ReadTools.swift`'s `ReadPath` doc comment, and
/// refusing — never probing — a path that escapes the flow directory.
struct CatFlowInputFileTests {

    private func makeScope(root: URL, flowID: String = "flow-1") -> FlowScope {
        FlowScope(identity: flowID, workspace: FlowWorkspace(root: root), locationID: flowID)
    }

    private func row(_ task: String?, settings: String? = nil, refs: [Ref] = []) -> Row {
        Row(id: UUID(), task: task, settings: settings, refs: refs)
    }

    private func tempRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("catflow-input-\(UUID().uuidString)")
    }

    // MARK: - The basic present/missing cases

    @Test func targetResolvesAPresentFile() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        let dir = scope.workspace.directory(for: "flow-1")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try "hi".write(to: dir.appendingPathComponent("notes.txt"), atomically: true, encoding: .utf8)

        let r = row("Read Text", settings: "notes.txt")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [r])
        let url = FlowInputFile.target(for: r, document: doc, scope: scope)
        #expect(url?.lastPathComponent == "notes.txt")
        #expect(!FlowInputFile.isMissing(for: r, document: doc, scope: scope))
    }

    @Test func isMissingReportsAnAbsentFile() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        try FileManager.default.createDirectory(at: scope.workspace.directory(for: "flow-1"),
                                                withIntermediateDirectories: true)

        let r = row("Read Text", settings: "notes.txt")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [r])
        #expect(FlowInputFile.isMissing(for: r, document: doc, scope: scope))
    }

    @Test func notAReadTaskOrNoPathReportsNothing() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)

        let notRead = row("Save Text", settings: "out.md")
        let doc1 = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [notRead])
        #expect(FlowInputFile.target(for: notRead, document: doc1, scope: scope) == nil)
        #expect(!FlowInputFile.isMissing(for: notRead, document: doc1, scope: scope))

        let noPath = row("Read Text", settings: nil)
        let doc2 = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [noPath])
        #expect(FlowInputFile.target(for: noPath, document: doc2, scope: scope) == nil)
    }

    // MARK: - The escape hatch: refuse, never probe

    @Test func targetRefusesAnEscapingPathRatherThanProbingIt() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        try FileManager.default.createDirectory(at: scope.workspace.directory(for: "flow-1"),
                                                withIntermediateDirectories: true)

        let r = row("Read Text", settings: "../../etc/hosts")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [r])
        #expect(FlowInputFile.target(for: r, document: doc, scope: scope) == nil)
    }

    // MARK: - The precedence rule, both directions

    /// `Read PDF` checks upstream (`ReadPath.resolve`); auto-chained after `Read Files`
    /// (gives `.listOf(.file)`), it must report nothing even though its own settings still
    /// name a file nobody wrote — the exact "stale path" case the backlog calls out.
    @Test func rowFedByAMatchingUpstreamIsNotCheckedEvenWithAStalePath() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        let files = row("Read Files", settings: "docs/; pattern=*.pdf")
        let pdf = row("Read PDF", settings: "stale-name-nobody-wrote.pdf")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [files, pdf])
        #expect(FlowInputFile.target(for: pdf, document: doc, scope: scope) == nil)
        #expect(!FlowInputFile.isMissing(for: pdf, document: doc, scope: scope))
    }

    /// READ-UPSTREAM-1: `Read Text` used to be the one exception to the precedence rule
    /// above — even fed by a matching upstream row, it still checked its own (possibly
    /// stale) settings path, because its tool never looked at the upstream item at all
    /// (verified against `SpokenSummaryTools.swift` directly). That exception is gone —
    /// `ReadTextTool` now calls `ReadPath.resolve` with the default `checksUpstream: true`,
    /// so the advisory must agree, exactly like `Read PDF` above. This is the exact
    /// mismatch smoke row 97 hit (`77-TicketRouter`'s `Read Text (input:1)` inside `<each>`).
    @Test func readTextFedByAMatchingUpstreamIsNotCheckedEvenWithAStalePath() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        let files = row("Read Files", settings: "docs/; pattern=*.txt")
        let text = row("Read Text", settings: "stale-name-nobody-wrote.txt")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [files, text])
        #expect(FlowInputFile.target(for: text, document: doc, scope: scope) == nil)
        #expect(!FlowInputFile.isMissing(for: text, document: doc, scope: scope))
    }

    @Test func rowWithAnExplicitRefIsNeverCheckedRegardlessOfTask() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        let source = row("Read Text", settings: "notes.txt")
        let refd = row("Read PDF", settings: "stale.pdf", refs: [.rowRef(source.id)])
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [source, refd])
        #expect(FlowInputFile.target(for: refd, document: doc, scope: scope) == nil)
    }

    /// `chainBreak` on the *preceding* row stops the auto-chain (the blank-line rule) — the
    /// row after it is no longer "fed by upstream" and must check its own path again.
    @Test func chainBreakStopsAutoChainSoTheRowChecksItsOwnPath() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        var files = row("Read Files", settings: "docs/; pattern=*.pdf")
        files.chainBreak = true
        let pdf = row("Read PDF", settings: "stale.pdf")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [files, pdf])
        #expect(FlowInputFile.target(for: pdf, document: doc, scope: scope) != nil)
    }

    // MARK: - Read Index resolves against manifest.json

    @Test func readIndexChecksForManifestJSONInsideTheDirectory() throws {
        let base = tempRoot()
        defer { try? FileManager.default.removeItem(at: base) }
        let scope = makeScope(root: base)
        let dir = scope.workspace.directory(for: "flow-1")
        let idx = dir.appendingPathComponent("library.index", isDirectory: true)
        try FileManager.default.createDirectory(at: idx, withIntermediateDirectories: true)

        let r = row("Read Index", settings: "library.index")
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [r])
        #expect(FlowInputFile.isMissing(for: r, document: doc, scope: scope))   // no manifest.json yet

        try Data("{}".utf8).write(to: idx.appendingPathComponent("manifest.json"))
        #expect(!FlowInputFile.isMissing(for: r, document: doc, scope: scope))
    }

    // MARK: - Every task in the canonical eleven is classified, not assumed

    @Test func everyReadTaskIsClassifiedForUpstreamAwareness() {
        // READ-UPSTREAM-1: verified by reading each tool's own run(_:progress:) directly —
        // ReadPath.resolve-backed (ReadTools.swift, now every one of `Read Image`/`Read
        // Images`/`Read Files`/`Read PDF`/`Read Audio`/`Read Text`/`Read Video`/`Read Index`)
        // + Read Context's own inline check (ContextTools.swift, "FIX-11") + Read CSV/JSON via
        // RealExecutor.resolveFile — all eleven check upstream now. `tools/files.py::
        // _resolve_path` never had a settings-only exception; neither does any tool anymore.
        for task in SampleSeed.readTaskNames {
            #expect(FlowInputFile.checksUpstream(task), "\(task) should check upstream")
        }
    }
}
