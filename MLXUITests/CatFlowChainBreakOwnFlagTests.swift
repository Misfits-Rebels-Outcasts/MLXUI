import Testing
import Foundation
@testable import MLXUI

/// RP-3-FIX-1 (owner go-ahead 2026-10-20) — `Row.chainBreak` is the blank line above **this**
/// row (CatSerializer, FlowValidator:1061/1226, parser, interpreter). "The row above feeds me"
/// therefore needs `!row.chainBreak` on the current row; the row above's flag is irrelevant.
/// `FlowEditorModel.previousRow`, `FlowInputFile.previousTopLevelRow` and `RowSources.isFed`
/// used to read `previous.chainBreak`.
struct CatFlowChainBreakOwnFlagTests {

    /// Model-class rows get the catalog's default model, so the model warning does not mask the
    /// prompt/input sentences under test.
    @MainActor private func editor(_ text: String) throws -> FlowEditorModel {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
            .domains.flatMap { $0.allModels }
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        var doc = try CatParser.parse(text)
        for i in doc.rows.indices {
            guard let t = doc.rows[i].task, TaskCatalog.get(t)?.taskClass == .model, doc.rows[i].model == nil else { continue }
            doc.rows[i].model = FlowEditorModel.defaultModel(forTask: t, catalog: catalog, claimableModelIDs: claimable)
        }
        let e = FlowEditorModel(name: "cb", document: doc,
                                workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
        e.modelCatalog = catalog
        e.claimableModelIDs = claimable
        return e
    }

    /// An editor with a catalog, built row by row through `add(task:)` so model rows arrive
    /// with a pinned model (a parsed doc has no `models:` block, and that warning would mask
    /// the sentences under test).
    @MainActor private func built(_ tasks: [String], breakBefore: Int? = nil) throws -> FlowEditorModel {
        let e = try editor("mlxflow 0.8")
        for (i, t) in tasks.enumerated() {
            e.add(task: t)
            if i == breakBefore, let id = e.document.rows.last?.id { e.setChainBreak(true, for: id) }
        }
        return e
    }

    // (a) a promptless generator right after a blank line is not fed → "needs a prompt"
    @Test @MainActor func aPromptlessGeneratorAfterABlankLineIsWarned() throws {
        for g in ["Generate Image", "Generate Sound", "Generate Video"] {
            let broken = try built(["Read Text", g], breakBefore: 1)
            let row = try #require(broken.document.rows.last)
            broken.setInstruction(nil, for: row.id); broken.setSetting(key: "seed", value: "1", for: row.id)
            let w = broken.warning(for: row.id)
            #expect(w?.contains("needs a prompt") == true, "\(g): \(w ?? "nil")")
            // Contrast: without the blank line the row above feeds it, so no warning.
            let fed = try built(["Read Text", g])
            let frow = try #require(fed.document.rows.last)
            fed.setChainBreak(false, for: frow.id)      // RP-5: + Add step now arrives ticked (blank line); this row must be fed
            fed.setInstruction(nil, for: frow.id); fed.setSetting(key: "seed", value: "1", for: frow.id)
            #expect(fed.warning(for: frow.id) == nil, "\(g): \(fed.warning(for: frow.id) ?? "nil")")
        }
    }

    // (a') a row after a blank line is not checked against the row above the break. (The
    // editor's "can't take row N's output" branch sits behind the validator's own E201, which
    // fires on the same condition and ignores the break; FlowValidator is out of scope, so
    // this pins the editor's `previousRow`, the input of that branch, directly.)
    @Test @MainActor func aRowAfterABlankLineHasNoPreviousRowToCheckAgainst() throws {
        let e = try editor("mlxflow 0.8\n1. Read Image   photo.png\n\n2. Summarize   \"x\"\n3. Summarize   \"y\"")
        let rows = e.document.rows
        #expect(rows[1].chainBreak && !rows[0].chainBreak)
        #expect(e.previousRow(before: rows[1].id) == nil)                 // the blank line is above row 2
        #expect(e.previousRow(before: rows[2].id)?.id == rows[1].id)      // row 2's own flag is irrelevant to row 3
        #expect(e.previousRow(before: rows[0].id) == nil)
    }

    // (b) a row directly below a chain-start row is still fed by it
    @Test @MainActor func aRowBelowAChainStartRowIsStillFedByIt() throws {
        let e = try editor("""
        mlxflow 0.8
        1. Template   "hello"

        2. Generate Image   "a lighthouse"
        3. Save Image   out.png
        """)
        #expect(e.warning(for: e.document.rows[2].id) == nil)
        #expect(RowSources.isFed(e.document.rows[2], previous: e.document.rows[1]))
        #expect(!RowSources.isFed(e.document.rows[1], previous: e.document.rows[0]))
    }

    @Test func isFedReadsTheRowsOwnFlag() {
        let text = Row(id: UUID(), task: "Template", settings: "\"x\"", refs: [], chainBreak: true)
        let gen = Row(id: UUID(), task: "Generate Image", settings: nil, refs: [])
        #expect(RowSources.isFed(gen, previous: text))                       // the above row's flag is irrelevant
        var broken = gen; broken.chainBreak = true
        #expect(!RowSources.isFed(broken, previous: text))
    }

    @Test func readRowAdvisoryUsesTheUpstreamRuleAcrossAChainStart() throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("cb-\(UUID().uuidString)")
        let scope = FlowScope(identity: "f", workspace: FlowWorkspace(root: base), locationID: "f")
        let files = Row(id: UUID(), task: "Read Files", settings: "docs/; pattern=*.pdf", refs: [], chainBreak: true)
        let pdf = Row(id: UUID(), task: "Read PDF", settings: "stale.pdf", refs: [])
        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [files, pdf])
        #expect(FlowInputFile.target(for: pdf, document: doc, scope: scope) == nil)      // fed by Read Files
        var cut = pdf; cut.chainBreak = true
        let doc2 = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [files, cut])
        #expect(FlowInputFile.target(for: cut, document: doc2, scope: scope) != nil)     // own path again
    }
}
