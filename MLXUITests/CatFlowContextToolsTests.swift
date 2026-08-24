import Testing
import Foundation
@testable import MLXUI

/// CFM-R12-7 group b — the context tools: Save/Read move the journal JSON between a file
/// and a `context` item; Count Context counts its entries.
struct CatFlowContextToolsTests {

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-ctx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    @Test func saveThenReadRoundTripsTheJournal() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        try FileManager.default.createDirectory(at: ws.directory(for: "f"),
                                                withIntermediateDirectories: true)
        let journal = #"[{"label": "decisions", "content": "ship"}]"#

        let save = SaveContextTool(workspace: ws, flowID: "f", settings: "ctx.json")
        let saved = try await save.run(Asset(items: [Item(kind: .context, value: journal, path: nil, sourceText: nil)])) { _ in }
        #expect(saved.items.first?.value?.hasPrefix("saved to") == true)

        let read = ReadContextTool(workspace: ws, flowID: "f", settings: "ctx.json")
        let out = try await read.run(Asset(items: [])) { _ in }
        let item = try #require(out.items.first)
        #expect(item.kind == .context)
        #expect(item.value == journal)
    }

    @Test func countContextCountsEntries() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let journal = #"[{"label": "a", "content": "1"}, {"label": "b", "content": "2"}, {"label": "c", "content": "3"}]"#
        let tool = CountContextTool(workspace: ws, flowID: "f", settings: "entries")
        let out = try await tool.run(Asset(items: [Item(kind: .context, value: journal, path: nil, sourceText: nil)])) { _ in }
        #expect(out.items.first?.value == "3")
    }

    @Test func flowsUnblockedByContextAreRunnableNow() throws {
        // 18/46/50 needed Save/Read Context; 55 needed Compare (group a, landed after).
        let runnable = ["18-DocChat", "46-IndexSelfTest", "50-NightlyDrift", "55-ReceiptsLedger"]
        for fid in runnable {
            let doc = try GalleryLoader.loadDocument(flowID: fid)
            #expect(FlowRunner.canRun(doc) == .runnable, "\(fid) should be runnable now")
        }
    }
}
