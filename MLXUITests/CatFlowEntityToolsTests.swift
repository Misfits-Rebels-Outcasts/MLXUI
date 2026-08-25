import Testing
import Foundation
@testable import MLXUI

/// CFM-R12-7 group c — the entity tools: `Append Row` and `Merge Record` (a key join).
struct CatFlowEntityToolsTests {

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-entity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    @Test func appendRowAddsARowAndWidensTheTable() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let table = flowDir.appendingPathComponent("t.table")
        try TableTool.writeTable(columns: ["name", "sales"], rows: [["Ada", "10"]], to: table)

        let item = Item(kind: .table, value: nil, path: table, sourceText: nil)
        let out = try TableTool.appendRow(settings: "name=Lin; sales=25", from: item)
        let outTable = try #require(out.items.first)
        let (columns, rows) = try TableTool.readTable(from: try #require(outTable.path))
        #expect(columns == ["name", "sales"])
        #expect(rows.count == 2)
        #expect(rows[1][0] as? String == "Lin")
        #expect(rows[1][1] as? Int == 25)
    }

    @Test func mergeRecordJoinsOnTheKey() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let a = flowDir.appendingPathComponent("a.table")
        let b = flowDir.appendingPathComponent("b.table")
        try TableTool.writeTable(columns: ["id", "name"], rows: [["1", "Ada"], ["2", "Lin"]], to: a)
        try TableTool.writeTable(columns: ["id", "role"], rows: [["2", "builds"], ["1", "leads"]], to: b)

        let itemA = Item(kind: .table, value: nil, path: a, sourceText: nil)
        let itemB = Item(kind: .table, value: nil, path: b, sourceText: nil)
        let out = try TableTool.mergeRecord(settings: "key=id", inputs: [Asset(items: [itemA]), Asset(items: [itemB])])
        let (columns, rows) = try TableTool.readTable(from: try #require(out.items.first?.path))
        #expect(columns == ["id", "name", "role"])
        #expect(rows.count == 2)
        let ada = rows.first { $0[1] as? String == "Ada" }
        #expect(ada?[2] as? String == "leads")
    }

    /// CFM-R12-FIX-7: widening tables appends new columns in **source order** (the Python's
    /// insertion-ordered pairs) — a Set order would change the schema/bytes/cache key.
    @Test func appendRowWidensInSourceOrder() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let table = flowDir.appendingPathComponent("t.table")
        try TableTool.writeTable(columns: ["name"], rows: [["Ada"]], to: table)

        let item = Item(kind: .table, value: nil, path: table, sourceText: nil)
        let out = try TableTool.appendRow(settings: "sales=25; region=EU; tier=gold", from: item)
        let (columns, _) = try TableTool.readTable(from: try #require(out.items.first?.path))
        #expect(columns == ["name", "sales", "region", "tier"])
    }
}
