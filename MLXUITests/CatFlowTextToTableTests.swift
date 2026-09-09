import Testing
import Foundation
@testable import MLXUI

/// DA-6 (`RSI/DelegateDeciderBacklog.md`) — Text to Table (SPEC-Q112), ported from
/// `catflow-mlx/tests/test_engines_llm_text_to_table.py` (11 tests) plus a
/// `parse_delimited_table` set, at commit `3600a4a`. The fast path is exercised for real
/// (deterministic, no mocking); the model fallback injects `gate` / `field` as closures — the
/// same seam DA-3a uses.
struct CatFlowTextToTableTests {

    private func textAsset(_ values: String...) -> Asset {
        Asset(items: values.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }

    private func blobDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("t2t-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func expectStageFailure(containing needle: String, _ body: () throws -> Void) {
        do {
            try body()
            Issue.record("expected a throw")
        } catch let FlowError.stageFailure(_, message) {
            #expect(message.contains(needle), "message was: \(message)")
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    // MARK: - _parse_expected

    @Test func parseExpectedSplitsOnCommasAndStrips() throws {
        #expect(try TextToTableStage.parseExpected("expected=\"product, issue_type, severity\"")
                == ["product", "issue_type", "severity"])
    }

    @Test func parseExpectedNoSettingsRaises() {
        expectStageFailure(containing: "needs an `expected=`") {
            _ = try TextToTableStage.parseExpected(nil)
        }
    }

    @Test func parseExpectedMissingKeyRaises() {
        expectStageFailure(containing: "needs an `expected=`") {
            _ = try TextToTableStage.parseExpected("format=csv")
        }
    }

    @Test func parseExpectedDropsEmptyFieldsFromTrailingCommas() throws {
        #expect(try TextToTableStage.parseExpected("expected=\"name, date,\"") == ["name", "date"])
    }

    // MARK: - parse_delimited_table (the fast path, pure)

    private func strings(_ rows: [TableTool.Row]?) -> [[String?]]? {
        rows?.map { $0.map { $0 as? String } }
    }

    @Test func parseDelimitedTableParsesCleanCsv() {
        let rows = TableTool.parseDelimitedTable(source: "product,severity\nDashboard,High\nMobile App,High",
                                                 columns: ["product", "severity"])
        #expect(strings(rows) == [["Dashboard", "High"], ["Mobile App", "High"]])
    }

    @Test func parseDelimitedTableDropsRepeatedJoinTextHeaders() {
        // Gallery 28's shape: per-item `Table to Text format=csv` blocks stitched by Join Text,
        // each still carrying its header line.
        let source = "product,severity\r\nDashboard,High\r\nproduct,severity\r\nMobile App,High"
        let rows = TableTool.parseDelimitedTable(source: source, columns: ["product", "severity"])
        #expect(strings(rows) == [["Dashboard", "High"], ["Mobile App", "High"]])
    }

    @Test func parseDelimitedTableMatchesHeaderOrderInsensitivelyAndReorders() {
        // Header is `b,a`; `expected=` is `a, b` — cells must come back in the declared order.
        let rows = TableTool.parseDelimitedTable(source: "b,a\ny,x", columns: ["a", "b"])
        #expect(strings(rows) == [["x", "y"]])
    }

    @Test func parseDelimitedTableReturnsNilForARaggedRow() {
        #expect(TableTool.parseDelimitedTable(source: "a,b\n1,2\n3", columns: ["a", "b"]) == nil)
    }

    @Test func parseDelimitedTableReturnsNilWhenHeaderDoesNotMatchExpected() {
        #expect(TableTool.parseDelimitedTable(source: "x,y\n1,2", columns: ["a", "b"]) == nil)
        #expect(TableTool.parseDelimitedTable(source: "a,b,c\n1,2,3", columns: ["a", "b"]) == nil)
    }

    @Test func parseDelimitedTableCoercesCellsIncludingCurrency() {
        // `_coerce` (SPEC-Q114 inherited): `$5000` → 5000, a bare int stays an int.
        let rows = TableTool.parseDelimitedTable(source: "item,total\nCoffee,$5000\nTea,3",
                                                 columns: ["item", "total"])
        #expect(TableTool.number(rows?[0].last ?? nil) == 5000)
        #expect(TableTool.number(rows?[1].last ?? nil) == 3)
    }

    // MARK: - text_to_table: fast path (model: nil), through the stage

    @Test func toTableFastPathParsesCsvWithoutAModel() async throws {
        let out = try await TextToTableStage.toTable(
            inputs: [textAsset("product,severity\nDashboard,High\nMobile App,High")],
            settings: "expected=\"product, severity\"", model: nil, in: blobDir())
        let (columns, rows) = try TableTool.readTable(from: #require(out.items.first?.path))
        #expect(columns == ["product", "severity"])
        #expect(rows.map { $0.map { $0 as? String } } == [["Dashboard", "High"], ["Mobile App", "High"]])
    }

    @Test func toTableNoInputRaises() async {
        do {
            _ = try await TextToTableStage.toTable(
                inputs: [], settings: "expected=\"name, date\"", model: nil, in: blobDir())
            Issue.record("expected a throw")
        } catch let FlowError.stageFailure(_, message) {
            #expect(message.contains("input text asset"))
        } catch { Issue.record("wrong error: \(error)") }
    }

    @Test func toTableNoModelAndMessyTextRaisesNeedsAModel() async {
        do {
            _ = try await TextToTableStage.toTable(
                inputs: [textAsset("Ada joined 2024-01-01, a great addition to the team.")],
                settings: "expected=\"name, date\"", model: nil, in: blobDir())
            Issue.record("expected a throw")
        } catch let FlowError.stageFailure(_, message) {
            #expect(message.contains("needs a model"))
        } catch { Issue.record("wrong error: \(error)") }
    }

    // MARK: - text_to_table: model fallback (gate / field injected)

    @Test func toTableWithAModelSkipsTheFastPathEvenForCleanCsv() async throws {
        // Explicit beats implicit (SPEC-Q112/Q95): a row that names a model always calls it,
        // even when the fast path would have worked.
        let gateCalls = T2TCounter()
        let out = try await TextToTableStage.toTable(
            inputs: [textAsset("name,age\nAda,30")], settings: "expected=\"name, age\"",
            model: (gate: { _ in _ = gateCalls.next(); return "no" }, field: { _ in "" }),
            in: blobDir())
        let (_, rows) = try TableTool.readTable(from: #require(out.items.first?.path))
        #expect(gateCalls.value == 1)   // the model WAS consulted
        #expect(rows.isEmpty)           // ...and said "no records", not the fast path's [Ada, 30]
    }

    @Test func toTableModelFallbackExtractsRows() async throws {
        let gateCalls = T2TCounter()
        let out = try await TextToTableStage.toTable(
            inputs: [textAsset("Ada is here.")], settings: "expected=\"name\"",
            model: (gate: { _ in gateCalls.next() <= 1 ? "yes" : "no" }, field: { _ in "Ada" }),
            in: blobDir())
        let (columns, rows) = try TableTool.readTable(from: #require(out.items.first?.path))
        #expect(columns == ["name"])
        #expect(rows.map { $0.map { $0 as? String } } == [["Ada"]])
    }

    @Test func toTableFiresOnPromptForTheModelFallback() async throws {
        let seen = T2TRecorder()
        _ = try await TextToTableStage.toTable(
            inputs: [textAsset("x")], settings: "expected=\"name\"",
            model: (gate: { _ in "no" }, field: { _ in "" }),
            onPrompt: { seen.append($0) }, in: blobDir())
        #expect(seen.values.count == 1)
        #expect(seen.values[0].contains("mention a record"))
    }

    // MARK: - The real seam: the gallery-09/28/30 shape that throws today

    /// `Table to Text format=csv` → `Text to Table` with **no model named** — the shape that
    /// reaches `runModel → resolveModel(display: nil)` and throws `missingInlineValue` before
    /// DA-6. The `Text to Table` row inverts the CSV its own `Table to Text` sibling wrote.
    @Test func realExecutorRunsAModellessTextToTableAfterTableToTextCsv() async throws {
        let blob = FileManager.default.temporaryDirectory.appendingPathComponent("t2t-seam-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: blob, withIntermediateDirectories: true)
        let executor = RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory
                .appendingPathComponent("t2t-ws-\(UUID().uuidString)")),
            flowID: "da6",
            blobDirectory: blob,
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "none", kind: .llm) },
            installedModelIDs: [],
            catalog: [])

        let table = try TableTool.tableItem(
            columns: ["merchant", "total"],
            rows: [["Cafe" as Any?, 12 as Any?], ["Books" as Any?, 30 as Any?]],
            in: blob)
        let csv = try await executor.execute(
            path: "2", row: Row(task: "Table to Text", model: nil, settings: "format=csv"),
            inputs: [Asset(items: [table])], transcript: nil, context: nil, usedFlowContent: nil)
        let out = try await executor.execute(
            path: "3", row: Row(task: "Text to Table", model: nil, settings: "expected=\"merchant, total\""),
            inputs: [csv], transcript: nil, context: nil, usedFlowContent: nil)
        #expect(out.items.first?.kind == .table)
        let (columns, rows) = try TableTool.readTable(from: #require(out.items.first?.path))
        #expect(columns == ["merchant", "total"])
        #expect(rows.count == 2)
        #expect(TableTool.number(rows[0].last ?? nil) == 12)   // number survives the round-trip
    }

    // MARK: - SPEC-Q214: a model-less Text to Table row shows no "needs a model" warning

    @MainActor @Test func modellessTextToTableRowHasNoWarning() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
            .domains.flatMap { $0.allModels }
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        let warned = "Row 2 needs a model — pick one that runs on this Mac."

        let doc = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [
            Row(id: UUID(), task: "Read Text", settings: "in.csv"),
            Row(id: UUID(), task: "Text to Table", model: nil, settings: "expected=\"a, b\"", refs: []),
        ])
        let editor = FlowEditorModel(name: "q214", document: doc,
                                     workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
        editor.modelCatalog = catalog
        editor.claimableModelIDs = claimable
        #expect(editor.warning(for: doc.rows[1].id) != warned)

        // A Text to Table row that names a bogus model still warns.
        let doc2 = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [
            Row(id: UUID(), task: "Read Text", settings: "in.csv"),
            Row(id: UUID(), task: "Text to Table", model: "Nonexistent Model 99B", settings: "expected=\"a, b\"", refs: []),
        ])
        let editor2 = FlowEditorModel(name: "q214b", document: doc2,
                                      workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory))
        editor2.modelCatalog = catalog
        editor2.claimableModelIDs = claimable
        #expect(editor2.warning(for: doc2.rows[1].id) == warned)
    }
}

private final class T2TCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func next() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
    var value: Int { lock.lock(); defer { lock.unlock() }; return n }
}

private final class T2TRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _values: [String] = []
    func append(_ s: String) { lock.lock(); defer { lock.unlock() }; _values.append(s) }
    var values: [String] { lock.lock(); defer { lock.unlock() }; return _values }
}
