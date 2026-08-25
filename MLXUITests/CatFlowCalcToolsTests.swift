import Testing
import Foundation
import CoreGraphics
@testable import MLXUI

/// CFM-R12-7 group a — the data tools: `Calculate` pinned against the Python's 44-case calc
/// golden, `Compare` (deterministic branch), `Range` (literal sequence), `Chart` (table →
/// image).
struct CatFlowCalcToolsTests {

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-calc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    /// `calc_golden.json` — the Python's 44 cases, byte-for-byte.
    @Test func calculateMatchesThePythonGolden() throws {
        let filePath = #filePath
        let url = URL(fileURLWithPath: filePath)
        var dir = url.deletingLastPathComponent()
        while dir.lastPathComponent != "MLXUITests" { dir = dir.deletingLastPathComponent() }
        let goldenURL = dir.deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow/tools/calc_golden.json")
        let cases = try JSONDecoder().decode([CalcGolden].self, from: Data(contentsOf: goldenURL))
        var mismatches: [String] = []
        for c in cases {
            let got: String
            do {
                got = CalcEngine.formatNumber(try CalcEngine.evaluate(c.expr))
            } catch CalcEngine.CalcError.message(let message) {
                got = message
            } catch {
                got = "unexpected: \(error)"
            }
            if got != c.expected { mismatches.append("'\(c.expr)': got '\(got)' expected '\(c.expected)'") }
        }
        #expect(mismatches.isEmpty, "calc golden mismatches: \(mismatches.joined(separator: "; "))")
    }

    private struct CalcGolden: Decodable {
        let expr: String
        let expected: String
    }

    /// CFM-R12-FIX-8 — the four parity cases the golden doesn't cover.
    @Test func fix8ParityCases() {
        func run(_ expr: String) -> String {
            do { return CalcEngine.formatNumber(try CalcEngine.evaluate(expr)) }
            catch CalcEngine.CalcError.message(let m) { return m }
            catch { return "unexpected: \(error)" }
        }
        #expect(run("-7 % 3") == "2")                       // a: floored modulo
        #expect(run("2 ^ 100") == "1267650600228229401496703205376")   // b: no crash, exact
        #expect(run("round(2.675, 2)") == "2.67")           // d: banker's on the exact value
        #expect(run("round(0.1234565, 6)") == "0.123456")   // c: half-even
        #expect(run("0 ^ -1") == "division by zero")
    }

    @Test func calculateToolReturnsErrorsAsText() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let tool = CalculateTool(workspace: ws, flowID: "f", settings: "")
        let out = try await tool.run(Asset(items: [Item(kind: .text, value: "1 / 0", path: nil, sourceText: nil)])) { _ in }
        #expect(out.items.first?.value == "division by zero")
    }

    @Test func compareFiresTheDeclaredTag() throws {
        // The gallery quotes the comparator ("Compare  \"> 50\"; tags: escalate, normal"),
        // so the whole criterion is the first bare token.
        let row = Row(id: UUID(), task: "Compare", settings: "\"> 100\"", clause: .decide(edges: [
            ClauseEdge(tag: "over", target: .done),
            ClauseEdge(tag: "under", target: .done),
        ]))
        let input = Asset(items: [Item(kind: .text, value: "150", path: nil, sourceText: nil)])
        let result = try CompareTool.run(row: row, inputs: [input], path: "1")
        #expect(result.firedTag == "over")
        let under = try CompareTool.run(row: row, inputs: [Asset(items: [Item(kind: .text, value: "50", path: nil, sourceText: nil)])], path: "1")
        #expect(under.firedTag == "under")
    }

    @Test func rangeProducesTheLiteralSequence() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let tool = RangeTool(workspace: ws, flowID: "f", settings: "0..3")
        let out = try await tool.run(Asset(items: [])) { _ in }
        #expect(out.items.map { $0.value } == ["0", "1", "2", "3"])
        let stepped = RangeTool(workspace: ws, flowID: "f", settings: "1..7; step=2")
        let out2 = try await stepped.run(Asset(items: [])) { _ in }
        #expect(out2.items.map { $0.value } == ["1", "3", "5", "7"])
    }

    @Test func chartRendersAnImageFromTheTable() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let table = flowDir.appendingPathComponent("sales.table")
        try TableTool.writeTable(columns: ["month", "sales"],
                                 rows: [["Jan", "10"], ["Feb", "25"], ["Mar", "18"]],
                                 to: table)
        let tool = ChartTool(workspace: ws, flowID: "f", settings: "x=month; y=sales; kind=bar")
        let out = try await tool.run(Asset(items: [Item(kind: .table, value: nil, path: table, sourceText: nil)])) { _ in }
        let item = try out.items.first ?? { throw FlowError.stageFailure(row: "Chart", message: "no item") }()
        #expect(item.kind == .image)
        let path = try item.path ?? { throw FlowError.stageFailure(row: "Chart", message: "no path") }()
        let data = try Data(contentsOf: path)
        let cg = ImageLoader.decodedCGImage(from: data)
        #expect(cg != nil)
        #expect(cg?.width ?? 0 > 100)
    }
}
