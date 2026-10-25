import Foundation
import PDFKit
import Testing
@testable import MLXUI

/// WW4-2-1 -- proves the SPEC-Q237 parity harness itself, with mock engines. These tests make no claim about whether the
/// Q35 loop and a grammar one-shot agree on a real model; that is WW4-2-2's experiment.
struct ParityQ237HarnessTests {
    private static let columns3 = ["name", "date", "amount"]

    private func testCase(_ id: String = "t", settings: String = "name, date, amount", text: String? = "some text") -> ParityCase {
        ParityCase(id: id,
                   provenance: ParityProvenance(origin: "test", derivation: "unit test"),
                   settings: settings,
                   columns: settings.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) },
                   inputStatus: text == nil ? "needs-upstream-model" : "text",
                   inputText: text,
                   expectedShape: "")
    }

    private func table(_ columns: [String], _ rows: [[Any?]]) -> ParityTable { ParityTable(columns: columns, rows: rows) }

    // MARK: - canonical serialization

    @Test func canonicalJSONIsCompactAndOrdered() {
        let t = table(["a", "b"], [["x", nil], [5, "y/z"]])
        #expect(t.canonicalJSON == #"{"columns":["a","b"],"rows":[["x",null],[5,"y/z"]]}"#)
    }

    @Test func canonicalisationUnifiesUnicodeAndIntegralDoubles() {
        let composed = "caf\u{00E9}", decomposed = "cafe\u{0301}"
        #expect(table(["c"], [[composed]]) == table(["c"], [[decomposed]]))
        #expect(table(["c"], [[composed]]).canonicalJSON == table(["c"], [[decomposed]]).canonicalJSON)
        #expect(table(["n"], [[5.0]]) == table(["n"], [[5]]))
        #expect(table(["n"], [[5.5]]).canonicalJSON == #"{"columns":["n"],"rows":[[5.5]]}"#)
    }

    @Test func aStringAndANumberStayDifferent() {
        #expect(table(["n"], [["5"]]) != table(["n"], [[5]]))
    }

    @Test func canonicalJSONEscapesCommasQuotesAndNewlines() throws {
        let value = "Smith, \"Jones\" & Co.\nline two"
        let json = table(["c"], [[value]]).canonicalJSON
        let decoded = try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        let rows = try #require(decoded?["rows"] as? [[String]])
        #expect(rows == [[value]])
    }

    @Test func columnAndRowOrderAreSignificant() {
        #expect(table(["a", "b"], [["1", "2"]]) != table(["b", "a"], [["1", "2"]]))
        #expect(table(["a"], [["1"], ["2"]]).canonicalJSON != table(["a"], [["2"], ["1"]]).canonicalJSON)
    }

    @Test func aTableItemReadsBackThroughTheShippedTablePath() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("parity-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let item = try TableTool.tableItem(columns: ["a", "b"], rows: [["x", nil], ["y", "z"]], in: dir)
        #expect(try ParityTable(item: item) == table(["a", "b"], [["x", nil], ["y", "z"]]))
    }

    // MARK: - comparison

    @Test func identicalTablesCompareIdentical() {
        let t = table(["a"], [["1"]])
        let result = ParityComparison.compare(t, t)
        #expect(result.identical)
        #expect(result.first == nil)
    }

    @Test func aDifferingCellIsReportedWithItsPosition() {
        let a = table(["name", "date"], [["Ada", "1"], ["Grace", "2"], ["Alan", "3"]])
        let b = table(["name", "date"], [["Ada", "1"], ["Grace", "2"], ["Alan", "9"]])
        let result = ParityComparison.compare(a, b)
        #expect(!result.identical)
        #expect(result.first == .cell(row: 2, column: 1, name: "date", a: .string("3"), b: .string("9")))
    }

    @Test func theFirstDifferingCellIsRowMajor() {
        let a = table(["x", "y"], [["1", "2"], ["3", "4"]])
        let b = table(["x", "y"], [["1", "2"], ["0", "0"]])
        #expect(ParityComparison.compare(a, b).first == .cell(row: 1, column: 0, name: "x", a: .string("3"), b: .string("0")))
        let c = table(["x", "y"], [["1", "0"], ["0", "4"]])
        #expect(ParityComparison.compare(a, c).first == .cell(row: 0, column: 1, name: "y", a: .string("2"), b: .string("0")))
    }

    @Test func columnsAreComparedBeforeCellsAndRowCountAfterThem() {
        #expect(ParityComparison.compare(table(["a"], [["1"]]), table(["b"], [["2"]])).first == .columns(a: ["a"], b: ["b"]))
        let short = table(["a"], [["1"]])
        let long = table(["a"], [["1"], ["2"]])
        #expect(ParityComparison.compare(short, long).first == .rowCount(a: 1, b: 2))
        // A cell difference in the shared rows wins over a row-count difference.
        let changed = table(["a"], [["9"], ["2"]])
        #expect(ParityComparison.compare(short, changed).first == .cell(row: 0, column: 0, name: "a", a: .string("1"), b: .string("9")))
    }

    @Test func aMissingCellInARaggedRowIsADifference() {
        let a = table(["a", "b"], [["1", "2"]])
        let b = table(["a", "b"], [["1"]])
        #expect(ParityComparison.compare(a, b).first == .cell(row: 0, column: 1, name: "b", a: .string("2"), b: nil))
    }

    // MARK: - the two paths against the mock

    private let oneRecord = [["Ada Lovelace", "2026-03-14", "250 dollars"]]

    @Test func pathACountsCallsAndTokens() async throws {
        let engine = ScriptedParityEngine(columns: Self.columns3, records: oneRecord)
        let c = testCase(text: "Invoice from Ada Lovelace dated 2026-03-14 for 250 dollars.")
        let result = await ParityHarness.runQ35Loop(c, engine: engine)
        #expect(result.failure == nil)
        #expect(result.table == table(Self.columns3, oneRecord))

        // One record = gate + 3 field calls, then the closing gate: 5 calls.
        let source = try #require(c.inputText)
        let first = ExtractStructuredStage.moreRecordsPrompt(source: source, columns: Self.columns3, rows: [])
        let fields = Self.columns3.map { ExtractStructuredStage.fieldPrompt(source: source, columns: Self.columns3, rows: [], field: $0) }
        let last = ExtractStructuredStage.moreRecordsPrompt(source: source, columns: Self.columns3, rows: oneRecord)
        let prompts = [first] + fields + [last]
        let replies = ["yes"] + oneRecord[0] + ["no"]
        #expect(result.accounting.modelCalls == 5)
        #expect(result.accounting.promptTokens == prompts.map(ParityReply.approximateTokens).reduce(0, +))
        #expect(result.accounting.completionTokens == replies.map(ParityReply.approximateTokens).reduce(0, +))
        #expect(result.accounting.totalTokens == result.accounting.promptTokens + result.accounting.completionTokens)
    }

    @Test func pathBIsOneCall() async throws {
        let engine = ScriptedParityEngine(columns: Self.columns3, records: oneRecord)
        let c = testCase(text: "Invoice from Ada Lovelace dated 2026-03-14 for 250 dollars.")
        let result = await ParityHarness.runGrammarOneShot(c, engine: engine)
        #expect(result.failure == nil)
        #expect(result.table == table(Self.columns3, oneRecord))
        let prompt = ParityGrammarOneShot.prompt(source: try #require(c.inputText), columns: Self.columns3)
        #expect(result.accounting.modelCalls == 1)
        #expect(result.accounting.promptTokens == ParityReply.approximateTokens(prompt))
        #expect(result.accounting.completionTokens > 0)
    }

    @Test func zeroRecordsCostsOneGateCallOnPathA() async {
        let engine = ScriptedParityEngine(columns: Self.columns3, records: [])
        let result = await ParityHarness.runQ35Loop(testCase(), engine: engine)
        #expect(result.table == table(Self.columns3, []))
        #expect(result.accounting.modelCalls == 1)
    }

    @Test func wallTimeCoversTheModelCalls() async {
        let delay = Duration.milliseconds(30)
        let c = testCase()
        let a = await ParityHarness.runQ35Loop(c, engine: ScriptedParityEngine(columns: Self.columns3, records: oneRecord, delay: delay))
        let b = await ParityHarness.runGrammarOneShot(c, engine: ScriptedParityEngine(columns: Self.columns3, records: oneRecord, delay: delay))
        #expect(a.accounting.wallSeconds >= 5 * 0.03 * 0.9)
        #expect(b.accounting.wallSeconds >= 0.03 * 0.9)
        #expect(a.accounting.wallSeconds > b.accounting.wallSeconds)
    }

    /// The five hand-written edge cases with the records an engine that is right would give.
    private static let handWritten: [(id: String, settings: String, records: [[String]])] = [
        ("h-zero-records", "name, date, amount", []),
        ("h-one-record", "name, date, amount", [["Ada Lovelace", "2026-03-14", "250 dollars"]]),
        ("h-missing-field", "name, date, amount", [["Ada Lovelace", "2026-03-14", "250 dollars"], ["Grace Hopper", "", "410 dollars"]]),
        ("h-comma-quote", "company, item, total", [["Smith, Jones & Co.", "12\" copper pipes", "40.50 dollars"]]),
        ("h-many-records", "name, date, amount", (0..<7).map { ["Person \($0)", "2026-03-0\($0 + 1)", "\(100 + $0 * 10)"] }),
    ]

    @Test func mockPathsAgreeOnTheHandWrittenEdgeCases() async throws {
        let cases = try ParityCorpus.load()
        for edge in Self.handWritten {
            let c = try #require(cases.first { $0.id == edge.id })
            let columns = try ExtractStructuredStage.parseSchema(c.settings)
            let report = await ParityHarness.runCase(c) { _, _ in
                ScriptedParityEngine(columns: columns, records: edge.records)
            }
            #expect(report.identical == true, "\(edge.id): \(report.comparison?.first.map(String.init(describing:)) ?? "no comparison")")
            #expect(report.a?.table?.rows.count == edge.records.count, "\(edge.id)")
            #expect(report.a?.accounting.modelCalls == 1 + edge.records.count * (columns.count + 1) , "\(edge.id)")
            #expect(report.b?.accounting.modelCalls == 1, "\(edge.id)")
        }
    }

    @Test func aDeliberatelyDifferingCellIsDetectedAndNamed() async {
        var oneShot = oneRecord + [["Grace Hopper", "2026-04-01", "410 dollars"]]
        let records = oneShot
        oneShot[1][1] = "April 1"
        let report = await ParityHarness.runCase(testCase()) { _, _ in
            ScriptedParityEngine(columns: Self.columns3, records: records, oneShotRecords: oneShot)
        }
        #expect(report.identical == false)
        #expect(report.comparison?.first == .cell(row: 1, column: 1, name: "date", a: .string("2026-04-01"), b: .string("April 1")))
        #expect(report.a?.accounting.modelCalls == 9)   // 3 gates (yes, yes, no) + 2 x 3 field calls
        #expect(report.b?.accounting.modelCalls == 1)
    }

    @Test func aMissingRecordOnOnePathIsARowCountDifference() async {
        let records = oneRecord + [["Grace Hopper", "2026-04-01", "410 dollars"]]
        let report = await ParityHarness.runCase(testCase()) { _, _ in
            ScriptedParityEngine(columns: Self.columns3, records: records, oneShotRecords: [records[0]])
        }
        #expect(report.comparison?.first == .rowCount(a: 2, b: 1))
    }

    @Test func aMalformedOneShotReplyIsAFailedPathNotAThrow() async {
        let report = await ParityHarness.runCase(testCase()) { _, _ in
            FixedReplyParityEngine(structuredReply: "Sure! Here you go", columns: Self.columns3, records: oneRecord)
        }
        #expect(report.identical == false)
        #expect(report.a?.table == table(Self.columns3, oneRecord))
        #expect(report.b?.table == nil)
        guard case .pathFailed(let path, let reason)? = report.comparison?.first else {
            Issue.record("expected a failed path, got \(String(describing: report.comparison?.first))")
            return
        }
        #expect(path == .grammarOneShot)
        #expect(reason.contains("not a JSON object holding a records array"))
    }

    @Test func anEngineThatCannotBeBuiltIsAFailedPath() async {
        struct Unavailable: Error {}
        let report = await ParityHarness.runCase(testCase()) { _, path in
            if path == .grammarOneShot { throw Unavailable() }
            return ScriptedParityEngine(columns: Self.columns3, records: [])
        }
        #expect(report.identical == false)
        #expect(report.b?.accounting.modelCalls == 0)
    }

    @Test func aCaseWithoutInputTextIsSkippedAndBuildsNoEngine() async {
        let report = await ParityHarness.runCase(testCase(text: nil)) { _, _ in
            Issue.record("no engine should be built for a skipped case")
            return ScriptedParityEngine(columns: Self.columns3, records: [])
        }
        #expect(report.skipped != nil)
        #expect(report.identical == nil)
        #expect(ParityHarness.summary([report]).contains("skipped"))
    }

    @Test func theEngineIsBuiltOncePerPath() async {
        actor Seen { var paths: [ParityPath] = []; func add(_ p: ParityPath) { paths.append(p) } }
        let seen = Seen()
        _ = await ParityHarness.runCase(testCase()) { _, path in
            await seen.add(path)
            return ScriptedParityEngine(columns: Self.columns3, records: [])
        }
        #expect(await seen.paths == [.q35Loop, .grammarOneShot])
    }

    @Test func theSummaryNamesTheVerdictTheFirstDifferenceAndTheCost() async {
        let ok = await ParityHarness.runCase(testCase("ok")) { _, _ in ScriptedParityEngine(columns: Self.columns3, records: []) }
        let bad = await ParityHarness.runCase(testCase("bad")) { _, _ in
            ScriptedParityEngine(columns: Self.columns3, records: self.oneRecord, oneShotRecords: [])
        }
        let lines = ParityHarness.summary([ok, bad]).split(separator: "\n").map(String.init)
        #expect(lines[0].hasPrefix("ok: identical | A 1 calls"))
        #expect(lines[1].contains("bad: DIFFERS, first: row count differs: A 1 vs B 0"))
        #expect(lines[1].contains("B 1 calls"))
    }

    // MARK: - the grammar one-shot's schema and decoder

    @Test func theSchemaListsEveryColumnInOrderAndAllowsNothingElse() throws {
        let schema = ParityGrammarOneShot.schema(columns: ["name", "we\"ird"])
        let root = try #require(try JSONSerialization.jsonObject(with: Data(schema.utf8)) as? [String: Any])
        #expect(root["type"] as? String == "object")
        #expect(root["required"] as? [String] == ["records"])
        #expect(root["additionalProperties"] as? Bool == false)
        let records = try #require((root["properties"] as? [String: Any])?["records"] as? [String: Any])
        #expect(records["type"] as? String == "array")
        let items = try #require(records["items"] as? [String: Any])
        #expect(items["required"] as? [String] == ["name", "we\"ird"])
        #expect(items["additionalProperties"] as? Bool == false)
        let properties = try #require(items["properties"] as? [String: Any])
        #expect(Set(properties.keys) == ["name", "we\"ird"])
        #expect(schema.range(of: "\"name\"")!.lowerBound < schema.range(of: "we\\\"ird")!.lowerBound)
    }

    /// The real grammar run refuses a schema `ServeSchemaCheck` refuses, and that includes a non-object root.
    @Test func theSchemaPassesTheServersOwnSchemaCheck() {
        #expect(ServeSchemaCheck.check(ParityGrammarOneShot.schema(columns: ["a", "b", "c", "d"])) == .ok)
    }

    @Test func theDecoderIsStrict() throws {
        #expect(try ParityGrammarOneShot.decode(reply: #"{"records":[]}"#, columns: ["a"]) == [])
        #expect(try ParityGrammarOneShot.decode(reply: #"{"records":[{"b":"2","a":"1"}]}"#, columns: ["a", "b"]) == [["1", "2"]])
        #expect(throws: ParityError.self) { try ParityGrammarOneShot.decode(reply: #"[{"a":"1"}]"#, columns: ["a"]) }
        #expect(throws: ParityError.self) { try ParityGrammarOneShot.decode(reply: #"{"records":[],"x":1}"#, columns: ["a"]) }
        #expect(throws: ParityError.self) { try ParityGrammarOneShot.decode(reply: #"{"records":[{"a":1}]}"#, columns: ["a"]) }
        #expect(throws: ParityError.self) { try ParityGrammarOneShot.decode(reply: #"{"records":[{"a":"1","z":"2"}]}"#, columns: ["a"]) }
        #expect(throws: ParityError.self) { try ParityGrammarOneShot.decode(reply: #"{"records":[{"a":"1"}]}"#, columns: ["a", "b"]) }
    }

    @Test func aReplyCanCostSeveralCallsAndReportSetupTime() async throws {
        let meter = ParityMeter()
        await meter.record(ParityReply(text: "yes", promptTokens: 10, completionTokens: 1, modelCalls: 2, setupSeconds: 0.5))
        await meter.record(ParityReply(text: "x", promptTokens: 4, completionTokens: 2))
        let accounting = await meter.snapshot(wallSeconds: 1.5)
        #expect(accounting.modelCalls == 3)
        #expect(accounting.totalTokens == 17)
        #expect(accounting.setupSeconds == 0.5)
        #expect(accounting.wallSeconds == 1.5)
    }
}

/// The fixture: every `Extract Structured` case of the gallery and the corpus, accounted for and traceable.
struct ParityQ237CorpusTests {
    private let root = ParityCorpus.repoRoot

    private func cases() throws -> [ParityCase] { try ParityCorpus.load() }

    private func collapsed(_ text: String) -> String {
        text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    @Test func theFixtureLoadsWithUniqueIDsAndCoherentColumns() throws {
        let all = try cases()
        #expect(all.count == 23)
        #expect(Set(all.map(\.id)).count == all.count)
        for c in all {
            #expect(try ExtractStructuredStage.parseSchema(c.settings) == c.columns, "\(c.id)")
            #expect((c.inputText != nil) == (c.inputStatus == "text"), "\(c.id)")
        }
    }

    @Test func theFiveHandWrittenEdgeCasesCoverTheQ35List() throws {
        let hand = try cases().filter { $0.provenance.origin == "hand-written" }
        #expect(Set(hand.compactMap(\.provenance.q35EdgeCase)) == [
            "zero records", "one record", "a missing field", "a field holding a comma or quote", "more than 5 records",
        ])
        #expect(hand.count == 5)
    }

    /// Every gallery flow that has an `Extract Structured` row is in the fixture, and nothing else is claimed to be gallery.
    @Test func everyGalleryExtractStructuredFlowIsInTheFixture() throws {
        let gallery = root.appendingPathComponent("MLXUI/Resources/Gallery")
        let flows = try FileManager.default.contentsOfDirectory(atPath: gallery.path).filter { $0.hasSuffix(".cat") }
        var withRow: Set<String> = []
        for flow in flows {
            let text = try String(contentsOf: gallery.appendingPathComponent(flow), encoding: .utf8)
            if text.contains("Extract Structured") { withRow.insert(flow) }
        }
        let fixture = Set(try cases().filter { $0.provenance.origin == "gallery" }.compactMap(\.provenance.flow))
        #expect(!withRow.isEmpty)
        #expect(fixture == withRow, "gallery flows with an Extract Structured row: \(withRow.sorted()); fixture covers: \(fixture.sorted())")
    }

    /// Tripwire: the synced conformance corpus held no `Extract Structured` case when this fixture was built. If a sync brings
    /// one, the owner's ruling ("every Extract Structured case in the synced conformance corpus") needs it added to the fixture.
    @Test func theSyncedConformanceCorpusHasNoExtractStructuredCase() throws {
        let dir = root.appendingPathComponent("Fixtures/CatFlow/conformance")
        let files = try #require(FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil))
        var hits: [String] = []
        for case let url as URL in files where url.pathExtension == "cat" || url.pathExtension == "json" {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            if text.range(of: "extract structured", options: .caseInsensitive) != nil
                || text.contains("extract_structured") { hits.append(url.lastPathComponent) }
        }
        #expect(hits.isEmpty, "add these to ParityQ237Cases.json: \(hits)")
    }

    @Test func galleryProvenanceMatchesTheBundledFiles() throws {
        let gallery = root.appendingPathComponent("MLXUI/Resources/Gallery")
        for c in try cases() where c.provenance.origin == "gallery" {
            let flow = try #require(c.provenance.flow, "\(c.id)")
            let cat = try String(contentsOf: gallery.appendingPathComponent(flow), encoding: .utf8)
            #expect(collapsed(cat).contains("\"\(c.settings)\""), "\(c.id): the row's schema is not in \(flow)")

            let file = gallery.appendingPathComponent(try #require(c.provenance.inputFile, "\(c.id)"))
            #expect(FileManager.default.fileExists(atPath: file.path), "\(c.id)")
            guard let text = c.inputText else { continue }
            switch file.pathExtension {
            case "txt", "md":
                #expect(text == (try String(contentsOf: file, encoding: .utf8)), "\(c.id)")
            case "pdf":
                let document = try #require(PDFDocument(url: file), "\(c.id)")
                let pages = (0..<document.pageCount).compactMap { document.page(at: $0)?.string }
                // A whole invoice is the page text; a contract section is a trimmed slice of it.
                let whole = pages.joined(separator: "\n\n")
                #expect(c.id.hasPrefix("g06") ? whole.contains(text) : text == whole, "\(c.id)")
            default:
                Issue.record("\(c.id): a text input cannot come from a \(file.pathExtension) file")
            }
        }
    }

    @Test func theCasesWithoutTextAreTheOCRRows() throws {
        let missing = try cases().filter { $0.inputText == nil }
        #expect(missing.count == 5)
        #expect(Set(missing.compactMap(\.provenance.flow)) == ["30-ReceiptsExpense.cat", "55-ReceiptsLedger.cat"])
    }
}
