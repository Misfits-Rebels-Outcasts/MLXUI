import Testing
import Foundation
@testable import MLXUI

/// DA-3a (`RSI/DelegateDeciderBacklog.md`) — Extract Structured's schema mode, ported from
/// `catflow-mlx/tests/test_engines_llm_extract_structured.py` (the ~20 pure + mock-driven
/// tests) and `test_engines_real_engine_executor_extract_structured.py` (the executor
/// branch) at commit `3600a4a`. All mock-driven — no real model. The Python monkeypatches
/// `decide` / `_mlx_complete_line`; here the two model calls are injected closures
/// (`gate` / `field`), the same split `RerankStageTests` uses.
///
/// `test_more_records_prompt_first_call_unaffected_by_the_addendum` ports as an **exact
/// string** assertion — that is what pins the SPEC-Q35/Q90 wording against a future
/// paraphrase (the one defect no other test in this phase can catch).
struct CatFlowExtractStructuredTests {

    private func textAsset(_ values: String...) -> Asset {
        Asset(items: values.map { Item(kind: .text, value: $0, path: nil, sourceText: nil) })
    }

    private func blobDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("es-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - _parse_schema

    @Test func parseSchemaSplitsOnCommasAndStrips() throws {
        #expect(try ExtractStructuredStage.parseSchema("name, date, amount") == ["name", "date", "amount"])
    }

    @Test func parseSchemaNoSettingsRaises() {
        expectStageFailure(containing: "needs a schema") {
            _ = try ExtractStructuredStage.parseSchema(nil)
        }
    }

    @Test func parseSchemaBlankSettingsRaises() {
        expectStageFailure(containing: "needs a schema") {
            _ = try ExtractStructuredStage.parseSchema("   ")
        }
    }

    /// House style is `#expect(throws: SomeError.self)`; this keeps that spirit while also
    /// pinning the message (the ported Python tests use `pytest.raises(match=…)`).
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

    @Test func parseSchemaDropsEmptyFieldsFromTrailingCommas() throws {
        #expect(try ExtractStructuredStage.parseSchema("name, date,") == ["name", "date"])
    }

    @Test func parseSchemaStripsWrappingQuotes() throws {
        // The .cat grammar's own example — settings captured verbatim, quotes kept, so
        // without unquoting the leading/trailing `"` leak into the first/last field name.
        #expect(try ExtractStructuredStage.parseSchema("\"clause_type, risk_level, obligation, deadline\"")
                == ["clause_type", "risk_level", "obligation", "deadline"])
    }

    // MARK: - prompt builders

    @Test func moreRecordsPromptFirstCallSaysARecord() {
        let prompt = ExtractStructuredStage.moreRecordsPrompt(source: "some text", columns: ["name", "date"], rows: [])
        #expect(prompt.contains("mention a record"))
        #expect(prompt.contains("(none yet)"))
    }

    @Test func moreRecordsPromptLaterCallExcludesAlreadyExtractedExplicitly() {
        // SPEC-Q90 addendum: no longer "does the text mention another record" (read as "is
        // this data present at all" — trivially true once quoted from the text) — now
        // explicit that already-listed records don't count.
        let prompt = ExtractStructuredStage.moreRecordsPrompt(source: "some text", columns: ["name"], rows: [["Ada"]])
        #expect(prompt.contains("ALREADY been extracted -- do not count them again"))
        #expect(prompt.contains("distinct from all records already listed above"))
        #expect(prompt.contains("name=Ada"))
    }

    @Test func moreRecordsPromptFirstCallUnaffectedByTheAddendum() {
        // The addendum only changes the later-call (non-empty rows) branch — the first call
        // must render byte-identical to the original SPEC-Q35-tuned wording. This exact
        // string is the golden that pins the wording against a paraphrase.
        let prompt = ExtractStructuredStage.moreRecordsPrompt(source: "some text", columns: ["name", "date"], rows: [])
        #expect(prompt == """
        Text:
        some text

        Does the text mention a record with values for: name, date? Answer yes or no.

        (Records already extracted: (none yet))
        """)
    }

    @Test func fieldPromptIncludesFieldNameAndSource() {
        let prompt = ExtractStructuredStage.fieldPrompt(source: "the source text", columns: ["name", "date"], rows: [], field: "date")
        #expect(prompt.contains("\"date\" value"))
        #expect(prompt.contains("the source text"))
        #expect(prompt.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("date:"))
    }

    @Test func fieldPromptIncludesAlreadyExtractedRows() {
        let prompt = ExtractStructuredStage.fieldPrompt(source: "the source text", columns: ["name"], rows: [["Ada"]], field: "name")
        #expect(prompt.contains("name=Ada"))
    }

    /// The full later-call prompt, as a golden — the SPEC-Q90 wording, character for character.
    @Test func moreRecordsPromptLaterCallGolden() {
        let prompt = ExtractStructuredStage.moreRecordsPrompt(source: "some text", columns: ["name", "date"], rows: [["Ada", "2024"]])
        #expect(prompt == """
        Text:
        some text

        You are extracting every record with values for: name, date.
        Records already listed below have ALREADY been extracted -- do not count them again.
        Records already extracted:
        name=Ada, date=2024

        Is there one more record in the text, distinct from all records already listed above? Answer no if every matching record in the text is already listed above.
        Answer yes or no.
        """)
    }

    /// The field prompt, as a golden — the SPEC-Q35 wording, character for character.
    @Test func fieldPromptGolden() {
        let prompt = ExtractStructuredStage.fieldPrompt(source: "src", columns: ["name"], rows: [], field: "name")
        #expect(prompt == """
        Text:
        src

        Records already extracted:
        (none yet)

        What is the "name" value for the next record in the text, if any? Answer with just the value, nothing else. If there is none, answer with an empty string.

        name:
        """)
    }

    // MARK: - _normalize_field_value

    @Test func normalizeFieldValueStripsFieldLabelEcho() {
        #expect(ExtractStructuredStage.normalizeFieldValue("name: Ada Lovelace", field: "name") == "Ada Lovelace")
    }

    @Test func normalizeFieldValueIsCaseInsensitiveAndToleratesSpacing() {
        #expect(ExtractStructuredStage.normalizeFieldValue("  Name :  Ada  ", field: "name") == "Ada")
    }

    @Test func normalizeFieldValuePassesThroughABareValue() {
        #expect(ExtractStructuredStage.normalizeFieldValue("1815-12-10", field: "birth_date") == "1815-12-10")
    }

    @Test(arguments: ["empty string", "N/A", "none", "Not present", "\"empty string\""])
    func normalizeFieldValueCanonicalizesEmptySentinels(_ sentinel: String) {
        #expect(ExtractStructuredStage.normalizeFieldValue(sentinel, field: "name") == "")
    }

    // MARK: - extract (gate / field injected)

    @Test func extractNoInputRaises() async {
        do {
            _ = try await ExtractStructuredStage.extract(
                inputs: [], settings: "name, date",
                gate: { _ in "no" }, field: { _ in "" }, in: blobDir())
            Issue.record("expected a throw")
        } catch let FlowError.stageFailure(_, message) {
            #expect(message.contains("input text asset"), "message was: \(message)")
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    @Test func extractZeroRecords() async throws {
        let out = try await ExtractStructuredStage.extract(
            inputs: [textAsset("nothing relevant here")], settings: "name, date",
            gate: { _ in "no" },
            field: { _ in Issue.record("no fields should be requested for zero records"); return "" },
            in: blobDir())
        let item = out.items[0]
        #expect(item.kind == .table)
        let (columns, rows) = try TableTool.readTable(from: #require(item.path))
        #expect(columns == ["name", "date"])
        #expect(rows.isEmpty)
    }

    @Test func extractExtractsTwoRowsThenStops() async throws {
        let gateCalls = ExtractStructuredTestCounter()
        let fieldCalls = ExtractStructuredTestCounter()
        let out = try await ExtractStructuredStage.extract(
            inputs: [textAsset("Ada joined 2024-01-01. Grace joined 2024-02-02.")], settings: "name, date",
            gate: { _ in gateCalls.next() <= 2 ? "yes" : "no" },   // yes, yes, no
            field: { _ in
                let n = fieldCalls.next() - 1
                let values = [["Ada", "2024-01-01"], ["Grace", "2024-02-02"]]
                return values[n / 2][n % 2]
            },
            in: blobDir())
        let (columns, rows) = try TableTool.readTable(from: #require(out.items[0].path))
        #expect(columns == ["name", "date"])
        #expect(rows.count == 2)
        #expect(rows[0].map { $0 as? String } == ["Ada", "2024-01-01"])
        #expect(rows[1].map { $0 as? String } == ["Grace", "2024-02-02"])
        #expect(gateCalls.value == 3)
    }

    @Test func extractSecondGateCallExcludesAlreadyExtracted() async throws {
        // Wired end-to-end through the loop, not just the standalone prompt-builder test —
        // the second "is there another record" call must actually carry the SPEC-Q90 wording.
        let seen = ExtractStructuredTestRecorder()
        let gateCalls = ExtractStructuredTestCounter()
        _ = try await ExtractStructuredStage.extract(
            inputs: [textAsset("Ada.")], settings: "name",
            gate: { prompt in seen.append(prompt); return gateCalls.next() == 1 ? "yes" : "no" },
            field: { _ in "Ada" },
            in: blobDir())
        #expect(seen.values.count == 2)
        #expect(seen.values[0].contains("Does the text mention a record"))
        #expect(seen.values[1].contains("ALREADY been extracted -- do not count them again"))
        #expect(seen.values[1].contains("name=Ada"))
    }

    @Test func extractCapsAtMaxRows() async throws {
        let out = try await ExtractStructuredStage.extract(
            inputs: [textAsset("endless records")], settings: "name",
            gate: { _ in "yes" }, field: { _ in "x" }, in: blobDir())
        let (_, rows) = try TableTool.readTable(from: #require(out.items[0].path))
        #expect(rows.count == ExtractStructuredStage.maxRows)
    }

    @Test func extractFiresOnPromptForEachSubCall() async throws {
        let seen = ExtractStructuredTestRecorder()
        _ = try await ExtractStructuredStage.extract(
            inputs: [textAsset("x")], settings: "name",
            gate: { _ in "no" }, field: { _ in "" },
            onPrompt: { seen.append($0) }, in: blobDir())
        #expect(seen.values.count == 1)   // just the one "more records?" check before stopping
        #expect(seen.values[0].contains("mention a record"))
    }

    // MARK: - The real seam: RealExecutor.execute builds the Extract Structured branch

    /// Replaces `test_engines_real_engine_executor_extract_structured.py` (whose subject —
    /// `_capture_frame` / `last_frame` — has no Swift equivalent). The Swift concern: the
    /// `engines.llm.extract_structured` branch resolves the model, runs the loop against the
    /// registry stage, and emits a `.table` **that `Table to Text` accepts** (the done-when).
    @Test func realExecutorRunsExtractStructuredAndTableToTextAcceptsTheResult() async throws {
        let entry = makeEntry(id: "es-model", modelType: .llm, source: .mlx,
                              hfModelId: "mlx-community/Test-Extract-4bit")
        let ws = FlowWorkspace(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("es-exec-\(UUID().uuidString)"))
        let blob = FileManager.default.temporaryDirectory.appendingPathComponent("es-blob-\(UUID().uuidString)")
        let executor = RealExecutor(
            workspace: ws, flowID: "da3a",
            blobDirectory: blob,
            makeModelStage: { _, _ in StubExtractStage() },
            installedModelIDs: [entry.id],
            catalog: [entry])

        let row = Row(task: "Extract Structured", model: entry.hfModelId, settings: "name, city")
        let result = try await executor.execute(
            path: "1", row: row, inputs: [textAsset("Ada lives in London. Grace lives in Baltimore.")],
            transcript: nil, context: nil, usedFlowContent: nil)
        #expect(result.items.first?.kind == .table)

        // Table to Text accepts it — chained as the next row, exactly as a real flow would.
        let text = try await executor.execute(
            path: "2", row: Row(task: "Table to Text", model: nil, settings: nil),
            inputs: [result], transcript: nil, context: nil, usedFlowContent: nil)
        #expect(text.items.first?.kind == .text)
        #expect(text.items.first?.value?.contains("name") == true)
    }

    // MARK: - DA-5: temp=0 is enforced, not just claimed

    /// The reviewer's key check: `StageConfig.temperature` **defaults to 0.7** — a `0` default
    /// would silently make every chat / summary / decider row greedy — and an explicit value
    /// sticks and is part of the config's `Hashable` identity (so `EngineCache` treats a
    /// temp-0 stage as distinct from a temp-0.7 one, per `differentConfigIsADistinctEngine`).
    @Test func stageConfigTemperatureDefaultsTo07AndIsPartOfIdentity() {
        #expect(StageConfig().temperature == 0.7)
        #expect(StageConfig.default.temperature == 0.7)
        #expect(StageConfig(temperature: 0).temperature == 0)
        #expect(StageConfig(temperature: 0) != StageConfig(temperature: 0.7))
        #expect(StageConfig(temperature: 0).hashValue != StageConfig(temperature: 0.7).hashValue)
    }

    /// DA-5: the `engines.llm.extract_structured` branch must build its stage with
    /// `temperature: 0` — the Python pins `temp=0.0` on both model calls, and a sampled gate
    /// makes the loop's termination non-deterministic. Captures the `StageConfig` the branch
    /// hands `makeModelStage`.
    @Test func extractStructuredBranchRequestsGreedyDecoding() async throws {
        let entry = makeEntry(id: "es-temp", modelType: .llm, source: .mlx,
                              hfModelId: "mlx-community/Test-Extract-Temp-4bit")
        let seen = ExtractStructuredTestConfigBox()
        let executor = RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory
                .appendingPathComponent("es-temp-\(UUID().uuidString)")),
            flowID: "da5",
            blobDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent("es-temp-blob-\(UUID().uuidString)"),
            makeModelStage: { _, config in seen.set(config); return StubExtractStage() },
            installedModelIDs: [entry.id],
            catalog: [entry])
        _ = try await executor.execute(
            path: "1", row: Row(task: "Extract Structured", model: entry.hfModelId, settings: "name"),
            inputs: [textAsset("Ada.")], transcript: nil, context: nil, usedFlowContent: nil)
        let config = try #require(seen.value)
        #expect(config.temperature == 0)
        #expect(config.maxTokens == ExtractStructuredStage.maxFieldTokens)   // the 32-token cap is untouched
    }
}

private final class ExtractStructuredTestConfigBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: StageConfig?
    func set(_ c: StageConfig) { lock.lock(); defer { lock.unlock() }; _value = c }
    var value: StageConfig? { lock.lock(); defer { lock.unlock() }; return _value }
}

/// A stub LLM stage for the executor seam: answers the yes/no gate ("yes" for the first two
/// checks, then "no") and returns a canned value for each field prompt — enough to prove the
/// branch assembles a real table, without weights.
private nonisolated struct StubExtractStage: PipelineStage {
    let id = "stub.extract.llm"
    let name = "Stub Extract LLM"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    /// One stage instance is built per `runModel` and reused for every gate + field call.
    let gateCount = ExtractStructuredTestCounter()

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        guard case .text(let prompt) = input else { throw StageError.unsupportedModel(id: id, kind: .llm) }
        if prompt.contains("Answer yes or no") || prompt.contains("Answer with exactly one") {
            return .text(gateCount.next() <= 2 ? "yes" : "no")
        }
        if prompt.contains("\"name\" value") { return .text("Ada") }
        if prompt.contains("\"city\" value") { return .text("London") }
        return .text("")
    }
}

/// Actor-free helpers — the injected closures are `@Sendable`, so a plain `var` capture races.
private final class ExtractStructuredTestCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    /// Post-increment: returns the count *including* this call (1 on the first call).
    func next() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
    var value: Int { lock.lock(); defer { lock.unlock() }; return n }
}

private final class ExtractStructuredTestRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _values: [String] = []
    func append(_ s: String) { lock.lock(); defer { lock.unlock() }; _values.append(s) }
    var values: [String] { lock.lock(); defer { lock.unlock() }; return _values }
}
