import Foundation
@testable import MLXUI

// WW4-2-1 -- the SPEC-Q237(c) parity harness: runs one `Extract Structured` case down both paths against a `ParityEngine` and
// compares the two tables. It makes no parity claim; WW4-2-2 plugs real engines in. Test target only: the grammar one-shot
// below is defined here and nothing in the shipping source changes (owner ruling 2026-10-25).

nonisolated enum ParityPath: String, Sendable {
    case q35Loop, grammarOneShot

    var label: String { self == .q35Loop ? "A (Q35 loop)" : "B (grammar one-shot)" }
}

// MARK: - The corpus

nonisolated struct ParityProvenance: Sendable, Decodable {
    var origin: String
    var derivation: String
    var flow: String?
    var row: String?
    var inputFile: String?
    var q35EdgeCase: String?
}

nonisolated struct ParityCase: Sendable, Decodable {
    var id: String
    var provenance: ParityProvenance
    /// The row's schema setting as `ExtractStructuredStage.parseSchema` takes it ("name, date, amount").
    var settings: String
    var columns: [String]
    /// `text`, or `needs-upstream-model` when the row's input is another model's output (OCR) and is not in the fixture.
    var inputStatus: String
    var inputText: String?
    /// A human note on what the text holds. Never a pass criterion.
    var expectedShape: String
}

nonisolated enum ParityCorpus {
    private struct Document: Decodable { var version: Int; var cases: [ParityCase] }

    static let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    static let fixtureURL = repoRoot.appendingPathComponent("MLXUITests/Fixtures/ParityQ237/ParityQ237Cases.json")

    static func load() throws -> [ParityCase] {
        try JSONDecoder().decode(Document.self, from: Data(contentsOf: fixtureURL)).cases
    }
}

// MARK: - Results

nonisolated struct ParityPathResult: Sendable {
    var table: ParityTable?
    var failure: String?
    var accounting: ParityAccounting
}

nonisolated struct ParityCaseReport: Sendable {
    var caseID: String
    /// Why the case was not run (its input is not in the fixture); both results and the comparison are then nil.
    var skipped: String?
    var a: ParityPathResult?
    var b: ParityPathResult?
    var comparison: ParityComparison?

    /// nil when skipped.
    var identical: Bool? { comparison?.identical }
}

// MARK: - The grammar one-shot (path B)

/// What a grammar-constrained single call would be asked and how its reply becomes a table. Implementer's call, pending owner
/// confirmation: the schema is the one SPEC-Q237(a) derives from the row ("an array of objects, one property per column",
/// every property a string, all required, nothing else allowed); the prompt is one plain instruction in the Q35 prompts'
/// short, direct register. Values are taken as the model wrote them: NO `normalizeFieldValue`, because a grammar one-shot
/// has no such step, and a sentinel like "n/a" surviving into a cell is exactly the kind of difference the experiment is for.
nonisolated enum ParityGrammarOneShot {
    /// A JSON Schema document, properties in column order.
    static func schema(columns: [String]) -> String {
        let properties = columns.map { "\(jsonString($0)):{\"type\":\"string\"}" }.joined(separator: ",")
        let required = columns.map(jsonString).joined(separator: ",")
        return "{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":{\(properties)},"
            + "\"required\":[\(required)],\"additionalProperties\":false}}"
    }

    static func prompt(source: String, columns: [String]) -> String {
        let fields = columns.joined(separator: ", ")
        return "Text:\n\(source)\n\n"
            + "Extract every record in the text with values for: \(fields).\n"
            + "Answer with a JSON array with one object per record, using exactly these keys: \(fields).\n"
            + "Use an empty string for a value that is not in the text. If there are no records, answer with []."
    }

    /// The reply as rows in `columns` order. Strict: anything but an array of objects whose values are exactly the columns'
    /// strings is a failure of the path, never repaired.
    static func decode(reply: String, columns: [String]) throws -> [[String]] {
        guard let data = reply.data(using: .utf8),
              let array = (try? JSONSerialization.jsonObject(with: data)) as? [Any] else {
            throw ParityError("reply is not a JSON array")
        }
        return try array.enumerated().map { index, element in
            guard let object = element as? [String: Any] else { throw ParityError("element \(index) is not an object") }
            if let extra = object.keys.first(where: { !columns.contains($0) }) {
                throw ParityError("element \(index) has an unexpected key \(extra)")
            }
            return try columns.map { column in
                guard let value = object[column] else { throw ParityError("element \(index) is missing \(column)") }
                guard let text = value as? String else { throw ParityError("element \(index) \(column) is not a string") }
                return text
            }
        }
    }

    private static func jsonString(_ text: String) -> String {
        guard let data = try? JSONEncoder().encode(text) else { return "\"\"" }
        return String(decoding: data, as: UTF8.self)
    }
}

nonisolated struct ParityError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

// MARK: - The harness

nonisolated enum ParityHarness {
    /// Path A: the shipped `ExtractStructuredStage.extract`, unmodified, with the engine behind its two closures. The table is
    /// read back from the file the stage wrote, so the real `TableTool` path is in the loop.
    static func runQ35Loop(_ testCase: ParityCase, engine: any ParityEngine) async -> ParityPathResult {
        let meter = ParityMeter()
        let start = ContinuousClock.now
        var table: ParityTable?
        var failure: String?
        let blobs = FileManager.default.temporaryDirectory.appendingPathComponent("parity-q237-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: blobs) }
        do {
            let asset = try await ExtractStructuredStage.extract(
                inputs: [textAsset(testCase.inputText ?? "")],
                settings: testCase.settings,
                gate: { prompt in
                    let reply = try await engine.gate(prompt: prompt)
                    await meter.record(reply)
                    return reply.text
                },
                field: { prompt in
                    let reply = try await engine.line(prompt: prompt)
                    await meter.record(reply)
                    return reply.text
                },
                in: blobs)
            guard let item = asset.items.first else { throw ParityError("the stage returned no item") }
            table = try ParityTable(item: item)
        } catch {
            failure = "\(error)"
        }
        return ParityPathResult(table: table, failure: failure,
                                accounting: await meter.snapshot(wallSeconds: seconds(since: start)))
    }

    /// Path B: ONE call asking for the whole table, decoded strictly.
    static func runGrammarOneShot(_ testCase: ParityCase, engine: any ParityEngine) async -> ParityPathResult {
        let meter = ParityMeter()
        let start = ContinuousClock.now
        var table: ParityTable?
        var failure: String?
        do {
            let columns = try ExtractStructuredStage.parseSchema(testCase.settings)
            let reply = try await engine.structured(
                prompt: ParityGrammarOneShot.prompt(source: testCase.inputText ?? "", columns: columns),
                schema: ParityGrammarOneShot.schema(columns: columns))
            await meter.record(reply)
            let rows = try ParityGrammarOneShot.decode(reply: reply.text, columns: columns)
            table = ParityTable(columns: columns, rows: rows.map { $0.map { $0 as Any? } })
        } catch {
            failure = "\(error)"
        }
        return ParityPathResult(table: table, failure: failure,
                                accounting: await meter.snapshot(wallSeconds: seconds(since: start)))
    }

    /// Both paths for one case. `makeEngine` is called once per path, so a stateful mock starts clean; a real run returns the
    /// one loaded engine both times. A path that fails (a throw, a malformed one-shot reply) is a recorded result and a
    /// difference, never an exception.
    static func runCase(
        _ testCase: ParityCase,
        makeEngine: (ParityCase, ParityPath) async throws -> any ParityEngine
    ) async -> ParityCaseReport {
        guard testCase.inputText != nil else {
            return ParityCaseReport(caseID: testCase.id, skipped: "input is \(testCase.inputStatus)")
        }
        let a = await run(.q35Loop, testCase, makeEngine)
        let b = await run(.grammarOneShot, testCase, makeEngine)
        var report = ParityCaseReport(caseID: testCase.id, a: a, b: b)
        if let reason = a.failure {
            report.comparison = ParityComparison(first: .pathFailed(path: .q35Loop, reason: reason))
        } else if let reason = b.failure {
            report.comparison = ParityComparison(first: .pathFailed(path: .grammarOneShot, reason: reason))
        } else if let left = a.table, let right = b.table {
            report.comparison = ParityComparison.compare(left, right)
        }
        return report
    }

    static func runCorpus(
        _ cases: [ParityCase],
        makeEngine: (ParityCase, ParityPath) async throws -> any ParityEngine
    ) async -> [ParityCaseReport] {
        var reports: [ParityCaseReport] = []
        for testCase in cases { reports.append(await runCase(testCase, makeEngine: makeEngine)) }
        return reports
    }

    /// One line per case for a journal: verdict, the first difference, and per-path calls / tokens / seconds.
    static func summary(_ reports: [ParityCaseReport]) -> String {
        reports.map { report in
            guard let comparison = report.comparison, let a = report.a, let b = report.b else {
                return "\(report.caseID): skipped (\(report.skipped ?? "no result"))"
            }
            let verdict = comparison.first.map { "DIFFERS, first: \($0)" } ?? "identical"
            return "\(report.caseID): \(verdict) | A \(cost(a.accounting)) | B \(cost(b.accounting))"
        }.joined(separator: "\n")
    }

    private static func cost(_ accounting: ParityAccounting) -> String {
        "\(accounting.modelCalls) calls, \(accounting.totalTokens) tokens, \(String(format: "%.3f", accounting.wallSeconds)) s"
    }

    private static func run(
        _ path: ParityPath, _ testCase: ParityCase,
        _ makeEngine: (ParityCase, ParityPath) async throws -> any ParityEngine
    ) async -> ParityPathResult {
        do {
            let engine = try await makeEngine(testCase, path)
            return path == .q35Loop
                ? await runQ35Loop(testCase, engine: engine)
                : await runGrammarOneShot(testCase, engine: engine)
        } catch {
            return ParityPathResult(table: nil, failure: "engine unavailable: \(error)", accounting: ParityAccounting())
        }
    }

    private static func textAsset(_ text: String) -> Asset {
        Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = ContinuousClock.now - start
        return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    }
}
