import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R7-1: `FlowInterpreter` reproduces the Python's run traces byte-for-byte. The
/// conformance corpus's `trace.events` (22 `mlxflow 0.4` flows) are the ground truth — the
/// same traces `catflow-mlx/tests/test_conformance.py::test_conformance_trace_case` asserts
/// the Python interpreter produces.
struct CatFlowInterpreterTraceTests {

    private var fixturesDir: URL {
        let filePath = #filePath
        return URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow")
    }

    private func canonicalJSON(_ object: Any) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return String(data: data, encoding: .utf8) ?? "?"
    }

    /// Serialize one interpreter event into the Python's wire dict.
    private func eventDict(_ event: FlowInterpreter.PathEvent) -> [String: Any] {
        var d: [String: Any] = ["type": event.kind.rawValue]
        switch event.kind {
        case .rowStarted, .rowCompleted, .budgetForced, .flagRaised:
            d["path"] = event.path
        case .eachItemStarted, .eachItemCompleted:
            d["path"] = event.path
            d["index"] = event.index ?? 0
            d["total"] = event.total ?? 0
        case .runCompleted:
            break
        case .transcriptAppended:
            d["path"] = event.path
        case .journalAppended:
            d["path"] = event.path
        case .runParked, .rowFailed, .rowSkipped, .cacheHit, .runResumed, .effectStaged:
            d["path"] = event.path
        }
        if event.kind == .rowCompleted {
            d["output"] = ["items": event.output?.items.map { item in
                var itemDict: [String: Any] = ["kind": item.kind.rawValue]
                if let value = item.value { itemDict["value"] = value } else { itemDict["value"] = NSNull() }
                if let path = item.path { itemDict["path"] = path.lastPathComponent } else { itemDict["path"] = NSNull() }
                return itemDict
            } ?? []]
            if let firedTag = event.firedTag { d["fired_tag"] = firedTag }
            if let context = event.context {
                d["context"] = context.map { ["label": $0.label, "content": $0.content] }
            }
            d["duration_ms"] = event.durationMS
        }
        if event.kind == .budgetForced, let edge = event.edge {
            d["edge"] = edge
        }
        if event.kind == .flagRaised {
            d["code"] = event.code ?? ""
            d["message"] = event.message ?? ""
        }
        if event.kind == .transcriptAppended, let t = event.transcript {
            d["entry"] = ["tool": t.tool, "input": t.input, "observation": t.observation]
        }
        if event.kind == .journalAppended, let entry = event.entry {
            d["entry_index"] = event.entryIndex ?? 0
            d["entry"] = ["label": entry.label, "content": entry.content]
        }
        if event.kind == .runParked {
            d["prompt"] = event.parkPrompt ?? ""
            d["policy"] = event.parkPolicy ?? ""
        }
        if event.kind == .rowFailed {
            d["error"] = event.error ?? ""
        }
        // DA-10 / SPEC-Q216: `row_skipped` carries the drop reason under `reason` (matching
        // `_ItemFailed(reason)` / F003's `reason=`). No conformance trace exercises it yet.
        if event.kind == .rowSkipped {
            d["reason"] = event.error ?? ""
        }
        if event.kind == .effectStaged, let staged = event.staged {
            d["id"] = staged.id
            d["kind"] = staged.kind
            d["summary"] = staged.summary
        }
        return d
    }

    @Test func runTracesMatchPython() async throws {
        let conformanceDir = fixturesDir.appendingPathComponent("conformance")
        let expectFiles = try FileManager.default.contentsOfDirectory(atPath: conformanceDir.path)
            .filter { $0.hasSuffix(".expect.json") }
            .sorted()
        var traced = 0
        var checked = 0
        var failingFlows: [String] = []
        for expectFile in expectFiles {
            let name = String(expectFile.dropLast(".expect.json".count))
            let expect = try JSONSerialization.jsonObject(
                with: Data(contentsOf: conformanceDir.appendingPathComponent(expectFile)))
                as? [String: Any]
            guard let trace = expect?["trace"] as? [String: Any],
                  let traceEvents = trace["events"] as? [[String: Any]], !traceEvents.isEmpty else {
                continue   // no v04 trace for this flow (guide flows A–F are `run`-only)
            }
            traced += 1

            let doc = try CatParser.parse(
                try String(contentsOf: conformanceDir.appendingPathComponent("\(name).cat"), encoding: .utf8))
            let base = FileManager.default.temporaryDirectory
                .appendingPathComponent("catflow-trace-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: base) }
            let script = (trace["decider_script"] as? [String: [String]]) ?? [:]
            let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"), deciderScript: script)

            let occurrence: FlowInterpreter.Occurrence?
            if let occ = trace["occurrence"] as? [String: Any] {
                occurrence = FlowInterpreter.Occurrence(path: (occ["path"] as? String).map(URL.init(fileURLWithPath:)),
                                                        tick: occ["tick"] as? String,
                                                        payload: occ["payload"] as? String)
            } else {
                occurrence = nil
            }

            let events = try await FlowInterpreter.run(doc, executor: mock, definitions: doc.definitions,
                                                       presets: doc.presets, occurrence: occurrence)
            let actual = try events.map { try canonicalJSON(eventDict($0)) }
            let expected = try traceEvents.map { try canonicalJSON($0) }
            #expect(actual == expected, "\(name): interpreter run trace differs from Python")
            if actual != expected {
                failingFlows.append(name)
                var diff = "=== \(name) ===\n"
                let n = max(actual.count, expected.count)
                for i in 0..<n {
                    let a = i < actual.count ? actual[i] : "<none>"
                    let e = i < expected.count ? expected[i] : "<none>"
                    if a != e { diff += "idx \(i)\n  exp: \(e)\n  got: \(a)\n" }
                }
                try? diff.write(to: FileManager.default.temporaryDirectory.appendingPathComponent("r7-fix-diag.txt"),
                                atomically: true, encoding: .utf8)
            }
            checked += 1
        }
        #expect(traced == 22, "expected all 22 trace-carrying conformance flows")
        #expect(checked == traced)
        try? (failingFlows.joined(separator: "\n") + "\n").write(
            to: FileManager.default.temporaryDirectory.appendingPathComponent("r7-fix-diag.txt"),
            atomically: true, encoding: .utf8)
    }

    /// CFM-R7-1: the interpreter's composite (`definitions:`), fork, goto-done, decider,
    /// journal, trigger, human-park, preset, and `uses:` inlining traces reproduce Python
    /// run traces generated from the same `run_v04_flat` + mock executor (stored in
    /// `Fixtures/CatFlow/traces/`).
    @Test func compositeAndClauseTracesMatchPython() async throws {
        let tracesDir = fixturesDir.appendingPathComponent("traces")
        let cats = try FileManager.default.contentsOfDirectory(atPath: tracesDir.path)
            .filter { $0.hasSuffix(".cat") }
            .filter { FileManager.default.fileExists(atPath: tracesDir.appendingPathComponent(String($0.dropLast(".cat".count)) + ".events.json").path) }
            .sorted()
        #expect(cats.count == 13)

        var checked = 0
        for cat in cats {
            let name = String(cat.dropLast(".cat".count))
            let source = try String(contentsOf: tracesDir.appendingPathComponent(cat), encoding: .utf8)
            let doc = try CatParser.parse(source)
            let expected = try JSONSerialization.jsonObject(
                with: Data(contentsOf: tracesDir.appendingPathComponent("\(name).events.json")))
                as? [[String: Any]] ?? []
            let base = FileManager.default.temporaryDirectory
                .appendingPathComponent("catflow-trace2-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: base) }
            let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))

            let occurrence: FlowInterpreter.Occurrence?
            if name == "r19_name_placeholder" {
                occurrence = FlowInterpreter.Occurrence(path: base.appendingPathComponent("quarterly-report.pdf"))
            } else {
                occurrence = nil
            }
            var usesGraph: [String: FlowInterpreter.UsedFlow] = [:]
            if name == "uses_inlining" {
                let lib = try CatParser.parse(try String(contentsOf: tracesDir.appendingPathComponent("uses-lib.cat"),
                                                         encoding: .utf8))
                usesGraph["Tidy"] = FlowInterpreter.UsedFlow(params: lib.params, rows: lib.rows,
                                                             definitions: lib.definitions, presets: lib.presets)
            }
            // human_resumed drives a live answer for row 3 (Ask Human) → RunResumed.
            let answers: [String: FlowInterpreter.HumanAnswer]
            if name == "human_resumed" {
                answers = ["3": FlowInterpreter.HumanAnswer(tag: "approve", text: nil)]
            } else {
                answers = [:]
            }
            let events = try await FlowInterpreter.run(doc, executor: mock, definitions: doc.definitions,
                                                       presets: doc.presets, usesGraph: usesGraph,
                                                       occurrence: occurrence, answers: answers)
            let actual = try events.map { try canonicalJSON(eventDict($0)) }
            let expectedJSON = try expected.map { try canonicalJSON($0) }
            #expect(actual == expectedJSON, "\(name): interpreter trace differs from Python")
            checked += 1
        }
        #expect(checked == cats.count)
    }

    /// CFM-R7 exit: `02-MeetingMinutes` — a `<parallel>` of three `Summarize` chains into a
    /// `Merge` — runs end-to-end under the mock interpreter (R7-2: chains run sequentially).
    @Test func meetingMinutesParallelRunsEndToEnd() async throws {
        let filePath = #filePath
        let gallery = URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/Gallery/02-MeetingMinutes.cat")
        let doc = try CatParser.parse(try String(contentsOf: gallery, encoding: .utf8))
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-mm-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))

        let events = try await FlowInterpreter.run(doc, executor: mock)

        // The block row 3 and its 4 children each start/complete; the 3 chains fan out via
        // the nested scopes (children 1, 2-3, 4), then Merge (4) and Save Text (5).
        let started = events.filter { $0.kind == .rowStarted }.map(\.path)
        let completed = events.filter { $0.kind == .rowCompleted }.map(\.path)
        // Starts: the block row 3 opens before its children. Completions: a block closes
        // *after* its children (the Python's order — row_completed "3" trails 3.1–3.4).
        #expect(started == ["1", "2", "3", "3.1", "3.2", "3.3", "3.4", "4", "5"])
        #expect(completed == ["1", "2", "3.1", "3.2", "3.3", "3.4", "3", "4", "5"])
        #expect(events.last?.kind == .runCompleted)
        // The parallel block's output feeds Merge: 3 chains × 1 text each.
        let blockOutput = events.first { $0.path == "3" && $0.kind == .rowCompleted }?.output
        #expect(blockOutput?.items.count == 3)
    }
}
