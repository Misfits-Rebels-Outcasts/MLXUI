import Testing
import Foundation
@testable import MLXUI

/// Pins the CFM-R7-FIX divergences the conformance corpus doesn't cover — each is a real
/// behavior the reviewer confirmed against the Python reference (items 5–13 + the minors),
/// asserted here so they can't silently regress. See `RSI/DelegateMergeBacklog.md` CFM-R7-FIX.
struct CatFlowR7FixTests {

    private func run(_ text: String,
                     definitions: [String: CompositeDef] = [:],
                     presets: [String: PresetDecl] = [:],
                     usesGraph: [String: FlowInterpreter.UsedFlow] = [:],
                     deciderScript: [String: [String]] = [:],
                     answers: [String: FlowInterpreter.HumanAnswer] = [:],
                     maxSteps: Int = FlowInterpreter.defaultMaxSteps) async throws -> [FlowInterpreter.PathEvent] {
        let doc = try CatParser.parse(text)
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-r7fix-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"),
                                deciderScript: deciderScript)
        return try await FlowInterpreter.run(doc, executor: mock, maxSteps: maxSteps,
                                             definitions: definitions, presets: presets,
                                             usesGraph: usesGraph, answers: answers)
    }

    /// FIX-5: `on_budget=fail` raises `budgetExceeded` — it must not silently proceed or
    /// re-run the row until `maxSteps`.
    @Test func onBudgetFailRaisesBudgetExceeded() async throws {
        // The Gate self-loops on `yes` and caps at 2 visits with on_budget=fail — the third
        // visit must raise, not proceed normally or spin to maxSteps.
        let text = """
        mlxflow 0.8
        1. Template    "go"
        2. Gate   (1)  "loop?" ; tags: yes, no ; max_visits=2 ; on_budget=fail
           -> { yes: 2 | no: 3 }
        3. Save Text   out.txt
        """
        do {
            _ = try await run(text, deciderScript: ["2": ["yes", "yes"]], maxSteps: 50)
            Issue.record("expected budgetExceeded, but the run completed")
        } catch let error as FlowError {
            guard case .budgetExceeded(let row, let leq) = error else {
                Issue.record("expected budgetExceeded, got \(error)")
                return
            }
            #expect(row == "2")
            #expect(leq == 2)
        }
    }

    /// FIX-6: an `<each on_error=skip>` item whose body fails flags F003 before the item
    /// completes — the drop is never silent.
    @Test func eachOnErrorSkipFlagsF003() async throws {
        // `Foo` is not a catalog task, so every item fails; on_error=skip absorbs each as an
        // F003 flag and the run continues (the mock's Split yields 3 items).
        let text = """
        mlxflow 0.8
        1. Template    "a; b"
        2. Split   (1)  by=lines
        3. <each try_em>   (2) ; on_error=skip
            1. Foo      {item}
        4. Join Text   (3)
        """
        let events = try await run(text)
        let flags = events.filter { $0.kind == .flagRaised && $0.code == "F003" }
        #expect(flags.count == 3, "expected one F003 per skipped item")
        #expect(flags[0].message?.contains("Item 1 of 3") == true)
        #expect(flags[0].message?.contains("was skipped") == true)
        #expect(flags[0].message?.contains("2 delivered") == true)
        // The run still completes (F003 is non-fatal).
        #expect(events.last?.kind == .runCompleted)
        // No item's row ever completes — each failed item is skipped, not delivered.
        #expect(!events.contains { $0.kind == .rowCompleted && $0.path == "3.1" })
    }

    /// DA-10 (SPEC-Q216): a skipped item must emit a **terminal** `row_skipped` event
    /// carrying the reason. Before DA-10 the failing row kept its `row_started` with no
    /// terminator — the UI dot stuck on ● forever and the reason never surfaced on the row.
    @Test func eachOnErrorSkipEmitsTerminalRowSkippedWithReason() async throws {
        let text = """
        mlxflow 0.8
        1. Template    "a; b"
        2. Split   (1)  by=lines
        3. <each try_em>   (2) ; on_error=skip
            1. Foo      {item}
        4. Join Text   (3)
        """
        let events = try await run(text)

        // One `row_skipped` per failed item, on the body row's path, carrying the reason.
        let skipped = events.filter { $0.kind == .rowSkipped && $0.path == "3.1" }
        #expect(skipped.count == 3, "expected one row_skipped per skipped item")
        #expect(skipped.allSatisfy { !($0.error ?? "").isEmpty }, "row_skipped carries a reason")

        // The seam guarantee: no row inside the block ends the run started-but-not-terminated.
        let terminal: Set<FlowInterpreter.PathEvent.Kind> = [.rowCompleted, .rowFailed, .rowSkipped]
        for (i, ev) in events.enumerated() where ev.kind == .rowStarted && ev.path.hasPrefix("3.") {
            let terminated = events[(i + 1)...].contains { $0.path == ev.path && terminal.contains($0.kind) }
            #expect(terminated, "row \(ev.path) started at idx \(i) with no terminal event")
        }

        // Unchanged: F003 still flags, the run still completes, the row never "completes".
        #expect(events.contains { $0.kind == .flagRaised && $0.code == "F003" })
        #expect(events.last?.kind == .runCompleted)
        #expect(!events.contains { $0.kind == .rowCompleted && $0.path == "3.1" })
    }

    /// FIX-7: a `<parallel>` chain's `· ctx` reads its own snapshot — a sibling's `ctx+`
    /// entry never leaks into another chain's context.
    @Test func parallelChainsDoNotShareJournal() async throws {
        // Chain A (child 1.1) appends via `; ctx+`; chain B's `· ctx` (child 1.2) must read
        // an empty journal, not A's entry.
        let text = """
        mlxflow 0.8
        1. <parallel two>
            1. Draft    "a" ; ctx+

            2. Draft    "b" ; ctx
        2. Save Text   out.txt
        """
        let events = try await run(text)
        guard let read = events.first(where: { $0.kind == .rowCompleted && $0.path == "1.2" }) else {
            Issue.record("missing child 1.2 completion")
            return
        }
        // The `· ctx` row carries an empty context — the sibling's `ctx+` entry is not visible.
        #expect(read.context?.isEmpty == true)
        // Chain A's append still committed onto the real journal.
        #expect(events.filter { $0.kind == .journalAppended }.count == 1)
    }

    /// FIX-11: `Read Context` hydrates the run's live journal — a later `· ctx` sees the
    /// loaded entries.
    @Test func readContextHydratesJournal() async throws {
        let text = """
        mlxflow 0.8
        1. Read Context   journal.json
        2. Draft   "hi" ; ctx
        """
        let doc = try CatParser.parse(text)
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-r7fix-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let stub = StubReadContext(inner: mock, json: #"[{"label":"Draft","content":"hi"}]"#)
        let events = try await FlowInterpreter.run(doc, executor: stub)
        guard let read = events.first(where: { $0.kind == .rowCompleted && $0.path == "2" }) else {
            Issue.record("missing row 2 completion")
            return
        }
        #expect(read.context?.count == 1)
        #expect(read.context?.first?.label == "Draft")
    }

    /// FIX-12: the cache key folds context/transcript/used_flow_content ingredients (with
    /// the frame version frame tasks always carry), pinned against Python-produced values.
    @Test func cacheKeyFoldsContextTranscriptAndUsedFlow() throws {
        let input = Asset(items: [Item(kind: .text, value: "hello", path: nil, sourceText: nil)])
        let context = [("Draft", "[mock:text] 1|Draft||x#0"),
                       ("Save Text", "[mock:status] 2|Save Text||out.txt#0")]
        let transcript = [FlowInterpreter.TranscriptEntry(tool: "search", input: "q", observation: "found")]
        let thinkFrame = CacheKey.frameVersion(task: "Think")
        let draftFrame = CacheKey.frameVersion(task: "Draft")

        let base = try CacheKey.cacheKey(task: "Think", model: "mlx-community/Qwen3-8B-4bit",
                                         settings: "tools: search, final", inputs: [input], realism: "mock",
                                         frameVersion: thinkFrame)
        #expect(base == "45d939542f5b73ac2d08c711be856e1178816360e5257baaeacadff1777b06c5")

        let withCtx = try CacheKey.cacheKey(task: "Think", model: "mlx-community/Qwen3-8B-4bit",
                                            settings: "tools: search, final", inputs: [input], realism: "mock",
                                            frameVersion: thinkFrame,
                                            contextVersion: CacheKey.journalVersion(context))
        #expect(withCtx == "eb009ab7cea12a736305528420cd458fddae063cf817c93564567c2a17190b6d")

        let withTr = try CacheKey.cacheKey(task: "Think", model: "mlx-community/Qwen3-8B-4bit",
                                           settings: "tools: search, final", inputs: [input], realism: "mock",
                                           frameVersion: thinkFrame,
                                           transcriptVersion: CacheKey.transcriptVersion(transcript))
        #expect(withTr == "52a9b0f0b6e6099ae269dbdaac5c37512f7b11679f7be25685e925e54a4d3410")

        let withUsed = try CacheKey.cacheKey(task: "Draft", model: "mlx-community/Qwen3-8B-4bit",
                                             settings: "greeting=Hello", inputs: [input], realism: "mock",
                                             frameVersion: draftFrame,
                                             usedFlowContent: "mlxflow 0.8\n1. Template ...")
        // CFM-R18-6: `usedFlowContent` header flipped `catflow`->`mlxflow`; re-pinned against Python.
        #expect(withUsed == "19814663190d3fd171ad24bf454c2f92c84c2ea5784973e314305cef4fcc6ba3")

        let allThree = try CacheKey.cacheKey(task: "Think", model: "mlx-community/Qwen3-8B-4bit",
                                             settings: "tools: search, final", inputs: [input], realism: "mock",
                                             frameVersion: thinkFrame,
                                             contextVersion: CacheKey.journalVersion(context),
                                             transcriptVersion: CacheKey.transcriptVersion(transcript),
                                             usedFlowContent: "mlxflow 0.8\n1. Template ...")
        #expect(allThree == "8b7f0c5e502bfaf9fd93cb9a25f10f2a0d89cf250ac919da0e3c29344a07e80b")

        // The version helpers themselves match Python (`journal_version`/`transcript_version`).
        #expect(CacheKey.journalVersion(context) == "2:7daf46283e728811bd2e92a760f348c1f9ea9b8e7a75d385ec804aaae21cd0ac")
        #expect(CacheKey.transcriptVersion(transcript) == "1:0a13a9fb3abe10c79377204593dd179f3bafc734b82e31a991911e617c705cbb")
    }

    /// FIX-13: a `uses:` entry's body runs under its own scope — its own composites resolve
    /// against the used flow's `definitions`, never the caller's.
    @Test func usesRunsUnderItsOwnScope() async throws {
        let libText = """
        mlxflow 0.8
        1. Greet   "hello"

        definitions:
          composite Greet   text -> [text]
          params: greeting = "hi"
             1. Template   {greeting}
        """
        let libDoc = try CatParser.parse(libText)
        let used = FlowInterpreter.UsedFlow(params: libDoc.params, rows: libDoc.rows,
                                            definitions: libDoc.definitions, presets: libDoc.presets,
                                            nested: [:], sourceText: libText)
        let caller = """
        mlxflow 0.8
        1. LibFlow
        2. Save Text   out.txt
        """
        // The caller declares no `definitions:` of its own — `Greet` inside the used flow's
        // body can only resolve against the used flow's scope.
        let events = try await run(caller, usesGraph: ["LibFlow": used])
        #expect(events.last?.kind == .runCompleted)
        // The used flow's body row 1 (Greet) expanded under the used flow's own definitions:
        // its child Template ran as 1.1.1 (the composite's single body row).
        #expect(events.contains { $0.kind == .rowCompleted && $0.path == "1.1.1" })
        #expect(events.contains { $0.kind == .rowCompleted && $0.path == "1.1" })
    }
}

/// Wraps a mock so `Read Context` returns supplied journal JSON (the default mock returns
/// `[]`, which would exercise nothing).
private struct StubReadContext: FlowExecutor, @unchecked Sendable {
    let inner: MockExecutor
    let json: String

    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset {
        if row.task == "Read Context" {
            return Asset(items: [Item(kind: .context, value: json, path: nil, sourceText: nil)])
        }
        return try await inner.execute(path: path, row: row, inputs: inputs,
                                       transcript: transcript, context: context,
                                       usedFlowContent: usedFlowContent)
    }

    var lastCacheHit: Bool { inner.lastCacheHit }
    var lastTag: String? { inner.lastTag }
    var lastTimeoutFlag: (code: String, message: String)? { inner.lastTimeoutFlag }
    var lastStaged: (id: String, kind: String, summary: String)? { inner.lastStaged }
}
