import Testing
import Foundation
@testable import MLXUI

/// CL-6 (`RSI/DelegateCLMBacklog.md` §CL-0 gate C, ruled (1) — one shared decider path) —
/// `RealExecutor.runDecisionDecider`'s CLM half, exercised through the real `execute(...)`
/// entry point with a fake `askCLM` closure (never a real `CLMEngine`/checkpoint), mirroring
/// `LayaDeciderTests.swift`'s own pattern for Laya. `LayaDeciderTests` proves Laya's behavior
/// is byte-identical to LY-7's; this file proves CLM's question-building (CL-5-FIX-1's own
/// lesson: `choice` criteria must be a JSON *object*, `score` criteria an *array*) and the
/// shared refusal/routing machinery both engines now go through.
struct CLMDeciderTests {
    private func clmEntry() -> ModelEntry {
        makeEntry(id: "RealityCat--CLM-v0.1-8B-MLX-8bit", family: "CLM", displayName: "CLM 8B",
                 modelType: .decision, source: .mlx, ramGB: 12.2, downloadSizeGB: 8.12,
                 hfRepo: "RealityCat", hfModelId: "RealityCat/CLM-v0.1-8B-MLX-8bit")
    }

    /// A reference box so the `makeModelStage` fake can report back through a `let` capture —
    /// same pattern as `LayaDeciderTests.StageFlag`.
    private final class StageFlag: @unchecked Sendable {
        var built = false
    }

    /// A lock-guarded recorder for the `askCLM` fake's own call arguments — a plain captured
    /// `var` isn't Sendable-safe across an escaping `@Sendable` closure boundary.
    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _calls: [(type: LayaQuestionType, instructions: CLMJSON, criteria: CLMJSON)] = []
        func record(type: LayaQuestionType, instructions: CLMJSON, criteria: CLMJSON) {
            lock.withLock { _calls.append((type, instructions, criteria)) }
        }
        var calls: [(type: LayaQuestionType, instructions: CLMJSON, criteria: CLMJSON)] { lock.withLock { _calls } }
    }

    private func makeExecutor(
        catalog: [ModelEntry], installedModelIDs: Set<String>,
        askCLM: @escaping @Sendable (URL, CLMJSON, LayaQuestionType, CLMJSON, CLMJSON) async throws -> LayaAnswer,
        stageWasBuilt: StageFlag
    ) throws -> RealExecutor {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("clm-decider-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        var executor = RealExecutor(
            workspace: ws, flowID: "probe", blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in
                stageWasBuilt.built = true
                throw StageError.unsupportedModel(id: "probe", kind: .llm)
            },
            installedModelIDs: installedModelIDs, catalog: catalog)
        executor.askCLM = askCLM
        return executor
    }

    private func fakeChoiceAnswer(label: String, tags: [String]) -> LayaAnswer {
        let index = tags.firstIndex(of: label) ?? 0
        var probs = [Double](repeating: 0.1 / Double(max(tags.count - 1, 1)), count: tags.count)
        probs[index] = 0.9
        var answer = LayaAnswer(type: .choice, confidence: 0.8, probabilities: probs,
                                optionLabels: tags, stateTruncated: false)
        answer.choiceLabel = label
        return answer
    }

    // MARK: - Classify/Gate: choice criteria as an OBJECT, {tag: ""} per declared tag, in order

    @Test func classifyFiresChoiceWithObjectCriteria() async throws {
        let recorder = Recorder()
        let executor = try makeExecutor(
            catalog: [clmEntry()], installedModelIDs: [clmEntry().id],
            askCLM: { _, _, type, instructions, criteria in
                recorder.record(type: type, instructions: instructions, criteria: criteria)
                return self.fakeChoiceAnswer(label: "billing", tags: ["billing", "technical"])
            },
            stageWasBuilt: StageFlag())

        let row = Row(id: UUID(), task: "Classify", model: "CLM 8B", settings: "Who owns this?",
                     tags: ["billing", "technical"])
        let input = Asset(items: [Item(kind: .text, value: "my invoice was charged twice", path: nil, sourceText: nil)])
        let output = try await executor.execute(path: "1", row: row, inputs: [input],
                                                transcript: nil, context: nil, usedFlowContent: nil)

        #expect(output.items.first?.value == "my invoice was charged twice")
        #expect(executor.lastTag == "billing")
        let call = try #require(recorder.calls.first)
        #expect(call.type == .choice)
        guard case .object(let fields) = call.criteria else {
            Issue.record("Classify's criteria must be a JSON object, got \(call.criteria)")
            return
        }
        #expect(fields.map(\.key) == ["billing", "technical"], "tags in declared order")
        for field in fields {
            #expect(field.value == .string(""), "an empty description per declared tag (candidate text is the tag itself)")
        }
    }

    // MARK: - Score: criteria as an ARRAY of the declared tags in order

    @Test func scoreFiresWithOrderedArrayCriteria() async throws {
        let recorder = Recorder()
        let executor = try makeExecutor(
            catalog: [clmEntry()], installedModelIDs: [clmEntry().id],
            askCLM: { _, _, type, instructions, criteria in
                recorder.record(type: type, instructions: instructions, criteria: criteria)
                var answer = LayaAnswer(type: .score, confidence: 0.6, probabilities: [0.1, 0.2, 0.7],
                                        optionLabels: ["low", "mid", "high"], stateTruncated: false)
                answer.choiceLabel = "high"
                return answer
            },
            stageWasBuilt: StageFlag())

        let row = Row(id: UUID(), task: "Score", model: "CLM 8B", settings: "How urgent?",
                     tags: ["low", "mid", "high"])
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        _ = try await executor.execute(path: "1", row: row, inputs: [input],
                                       transcript: nil, context: nil, usedFlowContent: nil)

        let call = try #require(recorder.calls.first)
        #expect(call.type == .score)
        #expect(call.criteria == .array([.string("low"), .string("mid"), .string("high")]),
                "the declared tags, in order, as a plain array — not the object shape Classify uses")
    }

    // MARK: - Judge/Think/Decide refuse, naming CLM 8B (not a hard-coded model)

    @Test func judgeThinkAndDecideRefuseForCLM() async throws {
        for task in ["Judge", "Think", "Decide"] {
            let stageWasBuilt = StageFlag()
            let executor = try makeExecutor(
                catalog: [clmEntry()], installedModelIDs: [clmEntry().id],
                askCLM: { _, _, _, _, _ in
                    Issue.record("\(task): askCLM must never be called for a refused task")
                    return LayaAnswer(type: .choice, confidence: 0, probabilities: [], optionLabels: [], stateTruncated: false)
                },
                stageWasBuilt: stageWasBuilt)

            let row = Row(id: UUID(), task: task, model: "CLM 8B", settings: "x", tags: ["a", "b"])
            let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
            do {
                _ = try await executor.execute(path: "1", row: row, inputs: [input],
                                               transcript: nil, context: nil, usedFlowContent: nil)
                Issue.record("\(task) should have thrown")
            } catch FlowError.stageFailure(_, let message) {
                #expect(message == "CLM 8B doesn't serve \(task) — pick a language model for this row.", "\(task): got '\(message)'")
            } catch {
                Issue.record("\(task): wrong error type: \(error)")
            }
            #expect(!stageWasBuilt.built, "\(task): must never build an LLM stage for CLM")
        }
    }

    // MARK: - No option-count ceiling: a 30-tag Classify is accepted for CLM

    @Test func thirtyTagClassifyIsAcceptedForCLM() async throws {
        let tags = (0 ..< 30).map { "tag\($0)" }
        let executor = try makeExecutor(
            catalog: [clmEntry()], installedModelIDs: [clmEntry().id],
            askCLM: { _, _, _, _, criteria in
                guard case .object(let fields) = criteria, fields.count == 30 else {
                    Issue.record("expected all 30 tags in the criteria object, got \(criteria)")
                    return LayaAnswer(type: .choice, confidence: 0, probabilities: [], optionLabels: [], stateTruncated: false)
                }
                return self.fakeChoiceAnswer(label: "tag0", tags: tags)
            },
            stageWasBuilt: StageFlag())

        let row = Row(id: UUID(), task: "Classify", model: "CLM 8B", settings: "x", tags: tags)
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        let output = try await executor.execute(path: "1", row: row, inputs: [input],
                                                 transcript: nil, context: nil, usedFlowContent: nil)
        #expect(output.items.first?.value == "text", "a 30-tag row must not be refused (Laya's ceiling doesn't apply to CLM)")
        #expect(executor.lastTag == "tag0")
    }

    // MARK: - An unknown `.decision` family (neither Laya nor CLM) still refuses clearly

    @Test func unknownDecisionFamilyRefusesNamingTheModel() async throws {
        let otherEntry = makeEntry(id: "other--decision-mlx", family: "OtherDecisionFamily",
                                   displayName: "Other Decider", modelType: .decision, source: .mlx,
                                   hfRepo: "other", hfModelId: "other/decision-mlx")
        let stageWasBuilt = StageFlag()
        let executor = try makeExecutor(
            catalog: [otherEntry], installedModelIDs: [otherEntry.id],
            askCLM: { _, _, _, _, _ in
                Issue.record("askCLM must never be called for an unclaimed decision family")
                return LayaAnswer(type: .choice, confidence: 0, probabilities: [], optionLabels: [], stateTruncated: false)
            },
            stageWasBuilt: stageWasBuilt)

        let row = Row(id: UUID(), task: "Classify", model: "Other Decider", settings: "x", tags: ["a", "b"])
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        do {
            _ = try await executor.execute(path: "1", row: row, inputs: [input],
                                           transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("should have thrown")
        } catch FlowError.stageFailure(_, let message) {
            #expect(message == "Other Decider isn't available in workflows yet — "
                    + "pick Laya 0.4B, CLM 8B, or a language model for this row.")
        } catch {
            Issue.record("wrong error type: \(error)")
        }
        #expect(!stageWasBuilt.built, "an unclaimed decision family must never build an LLM stage")
    }

    // MARK: - Through CachingExecutor(RealExecutor): CLM fires its tag, cold and warm

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var _count = 0
        func increment() { lock.withLock { _count += 1 } }
        var count: Int { lock.withLock { _count } }
    }

    /// CACHE-SIGNALS-1's own lesson: the real Run UI wraps `RealExecutor` in `CachingExecutor`,
    /// which used to drop `lastTag` entirely — a decider-routed block would silently end right
    /// after the decider on every real run, hit or miss. This proves CLM's own tag survives
    /// that wrapper, cold (a genuine miss) and warm (a genuine cache hit).
    @Test func clmBoundClassifyReachesTheRightBranchColdAndWarmThroughTheCachingWrapper() async throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("clm-caching-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let store = FlowCacheStore(root: base.appendingPathComponent("cache"))
        let flowID = "f"

        let clm = clmEntry()
        let calls = Counter()
        var real = RealExecutor(
            workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "probe", kind: .llm) },
            installedModelIDs: [clm.id], catalog: [clm])
        real.askCLM = { _, _, _, _, _ in
            calls.increment()
            return self.fakeChoiceAnswer(label: "technical", tags: ["billing", "technical"])
        }
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real",
                                      catalog: [clm], runSeed: 0, workspace: ws, flowID: flowID)

        let text = """
        mlxflow 0.8
        1. Template   "Customer says the app keeps crashing"
        2. Classify   CLM 8B; "Which team? tags: billing, technical"
           -> { billing: 3 | technical: 4 }
        3. Template                "BILLING    {1}"
           -> 5
        4. Template                "TECHNICAL  {1}"
        5. Template                "{1}"
        """
        let doc = try CatParser.parse(text)

        func assertRun(_ events: [FlowInterpreter.PathEvent], label: String) {
            let failures = events.filter { $0.kind == .rowFailed }
            #expect(failures.isEmpty, "\(label): no row should fail: \(failures.map(\.error))")
            for path in ["2", "4", "5"] {
                #expect(events.contains { $0.kind == .rowCompleted && $0.path == path },
                        "\(label): row \(path) never completed")
            }
            #expect(!events.contains { $0.kind == .rowCompleted && $0.path == "3" },
                    "\(label): the BILLING branch must not fire for a technical ticket")
        }

        let cold = try await FlowInterpreter.run(doc, executor: caching)
        assertRun(cold, label: "cold")
        #expect(calls.count == 1, "cold: CLM must be asked for real")

        let warm = try await FlowInterpreter.run(doc, executor: caching)
        assertRun(warm, label: "warm")
        #expect(calls.count == 1, "warm: a cache hit must never call askCLM again")
        #expect(warm.contains { $0.kind == .cacheHit && $0.path == "2" },
                "warm: Classify's own path should be a cache hit")
    }
}
