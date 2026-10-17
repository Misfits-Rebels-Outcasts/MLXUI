import Testing
import Foundation
@testable import MLXUI

/// CF-7 — the third `DecisionEngine` conformer, exercised through the real `execute(...)` entry point
/// with a fake `askClef` (never a real 6 GB checkpoint), mirroring `CLMDeciderTests`/`LayaDeciderTests`.
///
/// What this file proves: Clef's question building (a `choice` criteria **object** `{tag: ""}` per
/// declared tag in declared order, a `score` **array** of ordered levels — the reference sorts only a
/// choice's options, `clef_mlx.py:83` vs `:146`), the refusals it shares with the other two families,
/// that it has **no option ceiling** while Laya's still fires, and that `stateTruncated` reaches the
/// run log under gate **F2**'s 8,192-token cap.
struct ClefDeciderTests {
    private func clefEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--clef-flash-4bit", family: "Clef", displayName: "Clef Flash",
                 modelType: .decision, source: .mlx, ramGB: 9.32, downloadSizeGB: 6.21,
                 hfRepo: "mlx-community", hfModelId: "mlx-community/clef-flash-4bit")
    }

    private func layaEntry() -> ModelEntry {
        makeEntry(id: "aac6fef--laya-mlx", family: "Laya", displayName: "Laya 0.4B",
                 modelType: .decision, source: .mlx, hfRepo: "aac6fef", hfModelId: "aac6fef/laya-mlx")
    }

    private final class StageFlag: @unchecked Sendable {
        var built = false
    }

    /// A lock-guarded recorder for the `askClef` fake's own call arguments.
    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _calls: [(type: ClefQuestionType, instructions: ClefJSON, criteria: ClefJSON)] = []
        func record(type: ClefQuestionType, instructions: ClefJSON, criteria: ClefJSON) {
            lock.withLock { _calls.append((type, instructions, criteria)) }
        }
        var calls: [(type: ClefQuestionType, instructions: ClefJSON, criteria: ClefJSON)] {
            lock.withLock { _calls }
        }
    }

    private func makeExecutor(
        catalog: [ModelEntry], installedModelIDs: Set<String>,
        askClef: @escaping @Sendable (URL, ClefJSON, ClefQuestionType, ClefJSON, ClefJSON) async throws -> LayaAnswer,
        stageWasBuilt: StageFlag
    ) throws -> RealExecutor {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("clef-decider-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        var executor = RealExecutor(
            workspace: ws, flowID: "probe", blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in
                stageWasBuilt.built = true
                throw StageError.unsupportedModel(id: "probe", kind: .llm)
            },
            installedModelIDs: installedModelIDs, catalog: catalog)
        executor.askClef = askClef
        return executor
    }

    private func fakeChoiceAnswer(label: String, tags: [String]) -> LayaAnswer {
        let index = tags.firstIndex(of: label) ?? 0
        var probabilities = [Double](repeating: 0.1 / Double(max(tags.count - 1, 1)), count: tags.count)
        probabilities[index] = 0.9
        var answer = LayaAnswer(type: .choice, confidence: 0.8, probabilities: probabilities,
                                optionLabels: tags, stateTruncated: false)
        answer.choiceLabel = label
        return answer
    }

    private func input() -> Asset {
        Asset(items: [Item(kind: .text, value: "my invoice was charged twice", path: nil, sourceText: nil)])
    }

    // MARK: - Classify/Gate: a choice with an OBJECT of {tag: ""}, in declared order

    @Test func classifyFiresChoiceWithObjectCriteria() async throws {
        let recorder = Recorder()
        let executor = try makeExecutor(
            catalog: [clefEntry()], installedModelIDs: [clefEntry().id],
            askClef: { _, _, type, instructions, criteria in
                recorder.record(type: type, instructions: instructions, criteria: criteria)
                return self.fakeChoiceAnswer(label: "billing", tags: ["billing", "technical"])
            },
            stageWasBuilt: StageFlag())

        let row = Row(id: UUID(), task: "Classify", model: "Clef Flash", settings: "Who owns this?",
                     tags: ["billing", "technical"])
        let output = try await executor.execute(path: "1", row: row, inputs: [input()],
                                                transcript: nil, context: nil, usedFlowContent: nil)

        #expect(output.items.first?.value == "my invoice was charged twice", "Classify passes the input through")
        #expect(executor.lastTag == "billing")
        let call = try #require(recorder.calls.first)
        #expect(call.type == .choice)
        #expect(call.instructions == .string("Who owns this?"))
        guard case .object(let fields) = call.criteria else {
            Issue.record("Classify's criteria must be a JSON object, got \(call.criteria)")
            return
        }
        #expect(fields.map(\.key) == ["billing", "technical"], "tags in declared order")
        for field in fields {
            #expect(field.value == .string(""), "an empty description per tag — the tag is the option text")
        }
    }

    @Test func gateAlsoSendsTheChoiceObject() async throws {
        let recorder = Recorder()
        let executor = try makeExecutor(
            catalog: [clefEntry()], installedModelIDs: [clefEntry().id],
            askClef: { _, _, type, instructions, criteria in
                recorder.record(type: type, instructions: instructions, criteria: criteria)
                return self.fakeChoiceAnswer(label: "yes", tags: ["yes", "no"])
            },
            stageWasBuilt: StageFlag())

        let row = Row(id: UUID(), task: "Gate", model: "Clef Flash", settings: "Allow?", tags: ["yes", "no"])
        _ = try await executor.execute(path: "1", row: row, inputs: [input()],
                                       transcript: nil, context: nil, usedFlowContent: nil)
        let call = try #require(recorder.calls.first)
        #expect(call.type == .choice)
        #expect(call.criteria == .object([
            ClefJSONField(key: "yes", value: .string("")), ClefJSONField(key: "no", value: .string("")),
        ]))
    }

    // MARK: - Score: the declared tags as ordered levels

    @Test func scoreFiresWithOrderedLevels() async throws {
        let recorder = Recorder()
        let executor = try makeExecutor(
            catalog: [clefEntry()], installedModelIDs: [clefEntry().id],
            askClef: { _, _, type, instructions, criteria in
                recorder.record(type: type, instructions: instructions, criteria: criteria)
                var answer = LayaAnswer(type: .score, confidence: 0.6, probabilities: [0.1, 0.2, 0.7],
                                        optionLabels: ["0", "1", "2"], stateTruncated: false)
                answer.scoreValue = 1.6
                answer.scoreLevels = ["low", "mid", "high"]
                return answer
            },
            stageWasBuilt: StageFlag())

        let row = Row(id: UUID(), task: "Score", model: "Clef Flash", settings: "How urgent?",
                     tags: ["low", "mid", "high"])
        _ = try await executor.execute(path: "1", row: row, inputs: [input()],
                                       transcript: nil, context: nil, usedFlowContent: nil)

        let call = try #require(recorder.calls.first)
        #expect(call.type == .score)
        #expect(call.criteria == .array([.string("low"), .string("mid"), .string("high")]),
                "ordered levels, low → high — the reference orders a score's list, it sorts only a choice's ids")
        #expect(executor.lastDeciderDetail?.expectedLevel == 1.6)
    }

    // MARK: - Judge / Think / Decide refuse, naming Clef Flash

    @Test func judgeThinkAndDecideRefuseForClef() async throws {
        for task in ["Judge", "Think", "Decide"] {
            let stageWasBuilt = StageFlag()
            let executor = try makeExecutor(
                catalog: [clefEntry()], installedModelIDs: [clefEntry().id],
                askClef: { _, _, _, _, _ in
                    Issue.record("\(task): askClef must never be called for a refused task")
                    return LayaAnswer(type: .choice, confidence: 0, probabilities: [], optionLabels: [],
                                      stateTruncated: false)
                },
                stageWasBuilt: stageWasBuilt)

            let row = Row(id: UUID(), task: task, model: "Clef Flash", settings: "x", tags: ["a", "b"])
            do {
                _ = try await executor.execute(path: "1", row: row, inputs: [input()],
                                               transcript: nil, context: nil, usedFlowContent: nil)
                Issue.record("\(task) should have thrown")
            } catch FlowError.stageFailure(_, let message) {
                #expect(message == "Clef Flash doesn't serve \(task) — pick a language model for this row.",
                        "\(task): got '\(message)'")
            } catch {
                Issue.record("\(task): wrong error type: \(error)")
            }
            #expect(!stageWasBuilt.built, "\(task): must never build an LLM stage for Clef")
        }
    }

    // MARK: - No option ceiling for Clef; Laya's still fires

    @Test func thirtyTagClassifyIsAcceptedForClefWhileLayasCeilingStillFires() async throws {
        let tags = (0 ..< 30).map { "tag\($0)" }

        let clefExecutor = try makeExecutor(
            catalog: [clefEntry()], installedModelIDs: [clefEntry().id],
            askClef: { _, _, _, _, criteria in
                guard case .object(let fields) = criteria, fields.count == 30 else {
                    Issue.record("expected all 30 tags in the criteria object, got \(criteria)")
                    return LayaAnswer(type: .choice, confidence: 0, probabilities: [], optionLabels: [],
                                      stateTruncated: false)
                }
                return self.fakeChoiceAnswer(label: "tag0", tags: tags)
            },
            stageWasBuilt: StageFlag())
        let clefRow = Row(id: UUID(), task: "Classify", model: "Clef Flash", settings: "x", tags: tags)
        let output = try await clefExecutor.execute(path: "1", row: clefRow, inputs: [input()],
                                                    transcript: nil, context: nil, usedFlowContent: nil)
        #expect(output.items.first?.value == "my invoice was charged twice",
                "a 30-tag row must not be refused (Laya's ceiling doesn't apply to Clef)")
        #expect(clefExecutor.lastTag == "tag0")

        // The same row through Laya: the engine's own ceiling surfaces as a stage failure.
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("laya-ceiling-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        var layaExecutor = RealExecutor(
            workspace: FlowWorkspace(root: base.appendingPathComponent("flows")), flowID: "probe",
            blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "probe", kind: .llm) },
            installedModelIDs: [layaEntry().id], catalog: [layaEntry()])
        layaExecutor.askLaya = { _, _, _ in
            throw LayaPromptError.tooManyOptions(questionID: "0", optionCount: 30)
        }
        let layaRow = Row(id: UUID(), task: "Classify", model: "Laya 0.4B", settings: "x", tags: tags)
        do {
            _ = try await layaExecutor.execute(path: "1", row: layaRow, inputs: [input()],
                                               transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("Laya must still refuse 30 options")
        } catch FlowError.stageFailure(_, let message) {
            #expect(message.contains("30"), "Laya's refusal names the count: \(message)")
        }
    }

    // MARK: - decisionEngineBoundModel resolves the Clef entry

    @Test func decisionEngineBoundModelCoversAllThreeFamilies() {
        let catalog = [layaEntry(), clefEntry()]
        let clefRow = Row(id: UUID(), task: "Classify", model: "Clef Flash", settings: "x", tags: ["a", "b"])
        let layaRow = Row(id: UUID(), task: "Classify", model: "Laya 0.4B", settings: "x", tags: ["a", "b"])
        #expect(RealExecutor.decisionEngineBoundModel(for: clefRow, catalog: catalog)?.id == clefEntry().id)
        #expect(RealExecutor.decisionEngineBoundModel(for: layaRow, catalog: catalog)?.family == "Laya")
        // The routing set and the refusal's sentence are the same three families.
        #expect(Set(RealExecutor.decisionEngineFamilyOrder.map(\.family)) == RealExecutor.decisionEngineFamilies)
        #expect(RealExecutor.servedDeciderNames == ["Laya 0.4B", "CLM 8B", "Clef Flash"])
    }

    // MARK: - Truncation reaches the run log, under F2's cap

    @Test func truncationReachesTheRunLogWithF2sCap() async throws {
        let executor = try makeExecutor(
            catalog: [clefEntry()], installedModelIDs: [clefEntry().id],
            askClef: { _, _, _, _, _ in
                var answer = LayaAnswer(type: .choice, confidence: 0.7, probabilities: [0.7, 0.3],
                                        optionLabels: ["billing", "technical"], stateTruncated: true)
                answer.choiceLabel = "billing"
                return answer
            },
            stageWasBuilt: StageFlag())

        let row = Row(id: UUID(), task: "Classify", model: "Clef Flash", settings: "x",
                     tags: ["billing", "technical"])
        // Gate F2 (owner ruling 2026-10-17): the engine's budget is 8,192 tokens, not F1's 16,384 —
        // this is the cap the flag comes from, and the Run screen's note quotes.
        #expect(ClefPrompt.defaultMaxLength == 8_192)
        _ = try await executor.execute(path: "1", row: row, inputs: [input()],
                                       transcript: nil, context: nil, usedFlowContent: nil)
        let detail = try #require(executor.lastDeciderDetail)
        #expect(detail.stateTruncated, "the truncation flag reaches the run log")
        #expect(detail.tag == "billing")
        #expect(detail.probabilities.map(\.label) == ["billing", "technical"])
    }
}
