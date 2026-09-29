import Testing
import Foundation
@testable import MLXUI

/// LY-7 (`RSI/DelegateLayaBacklog.md`, Phase LY-B) — `RealExecutor.runLayaDecider`, exercised
/// through the real `execute(...)` entry point with a fake `askLaya` closure (never a real
/// `LayaEngine`/checkpoint) — the seam LY-7 itself asked for so this cycle needs no real model
/// weights. `makeModelStage` is a second fake that records whether it was ever called, so a
/// refusal path can prove it never built an LLM stage.
struct LayaDeciderTests {
    private func layaEntry() -> ModelEntry {
        makeEntry(id: "aac6fef--laya-mlx", family: "Laya", displayName: "Laya 0.4B",
                 modelType: .decision, source: .mlx, ramGB: 1.3, downloadSizeGB: 0.85,
                 hfRepo: "aac6fef", hfModelId: "aac6fef/laya-mlx")
    }

    private func clmEntry() -> ModelEntry {
        makeEntry(id: "RealityCat--CLM-v0.1-8B-MLX-8bit", family: "CLM", displayName: "CLM 8B",
                 modelType: .decision, source: .mlx, ramGB: 12.2, downloadSizeGB: 8.12,
                 hfRepo: "RealityCat", hfModelId: "RealityCat/CLM-v0.1-8B-MLX-8bit")
    }

    /// A `RealExecutor` with a fake `askLaya` (never loads a real checkpoint) and a
    /// `makeModelStage` that records whether it was ever called — proof a refusal never built
    /// an LLM stage either.
    private func makeExecutor(
        catalog: [ModelEntry], installedModelIDs: Set<String>,
        askLaya: @escaping @Sendable (URL, String, [LayaQuestion]) async throws -> [LayaAnswer],
        stageWasBuilt: StageFlag
    ) throws -> RealExecutor {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("laya-decider-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        var executor = RealExecutor(
            workspace: ws, flowID: "probe", blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in
                stageWasBuilt.built = true
                throw StageError.unsupportedModel(id: "probe", kind: .llm)
            },
            installedModelIDs: installedModelIDs, catalog: catalog)
        executor.askLaya = askLaya
        return executor
    }

    /// A reference box so the `makeModelStage` fake can report back through a `let` capture.
    private final class StageFlag: @unchecked Sendable {
        var built = false
    }

    private func fakeAnswer(
        type: LayaQuestionType, confidence: Double, probabilities: [Double],
        choiceLabel: String? = nil, scoreValue: Double? = nil, stateTruncated: Bool = false
    ) -> LayaAnswer {
        var answer = LayaAnswer(type: type, confidence: confidence, probabilities: probabilities,
                                optionLabels: [], stateTruncated: stateTruncated)
        answer.choiceLabel = choiceLabel
        answer.scoreValue = scoreValue
        return answer
    }

    // MARK: - Classify/Gate → choice, pass-through payload

    @Test func classifyFiresChoiceAndPassesInputThrough() async throws {
        let stageWasBuilt = StageFlag()
        let executor = try makeExecutor(
            catalog: [layaEntry()], installedModelIDs: [layaEntry().id],
            askLaya: { _, _, _ in
                [self.fakeAnswer(type: .choice, confidence: 0.8, probabilities: [0.8, 0.2], choiceLabel: "billing")]
            },
            stageWasBuilt: stageWasBuilt)

        let row = Row(id: UUID(), task: "Classify", model: "Laya 0.4B", settings: "Who should handle this?",
                     tags: ["billing", "technical"])
        let input = Asset(items: [Item(kind: .text, value: "my invoice was charged twice", path: nil, sourceText: nil)])
        let output = try await executor.execute(path: "1", row: row, inputs: [input],
                                                transcript: nil, context: nil, usedFlowContent: nil)

        #expect(output.items.first?.value == "my invoice was charged twice")
        #expect(executor.lastTag == "billing")
        #expect(!stageWasBuilt.built, "Classify on a Laya row must never build an LLM stage")
        let detail = try #require(executor.lastDeciderDetail)
        #expect(detail.tag == "billing")
        #expect(detail.confidence == 0.8)
        #expect(detail.expectedLevel == nil)
        #expect(detail.stateTruncated == false)
    }

    // MARK: - Score → argmax level, never the interpolated value (gate F)

    @Test func scoreFiresArgmaxLevelNotTheInterpolatedValue() async throws {
        let executor = try makeExecutor(
            catalog: [layaEntry()], installedModelIDs: [layaEntry().id],
            askLaya: { _, _, _ in
                [self.fakeAnswer(type: .score, confidence: 0.5, probabilities: [0.1, 0.7, 0.2], scoreValue: 1.1)]
            },
            stageWasBuilt: StageFlag())

        let row = Row(id: UUID(), task: "Score", model: "Laya 0.4B", settings: "How urgent?",
                     tags: ["low", "mid", "high"])
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        _ = try await executor.execute(path: "1", row: row, inputs: [input],
                                       transcript: nil, context: nil, usedFlowContent: nil)

        // argmax(0.1, 0.7, 0.2) is index 1 -> "mid", never "1.1" or a level built from it.
        #expect(executor.lastTag == "mid")
        let detail = try #require(executor.lastDeciderDetail)
        #expect(detail.tag == "mid")
        #expect(detail.expectedLevel == 1.1, "the expected level is logged, but must not be the fired tag")
    }

    // MARK: - Judge/Think/Decide refuse for Laya

    @Test func judgeThinkAndDecideRefuseForLaya() async throws {
        for task in ["Judge", "Think", "Decide"] {
            let stageWasBuilt = StageFlag()
            let executor = try makeExecutor(
                catalog: [layaEntry()], installedModelIDs: [layaEntry().id],
                askLaya: { _, _, _ in
                    Issue.record("\(task): askLaya must never be called for a refused task")
                    return []
                },
                stageWasBuilt: stageWasBuilt)

            let row = Row(id: UUID(), task: task, model: "Laya 0.4B", settings: "x", tags: ["a", "b"])
            let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
            do {
                _ = try await executor.execute(path: "1", row: row, inputs: [input],
                                               transcript: nil, context: nil, usedFlowContent: nil)
                Issue.record("\(task) should have thrown")
            } catch FlowError.stageFailure(_, let message) {
                #expect(message.contains("doesn't serve \(task)"), "\(task): got '\(message)'")
                #expect(message.contains("Laya 0.4B"))
            } catch {
                Issue.record("\(task): wrong error type: \(error)")
            }
            #expect(!stageWasBuilt.built, "\(task): must never build an LLM stage for Laya")
        }
    }

    // MARK: - The `tags:` clause is stripped from the criterion text (LY-7's own pinned fixture)

    @Test func criterionStripsTheDeclaredTagsClauseFromInstructions() async throws {
        var capturedInstructions: String?
        let executor = try makeExecutor(
            catalog: [layaEntry()], installedModelIDs: [layaEntry().id],
            askLaya: { _, _, questions in
                capturedInstructions = questions.first?.instructions
                return [self.fakeAnswer(type: .choice, confidence: 0.9, probabilities: [0.9, 0.1], choiceLabel: "act")]
            },
            stageWasBuilt: StageFlag())

        // LY-7's own fixture: "Urgency? tags: act, read, ignore" -> "Urgency?".
        let row = Row(id: UUID(), task: "Classify", model: "Laya 0.4B",
                     settings: "Urgency? tags: act, read, ignore", tags: ["act", "read", "ignore"])
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        _ = try await executor.execute(path: "1", row: row, inputs: [input],
                                       transcript: nil, context: nil, usedFlowContent: nil)

        #expect(capturedInstructions == "Urgency?")
    }

    // MARK: - Too many options for the token budget

    @Test func tooManyOptionsRefusesNamingTheCount() async throws {
        let executor = try makeExecutor(
            catalog: [layaEntry()], installedModelIDs: [layaEntry().id],
            askLaya: { _, _, _ in throw LayaPromptError.tooManyOptions(questionID: "0", optionCount: 30) },
            stageWasBuilt: StageFlag())

        let row = Row(id: UUID(), task: "Classify", model: "Laya 0.4B", settings: "x",
                     tags: (0 ..< 30).map { "tag\($0)" })
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        do {
            _ = try await executor.execute(path: "1", row: row, inputs: [input],
                                           transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("should have thrown")
        } catch FlowError.stageFailure(_, let message) {
            #expect(message.contains("30"), "message should name the option count: \(message)")
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    // MARK: - stateTruncated is logged, non-blocking

    @Test func stateTruncatedIsLoggedButDoesNotBlockTheRow() async throws {
        let executor = try makeExecutor(
            catalog: [layaEntry()], installedModelIDs: [layaEntry().id],
            askLaya: { _, _, _ in
                [self.fakeAnswer(type: .choice, confidence: 0.6, probabilities: [0.6, 0.4],
                                 choiceLabel: "a", stateTruncated: true)]
            },
            stageWasBuilt: StageFlag())

        let row = Row(id: UUID(), task: "Gate", model: "Laya 0.4B", settings: "x", tags: ["a", "b"])
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        let output = try await executor.execute(path: "1", row: row, inputs: [input],
                                                 transcript: nil, context: nil, usedFlowContent: nil)

        #expect(output.items.first?.value == "text", "truncation must not block the row")
        #expect(try #require(executor.lastDeciderDetail).stateTruncated == true)
    }

    // MARK: - A CLM-family `.decision` entry is refused, not routed to Laya or an LLM stage

    @Test func clmFamilyDecisionEntryRefusesInWorkflows() async throws {
        let stageWasBuilt = StageFlag()
        let executor = try makeExecutor(
            catalog: [clmEntry()], installedModelIDs: [clmEntry().id],
            askLaya: { _, _, _ in
                Issue.record("askLaya must never be called for a CLM-family row")
                return []
            },
            stageWasBuilt: stageWasBuilt)

        let row = Row(id: UUID(), task: "Classify", model: "CLM 8B", settings: "x", tags: ["a", "b"])
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        do {
            _ = try await executor.execute(path: "1", row: row, inputs: [input],
                                           transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("should have thrown")
        } catch FlowError.stageFailure(_, let message) {
            #expect(message == "CLM 8B isn't available in workflows yet — pick Laya 0.4B or a language model for this row.")
        } catch {
            Issue.record("wrong error type: \(error)")
        }
        #expect(!stageWasBuilt.built, "a refused CLM row must never build an LLM stage")
    }
}
