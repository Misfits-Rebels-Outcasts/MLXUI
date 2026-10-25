import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXUI

// WW4-2-2 -- the owner-run real-model half of the SPEC-Q237(c) parity experiment (KICKOFF rules 8 and 16). One checkpoint per
// run, one family per invocation (the owner's Mac has 19 GB), exactly like `ServeStructuredRealModelTests`:
//
//   TEST_RUNNER_MLXUI_TEST_PARITY_DIR=<installed checkpoint folder>   [TEST_RUNNER_MLXUI_TEST_PARITY_LABEL=gemma4-e4b]
//
// with the sandbox overridden on the command line (CODE_SIGN_ENTITLEMENTS=MLXUI/MLXUI-Direct.entitlements ENABLE_APP_SANDBOX=NO)
// because the sandboxed host cannot read the Direct store. The exact commands are in the WW4-2-2 journal. Without the variable the
// suite is skipped, and a skipped run does not count.
//
// This test never asserts parity: a difference between the two paths is the result. It asserts only that the run completed, that
// every path of every runnable case left a recorded result, that the grammar side ran with the prompt cache off, and that nothing
// outside a temp directory was written. Output is `PARITYQ237 ` lines on stdout and the same data as JSON in a temp file.

nonisolated enum ParityRealModel {
    static let environment = ProcessInfo.processInfo.environment
    static let directory: URL? = environment["MLXUI_TEST_PARITY_DIR"].map { URL(fileURLWithPath: $0) }
    /// The owner's ruling (2026-10-25): nothing over 8 GB.
    static let maxBytes: Int64 = 8 << 30

    /// Total bytes of the regular files under `directory`.
    static func size(of directory: URL) -> Int64 {
        var total: Int64 = 0
        guard let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { return 0 }
        for case let url as URL in files {
            let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true { total += Int64(values?.fileSize ?? 0) }
        }
        return total
    }

    /// `relative path -> size`, for the "the source was never touched" check.
    static func listing(of directory: URL) -> [String: Int64] {
        var result: [String: Int64] = [:]
        guard let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey]) else { return result }
        for case let url as URL in files {
            let relative = String(url.path.dropFirst(directory.path.count))
            result[relative] = Int64((try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? -1)
        }
        return result
    }
}

// MARK: - Path A's real engine: the shipped loop's model calls

/// Counts the real model calls and tokens behind one `ParityEngine` method. An actor, because the shipped stage's generator is
/// `@Sendable`. Serial use only: the harness runs one call at a time.
actor ParityCallTally {
    private var calls = 0
    private var promptTokens = 0
    private var completionTokens = 0

    func add(prompt: Int, completion: Int) {
        calls += 1
        promptTokens += prompt
        completionTokens += completion
    }

    /// Returns what accumulated since the last `take` and starts again.
    func take() -> (calls: Int, prompt: Int, completion: Int) {
        defer { calls = 0; promptTokens = 0; completionTokens = 0 }
        return (calls, promptTokens, completionTokens)
    }
}

/// The Q35 side: the same two model calls `RealExecutor.llmExtractionCalls` bridges onto a stage, over the same engine.
///
/// What ships (verified against the source, not assumed): `RealExecutor.makeExtractionStage` builds the stage with
/// `StageConfig(maxTokens: ExtractStructuredStage.maxFieldTokens (32), temperature: 0)`; `ChatSDK.makeStage` turns that into an
/// `LLMStage` whose generator is `LLMEngine.generate(prompt:modelDir:maxTokens:temperature:footprintBytes:)`. This adapter builds
/// the identical `LLMStage` over the same `LLMEngine.generate`, differing only in the model directory (the temp copy), the pool
/// (the one both paths share) and a counting wrapper. `LLMEngine.generate` is also where the shipped paths get their handling of
/// Qwen3's thinking (`enable_thinking: false` in the template context, then `stripThinkBlock`) and decode raw token ids; Gemma's
/// stop token is the one the loaded configuration carries, because `generateTokens` stops on it. `gate` is the shipped
/// `RealExecutor.fireTag` verbatim (yes/no, one stricter retry); `line` is the shipped closure's first-line cut (copied: the original
/// is a private nested closure, and changing shipping source for the experiment is out of scope).
/// The prompt cache is not involved on this path: `LLMEngine.generate` never touches it.
///
/// Token counts, implementer's call pending owner confirmation: prompt = the chat-templated prompt's token count from the
/// installed tokenizer (`ServePromptCounter`, the same template and context the generation uses); completion = the returned text
/// re-encoded (think block already stripped), so it is an estimate of the generated tokens, within a stop token or two.
struct RealQ35ParityEngine: ParityEngine {
    let directory: URL
    let pool: ModelContainerPool
    let tally = ParityCallTally()

    private var stage: LLMStage {
        let directory = directory, pool = pool, tally = tally
        return LLMStage(id: "parity-q35", name: "parity-q35", systemPrompt: nil) { prompt in
            let text = try await LLMEngine.generate(
                prompt: prompt, modelDir: directory,
                maxTokens: ExtractStructuredStage.maxFieldTokens, temperature: 0, pool: pool)
            let counter = ServePromptCounter.shared
            let rendered = try? await counter.count(
                directory: directory, messages: [["role": "user", "content": prompt]], tools: nil,
                context: LLMEngine.noThinkingContext)
            let completion = await counter.tokenizer(for: directory)?.encode(text: text, addSpecialTokens: false).count ?? 0
            await tally.add(prompt: rendered.flatMap { $0 } ?? 0, completion: completion)
            return text
        }
    }

    func gate(prompt: String) async throws -> ParityReply {
        _ = await tally.take()
        let (tag, _) = try await RealExecutor.fireTag(
            stage: stage, prompt: prompt, tags: ExtractStructuredStage.continueTags, rowLabel: "parity")
        return await reply(tag)
    }

    func line(prompt: String) async throws -> ParityReply {
        _ = await tally.take()
        let result = try await stage.run(.text(prompt), progress: { _ in })
        var text = ""
        if case .text(let full) = result {
            text = full.split(separator: "\n", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? full
        }
        return await reply(text)
    }

    func structured(prompt: String, schema: String) async throws -> ParityReply {
        throw ParityError("the Q35 engine has no grammar call")
    }

    private func reply(_ text: String) async -> ParityReply {
        let taken = await tally.take()
        return ParityReply(text: text, promptTokens: taken.prompt, completionTokens: taken.completion, modelCalls: taken.calls)
    }
}

// MARK: - Path B's real engine: one grammar-constrained generation

/// The grammar side: `MLXChatBackend` itself, so the grammar is built exactly as `prepareStructure` builds it (allow-list, schema
/// checks, the model's own stop token, vocabulary, compile) and run exactly as the Local Server runs a `response_format` request:
/// one generation, `temperature: 0` (greedy), thinking off (a structured reply is never a thinking reply: the request leaves
/// `reasoning` unset), on the same pool and directory as path A so the model is loaded once. The prompt cache is switched off for
/// the process by the test (`PromptCache.overrideForTesting = false`), and this engine records every reply's `cachedTokens` so the
/// test can show it stayed 0.
/// `maxTokens` is implementer's call, pending owner confirmation: 2048, far above 20 rows of a short schema; a reply the grammar
/// could not finish inside it is `incomplete`, fails the strict decode and is reported as a failed path, not repaired.
actor RealGrammarParityEngine: ParityEngine {
    static let maxTokens = 2048
    private let backend: MLXChatBackend
    private(set) var cachedTokens = 0
    private(set) var incompleteReplies = 0
    private(set) var finishReasons: [String] = []

    init(directory: URL, pool: ModelContainerPool) {
        var backend = MLXChatBackend(directory: directory, pool: pool)
        backend.grammarCache = ServeGrammarCache()
        self.backend = backend
    }

    nonisolated var modelType: String { backend.checkpointModelType }

    func gate(prompt: String) async throws -> ParityReply { throw ParityError("the grammar engine has no Q35 call") }
    func line(prompt: String) async throws -> ParityReply { throw ParityError("the grammar engine has no Q35 call") }

    func structured(prompt: String, schema: String) async throws -> ParityReply {
        var request = ServeRequest(model: "parity", messages: [ServeMessage(role: .user, text: prompt)])
        request.temperature = 0
        request.maxTokens = Self.maxTokens
        request.structure = .jsonSchema(name: "records", schema: schema, strict: true)

        let prepareStart = ContinuousClock.now
        let grammar = try await backend.prepareStructure(request)
        let setup = MLXChatBackend.seconds(ContinuousClock.now - prepareStart)

        var text = ""
        var usage: ServeUsage?
        var finish: ServeFinishReason?
        var failure: ServeError?
        for try await event in backend.generate(request, grammar: grammar) {
            switch event {
            case .textDelta(let piece): text += piece
            case .usage(let value): usage = value
            case .finish(let reason): finish = reason
            case .error(let error): failure = error
            default: break
            }
        }
        await grammar?.finish(clean: false)          // what the pipeline does after the backend returns (the precedent's run helper)
        if let failure { throw ParityError("the backend reported \(failure.code)") }
        cachedTokens += usage?.cachedTokens ?? 0
        if usage?.structuredOutcome != .complete { incompleteReplies += 1 }
        finishReasons.append(finish.map { "\($0)" } ?? "none")
        return ParityReply(text: text, promptTokens: usage?.promptTokens ?? 0, completionTokens: usage?.completionTokens ?? 0,
                           modelCalls: 1, setupSeconds: setup)
    }
}

// MARK: - The test

@Suite(.serialized, .enabled(if: ParityRealModel.directory != nil))
struct ParityQ237RealModelTests {
    private static func line(_ fields: String) { print("PARITYQ237 \(fields)") }

    private static func fmt(_ seconds: Double) -> String { String(format: "%.3f", seconds) }

    private static func quoted(_ text: String) -> String {
        "\"" + text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n") + "\""
    }

    private static func accountingJSON(_ a: ParityAccounting) -> [String: Any] {
        ["calls": a.modelCalls, "promptTokens": a.promptTokens, "completionTokens": a.completionTokens, "totalTokens": a.totalTokens,
         "wallSeconds": a.wallSeconds, "setupSeconds": a.setupSeconds]
    }

    private static func pathJSON(_ result: ParityPathResult) -> [String: Any] {
        var json: [String: Any] = ["accounting": accountingJSON(result.accounting)]
        if let table = result.table { json["table"] = table.canonicalJSON }
        if let failure = result.failure { json["failure"] = failure }
        return json
    }

    private static func add(_ total: inout ParityAccounting, _ part: ParityAccounting) {
        total.modelCalls += part.modelCalls
        total.promptTokens += part.promptTokens
        total.completionTokens += part.completionTokens
        total.wallSeconds += part.wallSeconds
        total.setupSeconds += part.setupSeconds
    }

    @Test func everyRunnableCaseRunsDownBothPathsAndTheResultsAreRecorded() async throws {
        let source = try #require(ParityRealModel.directory)
        let label = ParityRealModel.environment["MLXUI_TEST_PARITY_LABEL"] ?? source.lastPathComponent

        // Owner ruling 2026-10-25: nothing over 8 GB.
        let bytes = ParityRealModel.size(of: source)
        guard bytes <= ParityRealModel.maxBytes else {
            Issue.record("refusing \(source.lastPathComponent): \(bytes) bytes is over the 8 GB limit (owner ruling 2026-10-25)")
            return
        }
        #expect(FileManager.default.fileExists(atPath: source.appendingPathComponent(".installed").path),
                "\(source.path) is not an installed checkpoint (no .installed marker)")

        // The source is never touched: the run uses a clone (copyItem is an APFS clone on one volume), removed in the defer.
        let before = ParityRealModel.listing(of: source)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("parityq237-model-\(UUID().uuidString)")
        let copy = root.appendingPathComponent(source.lastPathComponent)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.copyItem(at: source, to: copy)

        // Prompt cache OFF for the process. Path A (`LLMEngine.generate`) never uses it; path B goes through
        // `MLXChatBackend.generate`, which consults `PromptCache.isEnabled`, so the override is what keeps both comparable.
        PromptCache.overrideForTesting = false
        defer { PromptCache.overrideForTesting = nil }

        let pool = await MainActor.run { ModelContainerPool(budgetBytes: 16 << 30) }
        let grammarEngine = RealGrammarParityEngine(directory: copy, pool: pool)
        let q35Engine = RealQ35ParityEngine(directory: copy, pool: pool)
        let modelType = grammarEngine.modelType

        // Load time, separately: the first container for the copy (weights into the pool; both paths then share it).
        let loadStart = ContinuousClock.now
        let container = try await pool.container(directory: copy)
        let loadSeconds = MLXChatBackend.seconds(ContinuousClock.now - loadStart)

        // The structured stop token, read from the installed tokenizer with the allow-list's own reader (precedent: the WW4-1-3
        // stop-token test). Gemma 4 is 106 `<turn|>` and Qwen3 is 151645 `<|im_end|>` per the allow-list's recorded checks.
        let tokenizer = try await HFTokenizerLoader().load(from: copy)
        let stops = ServeGrammarEOS.stopIDs(directory: copy, tokenizer: tokenizer)
        let pick = try #require(ServeGrammarEOS.pick(directory: copy, tokenizer: tokenizer))
        let token = tokenizer.convertIdToToken(pick) ?? "nil"
        let pickIsStop = await container.perform { context in
            ServeGrammarEOS.pickIsStopToken(pick, configuration: context.configuration, tokenizer: context.tokenizer)
        }
        Self.line("STOP type=\(modelType) stopIDs=\(stops.sorted()) pick=\(pick) token=\(Self.quoted(token)) pickIsStopToken=\(pickIsStop)")
        #expect(pickIsStop, "the pick must be in the loaded model's own stop set")
        #expect(ServeGrammarEOS.turnEnders.contains(token), "\(token) is not a known turn-ender spelling")
        let recorded = ["gemma4": 106, "qwen3": 151645]
        if let expected = recorded[modelType] { #expect(pick == expected, "\(modelType): picked \(pick), the allow-list recorded \(expected)") }
        #expect(ServeStructuredOutputFamilies.allows(modelType: modelType), "\(modelType) is not on the structured-output allow-list")

        // Warm-up on both real paths, off the clock and out of the results (first-use costs: kernels, the tokenizer, the grammar
        // vocabulary). Failures here are noise: the cases below report any real fault.
        _ = try? await q35Engine.line(prompt: "Reply with one word: ready")
        _ = try? await grammarEngine.structured(prompt: "Text: a b\n\nAnswer with {\"records\":[]}",
                                                schema: ParityGrammarOneShot.schema(columns: ["a"]))

        let cases = try ParityCorpus.load()
        let runnable = cases.filter { $0.inputText != nil }
        let notRun = cases.filter { $0.inputText == nil }
        Self.line("MODEL label=\(label) dir=\(source.lastPathComponent) type=\(modelType) bytes=\(bytes) loadSeconds=\(Self.fmt(loadSeconds)) "
                  + "cases=\(cases.count) runnable=\(runnable.count) notRun=\(notRun.count) maxFieldTokens=\(ExtractStructuredStage.maxFieldTokens) "
                  + "oneShotMaxTokens=\(RealGrammarParityEngine.maxTokens) temperature=0 promptCache=off")

        let reports = await ParityHarness.runCorpus(cases) { _, path -> any ParityEngine in
            if path == .q35Loop { return q35Engine }
            return grammarEngine
        }

        var caseJSON: [[String: Any]] = []
        var identical = 0, different = 0, failedPaths = 0
        var totalA = ParityAccounting(), totalB = ParityAccounting()
        for report in reports {
            guard let comparison = report.comparison, let a = report.a, let b = report.b else {
                Self.line("CASE id=\(report.caseID) status=not-run reason=\(Self.quoted("input is another model's output"))")
                caseJSON.append(["id": report.caseID, "status": "not-run", "reason": "input is another model's output"])
                continue
            }
            let verdict: String
            switch comparison.first {
            case nil: identical += 1; verdict = "yes"
            case .pathFailed?: failedPaths += 1; verdict = "no"
            default: different += 1; verdict = "no"
            }
            Self.add(&totalA, a.accounting)
            Self.add(&totalB, b.accounting)
            let first = comparison.first.map { Self.quoted("\($0)") } ?? "none"
            Self.line("CASE id=\(report.caseID) status=run identical=\(verdict) first=\(first) "
                      + "A.calls=\(a.accounting.modelCalls) A.tokens=\(a.accounting.totalTokens) A.wall=\(Self.fmt(a.accounting.wallSeconds)) "
                      + "B.calls=\(b.accounting.modelCalls) B.tokens=\(b.accounting.totalTokens) B.wall=\(Self.fmt(b.accounting.wallSeconds)) "
                      + "B.setup=\(Self.fmt(b.accounting.setupSeconds))")
            var entry: [String: Any] = ["id": report.caseID, "status": "run", "identical": verdict == "yes",
                                        "A": Self.pathJSON(a), "B": Self.pathJSON(b)]
            if let first = comparison.first { entry["firstDifference"] = "\(first)" }
            caseJSON.append(entry)
        }
        let ran = identical + different + failedPaths
        Self.line("SUMMARY label=\(label) ran=\(ran) notRun=\(notRun.count) of=\(cases.count) identical=\(identical) different=\(different) "
                  + "failedPaths=\(failedPaths)")
        Self.line("TOTAL path=A(q35-loop) calls=\(totalA.modelCalls) tokens=\(totalA.totalTokens) wall=\(Self.fmt(totalA.wallSeconds))")
        Self.line("TOTAL path=B(grammar-one-shot) calls=\(totalB.modelCalls) tokens=\(totalB.totalTokens) wall=\(Self.fmt(totalB.wallSeconds)) "
                  + "setup=\(Self.fmt(totalB.setupSeconds))")
        Self.line("NOTE the \(notRun.count) not-run cases are the OCR rows of flows 30 and 55: their input is another model's output "
                  + "(GLM-OCR) and is not run here; results are for \(label), not the gallery's models (Qwen3 8B / Ministral 3B)")

        // The same data as JSON, in a temp file whose path is the last line printed.
        let cachedTokens = await grammarEngine.cachedTokens
        let incomplete = await grammarEngine.incompleteReplies
        let finishes = await grammarEngine.finishReasons
        let document: [String: Any] = [
            "label": label, "directory": source.lastPathComponent, "modelType": modelType, "bytes": bytes,
            "loadSeconds": loadSeconds, "temperature": 0, "promptCache": "off", "maxFieldTokens": ExtractStructuredStage.maxFieldTokens,
            "oneShotMaxTokens": RealGrammarParityEngine.maxTokens,
            "stop": ["pick": pick, "token": token, "stopIDs": stops.sorted()],
            "summary": ["ran": ran, "notRun": notRun.count, "cases": cases.count, "identical": identical, "different": different,
                        "failedPaths": failedPaths],
            "totals": ["A": Self.accountingJSON(totalA), "B": Self.accountingJSON(totalB)],
            "grammarRun": ["cachedTokens": cachedTokens, "incompleteReplies": incomplete, "finishReasons": finishes],
            "cases": caseJSON,
        ]
        let output = FileManager.default.temporaryDirectory.appendingPathComponent(
            "parityq237-\(label)-\(Int(Date().timeIntervalSince1970)).json")
        try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys]).write(to: output)
        Self.line("JSON \(output.path)")

        // What this test does assert: completion, recorded results, cache off, nothing written beside the clone.
        #expect(reports.count == cases.count)
        #expect(ran == runnable.count, "every runnable case must have produced a comparison")
        for report in reports where report.skipped == nil {
            #expect(report.a?.table != nil || report.a?.failure != nil, "\(report.caseID): path A left no result")
            #expect(report.b?.table != nil || report.b?.failure != nil, "\(report.caseID): path B left no result")
        }
        #expect(cachedTokens == 0, "the prompt cache served \(cachedTokens) tokens: it was not off")
        #expect(FileManager.default.fileExists(atPath: output.path))
        #expect(ParityRealModel.listing(of: source) == before, "the source checkpoint folder changed")
    }
}
