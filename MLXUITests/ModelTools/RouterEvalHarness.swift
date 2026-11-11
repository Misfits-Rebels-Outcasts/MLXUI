import Foundation
import MLXLMCommon
@testable import MLXUI

// MT4-6 — the router eval harness (journal 2026-731).
//
// Runs any `ChatRouter` over the 30 cases in `RouterEvalCases.json` with the WORKERS MOCKED: every tool the policy offers is
// replaced by an `EvalMockTool` that records the call and returns canned output (it never touches the clipboard, the network,
// a file or a model). The turn handed to the router is built the way a Chat-page turn is: `ToolExposurePolicy` over the real
// default tool set (`ModelRunner.defaultTools()`), the model tools as `ModelTool(spec:, advertisesAssets: true)`, the real
// `ChatEnvironment` instructions with the asset notes.
//
// Four metrics per router (backlog MT4-6), plus the policy check. Nothing here changes a policy or a router.

// MARK: - Cases

nonisolated struct RouterEvalCase: Decodable, Sendable {
    struct Expected: Decodable, Sendable {
        let tool: String?
        var accept: [String]? = nil
        var args: [String: [String]]? = nil
    }
    let id: String
    let group: String
    let prompt: String
    var language: String? = nil
    /// Kinds attached this turn: "image", "audio", "text-oversized".
    var attached: [String]? = nil
    /// Kinds from earlier turns of the conversation.
    var earlier: [String]? = nil
    /// Overrides of the default status profile: "ready" | "needsDownload" | "cantRun" | "none".
    var statuses: [String: String]? = nil
    let expected: Expected
    var example: [String: JSONValue]? = nil
    var knownPolicyGap: String? = nil
    /// The gap also applies with the gate off (a generator cue, not a gated tool's).
    var gapWithoutGate: Bool? = nil

    var expectedTool: String? { expected.tool }
}

nonisolated struct RouterEvalSuite: Decodable, Sendable {
    let version: Int
    let cases: [RouterEvalCase]

    /// The checked-in file, next to this one (the tests read fixtures by `#filePath`, as the CatFlow goldens do).
    static func load(file: String = #filePath) throws -> RouterEvalSuite {
        let url = URL(fileURLWithPath: file).deletingLastPathComponent().appendingPathComponent("RouterEvalCases.json")
        return try JSONDecoder().decode(RouterEvalSuite.self, from: Data(contentsOf: url))
    }
}

// MARK: - Setup (which router configuration the policy sees)

nonisolated struct RouterEvalSetup: Sendable {
    let name: String
    let router: ToolExposurePolicy.Router
    let gateAmbiguousTools: Bool
    /// Install answer the mocked host gives: `false` = the user declines the download (the default).
    var acceptInstalls = false

    /// What Chat sends Apple Intelligence today: budget 6, gate on.
    static let appleIntelligence = RouterEvalSetup(name: "Apple Intelligence", router: .appleIntelligence, gateAmbiguousTools: true)
    /// What Chat sends an MLX router today: budget 10, gate off.
    static let mlxToday = RouterEvalSetup(name: "MLX (gate off, today)", router: .mlx(hasVision: false), gateAmbiguousTools: false)
    /// The measurement the lead asked for: the same MLX router with `gateAmbiguousTools` on.
    static let mlxGated = RouterEvalSetup(name: "MLX (gate on)", router: .mlx(hasVision: false), gateAmbiguousTools: true)
}

// MARK: - Recorder and mocked workers

nonisolated final class EvalRecorder: @unchecked Sendable {
    struct Call: Sendable { let name: String; let arguments: [String: JSONValue] }
    private let lock = NSLock()
    private var _calls: [Call] = []
    private var _installAsks: [String] = []
    private var _executed: [String] = []
    var calls: [Call] { lock.withLock { _calls } }
    var installAsks: [String] { lock.withLock { _installAsks } }
    /// Tools whose mocked worker "ran" (not asked-and-declined).
    var executed: [String] { lock.withLock { _executed } }
    func record(_ name: String, _ arguments: [String: JSONValue]) { lock.withLock { _calls.append(Call(name: name, arguments: arguments)) } }
    func askInstall(_ name: String) { lock.withLock { _installAsks.append(name) } }
    func ran(_ name: String) { lock.withLock { _executed.append(name) } }
}

/// Stands in for a tool: same name, description and parameters (so the router sees the real schema), no real work.
nonisolated struct EvalMockTool: AgentTool {
    let base: any AgentTool
    let recorder: EvalRecorder
    /// The tool's model is not installed: the host would put the install card up before running.
    let needsDownload: Bool
    let acceptInstall: Bool

    var name: String { base.name }
    var toolDescription: String { base.toolDescription }
    var parameters: [ToolParameter] { base.parameters }
    var requiresApproval: Bool { false }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        recorder.record(name, arguments)
        if needsDownload {
            recorder.askInstall(name)
            if !acceptInstall { return ModelToolHost.declineSentence(for: "the model for \(name)") }
        }
        recorder.ran(name)
        return Self.output(for: name)
    }

    /// Canned, informative, short results (a vague one makes a model re-call the tool).
    static func output(for tool: String) -> String {
        switch tool {
        case "calculator": return "397.8"
        case "datetime": return "2026-11-11 14:05 (Wednesday), time zone Asia/Kuala_Lumpur"
        case "read_clipboard": return "Clipboard text: \"Quarterly report draft v3\""
        case "write_clipboard": return "Copied to the clipboard."
        case "fetch_url": return "Page text: Plans: Basic $5 per month, Pro $12 per month."
        case "run_javascript": return "2,4,6"
        case "embed_text": return "Cosine similarity 0.31."
        case "summarize": return "Summary: the committee cut travel spending, kept training and postponed the lease vote."
        case "semantic_search": return "Top match (0.82), notes/trip.md: Trip budget is 1,200 total."
        case "ocr_image": return "Recognised text: TOTAL 42.00, 3 items."
        case "describe_image": return "A busy street market at dusk."
        case "transcribe_audio": return "Transcript: Welcome everyone. First item is the budget."
        case "translate": return "Selamat pagi, apa khabar?"
        case "speak": return "Saved audio as asset a2 (audio, 3 s)."
        case "generate_image": return "Saved image as asset a2 (image)."
        case "generate_music": return "Saved audio as asset a2 (audio, 10 s)."
        case "upscale_image", "segment_image", "estimate_depth": return "Saved image as asset a2 (image)."
        case "list_capabilities": return "ocr_image: ready. transcribe_audio: ready. speak: ready. describe_image: needs a download."
        case "read_asset": return "Characters 0-1200 of 48,000: Refund policy. Refunds are issued within 14 days."
        default: return "Done."
        }
    }
}

// MARK: - The turn the router is handed

nonisolated struct RouterEvalPlan: Sendable {
    let testCase: RouterEvalCase
    let setup: RouterEvalSetup
    let policy: ToolExposurePolicy.Output
    let tools: [any AgentTool]
    let instructions: String?
    let recorder: EvalRecorder

    var offered: [String] { policy.tools }
    var expectedOffered: Bool { testCase.expectedTool.map { offered.contains($0) } ?? true }

    func turn() -> RouterTurn {
        RouterTurn(instructions: instructions, history: .folded([]), prompt: testCase.prompt, tools: tools)
    }
}

@MainActor
enum RouterEvalPlanner {

    /// The default profile of a typical Mac: the OCR, ASR and TTS models installed, every other model tool a download away.
    static let defaultStatuses: [String: String] = [
        "ocr_image": "ready", "transcribe_audio": "ready", "speak": "ready",
        "describe_image": "needsDownload", "translate": "needsDownload", "generate_image": "needsDownload",
        "upscale_image": "needsDownload", "segment_image": "needsDownload", "estimate_depth": "needsDownload",
        "generate_music": "needsDownload",
    ]

    static func statuses(for testCase: RouterEvalCase) -> [String: ModelToolStatus] {
        var raw = defaultStatuses
        for (tool, status) in testCase.statuses ?? [:] { raw[tool] = status }
        var out: [String: ModelToolStatus] = [:]
        for (tool, status) in raw {
            switch status {
            case "ready": out[tool] = .ready(dummyModel)
            case "needsDownload": out[tool] = .needsDownload(dummyModel, gb: 2.0)
            case "cantRun": out[tool] = .cantRun(reason: "Needs about 24.0 GB of memory, and this Mac has 19.0 GB.")
            default: out[tool] = ModelToolStatus.none
            }
        }
        return out
    }

    static func plan(_ testCase: RouterEvalCase, setup: RouterEvalSetup, recorder: EvalRecorder = EvalRecorder()) -> RouterEvalPlan {
        let statuses = statuses(for: testCase)
        let attached = Set(testCase.attached ?? [])
        func kinds(_ names: [String]) -> Set<Kind> {
            Set(names.compactMap { name -> Kind? in
                switch name {
                case "image": return .image
                case "audio": return .audio
                default: return nil
                }
            })
        }
        let turnKinds = kinds(testCase.attached ?? [])
        let conversationKinds = turnKinds.union(kinds(testCase.earlier ?? []))

        let allTools = ModelRunner.defaultTools()
        let enabledNames = ModelRunner.defaultEnabledToolNames()
        // MT4-6b: `echo` is the DEBUG-only demo tool; leave it out so a Debug run sees the offered set a Release build has.
        let enabled = allTools.filter { enabledNames.contains($0.name) && $0.name != "echo" }
        let policy = ToolExposurePolicy.select(ToolExposurePolicy.Input(
            router: setup.router, modelToolsEnabled: true,
            conversationAssetKinds: conversationKinds, turnAssetKinds: turnKinds,
            hasOversizedTextAsset: attached.contains("text-oversized"),
            prompt: testCase.prompt, statuses: statuses, enabledTools: enabled.map(\.name),
            gateAmbiguousTools: setup.gateAmbiguousTools))

        // The same name-to-tool mapping `ChatToolSet.select` applies (a model-tool name is a `ModelTool` built for a host-bound chat).
        let byName = Dictionary(enabled.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        let tools: [any AgentTool] = policy.tools.compactMap { name in
            let base: (any AgentTool)?
            if let spec = ModelToolSpec.spec(named: name) { base = ModelTool(spec: spec, advertisesAssets: true) } else { base = byName[name] }
            guard let base else { return nil }
            var needsDownload = false
            if case .needsDownload? = statuses[name], ModelToolSpec.spec(named: name)?.task != nil { needsDownload = true }
            return EvalMockTool(base: base, recorder: recorder, needsDownload: needsDownload, acceptInstall: setup.acceptInstalls)
        }

        let assetLines = assetNote(for: testCase)
        #if DIRECT_BUILD
        let edition = FileScope.Edition.direct
        #else
        let edition = FileScope.Edition.appStore
        #endif
        let scope = FileScope.forConversation(attached: [], workingFolder: nil, grants: [], edition: edition)
        let instructions = ChatEnvironment.instructions(
            scope: scope, edition: edition, fileToolsEnabled: enabledNames, writeEnabled: false, assetNote: assetLines)
        return RouterEvalPlan(testCase: testCase, setup: setup, policy: policy, tools: tools, instructions: instructions, recorder: recorder)
    }

    private static func assetNote(for testCase: RouterEvalCase) -> String? {
        let files: [(String, String, String)] = (testCase.attached ?? []).compactMap { kind in
            switch kind {
            case "image": return ("receipt.jpg", "image", "a1")
            case "audio": return ("meeting.m4a", "audio", "a1")
            case "text-oversized": return ("policy.txt", "text", "a1")
            default: return nil
            }
        }
        let attachments = files.map { name, kind, handle in
            ChatAttachment(name: name, storedPath: "/private/tmp/\(name)", mode: .reference, byteCount: 100_000,
                           assetHandle: handle, assetKind: kind)
        }
        return ChatAttachment.assetNote(for: attachments, assets: nil)
    }

    static let dummyModel = ModelEntry(
        id: "eval--dummy", family: "Eval", displayName: "Eval Model", paramSize: "1B", paramCountB: 1, modelType: .llm,
        source: .mlx, format: "mlx", platforms: ["macos"], minMacOSVersion: nil, hfRepo: "", hfModelId: "", ramGB: 2,
        downloadSizeGB: 2, contextWindow: 8_192, license: nil, licenseUrl: nil, description: nil, summary: nil,
        descriptionSource: nil, architecture: "Eval", languages: nil, lastUpdated: nil, taskTags: nil,
        benchmarks: nil, speedTokensPerSec: nil, speedHardware: nil, speedEstimated: nil, communityDownloads: nil,
        communityLikes: nil, variants: nil)
}

// MARK: - Validation and matching

nonisolated enum RouterEvalValidator {
    /// `nil` = the arguments are valid for `tool`; else the sentence a router would be given back. A model tool is checked
    /// against its spec (names, types, ranges) and must carry one input (`asset`, `path`, `url` or `text`) when it takes one.
    static func problem(_ arguments: [String: JSONValue], for tool: any AgentTool) -> String? {
        let base = (tool as? EvalMockTool)?.base ?? tool
        if let model = base as? ModelTool {
            if case .failure(let failure) = ModelToolRunner.validate(arguments, for: model.spec) { return failure.text }
            let inputs = ModelToolRunner.inputArgumentNames(for: model.spec)
            if !inputs.isEmpty, !inputs.contains(where: { arguments[$0].map(isPresent) ?? false }) {
                return "\(model.spec.name) needs an input: \(inputs.joined(separator: ", "))."
            }
            return nil
        }
        if case .failure(let failure) = AppleFoundationRouter.ToolBridge.validate(arguments, against: base) { return failure.description }
        return nil
    }

    private static func isPresent(_ value: JSONValue) -> Bool {
        switch value {
        case .null: return false
        case .string(let s): return !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        default: return true
        }
    }

    static func text(of value: JSONValue) -> String {
        switch value {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return d == d.rounded() ? String(Int(d)) : String(d)
        case .bool(let b): return String(b)
        default: return ""
        }
    }

    /// Every expected key is present and its value contains one of the acceptable substrings (case-insensitive).
    static func keyArgsMatch(_ arguments: [String: JSONValue], expected: [String: [String]]?) -> Bool {
        guard let expected else { return true }
        for (key, options) in expected {
            guard let value = arguments[key] else { return false }
            let got = text(of: value).lowercased()
            if !options.isEmpty, !options.contains(where: { got.contains($0.lowercased()) }) { return false }
        }
        return true
    }
}

// MARK: - Running one router

nonisolated struct RouterEvalOutcome: Sendable {
    let id: String
    let group: String
    let language: String?
    let expectedTool: String?
    let offered: [String]
    let expectedOffered: Bool
    let knownPolicyGap: String?
    let calls: [EvalRecorder.Call]
    let text: String
    let error: String?
    let installAsks: [String]
    let expectedNeedsDownload: Bool
    let seconds: Double
    let firstEventSeconds: Double?
    let toolChoiceOK: Bool
    /// One entry per tool call: nil = valid, else the problem.
    let argProblems: [String?]
    /// Nil when no key arguments were expected (or the wrong tool was chosen).
    let keyArgsOK: Bool?

    var firstTool: String? { calls.first?.name }
    var answeredDirectly: Bool { calls.isEmpty && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
}

enum RouterEvalRunner {

    @MainActor
    static func run(router: any ChatRouter, setup: RouterEvalSetup, cases: [RouterEvalCase],
                    progress: ((RouterEvalOutcome) -> Void)? = nil) async -> [RouterEvalOutcome] {
        var outcomes: [RouterEvalOutcome] = []
        for testCase in cases {
            let outcome = await runOne(router: router, setup: setup, testCase: testCase)
            progress?(outcome)
            outcomes.append(outcome)
        }
        return outcomes
    }

    @MainActor
    static func runOne(router: any ChatRouter, setup: RouterEvalSetup, testCase: RouterEvalCase) async -> RouterEvalOutcome {
        let recorder = EvalRecorder()
        let plan = RouterEvalPlanner.plan(testCase, setup: setup, recorder: recorder)
        let clock = ContinuousClock()
        let start = clock.now
        var firstEvent: Double?
        var started: [EvalRecorder.Call] = []
        var text = ""
        var error: String?
        do {
            for try await event in router.streamTurn(plan.turn()) {
                if firstEvent == nil { firstEvent = seconds(since: start, clock) }
                switch event {
                case .text(let delta): text += delta
                case .toolStarted(let name, let arguments): started.append(.init(name: name, arguments: arguments))
                case .contextExceeded(let limit): error = "context exceeded (limit \(limit.map(String.init) ?? "unknown"))"
                default: break
                }
            }
        } catch {
            // The error is an outcome of the turn, not a harness failure.
            let message = (error as? LocalizedError)?.errorDescription ?? String(describing: error)
            return finish(plan, started, text, message, firstEvent, seconds(since: start, clock))
        }
        return finish(plan, started, text, error, firstEvent, seconds(since: start, clock))
    }

    private static func seconds(since start: ContinuousClock.Instant, _ clock: ContinuousClock) -> Double {
        let d = start.duration(to: clock.now)
        return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }

    @MainActor
    private static func finish(_ plan: RouterEvalPlan, _ calls: [EvalRecorder.Call], _ text: String, _ error: String?,
                               _ firstEvent: Double?, _ seconds: Double) -> RouterEvalOutcome {
        let testCase = plan.testCase
        let first = calls.first?.name
        let accept = Set(testCase.expected.accept ?? [])
        let toolChoiceOK: Bool
        if let expected = testCase.expectedTool {
            toolChoiceOK = first == expected || (first.map(accept.contains) ?? accept.contains("none"))
        } else {
            toolChoiceOK = calls.isEmpty
        }
        let byName = Dictionary(plan.tools.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        let problems: [String?] = calls.map { call in
            guard let tool = byName[call.name] else { return "\(call.name) was not offered this turn." }
            return RouterEvalValidator.problem(call.arguments, for: tool)
        }
        var keyArgsOK: Bool?
        if let expected = testCase.expectedTool, first == expected, testCase.expected.args != nil, let firstCall = calls.first {
            keyArgsOK = RouterEvalValidator.keyArgsMatch(firstCall.arguments, expected: testCase.expected.args)
        }
        var expectedNeedsDownload = false
        if let expected = testCase.expectedTool, case .needsDownload? = RouterEvalPlanner.statuses(for: testCase)[expected],
           ModelToolSpec.spec(named: expected)?.task != nil { expectedNeedsDownload = true }
        return RouterEvalOutcome(
            id: testCase.id, group: testCase.group, language: testCase.language, expectedTool: testCase.expectedTool,
            offered: plan.offered, expectedOffered: plan.expectedOffered, knownPolicyGap: testCase.knownPolicyGap,
            calls: calls, text: text, error: error, installAsks: plan.recorder.installAsks,
            expectedNeedsDownload: expectedNeedsDownload, seconds: seconds, firstEventSeconds: firstEvent,
            toolChoiceOK: toolChoiceOK, argProblems: problems, keyArgsOK: keyArgsOK)
    }
}

// MARK: - Report

nonisolated struct RouterEvalReport: Sendable {
    let routerName: String
    let outcomes: [RouterEvalOutcome]

    struct Rate: Sendable {
        let hits: Int
        let total: Int
        var percent: Double { total == 0 ? 0 : Double(hits) * 100 / Double(total) }
        var text: String { total == 0 ? "n/a" : String(format: "%d/%d (%.0f%%)", hits, total, percent) }
    }

    /// Tool-choice accuracy over every case (a "no tool" case is right when no tool was called).
    var toolChoice: Rate { Rate(hits: outcomes.filter(\.toolChoiceOK).count, total: outcomes.count) }
    /// Share of tool calls whose arguments are valid against the tool's schema.
    var validArguments: Rate {
        let all = outcomes.flatMap(\.argProblems)
        return Rate(hits: all.filter { $0 == nil }.count, total: all.count)
    }
    /// Of the correct tool choices that had key arguments to check, how many carried them.
    var keyArguments: Rate {
        let checked = outcomes.compactMap(\.keyArgsOK)
        return Rate(hits: checked.filter { $0 }.count, total: checked.count)
    }
    /// Of the cases that needed no tool: answered directly, with text and no tool call.
    var answeredWithoutTool: Rate {
        let none = outcomes.filter { $0.expectedTool == nil }
        return Rate(hits: none.filter(\.answeredDirectly).count, total: none.count)
    }
    /// Of the cases whose right tool needs a download: the router called it, so the host asked first.
    var askedBeforeDownload: Rate {
        let needs = outcomes.filter(\.expectedNeedsDownload)
        return Rate(hits: needs.filter { !$0.installAsks.isEmpty }.count, total: needs.count)
    }
    /// Install cards a router would have raised for a tool the case did not call for.
    var unneededInstallAsks: Int {
        outcomes.reduce(0) { total, outcome in
            total + outcome.installAsks.filter { $0 != outcome.expectedTool }.count
        }
    }
    /// Policy check: the offered set contains the expected tool.
    var policyOffered: Rate { Rate(hits: outcomes.filter(\.expectedOffered).count, total: outcomes.count) }
    var errors: [RouterEvalOutcome] { outcomes.filter { $0.error != nil } }

    func byGroup() -> [(group: String, rate: Rate)] {
        let groups = Array(Set(outcomes.map(\.group))).sorted()
        return groups.map { g in
            let rows = outcomes.filter { $0.group == g }
            return (g, Rate(hits: rows.filter(\.toolChoiceOK).count, total: rows.count))
        }
    }

    func render(detail: Bool = true) -> String {
        var lines = ["ROUTER-EVAL \(routerName)",
                     "  tool choice            \(toolChoice.text)",
                     "  valid arguments        \(validArguments.text)",
                     "  key arguments          \(keyArguments.text)",
                     "  answered, no tool      \(answeredWithoutTool.text)",
                     "  asked before download  \(askedBeforeDownload.text)   (install cards for a tool not called for: \(unneededInstallAsks))",
                     "  policy offered         \(policyOffered.text)",
                     "  by group               " + byGroup().map { "\($0.group) \($0.rate.text)" }.joined(separator: "; ")]
        let secs = outcomes.map(\.seconds)
        if !secs.isEmpty {
            let sorted = secs.sorted()
            lines.append(String(format: "  seconds per case       median %.2f, max %.2f", sorted[sorted.count / 2], sorted.last ?? 0))
        }
        if detail {
            for o in outcomes {
                let calls = o.calls.map { call in
                    call.name + "(" + call.arguments.keys.sorted().map { "\($0)=\(RouterEvalValidator.text(of: call.arguments[$0] ?? .null).prefix(24))" }.joined(separator: ",") + ")"
                }.joined(separator: " > ")
                let flags = [o.toolChoiceOK ? "ok" : "WRONG",
                             o.expectedOffered ? nil : "not-offered",
                             o.argProblems.contains { $0 != nil } ? "bad-args" : nil,
                             o.keyArgsOK == false ? "key-args-miss" : nil,
                             o.error.map { "error: \($0)" }].compactMap { $0 }.joined(separator: " ")
                lines.append("  \(o.id) [\(o.group)] expected \(o.expectedTool ?? "none") | got \(calls.isEmpty ? "no tool" : calls) | \(flags)")
            }
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Scripted routers (harness self-tests, no model)

/// A router that decides from the offered tools. `decide` returns the call to make (or nil to answer directly).
nonisolated struct ScriptedEvalRouter: ChatRouter {
    let identity = RouterIdentity(displayName: "Scripted", kind: .system, contextSize: nil)
    let decide: @Sendable (RouterTurn) -> (name: String, arguments: [String: JSONValue])?

    func streamTurn(_ turn: RouterTurn) -> AsyncThrowingStream<RouterEvent, Error> {
        let decide = decide
        return AsyncThrowingStream { continuation in
            let task = Task {
                if let call = decide(turn), let tool = turn.tools.first(where: { $0.name == call.name }) {
                    continuation.yield(.toolStarted(name: call.name, arguments: call.arguments))
                    let result = (try? await tool.execute(arguments: call.arguments)) ?? "error"
                    continuation.yield(.toolFinished(name: call.name, outcome: .ok(result)))
                    continuation.yield(.text("Done: \(result.prefix(40))"))
                } else {
                    continuation.yield(.text("I answered without a tool."))
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// The perfect router given what it is offered: calls the expected tool with the case's example arguments when the policy
    /// offered it, answers directly otherwise.
    static func oracle(cases: [RouterEvalCase]) -> ScriptedEvalRouter {
        let byPrompt = Dictionary(cases.map { ($0.prompt, $0) }, uniquingKeysWith: { first, _ in first })
        return ScriptedEvalRouter { turn in
            guard let testCase = byPrompt[turn.prompt], let tool = testCase.expectedTool else { return nil }
            return (tool, testCase.example ?? [:])
        }
    }
}

// MARK: - Router reload time (plan section 5)

/// Real-MLX only (owner-run, see `RouterEvalRealTests`): what it costs to bring a router back after a worker evicted it.
/// The sequence is: load the container (cold for this process), run one warm turn, release it, load it again, run one turn.
/// `reload` is the second load; `firstEventAfterReload` is what the user waits for on the next turn. The second load is
/// page-cache warm unless the owner purges the cache first (`sudo purge`), so a cold figure is a separate owner reading.
nonisolated struct RouterReloadMeasurement: Sendable {
    let directory: String
    let firstLoadSeconds: Double
    let warmFirstEventSeconds: Double?
    let reloadSeconds: Double
    let firstEventAfterReloadSeconds: Double?

    var line: String {
        String(format: "ROUTER-RELOAD %@: first load %.2f s, warm first event %@, reload after eviction %.2f s, first event after reload %@",
               (directory as NSString).lastPathComponent, firstLoadSeconds,
               warmFirstEventSeconds.map { String(format: "%.2f s", $0) } ?? "n/a", reloadSeconds,
               firstEventAfterReloadSeconds.map { String(format: "%.2f s", $0) } ?? "n/a")
    }
}
