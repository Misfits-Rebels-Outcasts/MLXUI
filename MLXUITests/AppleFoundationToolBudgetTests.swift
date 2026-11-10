import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// MT4-1 — the 1,500-token tool budget is enforced at run time by `AppleFoundationRouter` (trimming from the end of the policy order).
@MainActor
@Suite(.serialized)
struct AppleFoundationToolBudgetTests {

    // MARK: pure trim

    /// 200 base + 300 per tool.
    private let linear: @Sendable ([String]) async -> Int? = { 200 + $0.count * 300 }

    @Test func aSetThatFitsIsKeptAsIs() async {
        let r = await AppleFoundationToolTrim.trim(tools: ["a", "b", "c"], protected: [], budget: 1_500, count: linear)
        #expect(r.kept == ["a", "b", "c"] && r.dropped.isEmpty && r.tokens == 1_100 && r.countCalls == 1)
    }

    @Test func toolsAreDroppedFromTheEndUntilItFits() async {
        let r = await AppleFoundationToolTrim.trim(tools: ["a", "b", "c", "d", "e", "f"], protected: [], budget: 1_500, count: linear)
        #expect(r.kept == ["a", "b", "c", "d"])
        #expect(r.dropped == ["f", "e"])
        #expect(r.tokens == 1_400 && r.tokensBefore == 2_000)
    }

    @Test func aProtectedToolIsNeverDropped() async {
        let r = await AppleFoundationToolTrim.trim(tools: ["a", "b", "list_capabilities"], protected: ["list_capabilities"], budget: 700, count: linear)
        #expect(r.kept == ["list_capabilities"] && r.dropped == ["b", "a"])
    }

    @Test func aSingleToolOverBudgetIsKept() async {
        let heavy: @Sendable ([String]) async -> Int? = { _ in 5_000 }
        let r = await AppleFoundationToolTrim.trim(tools: ["only"], protected: [], budget: 1_500, count: heavy)
        #expect(r.kept == ["only"] && r.dropped.isEmpty)
        let many = await AppleFoundationToolTrim.trim(tools: ["a", "b", "c"], protected: [], budget: 1_500, count: heavy)
        #expect(many.kept == ["a"])                     // the top-priority tool stays
    }

    @Test func whenCountingIsUnavailableNothingIsTrimmed() async {
        let none: @Sendable ([String]) async -> Int? = { _ in nil }
        let r = await AppleFoundationToolTrim.trim(tools: ["a", "b", "c", "d", "e", "f"], protected: [], budget: 1_500, count: none)
        #expect(r.kept.count == 6 && r.dropped.isEmpty && r.tokens == nil)
    }

    // MARK: the router

    nonisolated final class CountingChatter: AppleFoundationToolChatting, @unchecked Sendable {
        let contextSize: Int? = 8192
        private let lock = NSLock()
        private var calls = 0
        private(set) var toolSets: [[AppleFoundationToolSpec]] = []
        private(set) var instructions: [String?] = []
        let cost: (@Sendable (String?, [AppleFoundationToolSpec]) -> Int?)
        init(cost: @escaping @Sendable (String?, [AppleFoundationToolSpec]) -> Int?) { self.cost = cost }
        var countCalls: Int { lock.withLock { calls } }
        func makeSession(instructions: String?, tools: [AppleFoundationToolSpec],
                         handler: @escaping AppleFoundationToolHandler) -> any AppleFoundationToolSession {
            lock.withLock { toolSets.append(tools); self.instructions.append(instructions) }
            return AppleFoundationRouterTests.FakeSession(script: { _, _ in [.snapshot("ok")] }, handler: handler)
        }
        func toolTokenCount(instructions: String?, tools: [AppleFoundationToolSpec]) async -> Int? {
            lock.withLock { calls += 1 }
            return cost(instructions, tools)
        }
    }

    private func drain(_ stream: AsyncThrowingStream<RouterEvent, Error>) async throws {
        for try await _ in stream {}
    }

    private let sixTools: [any AgentTool] = (1...6).map { AppleFoundationRouterTests.NamedTool(name: "t\($0)") }
    private let linearCost: @Sendable (String?, [AppleFoundationToolSpec]) -> Int? = { i, t in 200 + t.count * 300 + (i?.count ?? 0) / 4 }

    @Test func theRouterDropsFromTheEndAndNamesTheDroppedToTheModel() async throws {
        let chatter = CountingChatter(cost: linearCost)
        let router = AppleFoundationRouter(chatter: chatter)
        try await drain(router.streamTurn(RouterTurn(instructions: "Be brief.", prompt: "hi", tools: sixTools)))
        #expect(chatter.toolSets.first?.map(\.name) == ["t1", "t2", "t3", "t4"])
        let text = try #require(chatter.instructions.first ?? nil)
        #expect(text.contains("not available in this chat") && text.contains("t5") && text.contains("t6"))
        let report = try #require(router.lastTrimReport)
        #expect(report.tokensBefore != nil && report.tokensAfter != nil && report.dropped == ["t6", "t5"])
        #expect((report.tokensAfter ?? 9_999) <= AppleFoundationToolBudget.tokenBudget)
    }

    @Test func aSetThatFitsIsSentWhole() async throws {
        let chatter = CountingChatter(cost: { _, _ in 900 })
        let router = AppleFoundationRouter(chatter: chatter)
        try await drain(router.streamTurn(RouterTurn(prompt: "hi", tools: sixTools)))
        #expect(chatter.toolSets.first?.count == 6)
        #expect(router.lastTrimReport?.dropped.isEmpty == true)
    }

    @Test func withoutTokenCountingOnlyTheCountCapApplies() async throws {
        let chatter = CountingChatter(cost: { _, _ in nil })
        let router = AppleFoundationRouter(chatter: chatter)
        try await drain(router.streamTurn(RouterTurn(prompt: "hi", tools: sixTools)))
        #expect(chatter.toolSets.first?.count == 6)
        #expect(router.lastTrimReport?.tokensAfter == nil)
    }

    @Test func listCapabilitiesIsNeverDropped() async throws {
        let chatter = CountingChatter(cost: linearCost)
        let router = AppleFoundationRouter(chatter: chatter)
        let tools: [any AgentTool] = ["a", "b", "c", "d", "e", "list_capabilities"].map { AppleFoundationRouterTests.NamedTool(name: $0) }
        try await drain(router.streamTurn(RouterTurn(prompt: "hi", tools: tools)))
        let names = chatter.toolSets.first?.map(\.name) ?? []
        #expect(names.contains("list_capabilities"))
        #expect((router.lastTrimReport?.tokensAfter ?? 9_999) <= AppleFoundationToolBudget.tokenBudget)
    }

    @Test func aRepeatedToolSetIsNotCountedAgain() async throws {
        let chatter = CountingChatter(cost: linearCost)
        let router = AppleFoundationRouter(chatter: chatter)
        try await drain(router.streamTurn(RouterTurn(instructions: "Be brief.", prompt: "one", tools: sixTools)))
        let first = chatter.countCalls
        #expect(first > 1)
        try await drain(router.streamTurn(RouterTurn(instructions: "Be brief.", prompt: "two", tools: sixTools)))
        #expect(chatter.countCalls == first)
    }

    // MARK: owner-run real measurements (Apple Intelligence ready + TEST_RUNNER_MLXUI_TEST_REAL_AFM=1)

    /// Wraps the real chatter, timing and recording the counts the router asks for.
    nonisolated final class SpyChatter: AppleFoundationToolChatting, @unchecked Sendable {
        let real: any AppleFoundationToolChatting
        private let lock = NSLock()
        private(set) var latencies: [Double] = []
        init(_ real: any AppleFoundationToolChatting) { self.real = real }
        var contextSize: Int? { real.contextSize }
        func makeSession(instructions: String?, tools: [AppleFoundationToolSpec],
                         handler: @escaping AppleFoundationToolHandler) -> any AppleFoundationToolSession {
            real.makeSession(instructions: instructions, tools: tools, handler: handler)
        }
        func toolTokenCount(instructions: String?, tools: [AppleFoundationToolSpec]) async -> Int? {
            let t0 = ContinuousClock.now
            let n = await real.toolTokenCount(instructions: instructions, tools: tools)
            let d = ContinuousClock.now - t0
            lock.withLock { latencies.append(Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18) }
            return n
        }
    }

    @Test(.enabled(if: AppleFoundationRouterTests.realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realTheShippedToolSetFitsTheBudgetForRepresentativePrompts() async throws {
        let edition: String = {
            #if DIRECT_BUILD
            return "Direct"
            #else
            return "AppStore"
            #endif
        }()
        let prompts = ["What is the capital of France?", "What time is it?", "Copy hello-mt3 to my clipboard",
                       "summarize https://example.com/a",
                       "Write a Swift function that merges two sorted arrays, explain its complexity, and show a unit test."]
        let spy = SpyChatter(try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true)))
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("mt41-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: store); AppleFoundationAvailability.toolChatterOverride = nil }
        AppleFoundationAvailability.toolChatterOverride = spy
        let runner = ModelRunner()
        runner.prepare(for: AppleIntelligenceChat.entry, in: store)
        print("MT41 \(edition) enabled tools (\(runner.enabledToolNames.count)):", runner.availableTools.map(\.name).filter { runner.enabledToolNames.contains($0) })
        for prompt in prompts {
            runner.newConversation(for: AppleIntelligenceChat.modelID)
            let armed = ModelRunner.exposedTools(runner.availableTools.filter { runner.enabledToolNames.contains($0.name) }, prompt: prompt).map(\.name)
            await runner.send(prompt, for: AppleIntelligenceChat.entry)
            let deadline = ContinuousClock.now.advanced(by: .seconds(120))
            while runner.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(50)) }
            let r = runner.appleRouter()?.lastTrimReport
            print("MT41 \(edition) | \(prompt.prefix(40)) | policy:", armed, "| before:", r?.tokensBefore as Any, "after:", r?.tokensAfter as Any,
                  "dropped:", r?.dropped as Any, "count calls:", r?.countCalls as Any, "trim secs:", r?.seconds as Any,
                  "| reply ok:", runner.errorMessage == nil)
            if let after = r?.tokensAfter { #expect(after <= AppleFoundationToolBudget.tokenBudget) }
            #expect(runner.errorMessage == nil)
        }
        print("MT41 count latencies (s):", spy.latencies)
    }

    /// Measurement-only mirrors of `Apps/Direct` (those files are not in the App Store test host): same name, description and
    /// parameters as the shipped Direct tools, so the schema costs the same tokens. Not used by any other test.
    nonisolated struct DirectMirror: AgentTool {
        let name: String
        let toolDescription: String
        let parameters: [ToolParameter]
        func execute(arguments: [String: JSONValue]) async throws -> String { "" }
    }

    static var directMirrors: [any AgentTool] {
        let path = ScopedFileTools.relativePathDescription
        return [
            DirectMirror(name: "run_shell", toolDescription: """
                Run a shell command on the user's Mac and return its exit code, stdout and stderr. The user must approve each command before it runs. Prefer a single non-interactive command; commands that wait for input will time out. Only the first 80 lines of output are returned, so filter in the command itself (head, tail, grep, wc -l) when a command could print a lot.
                """, parameters: [
                    .required("command", type: .string, description: "The command line to run, e.g. 'git status --short'."),
                    .optional("working_directory", type: .string, description: "Optional — omit this unless the command must run somewhere specific. Defaults to this chat's working folder when it has one, otherwise the user's home directory. If given, an absolute path inside the home directory."),
                    .optional("timeout_seconds", type: .int, description: "Optional — omit this to use the 60 second default. Maximum 600. The command is killed if it runs longer."),
                ]),
            DirectMirror(name: "read_file", toolDescription: "Read a UTF-8 text file from the user's Mac. Returns the file's contents, truncated at 256 KB.",
                         parameters: [.required("path", type: .string, description: path)]),
            DirectMirror(name: "write_file", toolDescription: "Write UTF-8 text to a file on the user's Mac, creating or overwriting it. The user must approve each write.",
                         parameters: [.required("path", type: .string, description: path), .required("contents", type: .string, description: "Text to write.")]),
            DirectMirror(name: "edit_file", toolDescription: "Replace exact text in an existing UTF-8 file on the user's Mac. old_string must match the file exactly, including spaces and indentation, and must appear once unless replace_all is true. Copy it from read_file. Use this instead of write_file to change part of a file. The user must approve each edit.",
                         parameters: [.required("path", type: .string, description: path),
                                      .required("old_string", type: .string, description: "Exact text to replace, copied from read_file."),
                                      .required("new_string", type: .string, description: "Text to put in its place; \"\" deletes it."),
                                      .optional("replace_all", type: .bool, description: "Replace every occurrence instead of requiring a unique one.")]),
            DirectMirror(name: "list_directory", toolDescription: "List the files and folders in a directory on the user's Mac.",
                         parameters: [.optional("path", type: .string, description: ScopedFileTools.optionalPathDescription)]),
        ]
    }

    @Test(.enabled(if: AppleFoundationRouterTests.realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realPerToolCostsAndTheDirectWorstCase() async throws {
        let chatter = try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true))
        let direct = Self.directMirrors
        let registry = ModelRunner.defaultTools().filter { t in !direct.contains { $0.name == t.name } }
        var costs: [(String, Int)] = []
        let base = try #require(await chatter.toolTokenCount(instructions: nil, tools: []))
        for tool in registry + direct {
            let n = try #require(await chatter.toolTokenCount(instructions: nil, tools: [AppleFoundationRouter.spec(for: tool)]))
            costs.append((tool.name, n - base))
        }
        print("MT41 per-tool cost (tokens, over an empty tool list of \(base)):", costs.map { "\($0.0)=\($0.1)" }.joined(separator: ", "))
        // The Direct default-enabled plain-prompt set (what the policy hands Apple Intelligence on 'capital of France').
        let names = ["embed_text", "summarize", "semantic_search", "transcribe_audio", "ocr_image", "read_file"]
        let pool = direct + registry
        let plain = names.compactMap { n in pool.first { $0.name == n } }
        let plainCount = try #require(await chatter.toolTokenCount(instructions: nil, tools: plain.map(AppleFoundationRouter.spec(for:))))
        print("MT41 Direct plain-prompt set (no instructions):", plainCount)
        // Worst case: the six most expensive tools of the whole registry + Direct tools, all enabled by the user.
        let heaviest = costs.sorted { $0.1 > $1.1 }.prefix(6).map(\.0)
        let worst = heaviest.compactMap { n in pool.first { $0.name == n } }
        let worstCount = try #require(await chatter.toolTokenCount(instructions: nil, tools: worst.map(AppleFoundationRouter.spec(for:))))
        print("MT41 Direct worst-case six (\(heaviest)):", worstCount)
        // And the trimmer on that worst case with the router's own budget.
        let trimmed = await AppleFoundationToolTrim.trim(tools: heaviest, protected: [], budget: AppleFoundationToolBudget.tokenBudget) { kept in
            await chatter.toolTokenCount(instructions: nil, tools: kept.compactMap { n in worst.first { $0.name == n } }.map(AppleFoundationRouter.spec(for:)))
        }
        print("MT41 trimmed worst case:", trimmed.kept, "dropped:", trimmed.dropped, "tokens:", trimmed.tokens as Any)
        #expect((trimmed.tokens ?? 0) <= AppleFoundationToolBudget.tokenBudget || trimmed.kept.count == 1)
    }
}
