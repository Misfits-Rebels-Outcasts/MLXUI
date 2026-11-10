import Foundation
import Testing
import FoundationModels
import MLXLMCommon
@testable import MLXUI

/// MT3-2 — `AppleFoundationRouter`, entirely fixture-driven (no device): the seam is replaced by a scripted chatter.
/// The owner-run real tests at the bottom run only with `TEST_RUNNER_MLXUI_TEST_REAL_AFM=1` and Apple Intelligence ready.
@MainActor
@Suite(.serialized)
struct AppleFoundationRouterTests {

    // MARK: fixtures

    nonisolated struct EchoTool: AgentTool {
        let name = "echo"
        let toolDescription = "Echo the text back."
        let parameters: [ToolParameter] = [
            .required("text", type: .string, description: "text"),
            .optional("times", type: .int, description: "how many times"),
        ]
        func execute(arguments: [String: JSONValue]) async throws -> String {
            let n: Int
            if case .int(let i)? = arguments["times"] { n = i } else { n = 1 }
            return Array(repeating: "echo: \(arguments.string("text") ?? "")", count: n).joined(separator: " ")
        }
    }
    nonisolated struct GatedTool: AgentTool {
        let name = "gated"
        let toolDescription = "Needs approval."
        let parameters: [ToolParameter] = []
        var requiresApproval: Bool { true }
        func execute(arguments: [String: JSONValue]) async throws -> String { "ran" }
    }
    nonisolated struct BigTool: AgentTool {
        let name = "big"
        let toolDescription = "Returns a lot."
        let parameters: [ToolParameter] = []
        let size: Int
        func execute(arguments: [String: JSONValue]) async throws -> String { String(repeating: "x", count: size) }
    }
    nonisolated struct EmptyTool: AgentTool {
        let name = "empty"
        let toolDescription = "Returns nothing."
        let parameters: [ToolParameter] = []
        func execute(arguments: [String: JSONValue]) async throws -> String { "" }
    }
    nonisolated struct ThrowingTool: AgentTool {
        struct Boom: LocalizedError { var errorDescription: String? { "it broke" } }
        let name = "boom"
        let toolDescription = "Always fails."
        let parameters: [ToolParameter] = []
        func execute(arguments: [String: JSONValue]) async throws -> String { throw Boom() }
    }
    nonisolated struct ArrayTool: AgentTool {
        let name = "arr"
        let toolDescription = "Takes a list."
        let parameters: [ToolParameter] = [.required("items", type: .array(elementType: .string), description: "items")]
        func execute(arguments: [String: JSONValue]) async throws -> String { "" }
    }
    nonisolated struct NamedTool: AgentTool {
        let name: String
        let toolDescription = "A tool."
        let parameters: [ToolParameter] = []
        func execute(arguments: [String: JSONValue]) async throws -> String { name }
    }

    typealias Script = @Sendable (_ prompt: String, _ handler: AppleFoundationToolHandler) async throws -> [AppleFoundationChatEvent]

    /// A session that runs `script`, yields its events, and counts overlapping `respond`s (a programmer error in the SDK).
    nonisolated final class FakeSession: AppleFoundationToolSession, @unchecked Sendable {
        let script: Script
        let handler: AppleFoundationToolHandler
        private let lock = NSLock()
        private var running = 0
        private(set) var overlaps = 0
        private(set) var prompts: [String] = []
        init(script: @escaping Script, handler: @escaping AppleFoundationToolHandler) { self.script = script; self.handler = handler }
        func respond(prompt: String) -> AsyncThrowingStream<AppleFoundationChatEvent, Error> {
            lock.withLock { prompts.append(prompt); if running > 0 { overlaps += 1 }; running += 1 }
            let (stream, continuation) = AsyncThrowingStream<AppleFoundationChatEvent, Error>.makeStream()
            let task = Task {
                do {
                    for event in try await script(prompt, handler) { continuation.yield(event) }
                    lock.withLock { running -= 1 }
                    continuation.finish()
                } catch {
                    lock.withLock { running -= 1 }
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
            return stream
        }
    }

    nonisolated final class FakeChatter: AppleFoundationToolChatting, @unchecked Sendable {
        let contextSize: Int? = 8192
        var script: Script
        private let lock = NSLock()
        private(set) var sessions: [FakeSession] = []
        private(set) var instructions: [String?] = []
        private(set) var toolSets: [[AppleFoundationToolSpec]] = []
        init(script: @escaping Script = { _, _ in [.snapshot("ok")] }) { self.script = script }
        func makeSession(instructions: String?, tools: [AppleFoundationToolSpec],
                         handler: @escaping AppleFoundationToolHandler) -> any AppleFoundationToolSession {
            let s = FakeSession(script: script, handler: handler)
            lock.withLock { sessions.append(s); self.instructions.append(instructions); toolSets.append(tools) }
            return s
        }
        func toolTokenCount(instructions: String?, tools: [AppleFoundationToolSpec]) async -> Int? { nil }
        var sessionCount: Int { lock.withLock { sessions.count } }
    }

    nonisolated final class Box<T>: @unchecked Sendable {
        private let lock = NSLock(); private var items: [T] = []
        func add(_ t: T) { lock.withLock { items.append(t) } }
        var all: [T] { lock.withLock { items } }
    }

    private func collect(_ stream: AsyncThrowingStream<RouterEvent, Error>) async throws -> [RouterEvent] {
        var out: [RouterEvent] = []
        for try await e in stream { out.append(e) }
        return out
    }

    private func callTool(_ name: String, _ args: [String: JSONValue] = [:]) -> Script {
        { _, handler in
            let result = await handler(name, args)
            return [.snapshot("got: \(result)")]
        }
    }

    // MARK: schema adapter

    @Test func everyV1ModelToolIsExpressibleAsASeamSpec() {
        let tools: [any AgentTool] = ModelTool.all(advertisingAssets: true)
        let selection = AppleFoundationRouter.select(tools, limit: 100)
        #expect(selection.dropped.isEmpty)
        let spec = AppleFoundationRouter.spec(for: ModelTool(spec: ModelToolSpec.spec(named: "generate_image")!, advertisesAssets: true))
        #expect(spec.name == "generate_image")
        #expect(spec.properties.first { $0.name == "steps" }?.kind == .integer)
        #expect(spec.properties.first { $0.name == "asset" }?.kind == .string)
        #expect(spec.properties.allSatisfy { !$0.required })
    }

    @Test func aToolWithAnArrayParameterIsLeftOutAndNamed() {
        let selection = AppleFoundationRouter.select([EchoTool(), ArrayTool()], limit: 6)
        #expect(selection.included.map(\.name) == ["echo"])
        #expect(selection.dropped == ["arr"])
    }

    // MARK: the six-tool cap and the budget

    @Test func atMostSixToolsReachTheModelAndTheExtrasAreNamedInOrder() {
        let tools: [any AgentTool] = (1...9).map { NamedTool(name: "t\($0)") }
        let selection = AppleFoundationRouter.select(tools, limit: 6)
        #expect(selection.included.map(\.name) == ["t1", "t2", "t3", "t4", "t5", "t6"])
        #expect(selection.dropped == ["t7", "t8", "t9"])
    }

    @Test func theRouterSendsSixOfTwelveAndTellsTheModelWhichAreMissing() async throws {
        let chatter = FakeChatter()
        let router = AppleFoundationRouter(chatter: chatter)
        let tools: [any AgentTool] = ModelTool.all(advertisingAssets: true)
        _ = try await collect(router.streamTurn(RouterTurn(instructions: "Be brief.", prompt: "hi", tools: tools)))
        #expect(chatter.toolSets.first?.count == 6)
        #expect(chatter.toolSets.first?.map(\.name) == tools.prefix(6).map(\.name))
        let text = try #require(chatter.instructions.first ?? nil)
        #expect(text.hasPrefix("Be brief."))
        #expect(text.contains("not available in this chat"))
        #expect(text.contains("read_asset"))
    }

    @Test func aCallerCannotRaiseTheCapPastSix() async throws {
        let chatter = FakeChatter()
        let router = AppleFoundationRouter(chatter: chatter, maxTools: 20)
        let tools: [any AgentTool] = (1...9).map { NamedTool(name: "t\($0)") }
        _ = try await collect(router.streamTurn(RouterTurn(prompt: "hi", tools: tools)))
        #expect(chatter.toolSets.first?.count == 6)
    }

    @Test func theBudgetHelperAndThePinnedWorstCase() {
        #expect(AppleFoundationToolBudget.tokenBudget == 1_500)
        #expect(AppleFoundationToolBudget.maxTools == 6)
        #expect(AppleFoundationToolBudget.fits(tokens: AppleFoundationToolBudget.measuredWorstCaseSixToolTokens))
        #expect(AppleFoundationToolBudget.headroom(tokens: 1_099) == 401)
        #expect(!AppleFoundationToolBudget.fits(tokens: 1_944))      // the whole table does not, which is why the cap exists
        #expect(ModelToolSpec.all.count + 1 > AppleFoundationToolBudget.maxTools)   // +1: date_time
    }

    // MARK: argument validation (corrective strings, never throws)

    private func runOneCall(tool: any AgentTool, name: String, args: [String: JSONValue],
                            approve: @escaping @Sendable (any AgentTool, [String: JSONValue]) async -> Bool = { _, _ in true },
                            maxToolCalls: Int = 6) async throws -> (events: [RouterEvent], toolEvents: [AgentToolEvent], reply: String) {
        let chatter = FakeChatter(script: callTool(name, args))
        let router = AppleFoundationRouter(chatter: chatter, maxToolCalls: maxToolCalls)
        let log = Box<AgentToolEvent>()
        let events = try await collect(router.streamTurn(
            RouterTurn(prompt: "go", tools: [tool], approve: approve, onToolEvent: { log.add($0) })))
        var reply = ""
        for case .text(let t) in events { reply += t }
        return (events, log.all, reply)
    }

    @Test func aGoodCallRunsRaisesEventsFromInsideTheCallAndReturnsTheOutput() async throws {
        let r = try await runOneCall(tool: EchoTool(), name: "echo", args: ["text": .string("a"), "times": .int(2)])
        #expect(r.events == [
            .toolStarted(name: "echo", arguments: ["text": .string("a"), "times": .int(2)]),
            .toolFinished(name: "echo", outcome: .ok("echo: a echo: a")),
            .text("got: echo: a echo: a"),
        ])
        #expect(r.toolEvents.count == 2)
    }

    @Test func aWholeNumberSentAsADoubleIsAccepted() async throws {
        let r = try await runOneCall(tool: EchoTool(), name: "echo", args: ["text": .string("a"), "times": .double(2)])
        #expect(r.reply == "got: echo: a echo: a")
    }

    @Test func aNullOptionalIsTreatedAsAbsent() async throws {
        let r = try await runOneCall(tool: EchoTool(), name: "echo", args: ["text": .string("a"), "times": .null])
        #expect(r.reply == "got: echo: a")
    }

    @Test func aMissingRequiredArgumentIsACorrectiveStringNotAThrow() async throws {
        let r = try await runOneCall(tool: EchoTool(), name: "echo", args: [:])
        #expect(r.reply.hasPrefix("got: Error: echo needs the \"text\" argument"))
        #expect(r.events.contains(.toolFinished(name: "echo", outcome: .failed("echo needs the \"text\" argument (text)"))))
    }

    @Test func anUnknownArgumentNamesTheValidOnes() async throws {
        let r = try await runOneCall(tool: EchoTool(), name: "echo", args: ["text": .string("a"), "volume": .int(3)])
        #expect(r.reply.contains("unknown argument \"volume\" for echo. Valid arguments: text, times."))
    }

    @Test func aWrongTypeSaysWhatIsExpected() async throws {
        let r = try await runOneCall(tool: EchoTool(), name: "echo", args: ["text": .string("a"), "times": .string("two")])
        #expect(r.reply.contains("times must be a whole number."))
        let r2 = try await runOneCall(tool: EchoTool(), name: "echo", args: ["text": .int(5)])
        #expect(r2.reply.contains("text must be text."))
    }

    @Test func anUnknownToolNamesTheAvailableOnes() async throws {
        let r = try await runOneCall(tool: EchoTool(), name: "nope", args: [:])
        #expect(r.reply.contains("there is no tool named \"nope\". Available tools: echo."))
        #expect(r.events == [.toolFinished(name: "nope", outcome: .unknownTool), .text(r.reply)])
    }

    @Test func aThrowingToolBecomesAnErrorStringAndAFailedEvent() async throws {
        let r = try await runOneCall(tool: ThrowingTool(), name: "boom", args: [:])
        #expect(r.reply == "got: Error: it broke")
        #expect(r.events.contains(.toolFinished(name: "boom", outcome: .failed("it broke"))))
    }

    @Test func aRealModelToolRefusalPassesThroughAsItsOwnSentence() async throws {
        // translate with no `to` -> the MT1 tool's own corrective text (or the adapter's), never a throw.
        let tool = ModelTool(spec: ModelToolSpec.spec(named: "translate")!, advertisesAssets: true)
        let r = try await runOneCall(tool: tool, name: "translate", args: ["text": .string("hello")])
        #expect(r.reply.hasPrefix("got: Error:") || r.reply.contains("to"))
    }

    // MARK: approvals

    @Test func aDeclinedApprovalNeverRunsTheToolAndSaysSo() async throws {
        let asked = Box<String>()
        let r = try await runOneCall(tool: GatedTool(), name: "gated", args: [:], approve: { t, _ in asked.add(t.name); return false })
        #expect(asked.all == ["gated"])
        #expect(r.events.first == .toolFinished(name: "gated", outcome: .denied))
        #expect(r.reply.contains("declined"))
    }

    @Test func anApprovedGatedToolRunsAfterTheApprovalAwaited() async throws {
        let r = try await runOneCall(tool: GatedTool(), name: "gated", args: [:], approve: { _, _ in
            try? await Task.sleep(for: .milliseconds(30)); return true })
        #expect(r.events == [.toolStarted(name: "gated", arguments: [:]), .toolFinished(name: "gated", outcome: .ok("ran")), .text("got: ran")])
    }

    // MARK: cap and outputs

    @Test func theThirdCallPastACapOfTwoIsToldToStop() async throws {
        let chatter = FakeChatter(script: { _, handler in
            var out: [String] = []
            for _ in 0..<3 { out.append(await handler("echo", ["text": .string("x")])) }
            return [.snapshot(out.joined(separator: "|"))]
        })
        let router = AppleFoundationRouter(chatter: chatter, maxToolCalls: 2)
        let events = try await collect(router.streamTurn(RouterTurn(prompt: "go", tools: [EchoTool()])))
        #expect(events.contains(.toolFinished(name: "echo", outcome: .budgetExceeded)))
        guard case .text(let text)? = events.last else { Issue.record("no text"); return }
        #expect(text.contains("Stop calling tools"))
        #expect(events.filter { if case .toolStarted = $0 { return true } else { return false } }.count == 2)
    }

    @Test func theCapResetsOnTheNextTurn() async throws {
        let chatter = FakeChatter(script: callTool("echo", ["text": .string("x")]))
        let router = AppleFoundationRouter(chatter: chatter, maxToolCalls: 1)
        let id = UUID()
        for _ in 0..<2 {
            let events = try await collect(router.streamTurn(RouterTurn(history: .live(conversationID: id, fallback: []), prompt: "go", tools: [EchoTool()])))
            #expect(events.contains(.toolFinished(name: "echo", outcome: .ok("echo: x"))))
        }
        #expect(chatter.sessionCount == 1)
    }

    @Test func aHugeOutputIsCutWithANoteAndAnEmptyOneSaysSo() async throws {
        let big = try await runOneCall(tool: BigTool(size: 20_000), name: "big", args: [:])
        #expect(big.reply.contains("[Output cut: 14000 more characters not shown.]"))
        #expect(big.reply.count < 7_000)
        let empty = try await runOneCall(tool: EmptyTool(), name: "empty", args: [:])
        #expect(empty.reply == "got: The tool finished and returned no output.")
    }

    // MARK: sessions

    @Test func aLiveSessionIsReusedAndReceivesOnlyThePrompt() async throws {
        let chatter = FakeChatter()
        let router = AppleFoundationRouter(chatter: chatter)
        let id = UUID()
        let history = [RouterMessage(role: .user, content: "earlier"), RouterMessage(role: .assistant, content: "reply")]
        _ = try await collect(router.streamTurn(RouterTurn(history: .live(conversationID: id, fallback: history), prompt: "one")))
        _ = try await collect(router.streamTurn(RouterTurn(history: .live(conversationID: id, fallback: history + [.init(role: .user, content: "one")]), prompt: "two")))
        #expect(chatter.sessionCount == 1)
        #expect(chatter.sessions[0].prompts == [AppleFoundationRouter.foldPrompt(history: history, last: "one"), "two"])
    }

    @Test func aMissFoldsHistoryExactlyAsTheServerBackendDoes() {
        let history = [RouterMessage(role: .user, content: "My name is Sam."), RouterMessage(role: .assistant, content: "Nice to meet you, Sam.")]
        let serve = AFMBackend.prompt(
            history: [ServeMessage(role: .user, text: "My name is Sam."), ServeMessage(role: .assistant, text: "Nice to meet you, Sam.")],
            last: "What is my name?")
        #expect(AppleFoundationRouter.foldPrompt(history: history, last: "What is my name?") == serve)
        #expect(AppleFoundationRouter.foldPrompt(history: [], last: "hi") == "hi")
    }

    @Test func historySystemMessagesJoinTheInstructionsAndAreNotFolded() async throws {
        let chatter = FakeChatter()
        let router = AppleFoundationRouter(chatter: chatter)
        let history = [RouterMessage(role: .system, content: "Rule A"), RouterMessage(role: .user, content: "hi"), RouterMessage(role: .assistant, content: "yo")]
        _ = try await collect(router.streamTurn(RouterTurn(instructions: "Base", history: .folded(history), prompt: "next")))
        #expect(chatter.instructions.first == "Base\n\nRule A")
        #expect(chatter.sessions[0].prompts == [AppleFoundationRouter.foldPrompt(history: Array(history.dropFirst()), last: "next")])
    }

    @Test func aFoldedTurnNeverEntersThePool() async throws {
        let chatter = FakeChatter()
        let router = AppleFoundationRouter(chatter: chatter)
        _ = try await collect(router.streamTurn(RouterTurn(history: .folded([]), prompt: "a")))
        _ = try await collect(router.streamTurn(RouterTurn(history: .folded([]), prompt: "b")))
        #expect(chatter.sessionCount == 2)
        #expect(router.liveConversationIDs.isEmpty)
    }

    @Test func theLeastRecentlyUsedConversationIsEvictedPastCapacity() async throws {
        let chatter = FakeChatter()
        let router = AppleFoundationRouter(chatter: chatter, sessionCapacity: 2)
        let a = UUID(), b = UUID(), c = UUID()
        for id in [a, b, c] {
            _ = try await collect(router.streamTurn(RouterTurn(history: .live(conversationID: id, fallback: []), prompt: "x")))
        }
        #expect(router.liveConversationIDs == [b, c])
        _ = try await collect(router.streamTurn(RouterTurn(history: .live(conversationID: b, fallback: []), prompt: "y")))   // touch b
        #expect(router.liveConversationIDs == [c, b])
        _ = try await collect(router.streamTurn(RouterTurn(history: .live(conversationID: a, fallback: [.init(role: .user, content: "old")]), prompt: "z")))
        #expect(router.liveConversationIDs == [b, a])          // c was the least recently used
        #expect(chatter.sessions.last?.prompts.first?.hasPrefix("Conversation so far:") == true)   // a came back folded
    }

    @Test func theDefaultCapacityIsFour() {
        #expect(AppleFoundationRouter.defaultSessionCapacity == 4)
    }

    @Test func changingTheToolSetOrInstructionsStartsANewSession() async throws {
        let chatter = FakeChatter()
        let router = AppleFoundationRouter(chatter: chatter)
        let id = UUID()
        let live = RouterHistory.live(conversationID: id, fallback: [])
        _ = try await collect(router.streamTurn(RouterTurn(instructions: "A", history: live, prompt: "1", tools: [EchoTool()])))
        _ = try await collect(router.streamTurn(RouterTurn(instructions: "A", history: live, prompt: "2", tools: [EchoTool(), GatedTool()])))
        _ = try await collect(router.streamTurn(RouterTurn(instructions: "B", history: live, prompt: "3", tools: [EchoTool(), GatedTool()])))
        #expect(chatter.sessionCount == 3)
        #expect(router.liveConversationIDs == [id])
    }

    @Test func forgetDropsAConversationsSession() async throws {
        let chatter = FakeChatter()
        let router = AppleFoundationRouter(chatter: chatter)
        let id = UUID()
        _ = try await collect(router.streamTurn(RouterTurn(history: .live(conversationID: id, fallback: []), prompt: "x")))
        router.forget(conversationID: id)
        #expect(router.liveConversationIDs.isEmpty)
    }

    @Test func aSecondTurnOnABusyConversationUsesAFreshSessionNotTheBusyOne() async throws {
        let gate = Gate()
        let chatter = FakeChatter(script: { prompt, _ in
            if prompt.hasSuffix("slow") { await gate.wait() }
            return [.snapshot("done \(prompt)")]
        })
        let router = AppleFoundationRouter(chatter: chatter)
        let id = UUID()
        let live = RouterHistory.live(conversationID: id, fallback: [.init(role: .user, content: "h")])
        let first = Task { try await self.collect(router.streamTurn(RouterTurn(history: live, prompt: "slow"))) }
        try await Task.sleep(for: .milliseconds(60))
        let second = try await collect(router.streamTurn(RouterTurn(history: live, prompt: "quick")))
        await gate.open()
        _ = try await first.value
        #expect(second == [.text("done " + AppleFoundationRouter.foldPrompt(history: [.init(role: .user, content: "h")], last: "quick"))])
        #expect(chatter.sessions.allSatisfy { $0.overlaps == 0 })
        #expect(chatter.sessionCount == 2)
        #expect(router.liveConversationIDs == [id])
    }

    actor Gate {
        private var opened = false
        private var waiters: [CheckedContinuation<Void, Never>] = []
        func wait() async { if opened { return }; await withCheckedContinuation { waiters.append($0) } }
        func open() { opened = true; waiters.forEach { $0.resume() }; waiters = [] }
    }

    // MARK: text deltas

    @Test func snapshotsBecomeDeltasAndARevisionIsHeldBack() async throws {
        let chatter = FakeChatter(script: { _, _ in
            [.snapshot("He"), .snapshot("He"), .snapshot("Hello"), .snapshot("Help"), .snapshot("Hello, wor"), .snapshot("Hello, world")]
        })
        let router = AppleFoundationRouter(chatter: chatter)
        let events = try await collect(router.streamTurn(RouterTurn(prompt: "x")))
        #expect(events == [.text("He"), .text("llo"), .text(", wor"), .text("ld")])
    }

    @Test func usageIsPassedThrough() async throws {
        let chatter = FakeChatter(script: { _, _ in [.snapshot("a"), .usage(promptTokens: 900, completionTokens: 12)] })
        let events = try await collect(AppleFoundationRouter(chatter: chatter).streamTurn(RouterTurn(prompt: "x")))
        #expect(events == [.text("a"), .usage(promptTokens: 900, completionTokens: 12)])
    }

    // MARK: errors

    @Test func aContextOverflowBecomesAContextExceededEventAndDropsTheSession() async throws {
        let chatter = FakeChatter(script: { _, _ in throw AppleFoundationToolError.contextWindowExceeded(limit: 4096, tokenCount: 9000) })
        let router = AppleFoundationRouter(chatter: chatter)
        let id = UUID()
        let events = try await collect(router.streamTurn(RouterTurn(history: .live(conversationID: id, fallback: []), prompt: "x")))
        #expect(events == [.contextExceeded(limit: 4096)])
        #expect(router.liveConversationIDs.isEmpty)
    }

    @Test func aContextOverflowWithoutALimitUsesTheRoutersWindow() async throws {
        let chatter = FakeChatter(script: { _, _ in throw AppleFoundationToolError.contextWindowExceeded(limit: nil, tokenCount: nil) })
        let events = try await collect(AppleFoundationRouter(chatter: chatter).streamTurn(RouterTurn(prompt: "x")))
        #expect(events == [.contextExceeded(limit: 8192)])
    }

    @Test(arguments: [
        AppleFoundationToolError.guardrailViolation, .unsupportedLanguage, .refusal, .rateLimited, .timeout, .assetsUnavailable, .failed,
    ])
    func everyOtherFailureIsThrownTypedWithASentence(error: AppleFoundationToolError) async {
        let chatter = FakeChatter(script: { _, _ in throw error })
        var thrown: AppleFoundationToolError?
        do { _ = try await collect(AppleFoundationRouter(chatter: chatter).streamTurn(RouterTurn(prompt: "x"))) }
        catch let e as AppleFoundationToolError { thrown = e } catch {}
        #expect(thrown == error)
        #expect(!error.userMessage.isEmpty)
        #expect(error.userMessage.hasSuffix("."))
    }

    @Test func theSentencesAreDistinct() {
        let all: [AppleFoundationToolError] = [.contextWindowExceeded(limit: 8192, tokenCount: nil), .guardrailViolation, .unsupportedLanguage, .refusal, .rateLimited, .timeout, .assetsUnavailable, .failed]
        #expect(Set(all.map(\.userMessage)).count == all.count)
        #expect(AppleFoundationToolError.contextWindowExceeded(limit: 8192, tokenCount: nil).userMessage.contains("8192"))
    }

    @Test func cancellingTheConsumerCancelsTheTurn() async throws {
        let chatter = FakeChatter(script: { _, _ in try await Task.sleep(for: .seconds(30)); return [.snapshot("late")] })
        let router = AppleFoundationRouter(chatter: chatter)
        let task = Task { try await self.collect(router.streamTurn(RouterTurn(prompt: "x"))) }
        try await Task.sleep(for: .milliseconds(50))
        task.cancel()
        let result = try? await task.value
        #expect(result == nil || result == [])
    }

    @Test func identityIsTheSystemRouterWithTheWindow() {
        let router = AppleFoundationRouter(chatter: FakeChatter())
        #expect(router.identity == RouterIdentity(displayName: "Apple Intelligence", kind: .system, contextSize: 8192))
    }

    // MARK: the real framework's errors → the seam's typed errors (macOS 26+)

    @Test func the26xErrorsMapToTheTypedErrors() throws {
        guard #available(macOS 26, *) else { return }
        typealias G = LanguageModelSession.GenerationError
        let ctx = G.Context(debugDescription: "x")
        #expect(AppleFoundationErrorMapper.mapDetailed(G.exceededContextWindowSize(ctx)) == .contextWindowExceeded(limit: SystemLanguageModel.default.contextSize, tokenCount: nil))
        #expect(AppleFoundationErrorMapper.mapDetailed(G.guardrailViolation(ctx)) == .guardrailViolation)
        #expect(AppleFoundationErrorMapper.mapDetailed(G.unsupportedLanguageOrLocale(ctx)) == .unsupportedLanguage)
        #expect(AppleFoundationErrorMapper.mapDetailed(G.rateLimited(ctx)) == .rateLimited)
        #expect(AppleFoundationErrorMapper.mapDetailed(G.assetsUnavailable(ctx)) == .assetsUnavailable)
        #expect(AppleFoundationErrorMapper.mapDetailed(G.decodingFailure(ctx)) == .failed)
        #expect(AppleFoundationErrorMapper.mapDetailed(URLError(.badURL)) == .failed)
        #expect(AppleFoundationErrorMapper.mapDetailed(AppleFoundationToolError.timeout) == .timeout)
    }

    @Test func the27ErrorsMapToTheTypedErrors() throws {
        guard #available(macOS 27, *) else { return }
        #expect(AppleFoundationErrorMapper.mapDetailed(LanguageModelError.contextSizeExceeded(.init(contextSize: 8192, tokenCount: 40060, debugDescription: "x")))
                == .contextWindowExceeded(limit: 8192, tokenCount: 40060))
        #expect(AppleFoundationErrorMapper.mapDetailed(LanguageModelError.timeout(.init(debugDescription: "x"))) == .timeout)
        #expect(AppleFoundationErrorMapper.mapDetailed(LanguageModelError.rateLimited(.init(resetDate: nil, debugDescription: "x"))) == .rateLimited)
    }

    // MT3-FIX-3 — the overflow wording the framework used for ~570K tokens with tools armed.
    struct PrivateOverflowError: Error, CustomStringConvertible {
        var description: String { "Provided 570,100 tokens, but the maximum allowed is 8,192.::Provided 570,100 tokens, but the maximum allowed is 8,192.: The operation couldn\u{2019}t be completed." }
    }

    @Test func aPrivateFrameworkOverflowSentenceMapsToContextExceeded() {
        #expect(AppleFoundationErrorMapper.mapDetailed(PrivateOverflowError()) == .contextWindowExceeded(limit: 8192, tokenCount: nil))
        #expect(AppleFoundationErrorMapper.overflowLimit(inDescription: "Provided 9 tokens, but the maximum allowed is 4,096.") == 4096)
        #expect(AppleFoundationErrorMapper.overflowLimit(inDescription: "something else failed") == nil)
        #expect(AppleFoundationErrorMapper.mapDetailed(CocoaError(.fileNoSuchFile)) == .failed)
    }

    // MARK: owner-run real tests (Apple Intelligence ready + TEST_RUNNER_MLXUI_TEST_REAL_AFM=1)

    nonisolated static var realEnabled: Bool {
        ProcessInfo.processInfo.environment["MLXUI_TEST_REAL_AFM"] == "1"
            && AppleFoundationAvailability.currentReadiness(forceReal: true) == .ready
    }

    nonisolated struct DateTimeTool: AgentTool {
        let name = "date_time"
        let toolDescription = "Get the current date and time."
        let parameters: [ToolParameter] = []
        func execute(arguments: [String: JSONValue]) async throws -> String { "2026-11-10 14:05 (Tuesday), time zone Asia/Kuala_Lumpur" }
    }

    @Test(.enabled(if: realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realTheWorstCaseSixToolsFitTheBudget() async throws {
        let chatter = try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true))
        let names = ["list_capabilities", "generate_image", "transcribe_audio", "describe_image", "estimate_depth", "segment_image"]
        let tools: [any AgentTool] = names.compactMap { n in ModelToolSpec.spec(named: n).map { ModelTool(spec: $0, advertisesAssets: true) } }
        #expect(tools.count == 6)
        let specs = tools.map(AppleFoundationRouter.spec(for:))
        let instructions = "You are a helpful assistant in a Mac app. Use a tool when the user needs something you cannot do yourself, such as the time, reading an image, or transcribing audio. Files are referred to by handles such as a1. If a tool says an argument is wrong, fix it and try again."
        let count = try #require(await chatter.toolTokenCount(instructions: instructions, tools: specs))
        print("MT3-2 measured worst-case six tools + instructions:", count)
        #expect(AppleFoundationToolBudget.fits(tokens: count))
        #expect(abs(count - AppleFoundationToolBudget.measuredWorstCaseSixToolTokens) < 150)
    }

    @Test(.enabled(if: realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realAToolCallRaisesChipsAndASecondTurnReusesTheSession() async throws {
        let chatter = try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true))
        let router = AppleFoundationRouter(chatter: chatter)
        let id = UUID()
        let live = RouterHistory.live(conversationID: id, fallback: [])
        let tools: [any AgentTool] = [DateTimeTool()]
        let events = try await collect(router.streamTurn(RouterTurn(
            instructions: "You are a helpful assistant. Use the date_time tool for questions about the time.",
            history: live, prompt: "What time is it right now?", tools: tools)))
        print("MT3-2 real turn 1:", events)
        #expect(events.contains(.toolStarted(name: "date_time", arguments: [:])))
        #expect(events.contains { if case .toolFinished("date_time", .ok) = $0 { return true } else { return false } })
        var reply = ""
        for case .text(let t) in events { reply += t }
        #expect(reply.contains("14:05") || reply.lowercased().contains("2:05"))
        #expect(router.liveConversationIDs == [id])

        let second = try await collect(router.streamTurn(RouterTurn(
            instructions: "You are a helpful assistant. Use the date_time tool for questions about the time.",
            history: live, prompt: "Without using any tool: what was the first question I asked you in this conversation?", tools: tools)))
        print("MT3-2 real turn 2:", second)
        var reply2 = ""
        for case .text(let t) in second { reply2 += t }
        #expect(reply2.lowercased().contains("time"))
    }

    @Test(.enabled(if: realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realAnOverlongPromptIsAContextExceededEvent() async throws {
        let chatter = try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true))
        let router = AppleFoundationRouter(chatter: chatter)
        let big = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 4000)
        let events = try await collect(router.streamTurn(RouterTurn(prompt: "Summarize: " + big)))
        print("MT3-2 real overflow:", events)
        #expect(events.contains { if case .contextExceeded = $0 { return true } else { return false } })
    }

    /// MT3-FIX-3: ~20K and ~500K tokens, with and without tools armed, are a `.contextExceeded` event from the router.
    @Test(.enabled(if: realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"),
          arguments: [(90_000, false), (90_000, true), (2_250_000, false), (2_250_000, true)])
    func realAnyPromptPastTheWindowIsAContextExceededEvent(chars: Int, withTools: Bool) async throws {
        let chatter = try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true))
        let router = AppleFoundationRouter(chatter: chatter)
        let text = String(repeating: "The quick brown fox jumps over the lazy dog. Chapter marker: LONGFILE-OK.\n", count: chars / 75)
        let tools: [any AgentTool] = withTools ? [DateTimeTool()] : []
        let events = try await collect(router.streamTurn(RouterTurn(
            instructions: "You are helpful.", history: .live(conversationID: UUID(), fallback: []),
            prompt: "Summarize: " + text, tools: tools)))
        #expect(events.contains(.contextExceeded(limit: 8192)))
    }
}
