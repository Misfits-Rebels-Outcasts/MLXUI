import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// MT3-1 — the `ChatRouter` protocol shape and `MLXChatRouter`'s event mapping. Model-free: the router's
/// session factory is replaced by a fake session that runs a real `AgentSession.dispatch` for each scripted
/// tool call, so approvals, budget and the lifecycle events are the production ones.
@MainActor
@Suite(.serialized)
struct ChatRouterTests {

    nonisolated struct EchoTool: AgentTool {
        let name = "echo"
        let toolDescription = "Echo the text back."
        let parameters: [ToolParameter] = [.required("text", type: .string, description: "text")]
        func execute(arguments: [String: JSONValue]) async throws -> String { "echo: \(arguments.string("text") ?? "")" }
    }
    nonisolated struct GatedTool: AgentTool {
        let name = "gated"
        let toolDescription = "Needs approval."
        let parameters: [ToolParameter] = []
        var requiresApproval: Bool { true }
        func execute(arguments: [String: JSONValue]) async throws -> String { "ran" }
    }

    /// Streams `chunks`, running each scripted tool call through `AgentSession.dispatch` first.
    final class ScriptedSession: ChatStreamingSession, @unchecked Sendable {
        let chunks: [String]
        let calls: [(String, [String: JSONValue])]
        let turn: RouterTurn
        let onEvent: @Sendable (AgentToolEvent) -> Void
        var promptSeen: String?
        init(chunks: [String], calls: [(String, [String: JSONValue])], turn: RouterTurn,
             onEvent: @escaping @Sendable (AgentToolEvent) -> Void) {
            self.chunks = chunks; self.calls = calls; self.turn = turn; self.onEvent = onEvent
        }
        func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
            promptSeen = prompt
            let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
            let registry = ToolRegistry(turn.tools)
            let budget = ToolBudget(limit: 1)
            let approve = turn.approve
            let onEvent = onEvent
            let chunks = chunks
            let calls = calls
            let task = Task {
                for (name, args) in calls {
                    let call = ToolCall(function: .init(name: name, arguments: args))
                    _ = await AgentSession.dispatch(
                        call, registry: registry, budget: budget,
                        approve: { call, tool in await approve(tool, call.function.arguments) }, onEvent: onEvent)
                }
                for chunk in chunks { continuation.yield(chunk) }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
            return stream
        }
    }

    private func router(chunks: [String], calls: [(String, [String: JSONValue])] = []) -> MLXChatRouter {
        MLXChatRouter(identity: RouterIdentity(displayName: "Fake", kind: .mlx, contextSize: 4096)) { turn, onEvent in
            ScriptedSession(chunks: chunks, calls: calls, turn: turn, onEvent: onEvent)
        }
    }

    private func collect(_ stream: AsyncThrowingStream<RouterEvent, Error>) async throws -> [RouterEvent] {
        var out: [RouterEvent] = []
        for try await event in stream { out.append(event) }
        return out
    }

    @Test func identityIsAMLXRouterWithTheWindow() {
        let r = router(chunks: [])
        #expect(r.identity == RouterIdentity(displayName: "Fake", kind: .mlx, contextSize: 4096))
    }

    @Test func textChunksBecomeTextEventsInOrder() async throws {
        let events = try await collect(router(chunks: ["Hel", "lo"]).streamTurn(RouterTurn(prompt: "hi")))
        #expect(events == [.text("Hel"), .text("lo")])
    }

    @Test func thePromptReachesTheSessionUnchanged() async throws {
        let box = SessionBox()
        let r = MLXChatRouter(identity: RouterIdentity(displayName: "F", kind: .mlx, contextSize: nil)) { turn, onEvent in
            let s = ScriptedSession(chunks: ["x"], calls: [], turn: turn, onEvent: onEvent); box.session = s; return s
        }
        _ = try await collect(r.streamTurn(RouterTurn(instructions: "sys", prompt: "the prompt")))
        #expect(box.session?.promptSeen == "the prompt")
        #expect(box.session?.turn.instructions == "sys")
    }

    @Test func aToolCallRaisesStartedThenFinishedBeforeTheText() async throws {
        let r = router(chunks: ["done"], calls: [("echo", ["text": .string("a")])])
        let events = try await collect(r.streamTurn(RouterTurn(prompt: "go", tools: [EchoTool()])))
        #expect(events == [
            .toolStarted(name: "echo", arguments: ["text": .string("a")]),
            .toolFinished(name: "echo", outcome: .ok("echo: a")),
            .text("done"),
        ])
    }

    @Test func onToolEventStillSeesTheFullFidelityLifecycle() async throws {
        let log = EventBox()
        let turn = RouterTurn(prompt: "go", tools: [EchoTool()], onToolEvent: { log.add($0) })
        _ = try await collect(router(chunks: [], calls: [("echo", ["text": .string("a")])]).streamTurn(turn))
        let names = log.snapshot().map(Self.tag)
        #expect(names == ["started:echo", "finished:echo"])
    }

    @Test func aDeclinedApprovalIsDeniedAndNeverRuns() async throws {
        let asked = EventBox()
        let turn = RouterTurn(prompt: "go", tools: [GatedTool()],
                              approve: { tool, _ in asked.addName(tool.name); return false })
        let events = try await collect(router(chunks: [], calls: [("gated", [:])]).streamTurn(turn))
        #expect(events == [.toolFinished(name: "gated", outcome: .denied)])
        #expect(asked.names() == ["gated"])
    }

    @Test func anApprovedGatedToolRunsAndReceivesItsArguments() async throws {
        let turn = RouterTurn(prompt: "go", tools: [GatedTool()], approve: { _, _ in true })
        let events = try await collect(router(chunks: [], calls: [("gated", [:])]).streamTurn(turn))
        #expect(events == [.toolStarted(name: "gated", arguments: [:]), .toolFinished(name: "gated", outcome: .ok("ran"))])
    }

    @Test func unknownToolAndBudgetMapToTheirOutcomes() async throws {
        let r = router(chunks: [], calls: [("echo", ["text": .string("1")]), ("echo", ["text": .string("2")]), ("nope", [:])])
        let events = try await collect(r.streamTurn(RouterTurn(prompt: "go", tools: [EchoTool()])))
        #expect(events == [
            .toolStarted(name: "echo", arguments: ["text": .string("1")]),
            .toolFinished(name: "echo", outcome: .ok("echo: 1")),
            .toolFinished(name: "echo", outcome: .budgetExceeded),
            .toolFinished(name: "nope", outcome: .unknownTool),
        ])
    }

    @Test func everyAgentToolEventMapsToARouterEvent() {
        let args: [String: JSONValue] = ["k": .int(1)]
        #expect(AgentToolEvent.started(name: "t", arguments: args).routerEvent == .toolStarted(name: "t", arguments: args))
        #expect(AgentToolEvent.finished(name: "t", result: "r").routerEvent == .toolFinished(name: "t", outcome: .ok("r")))
        #expect(AgentToolEvent.failed(name: "t", message: "m").routerEvent == .toolFinished(name: "t", outcome: .failed("m")))
        #expect(AgentToolEvent.denied(name: "t").routerEvent == .toolFinished(name: "t", outcome: .denied))
        #expect(AgentToolEvent.unknownTool(name: "t").routerEvent == .toolFinished(name: "t", outcome: .unknownTool))
        #expect(AgentToolEvent.budgetExceeded(name: "t").routerEvent == .toolFinished(name: "t", outcome: .budgetExceeded))
    }

    @Test func historyFoldsFromLiveFallbackAndConvertsRoles() {
        let msgs = [RouterMessage(role: .user, content: "a"), RouterMessage(role: .assistant, content: "b")]
        let id = UUID()
        #expect(RouterHistory.folded(msgs).foldedMessages == msgs)
        #expect(RouterHistory.live(conversationID: id, fallback: msgs).foldedMessages == msgs)
        for role in [RouterMessage.Role.system, .user, .assistant, .tool] {
            let m = RouterMessage(role: role, content: "c")
            #expect(MLXChatRouter.routerMessage(MLXChatRouter.chatMessage(m)) == m)
        }
    }

    @Test func aFailingSessionFinishesTheStreamWithTheError() async {
        struct Boom: Error {}
        final class Failing: ChatStreamingSession, @unchecked Sendable {
            func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
                AsyncThrowingStream { $0.yield("part"); $0.finish(throwing: Boom()) }
            }
        }
        let r = MLXChatRouter(identity: RouterIdentity(displayName: "F", kind: .mlx, contextSize: nil)) { _, _ in Failing() }
        var got: [RouterEvent] = []
        var threw = false
        do { for try await e in r.streamTurn(RouterTurn(prompt: "x")) { got.append(e) } } catch { threw = true }
        #expect(got == [.text("part")])
        #expect(threw)
    }

    @Test func cancellingTheConsumerEndsTheTurn() async {
        final class Endless: ChatStreamingSession, @unchecked Sendable {
            func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
                AsyncThrowingStream { c in
                    let t = Task { while !Task.isCancelled { c.yield("."); try? await Task.sleep(for: .milliseconds(1)) }; c.finish() }
                    c.onTermination = { _ in t.cancel() }
                }
            }
        }
        let r = MLXChatRouter(identity: RouterIdentity(displayName: "F", kind: .mlx, contextSize: nil)) { _, _ in Endless() }
        let count = Counter()
        let task = Task { for try await _ in r.streamTurn(RouterTurn(prompt: "x")) { count.bump() } }
        try? await Task.sleep(for: .milliseconds(50))
        task.cancel()
        _ = try? await task.value
        let frozen = count.value
        try? await Task.sleep(for: .milliseconds(50))
        #expect(count.value == frozen)
        #expect(frozen > 0)
    }

    // MARK: helpers

    nonisolated final class SessionBox: @unchecked Sendable { var session: ScriptedSession? }

    nonisolated final class EventBox: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [AgentToolEvent] = []
        private var tools: [String] = []
        func add(_ e: AgentToolEvent) { lock.lock(); events.append(e); lock.unlock() }
        func addName(_ n: String) { lock.lock(); tools.append(n); lock.unlock() }
        func snapshot() -> [AgentToolEvent] { lock.lock(); defer { lock.unlock() }; return events }
        func names() -> [String] { lock.lock(); defer { lock.unlock() }; return tools }
    }
    nonisolated final class Counter: @unchecked Sendable {
        private let lock = NSLock(); private var n = 0
        func bump() { lock.lock(); n += 1; lock.unlock() }
        var value: Int { lock.lock(); defer { lock.unlock() }; return n }
    }
    nonisolated static func tag(_ e: AgentToolEvent) -> String {
        switch e {
        case .started(let n, _): return "started:\(n)"
        case .finished(let n, _): return "finished:\(n)"
        case .failed(let n, _): return "failed:\(n)"
        case .denied(let n): return "denied:\(n)"
        case .unknownTool(let n): return "unknown:\(n)"
        case .budgetExceeded(let n): return "budget:\(n)"
        }
    }
}
