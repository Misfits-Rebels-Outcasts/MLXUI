//
//  AppleFoundationRouter.swift
//  MLXUI — Model Tools in Chat, MT3-2 (plan §3.3; spike findings in journal 2026-713).
//
//  Apple Intelligence behind `ChatRouter`. **This file contains no FoundationModels code**: everything
//  framework-specific (the runtime-built `GenerationSchema`, `Tool`, `LanguageModelSession`, the error
//  types and every `@available`) lives in `FlowKit/AppleFoundationExecutor.swift` behind the untagged
//  `AppleFoundationToolChatting` seam, so this router and its tests run with a fixture chatter and no device.
//
//  What the router adds on top of the seam (MT3-0 findings):
//   - Adapters from `AgentTool` to seam tool specs; **argument validation inside the call that returns a
//     corrective string and never throws** (a throwing FM tool ends the whole turn with `ToolCallError`).
//   - `.toolStarted` / `.toolFinished` and `onToolEvent` are raised **from inside the call** (the framework
//     transcript only shows a call after it finished, so it can't drive a live chip); approvals run there too.
//   - A per-turn tool-call cap (every call counts, valid or not) and informative, bounded tool outputs: a vague
//     result made the model re-call a tool until the 8,192-token window overflowed mid-turn.
//   - At most 6 tools reach the model (the 1,500-token budget holds for the largest 6, not for the whole table).
//   - A live session per conversation (LRU 4), one turn at a time; a miss folds history into the prompt exactly as
//     `AFMBackend` does (a session built from a constructed `Transcript` ignores assistant turns, readme T7/F10).
//   - Snapshots → `.text` deltas; errors → `.contextExceeded` or a typed `AppleFoundationToolError`.
//

import Foundation
import MLXLMCommon

/// The context budget for Apple Intelligence (plan §3.3): instructions + tool schemas.
nonisolated enum AppleFoundationToolBudget {
    /// Tokens the instructions and all tool schemas together may take, leaving the rest of the 8,192 window for
    /// history, tool outputs and the reply.
    static let tokenBudget = 1_500
    /// The most tools the router sends the model.
    static let maxTools = 6
    /// Measured on macOS 27.0 (journal 2026-713): the largest 6-tool exposure of the v1 table with the asset input
    /// and a 110-token instruction block: list_capabilities + generate_image, transcribe_audio, describe_image,
    /// estimate_depth, segment_image.
    static let measuredWorstCaseSixToolTokens = 1_099

    static func fits(tokens: Int) -> Bool { tokens <= tokenBudget }
    static func headroom(tokens: Int) -> Int { tokenBudget - tokens }
}

extension AppleFoundationToolError {
    /// One sentence for the banner. The UI adds the buttons (MT3-3).
    nonisolated var userMessage: String {
        switch self {
        case .contextWindowExceeded(let limit, _):
            return "This chat is too long for Apple Intelligence" + (limit.map { " (its window is \($0) tokens)" } ?? "") + "."
        case .guardrailViolation: return "Apple Intelligence declined this request."
        case .unsupportedLanguage: return "Apple Intelligence doesn't support this language."
        case .refusal: return "Apple Intelligence refused to answer this."
        case .rateLimited: return "Apple Intelligence is busy right now. Try again in a moment."
        case .timeout: return "Apple Intelligence took too long to answer."
        case .assetsUnavailable: return "Apple Intelligence's model isn't available yet. macOS may still be downloading it."
        case .failed: return "Apple Intelligence failed to generate a reply."
        }
    }
}

nonisolated final class AppleFoundationRouter: ChatRouter, @unchecked Sendable {

    static let defaultMaxToolCalls = 6
    static let defaultSessionCapacity = 4
    /// A tool result longer than this is cut (with a note): the model's window is 8,192 tokens in total.
    static let maxToolOutputCharacters = 6_000

    let identity: RouterIdentity
    private let chatter: any AppleFoundationToolChatting
    private let maxTools: Int
    private let maxToolCalls: Int
    private let sessionCapacity: Int

    private let lock = NSLock()
    private var pool: [PoolEntry] = []          // least recently used first

    init(chatter: any AppleFoundationToolChatting,
         displayName: String = "Apple Intelligence",
         maxTools: Int = AppleFoundationToolBudget.maxTools,
         maxToolCalls: Int = AppleFoundationRouter.defaultMaxToolCalls,
         sessionCapacity: Int = AppleFoundationRouter.defaultSessionCapacity) {
        self.chatter = chatter
        self.identity = RouterIdentity(displayName: displayName, kind: .system, contextSize: chatter.contextSize)
        self.maxTools = min(maxTools, AppleFoundationToolBudget.maxTools)
        self.maxToolCalls = max(1, maxToolCalls)
        self.sessionCapacity = max(1, sessionCapacity)
    }

    // MARK: - Pool

    private final class PoolEntry: @unchecked Sendable {
        let key: UUID?
        let signature: String
        let session: any AppleFoundationToolSession
        let bridge: ToolBridge
        var busy = false
        init(key: UUID?, signature: String, session: any AppleFoundationToolSession, bridge: ToolBridge) {
            self.key = key; self.signature = signature; self.session = session; self.bridge = bridge
        }
    }

    /// Conversation ids that currently hold a live session, least recently used first (for tests and the menu).
    var liveConversationIDs: [UUID] { lock.withLock { pool.compactMap(\.key) } }

    /// Drop a conversation's live session (history was edited, the conversation was deleted).
    func forget(conversationID: UUID) { lock.withLock { pool.removeAll { $0.key == conversationID && !$0.busy } } }

    private func discard(_ entry: PoolEntry) { lock.withLock { pool.removeAll { $0 === entry } } }

    // MARK: - ChatRouter

    func streamTurn(_ turn: RouterTurn) -> AsyncThrowingStream<RouterEvent, Error> {
        let (stream, continuation) = AsyncThrowingStream<RouterEvent, Error>.makeStream()
        let task = Task { await self.run(turn, continuation) }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    private func run(_ turn: RouterTurn, _ continuation: AsyncThrowingStream<RouterEvent, Error>.Continuation) async {
        let selection = Self.select(turn.tools, limit: maxTools)
        let history = turn.history.foldedMessages
        let instructions = Self.composeInstructions(
            turn.instructions, historySystem: history.filter { $0.role == .system }.map(\.content),
            unavailable: selection.dropped)
        let specs = selection.included.map(Self.spec(for:))
        let signature = specs.map(\.name).joined(separator: ",") + "|" + (instructions ?? "")

        let (entry, prompt) = acquire(turn: turn, specs: specs, instructions: instructions, signature: signature,
                                      folded: history.filter { $0.role != .system })
        let state = ToolBridge.TurnState(
            tools: Dictionary(selection.included.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first }),
            approve: turn.approve, onToolEvent: turn.onToolEvent, continuation: continuation, maxCalls: maxToolCalls)
        entry.bridge.begin(state)
        defer {
            entry.bridge.end()
            lock.withLock { entry.busy = false }
        }

        var converter = SnapshotDeltaConverter()
        do {
            for try await event in entry.session.respond(prompt: prompt) {
                switch event {
                case .snapshot(let snapshot):
                    let delta = converter.push(snapshot)
                    if !delta.isEmpty, case .terminated = continuation.yield(.text(delta)) { return }
                case .usage(let promptTokens, let completionTokens):
                    continuation.yield(.usage(promptTokens: promptTokens, completionTokens: completionTokens))
                }
            }
            continuation.finish()
        } catch is CancellationError {
            continuation.finish()
        } catch let error as AppleFoundationToolError {
            discard(entry)         // a full or refused session is not worth keeping
            if case .contextWindowExceeded(let limit, _) = error {
                continuation.yield(.contextExceeded(limit: limit ?? identity.contextSize))
                continuation.finish()
            } else {
                continuation.finish(throwing: error)
            }
        } catch {
            discard(entry)
            continuation.finish(throwing: error)
        }
    }

    /// A live session for the conversation when the tool set and instructions still match and it isn't mid-turn;
    /// otherwise a new session whose history is folded into the prompt.
    private func acquire(turn: RouterTurn, specs: [AppleFoundationToolSpec], instructions: String?, signature: String,
                         folded: [RouterMessage]) -> (PoolEntry, String) {
        var key: UUID?
        if case .live(let id, _) = turn.history { key = id }
        let chatter = chatter
        return lock.withLock {
            if let key, let hit = pool.first(where: { $0.key == key }), hit.signature == signature, !hit.busy {
                hit.busy = true
                pool.removeAll { $0 === hit }
                pool.append(hit)
                return (hit, turn.prompt)
            }
            let bridge = ToolBridge()
            let session = chatter.makeSession(instructions: instructions, tools: specs,
                                              handler: { name, arguments in await bridge.handle(name, arguments) })
            // A busy session for the same conversation is left alone (a second respond would be a programmer error):
            // this turn runs on an unpooled fresh session.
            let poolable = key != nil && !pool.contains { $0.key == key && $0.busy }
            let entry = PoolEntry(key: poolable ? key : nil, signature: signature, session: session, bridge: bridge)
            entry.busy = true
            if poolable {
                pool.removeAll { $0.key == key }
                pool.append(entry)
                while pool.count > sessionCapacity, let victim = pool.firstIndex(where: { !$0.busy }) {
                    pool.remove(at: victim)
                }
            }
            return (entry, Self.foldPrompt(history: folded, last: turn.prompt))
        }
    }

    // MARK: - Pure helpers

    /// At most `limit` tools, in the order given; tools whose parameters AFM's schema can't express are left out.
    static func select(_ tools: [any AgentTool], limit: Int) -> (included: [any AgentTool], dropped: [String]) {
        var included: [any AgentTool] = []
        var dropped: [String] = []
        var seen = Set<String>()
        for tool in tools {
            guard seen.insert(tool.name).inserted else { continue }
            if included.count < limit, tool.parameters.allSatisfy({ kind(of: $0.type) != nil }) {
                included.append(tool)
            } else {
                dropped.append(tool.name)
            }
        }
        return (included, dropped)
    }

    static func kind(of type: ToolParameterType) -> AppleFoundationToolSpec.Kind? {
        switch type {
        case .string: return .string
        case .int: return .integer
        case .double: return .number
        case .bool: return .boolean
        default: return nil
        }
    }

    static func spec(for tool: any AgentTool) -> AppleFoundationToolSpec {
        AppleFoundationToolSpec(
            name: tool.name, description: tool.toolDescription,
            properties: tool.parameters.compactMap { p in
                kind(of: p.type).map { .init(name: p.name, description: p.description, kind: $0, required: p.isRequired) }
            })
    }

    static func composeInstructions(_ instructions: String?, historySystem: [String], unavailable: [String]) -> String? {
        var parts = ([instructions] + historySystem.map { Optional($0) }).compactMap { $0 }.filter { !$0.isEmpty }
        if !unavailable.isEmpty {
            parts.append("These tools are not available in this chat: \(unavailable.joined(separator: ", ")).")
        }
        return parts.isEmpty ? nil : parts.joined(separator: "\n\n")
    }

    /// The same framing `AFMBackend.prompt(history:last:)` uses (its test pins the equality).
    static func foldPrompt(history: [RouterMessage], last: String) -> String {
        guard !history.isEmpty else { return last }
        let lines = history.map { ($0.role == .user ? "User: " : "Assistant: ") + $0.content }.joined(separator: "\n")
        return "Conversation so far:\n\(lines)\n\nReply to the latest message from the user:\nUser: \(last)"
    }

    // MARK: - Tool bridge

    /// Receives every tool call of one session. One bridge per session; the per-turn state (approval, events, cap) is
    /// swapped in for each turn, because a live session outlives its turns.
    final class ToolBridge: @unchecked Sendable {
        struct TurnState: @unchecked Sendable {
            let tools: [String: any AgentTool]
            let approve: @Sendable (any AgentTool, [String: JSONValue]) async -> Bool
            let onToolEvent: @Sendable (AgentToolEvent) -> Void
            let continuation: AsyncThrowingStream<RouterEvent, Error>.Continuation
            let maxCalls: Int
        }

        private let lock = NSLock()
        private var state: TurnState?
        private var calls = 0

        func begin(_ state: TurnState) { lock.withLock { self.state = state; calls = 0 } }
        func end() { lock.withLock { state = nil } }

        func handle(_ name: String, _ rawArguments: [String: JSONValue]) async -> String {
            let (current, count): (TurnState?, Int) = lock.withLock { calls += 1; return (state, calls) }
            guard let state = current else { return "This turn has ended. Answer without calling tools." }
            func emit(_ event: AgentToolEvent) {
                state.onToolEvent(event)
                state.continuation.yield(event.routerEvent)
            }

            guard let tool = state.tools[name] else {
                emit(.unknownTool(name: name))
                let names = state.tools.keys.sorted().joined(separator: ", ")
                return "Error: there is no tool named \"\(name)\"." + (names.isEmpty ? " No tools are available." : " Available tools: \(names).")
            }
            if count > state.maxCalls {
                emit(.budgetExceeded(name: name))
                return "Tool-call limit reached for this reply. Stop calling tools and answer with what you already have."
            }
            if Task.isCancelled { return "The reply was stopped." }

            let arguments: [String: JSONValue]
            switch Self.validate(rawArguments, against: tool) {
            case .failure(let failure):
                emit(.started(name: name, arguments: rawArguments))
                emit(.failed(name: name, message: failure.description))
                return "Error: \(failure.description)"
            case .success(let cleaned):
                arguments = cleaned
            }

            if tool.requiresApproval, await state.approve(tool, arguments) == false {
                emit(.denied(name: name))
                return "The user declined to run this tool. Do not call it again; tell the user what you could not do."
            }

            emit(.started(name: name, arguments: arguments))
            do {
                let output: String
                if let timeout = tool.executionTimeout {
                    guard let result = try await AgentSession.withTimeout(
                        seconds: timeout, operation: { try await tool.execute(arguments: arguments) })
                    else {
                        let message = "timed out after \(Int(timeout))s"
                        emit(.failed(name: name, message: message))
                        return "Error: \(message)"
                    }
                    output = result
                } else {
                    output = try await tool.execute(arguments: arguments)
                }
                emit(.finished(name: name, result: output))
                return Self.modelResult(output)
            } catch {
                let message = error.localizedDescription
                emit(.failed(name: name, message: message))
                return "Error: \(message)"
            }
        }

        /// Informative and bounded: an empty result says so, a huge one is cut with a note.
        static func modelResult(_ output: String) -> String {
            if output.isEmpty { return "The tool finished and returned no output." }
            guard output.count > AppleFoundationRouter.maxToolOutputCharacters else { return output }
            return String(output.prefix(AppleFoundationRouter.maxToolOutputCharacters))
                + "\n[Output cut: \(output.count - AppleFoundationRouter.maxToolOutputCharacters) more characters not shown.]"
        }

        /// Names, required arguments and types against the tool's declared parameters. Failure is the corrective sentence.
        static func validate(_ arguments: [String: JSONValue], against tool: any AgentTool) -> Result<[String: JSONValue], ValidationFailure> {
            let declared = tool.parameters
            let given = arguments.filter { if case .null = $0.value { return false } else { return true } }
            let valid = declared.map(\.name)
            let unknown = given.keys.filter { !valid.contains($0) }.sorted()
            if !unknown.isEmpty {
                let list = unknown.map { "\"\($0)\"" }.joined(separator: ", ")
                return .failure(.init("unknown argument\(unknown.count == 1 ? "" : "s") \(list) for \(tool.name). Valid arguments: \(valid.isEmpty ? "none" : valid.joined(separator: ", "))."))
            }
            var cleaned: [String: JSONValue] = [:]
            for p in declared {
                guard let value = given[p.name] else {
                    if p.isRequired { return .failure(.init("\(tool.name) needs the \"\(p.name)\" argument (\(p.description))")) }
                    continue
                }
                switch (p.type, value) {
                case (.string, .string): cleaned[p.name] = value
                case (.int, .int): cleaned[p.name] = value
                case (.int, .double(let d)) where d == d.rounded() && abs(d) < 9e15: cleaned[p.name] = .int(Int(d))
                case (.double, .double): cleaned[p.name] = value
                case (.double, .int(let i)): cleaned[p.name] = .double(Double(i))
                case (.bool, .bool): cleaned[p.name] = value
                default:
                    return .failure(.init("\(p.name) must be \(describe(p.type))."))
                }
            }
            return .success(cleaned)
        }

        struct ValidationFailure: Error, CustomStringConvertible {
            let description: String
            init(_ description: String) { self.description = description }
        }

        static func describe(_ type: ToolParameterType) -> String {
            switch type {
            case .string: return "text"
            case .int: return "a whole number"
            case .double: return "a number"
            case .bool: return "true or false"
            default: return "a supported value"
            }
        }
    }
}
