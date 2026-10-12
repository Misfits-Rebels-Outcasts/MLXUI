//
//  AgentSession.swift
//  MLXUI — agentic chat tools, slice AG0.
//
//  Thin wrapper over `MLXLMCommon.ChatSession` that turns a `ToolRegistry` into the
//  `tools` + `toolDispatch` that `ChatSession` needs, and adds the app-side concerns
//  `ChatSession` doesn't cover: a per-turn call budget, an approval gate for
//  `requiresApproval` tools, graceful error capture, and lifecycle events for the UI.
//
//  `ChatSession` itself runs the loop (generate → parse tool call → dispatch → feed result
//  → continue), so this stays small. The dispatch logic is a `static` function so it can be
//  unit-tested without loading a model.
//
//  TOOL-CALL FORMAT (wired in AG1, rewritten in S2-1): the generate path selects the parser from
//  `ModelConfiguration.toolCallFormat ?? .json`. `LLMModelFactory` already infers that format from
//  the checkpoint at load (`ToolCallFormat.infer(from:configData:)`), and `ModelContainerPool` adds
//  the app's overrides on top — see `ToolCallFormatTable`. The app no longer keeps its own family
//  table here, and `Self.toolCallFormat(forModelType:)` below is only the no-`config.json`
//  convenience kept for AG0's test.
//

import Foundation
import MLXLMCommon

nonisolated final class AgentSession {
    private let model: ModelContainer
    private let instructions: String?
    private let history: [Chat.Message]
    private let registry: ToolRegistry
    /// The conversation's folder scope, bound as a `@TaskLocal` for every dispatch (CW-1 C1).
    /// `nil` — flows, the Local Server, tests — means the tools behave exactly as before.
    private let fileScope: FileScope?
    private let parameters: GenerateParameters
    private let maxToolCalls: Int
    private let approve: @Sendable (ToolCall, any AgentTool) async -> Bool
    private let onEvent: @Sendable (AgentToolEvent) -> Void

    /// - Parameters:
    ///   - model: a loaded `ModelContainer` (caller loads via the existing LLM path).
    ///   - history: prior turns to rehydrate the chat with (persistent conversations).
    ///   - registry: tools available this session (empty → plain chat, no tools advertised).
    ///   - fileScope: folders this conversation's file tools may use (CW-1). `nil` = no scope.
    ///   - maxToolCalls: cap on tool invocations per response (runaway-loop guard).
    ///   - approve: gate invoked for `requiresApproval` tools; return false to decline.
    ///   - onEvent: tool lifecycle callbacks for the chat UI / audit log.
    init(
        model: ModelContainer,
        instructions: String? = nil,
        history: [Chat.Message] = [],
        registry: ToolRegistry = ToolRegistry(),
        fileScope: FileScope? = nil,
        parameters: GenerateParameters = .init(),
        maxToolCalls: Int = 8,
        approve: @escaping @Sendable (ToolCall, any AgentTool) async -> Bool = { _, _ in true },
        onEvent: @escaping @Sendable (AgentToolEvent) -> Void = { _ in }
    ) {
        self.model = model
        self.instructions = instructions
        self.history = history
        self.registry = registry
        self.fileScope = fileScope
        self.parameters = parameters
        self.maxToolCalls = maxToolCalls
        self.approve = approve
        self.onEvent = onEvent
    }

    /// TCP-1 (`RSI/backlog.md`, journal `2026-230`): whether a turn goes through the
    /// tool-call scanner path. `ChatSession`'s generate path *always* runs mlx-swift-lm's
    /// `ToolCallProcessor`, which drops the text before a `<` that partially matches
    /// `<tool_call>` — so any reply containing HTML / XML / SVG / JSX loses `>` characters.
    /// A turn with no tools armed has no reason to pay that, so it takes a raw token stream.
    static func usesRawTextPath(_ registry: ToolRegistry) -> Bool { registry.isEmpty }

    /// Stream a response, running the tool loop as the model requests tools.
    func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
        guard !Self.usesRawTextPath(registry) else { return rawStreamResponse(to: prompt) }

        let registry = registry
        let approve = approve
        let onEvent = onEvent
        let fileScope = fileScope
        let budget = ToolBudget(limit: maxToolCalls)

        // The scope is bound *inside* the dispatch closure, not around `streamResponse`:
        // `ChatSession` may hop tasks to run the loop, and a `@TaskLocal` only flows down through
        // structured concurrency. Bound here it reaches `tool.execute` on both the untimed path
        // and the `withTimeout` task-group path (CW-1's propagation test is the guard).
        let dispatch: @Sendable (ToolCall) async throws -> String = { call in
            let result = await FileScope.$current.withValue(fileScope) {
                await AgentSession.dispatch(
                    call, registry: registry, budget: budget, approve: approve, onEvent: onEvent)
            }
            return result.modelResult
        }
        // ChatSession's history initializer rehydrates prior turns for persistent
        // conversations; the plain initializer starts fresh. Tools are armed here (the raw
        // path handled the empty case above), so `tools:` is always non-nil.
        let session = history.isEmpty
            ? ChatSession(
                model,
                instructions: instructions,
                generateParameters: parameters,
                tools: registry.toolSpecs,
                toolDispatch: dispatch)
            : ChatSession(
                model,
                instructions: instructions,
                history: history,
                generateParameters: parameters,
                tools: registry.toolSpecs,
                toolDispatch: dispatch)
        return session.streamResponse(to: prompt)
    }

    /// A plain chat turn, streamed straight from the token iterator with **no tool-call
    /// scanner in the path** (TCP-1). Mirrors `ChatSession`'s message assembly (system +
    /// history + user) but drives `MLXLMCommon.generateTokens` and decodes with
    /// `NaiveStreamingDetokenizer` alone — the detokenizer is lossless; only `ToolCallProcessor`
    /// layered on top of it drops the `>`. The chat sheet's history is always text-only
    /// (`Conversation.chatHistory`), so it flattens to Sendable `(role, content)` pairs.
    private func rawStreamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let model = model
        let parameters = parameters
        let systemText = instructions
        let priorTurns: [(role: Chat.Message.Role, content: String)] =
            history.map { ($0.role, $0.content) }

        let task = Task {
            do {
                try await model.perform { context in
                    var messages: [Chat.Message] = []
                    if let systemText { messages.append(.system(systemText)) }
                    for turn in priorTurns {
                        messages.append(Chat.Message(role: turn.role, content: turn.content))
                    }
                    messages.append(.user(prompt))

                    let input = try await context.processor.prepare(
                        input: UserInput(chat: messages))
                    let (tokens, genTask) = try MLXLMCommon.generateTokensTask(
                        input: input, parameters: parameters, context: context)

                    var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
                    for await item in tokens {
                        guard case .token(let id) = item else { continue }
                        detokenizer.append(token: id)
                        if let piece = detokenizer.next(), !piece.isEmpty {
                            if case .terminated = continuation.yield(piece) { break }
                        }
                    }
                    // The generation task can run briefly past an early break (KVCache in use).
                    await genTask.value
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    /// Resolve + run one tool call: unknown → error; over budget → error; approval-gated →
    /// ask; execute with error capture. Pure (no model), so it is unit-testable directly.
    static func dispatch(
        _ call: ToolCall,
        registry: ToolRegistry,
        budget: ToolBudget,
        approve: @Sendable (ToolCall, any AgentTool) async -> Bool,
        onEvent: @Sendable (AgentToolEvent) -> Void
    ) async -> ToolDispatchResult {
        let name = call.function.name
        let arguments = call.function.arguments

        guard let tool = registry.tool(named: name) else {
            onEvent(.unknownTool(name: name))
            return .unknownTool(name)
        }
        guard await budget.tryConsume() else {
            onEvent(.budgetExceeded(name: name))
            return .budgetExceeded
        }
        if tool.requiresApproval, await approve(call, tool) == false {
            onEvent(.denied(name: name))
            return .denied
        }

        onEvent(.started(name: name, arguments: arguments))
        do {
            let output: String
            if let timeout = tool.executionTimeout {
                guard let result = try await withTimeout(
                    seconds: timeout, operation: { try await tool.execute(arguments: arguments) })
                else {
                    let message = "timed out after \(Int(timeout))s"
                    onEvent(.failed(name: name, message: message))
                    return .failed(message)
                }
                output = result
            } else {
                output = try await tool.execute(arguments: arguments)
            }
            onEvent(.finished(name: name, result: output))
            return .ok(output)
        } catch {
            let message = error.localizedDescription
            onEvent(.failed(name: name, message: message))
            return .failed(message)
        }
    }

    /// Runs `operation`, returning its result, or `nil` if it exceeds `seconds` (AG6 safety rail).
    /// This bounds the AGENT LOOP: on timeout the caller reports a `.failed` and the model
    /// continues. It does NOT interrupt a cancellation-deaf synchronous body — a runaway
    /// `run_javascript` loop keeps spinning on its background task until the process exits, since
    /// JavaScriptCore has no public interrupt (see journal 2026-56). Pure, so it is unit-testable.
    static func withTimeout<T: Sendable>(
        seconds: Double,
        operation: @escaping @Sendable () async throws -> T
    ) async throws -> T? {
        try await withThrowingTaskGroup(of: T?.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(max(0, seconds) * 1_000_000_000))
                return nil
            }
            let first = try await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }

    /// Tool-call parser format for a given `modelType` string from the checkpoint config.
    ///
    /// **S2-1: this no longer keeps a table of its own.** mlx-swift-lm's `LLMModelFactory` already
    /// runs `ToolCallFormat.infer(from:configData:)` at load (`LLMModelFactory.swift:543`), so the
    /// app's own row — `qwen3_5` → `.xmlFunction` — only duplicated upstream (readme F3b; keeping
    /// the duplicate is what the S2-1 contract forbids). The answer now comes from
    /// `ToolCallFormatTable.resolved`, which defers to upstream and applies the app's overrides on
    /// top — none for any catalog model, see that table.
    ///
    /// Kept, unchanged in meaning, because AG0's test pins it and because it is the convenience for
    /// a caller with no `config.json` in hand. Without that data the `llama` family cannot be told
    /// apart (Llama 3 needs `vocab_size` / `rope_scaling`), and upstream's `infer` returns `nil`
    /// for it in exactly the same way.
    ///
    /// Known gap (unchanged): Qwen3-Coder and Qwen3-30B-A3B both report `qwen3_moe`, so the Coder
    /// variant can't be told apart from `modelType` alone — it would need the model id. Left as-is
    /// rather than guessed at; Coder is the rarer case and a wrong `.xmlFunction` is worse than a
    /// wrong `.json` (the JSON parser at least fails loudly on malformed input).
    static func toolCallFormat(forModelType modelType: String) -> ToolCallFormat? {
        ToolCallFormatTable.resolved(forModelType: modelType, configData: nil)
    }
}

/// The one thing `ModelRunner.send` needs from a streaming session.
///
/// MLX-ERR-3's test seam (owner ruling R-ME3-2, 2026-10-12: *"Use the one-method protocol for the
/// test seam, and assert that the stream stopped early."*). It exists so a test can hand `send` a
/// stream that reports an MLX error part-way through and assert the turn **stopped there** — which
/// is the observable difference R-ME3-1 creates, and the thing a closure-shaped seam could not
/// express. `AgentSession` is the only production conformer.
///
/// Deliberately not `Sendable`: `AgentSession` holds a `ModelContainer` and is not, and the session
/// is built and consumed inside one `ModelRunner` task, so nothing here crosses an isolation
/// boundary.
protocol ChatStreamingSession {
    func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error>
}

extension AgentSession: ChatStreamingSession {}
