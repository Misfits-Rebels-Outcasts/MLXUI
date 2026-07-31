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
//  TOOL-CALL FORMAT (wired in AG1): the generate path selects the parser from
//  `ModelConfiguration.toolCallFormat ?? .json`. Qwen3.5 emits `<tool_call><function=…>`
//  which is `ToolCallFormat.xmlFunction` — so the loaded container's configuration must be
//  set to `.xmlFunction` for the family (see `Self.toolCallFormat(forModelType:)`), or tool
//  calls won't be parsed. AG0 defines the mapping; AG1 applies it at load + wires the UI.
//

import Foundation
import MLXLMCommon

nonisolated final class AgentSession {
    private let model: ModelContainer
    private let instructions: String?
    private let registry: ToolRegistry
    private let parameters: GenerateParameters
    private let maxToolCalls: Int
    private let approve: @Sendable (ToolCall, any AgentTool) async -> Bool
    private let onEvent: @Sendable (AgentToolEvent) -> Void

    /// - Parameters:
    ///   - model: a loaded `ModelContainer` (caller loads via the existing LLM path).
    ///   - registry: tools available this session (empty → plain chat, no tools advertised).
    ///   - maxToolCalls: cap on tool invocations per response (runaway-loop guard).
    ///   - approve: gate invoked for `requiresApproval` tools; return false to decline.
    ///   - onEvent: tool lifecycle callbacks for the chat UI / audit log.
    init(
        model: ModelContainer,
        instructions: String? = nil,
        registry: ToolRegistry = ToolRegistry(),
        parameters: GenerateParameters = .init(),
        maxToolCalls: Int = 8,
        approve: @escaping @Sendable (ToolCall, any AgentTool) async -> Bool = { _, _ in true },
        onEvent: @escaping @Sendable (AgentToolEvent) -> Void = { _ in }
    ) {
        self.model = model
        self.instructions = instructions
        self.registry = registry
        self.parameters = parameters
        self.maxToolCalls = maxToolCalls
        self.approve = approve
        self.onEvent = onEvent
    }

    /// Stream a response, running the tool loop as the model requests tools.
    func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
        let registry = registry
        let approve = approve
        let onEvent = onEvent
        let budget = ToolBudget(limit: maxToolCalls)

        let session = ChatSession(
            model,
            instructions: instructions,
            generateParameters: parameters,
            tools: registry.isEmpty ? nil : registry.toolSpecs,
            toolDispatch: { call in
                let result = await AgentSession.dispatch(
                    call, registry: registry, budget: budget, approve: approve, onEvent: onEvent)
                return result.modelResult
            }
        )
        return session.streamResponse(to: prompt)
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
    /// Returns `nil` when the default (`.json`) is appropriate. Applied at load in AG1.
    ///
    /// Only *some* of the Qwen family uses the XML form. Per MLXLMCommon's own
    /// `ToolCallFormat` documentation:
    ///
    ///   - `.json`        — Llama, **Qwen** (incl. Qwen2.5 and Qwen3):
    ///                      `<tool_call>{"name": "f", "arguments": {…}}</tool_call>`
    ///   - `.xmlFunction` — Nemotron, **Qwen3 Coder, Qwen3.5**:
    ///                      `<tool_call><function=f><parameter=k>v</parameter></function>`
    ///
    /// Matching all of `qwen*` to `.xmlFunction` made plain Qwen3 emit a JSON tool
    /// call that the XML parser silently discarded — the model "decided" to call a
    /// tool and the transcript showed nothing. Found while bringing up `run_shell`
    /// on Qwen3-8B; see Design/dual-distribution.md § 5.
    ///
    /// Known gap: Qwen3-Coder and Qwen3-30B-A3B both report `qwen3_moe`, so the
    /// Coder variant can't be told apart from `modelType` alone — it would need the
    /// model id. Left as-is rather than guessed at; Coder is the rarer case and a
    /// wrong `.xmlFunction` is worse than a wrong `.json` (the JSON parser at least
    /// fails loudly on malformed input).
    static func toolCallFormat(forModelType modelType: String) -> ToolCallFormat? {
        let t = modelType.lowercased()
        if t.hasPrefix("qwen3_5") || t.hasPrefix("qwen3.5") { return .xmlFunction }
        return nil
    }
}
