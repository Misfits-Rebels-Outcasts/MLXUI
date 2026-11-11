//
//  ChatRouter.swift
//  MLXUI — Model Tools in Chat, MT3-1 (plan §3.3).
//
//  What "the thing that answers a chat turn" looks like to `ModelRunner`, so the MLX path
//  (`MLXChatRouter`, today's `AgentSession`) and a later Apple Intelligence router (MT3-2, behind
//  the `AppleFoundationExecutor` seam) are interchangeable. This file knows neither framework: no
//  `FoundationModels`, no `MLXLM*` session types. (`AgentTool` and `JSONValue` are the app's own tool
//  vocabulary, which both routers adapt.)
//
//  Events:
//   - `.text` carries a **delta** (a router whose backend streams snapshots diffs them first).
//   - `.toolStarted` / `.toolFinished` are raised by the router's tool adapter around the call
//     (MT3-0: the framework transcript only shows a tool call after it has finished, so it cannot
//     drive a live chip).
//   - `.usage` / `.contextExceeded` are for routers that can measure the window; the MLX router
//     emits neither (its budget lives in `ContextWindow`).
//  `RouterTurn.onToolEvent` is the full-fidelity lifecycle side channel (`AgentToolEvent`) that the
//  transcript and the audit log already consume; the stream's tool events are the router-neutral
//  summary of the same calls.
//

import Foundation
import MLXLMCommon

/// Who a router is, for the model menu and banners.
nonisolated struct RouterIdentity: Sendable, Equatable {
    enum Kind: String, Sendable, Equatable {
        /// An OS-provided model (Apple Intelligence).
        case system
        /// A local MLX chat model.
        case mlx
        /// A remote provider (later; the `offdevice` banner applies).
        case provider
    }

    let displayName: String
    let kind: Kind
    /// The usable window in tokens, when known.
    let contextSize: Int?
}

/// One history message, router-neutral.
nonisolated struct RouterMessage: Sendable, Equatable {
    enum Role: String, Sendable, Equatable { case system, user, assistant, tool }
    let role: Role
    let content: String
}

/// How earlier turns reach the router.
nonisolated enum RouterHistory: Sendable, Equatable {
    /// The earlier turns, folded into the new session / prompt. Always valid.
    case folded([RouterMessage])
    /// The router may reuse a live session it already holds for this conversation; `fallback` is what
    /// to fold if it holds none (relaunch, eviction, switching back).
    case live(conversationID: UUID, fallback: [RouterMessage])

    /// The messages to fold when no live session exists.
    var foldedMessages: [RouterMessage] {
        switch self {
        case .folded(let messages): return messages
        case .live(_, let fallback): return fallback
        }
    }
}

/// MT4-2: the per-turn context a tool needs when it runs: the chat's folder scope, the app's model catalog/install state and
/// the conversation's assets. A router whose tools run outside an `AgentSession` (Apple Intelligence) binds it around each
/// call with `run`, exactly as `AgentSession.streamResponse` does for the MLX router. All `nil` = no context (flows, tests).
nonisolated struct RouterToolContext: Sendable {
    var fileScope: FileScope?
    var modelToolRuntime: ModelToolRuntime?
    var modelToolHost: ModelToolHost?

    init(fileScope: FileScope? = nil, modelToolRuntime: ModelToolRuntime? = nil, modelToolHost: ModelToolHost? = nil) {
        self.fileScope = fileScope
        self.modelToolRuntime = modelToolRuntime
        self.modelToolHost = modelToolHost
    }

    /// Run `body` with the context bound as task-locals.
    func run<T: Sendable>(_ body: @Sendable () async throws -> T) async rethrows -> T {
        try await FileScope.$current.withValue(fileScope) {
            try await ModelToolRuntime.$current.withValue(modelToolRuntime) {
                try await ModelToolHost.$current.withValue(modelToolHost) {
                    try await body()
                }
            }
        }
    }
}

/// Everything one turn needs. `Sendable`: it crosses into the router's task.
nonisolated struct RouterTurn: Sendable {
    /// The system / environment text, if any.
    var instructions: String?
    var history: RouterHistory
    /// The user's turn, attachments already folded in.
    var prompt: String
    /// Tools armed for this turn (may be empty: a plain chat turn).
    var tools: [any AgentTool]
    /// Asked for a `requiresApproval` tool before it runs; false declines.
    var approve: @Sendable (_ tool: any AgentTool, _ arguments: [String: JSONValue]) async -> Bool
    /// Full-fidelity tool lifecycle (transcript cards, audit log).
    var onToolEvent: @Sendable (AgentToolEvent) -> Void
    /// MT4-2: bound around each tool call by a router that runs tools itself (Apple Intelligence). The MLX router binds the
    /// same three values through its `AgentSession` and ignores this.
    var toolContext: RouterToolContext?

    init(
        instructions: String? = nil,
        history: RouterHistory = .folded([]),
        prompt: String,
        tools: [any AgentTool] = [],
        approve: @escaping @Sendable (any AgentTool, [String: JSONValue]) async -> Bool = { _, _ in true },
        onToolEvent: @escaping @Sendable (AgentToolEvent) -> Void = { _ in },
        toolContext: RouterToolContext? = nil
    ) {
        self.toolContext = toolContext
        self.instructions = instructions
        self.history = history
        self.prompt = prompt
        self.tools = tools
        self.approve = approve
        self.onToolEvent = onToolEvent
    }
}

/// How a tool call ended, router-neutral.
nonisolated enum RouterToolOutcome: Sendable, Equatable {
    case ok(String)
    case failed(String)
    case denied
    case unknownTool
    case budgetExceeded
}

nonisolated enum RouterEvent: Sendable, Equatable {
    /// A delta of the reply text.
    case text(String)
    case toolStarted(name: String, arguments: [String: JSONValue])
    case toolFinished(name: String, outcome: RouterToolOutcome)
    /// Token counts for the turn, when the router can measure them.
    case usage(promptTokens: Int, completionTokens: Int)
    /// The conversation no longer fits; `limit` is the window when known.
    case contextExceeded(limit: Int?)
}

nonisolated protocol ChatRouter: Sendable {
    var identity: RouterIdentity { get }
    /// Stream one turn. The returned stream must be created **synchronously** from this call (the
    /// caller wraps the call in `MLXErrorGuard.guardedStream`'s `makeStream` closure so the producer
    /// task inherits mlx-swift's error handler, journal 2026-469). Cancelling the consumer cancels the turn.
    func streamTurn(_ turn: RouterTurn) -> AsyncThrowingStream<RouterEvent, Error>
}

extension AgentToolEvent {
    /// The router-neutral event for a lifecycle event.
    nonisolated var routerEvent: RouterEvent {
        switch self {
        case .started(let name, let arguments): return .toolStarted(name: name, arguments: arguments)
        case .finished(let name, let result): return .toolFinished(name: name, outcome: .ok(result))
        case .failed(let name, let message): return .toolFinished(name: name, outcome: .failed(message))
        case .denied(let name): return .toolFinished(name: name, outcome: .denied)
        case .unknownTool(let name): return .toolFinished(name: name, outcome: .unknownTool)
        case .budgetExceeded(let name): return .toolFinished(name: name, outcome: .budgetExceeded)
        }
    }
}
