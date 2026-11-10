//
//  MLXChatRouter.swift
//  MLXUI — Model Tools in Chat, MT3-1.
//
//  Today's `AgentSession` path behind the `ChatRouter` protocol. **No behaviour change**: the
//  session is built from exactly the inputs `ModelRunner.streamingSession` used to pass it (raw-token
//  path when no tools are armed — TCP-1; `ToolCallFormatTable` through the container; the approval
//  gate; the per-turn `ToolBudget`; the `FileScope` / `ModelToolRuntime` task-locals), and its text
//  chunks are forwarded as `.text`. A router is built **per turn**, because the leased container,
//  the file scope and the budget are per turn.
//

import Foundation
import MLXLMCommon

nonisolated struct MLXChatRouter: ChatRouter {
    /// Builds the session for a turn. `onEvent` is the router's wrapped lifecycle callback (it feeds
    /// `RouterTurn.onToolEvent` and the stream). Called synchronously inside `streamTurn`.
    /// Production builds an `AgentSession`; a test supplies a fake.
    typealias SessionFactory = @Sendable (
        _ turn: RouterTurn, _ onEvent: @escaping @Sendable (AgentToolEvent) -> Void
    ) -> any ChatStreamingSession

    let identity: RouterIdentity
    private let makeSession: SessionFactory

    /// Test seam / custom construction.
    init(identity: RouterIdentity, makeSession: @escaping SessionFactory) {
        self.identity = identity
        self.makeSession = makeSession
    }

    /// The production router: an `AgentSession` over a loaded (leased) container.
    init(
        identity: RouterIdentity,
        container: ModelContainer,
        fileScope: FileScope?,
        modelToolRuntime: ModelToolRuntime?,
        modelToolHost: ModelToolHost? = nil,
        parameters: GenerateParameters = GenerateParameters(maxTokens: 2048, temperature: 0.7),
        maxToolCalls: Int
    ) {
        self.identity = identity
        self.makeSession = { turn, onEvent in
            AgentSession(
                model: container,
                instructions: turn.instructions,
                history: turn.history.foldedMessages.map(Self.chatMessage),
                registry: ToolRegistry(turn.tools),
                fileScope: fileScope,
                modelToolRuntime: modelToolRuntime,
                modelToolHost: modelToolHost,
                // 512 was too tight once reasoning models entered the picture: Qwen3's <think> block alone can run
                // 250-500 tokens, so the generation was truncated *before* the tool call was emitted.
                parameters: parameters,
                maxToolCalls: maxToolCalls,
                approve: { call, tool in await turn.approve(tool, call.function.arguments) },
                onEvent: onEvent)
        }
    }

    func streamTurn(_ turn: RouterTurn) -> AsyncThrowingStream<RouterEvent, Error> {
        let (stream, continuation) = AsyncThrowingStream<RouterEvent, Error>.makeStream()
        let onToolEvent = turn.onToolEvent
        let makeSession = makeSession
        // The task is created here, inside the caller's guard closure, so it inherits mlx-swift's task-local error
        // handler (journal 2026-469), and the session it builds (a `ChatStreamingSession`, main-actor by the project
        // default) is built and consumed on one actor, as `ModelRunner` did before the router existed.
        let task = Task { @MainActor in
            let session = makeSession(turn) { event in
                // Same order as before for the transcript/audit log, then the neutral summary into the stream.
                onToolEvent(event)
                continuation.yield(event.routerEvent)
            }
            do {
                for try await chunk in session.streamResponse(to: turn.prompt) {
                    if case .terminated = continuation.yield(.text(chunk)) { break }
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    nonisolated static func chatMessage(_ message: RouterMessage) -> Chat.Message {
        let role: Chat.Message.Role
        switch message.role {
        case .system: role = .system
        case .user: role = .user
        case .assistant: role = .assistant
        case .tool: role = .tool
        }
        return Chat.Message(role: role, content: message.content)
    }

    nonisolated static func routerMessage(_ message: Chat.Message) -> RouterMessage {
        let role: RouterMessage.Role
        switch message.role {
        case .system: role = .system
        case .user: role = .user
        case .assistant: role = .assistant
        case .tool: role = .tool
        }
        return RouterMessage(role: role, content: message.content)
    }
}
