//
//  ChatTranscript.swift
//  MLXUI — agentic chat tools, slice AG1.
//
//  The renderable chat transcript: assistant/user text bubbles interleaved with tool-call
//  cards. `TranscriptBuilder` is a pure, `@MainActor`-free reducer that folds streamed text
//  chunks and `AgentToolEvent`s into an ordered `[TranscriptItem]` — so the interleaving
//  logic is unit-testable without a model or a view.
//
//  A tool event closes the currently-open assistant bubble, so the next text chunk starts a
//  fresh bubble below the card. This yields the natural order:
//  user → assistant… → [tool card] → assistant… .
//

import Foundation
import MLXLMCommon

/// One tool invocation shown inline in the chat transcript.
struct ToolActivity: Identifiable, Sendable, Equatable {
    /// Terminal + in-flight states a tool card can display.
    enum Status: Sendable, Equatable {
        case running
        case finished
        case failed(String)
        case denied
        case unknownTool
        case budgetExceeded
    }

    let id: UUID
    let name: String
    var arguments: [String: JSONValue]
    var status: Status
    /// Result string fed back to the model (shown in the card body when finished).
    var result: String?

    init(
        id: UUID = UUID(),
        name: String,
        arguments: [String: JSONValue],
        status: Status = .running,
        result: String? = nil
    ) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.status = status
        self.result = result
    }
}

/// An ordered element of the chat transcript: a text message or a tool-call card.
enum TranscriptItem: Identifiable, Sendable {
    case message(ChatMessage)
    case tool(ToolActivity)

    var id: UUID {
        switch self {
        case .message(let m): return m.id
        case .tool(let t): return t.id
        }
    }
}

/// Pure reducer that assembles the transcript from streamed text + tool lifecycle events.
/// No SwiftUI, no actor isolation — drive it directly from tests.
struct TranscriptBuilder {
    private(set) var items: [TranscriptItem] = []
    /// Index of the assistant bubble currently accumulating text, if any.
    private var openAssistantIndex: Int?

    var isEmpty: Bool { items.isEmpty }

    /// The last assistant text (used by the view to decide whether to show a "thinking" spinner).
    var lastAssistantText: String? {
        if let i = openAssistantIndex, case .message(let m) = items[i] { return m.text }
        return nil
    }

    mutating func reset() {
        items.removeAll()
        openAssistantIndex = nil
    }

    /// Append a user prompt; closes any open assistant bubble.
    mutating func addUserMessage(_ text: String) {
        items.append(.message(ChatMessage(role: .user, text: text)))
        openAssistantIndex = nil
    }

    /// Append a streamed assistant text chunk, extending the open bubble or starting a new one.
    mutating func appendAssistantChunk(_ chunk: String) {
        guard !chunk.isEmpty else { return }
        if let i = openAssistantIndex, case .message(var m) = items[i], m.role == .assistant {
            m.text += chunk
            items[i] = .message(m)
        } else {
            items.append(.message(ChatMessage(role: .assistant, text: chunk)))
            openAssistantIndex = items.count - 1
        }
    }

    /// Fold a tool lifecycle event into the transcript. Any event closes the open assistant
    /// bubble so subsequent text renders below the card.
    mutating func apply(_ event: AgentToolEvent) {
        switch event {
        case .started(let name, let arguments):
            items.append(.tool(ToolActivity(name: name, arguments: arguments, status: .running)))
        case .finished(let name, let result):
            updateLastRunningTool(named: name) { $0.status = .finished; $0.result = result }
        case .failed(let name, let message):
            updateLastRunningTool(named: name) { $0.status = .failed(message); $0.result = message }
        case .denied(let name):
            appendTerminalTool(name: name, status: .denied)
        case .unknownTool(let name):
            appendTerminalTool(name: name, status: .unknownTool)
        case .budgetExceeded(let name):
            appendTerminalTool(name: name, status: .budgetExceeded)
        }
        openAssistantIndex = nil
    }

    // MARK: - helpers

    /// `.finished` / `.failed` always follow a `.started`, so update that running card.
    private mutating func updateLastRunningTool(
        named name: String, _ mutate: (inout ToolActivity) -> Void
    ) {
        for i in items.indices.reversed() {
            if case .tool(var t) = items[i], t.name == name, t.status == .running {
                mutate(&t)
                items[i] = .tool(t)
                return
            }
        }
    }

    /// `.denied` / `.unknownTool` / `.budgetExceeded` fire without a `.started`, so add a card.
    private mutating func appendTerminalTool(name: String, status: ToolActivity.Status) {
        items.append(.tool(ToolActivity(name: name, arguments: [:], status: status)))
    }
}
