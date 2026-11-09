//
//  ChatController.swift
//  MLXUI — Model Tools, MT2-1 (`RSI/journal/2026-700-mt2-0-chat-state-spike.md`).
//
//  The one API chat views use. It owns the *surface lease* (which view may drive the runner), each
//  surface's model and composer draft, and the in-memory working sets parked while another surface
//  holds the runner. `ModelRunner` stays the single generation engine: one `isRunning`, one approval
//  slot, one audit log, `send` unchanged — and the `[Conversation]` array stays physically in it.
//
//  MT2-1 wires only the bench surface (`RunChatView`'s sheet), so the app behaves exactly as before.
//  The page surface and its lease rules exist so they can be tested before MT2-2 adds the page.
//  `@Observable`, no Combine.
//

import Foundation

/// Who is driving the chat runner.
enum ChatSurface: Hashable, Sendable {
    /// The per-model sheet (`RunChatView`).
    case bench
    /// The Chat sidebar page (MT2-2; no view yet).
    case page
}

/// Which conversations a surface lists.
enum ChatScope: Equatable, Sendable {
    case all
    case model(String)
}

/// A surface's conversations, parked in memory while another surface holds the runner. Swapped, never
/// reloaded: a blank conversation with attached folders is memory-only, so a disk reload would drop it.
struct ChatWorkingSet {
    var scope: ChatScope
    var conversations: [Conversation]
    var activeConversationID: UUID?
    var transcriptItems: [TranscriptItem]
}

/// What a composer holds for the next message.
struct ChatDraft: Equatable, Sendable {
    var text = ""
    var files: [URL] = []
}

enum ChatLease: Equatable, Sendable {
    case granted
    /// Another surface is generating; nothing was rebound.
    case busy(holder: ChatSurface)
}

@MainActor @Observable
final class ChatController {
    let runner: ModelRunner
    @ObservationIgnored private let conversationDirectory: URL

    private(set) var holder: ChatSurface?
    private var models: [ChatSurface: ModelEntry] = [:]
    private var drafts: [ChatSurface: ChatDraft] = [:]
    private var parked: [ChatSurface: ChatWorkingSet] = [:]
    private var startedBy: ChatSurface?

    init(runner: ModelRunner, conversationDirectory: URL = ConversationStore.defaultDirectory()) {
        self.runner = runner
        self.conversationDirectory = conversationDirectory
    }

    // MARK: - Lease

    /// The surface whose turn is in flight, or nil when nothing is generating.
    var generationOwner: ChatSurface? { runner.isRunning ? startedBy : nil }

    /// Bind `surface` (with `model`, for the bench) to the runner. Refused, with nothing rebound, while
    /// another surface is generating. Re-acquiring a held surface is the old `prepare(for:)` call.
    @discardableResult
    func acquire(_ surface: ChatSurface, model: ModelEntry? = nil) -> ChatLease {
        if let model { models[surface] = model }
        if let current = holder, current != surface {
            if runner.isRunning { return .busy(holder: current) }
            parked[current] = runner.release()
        }
        holder = surface
        switch surface {
        case .bench:
            if let set = parked.removeValue(forKey: .bench), let model = models[.bench], set.scope == .model(model.id) {
                runner.adopt(set)
            } else if let model = models[.bench] {
                runner.prepare(for: model, in: conversationDirectory)
            }
        case .page:
            if let set = parked.removeValue(forKey: .page) { runner.adopt(set) }
            else { runner.prepare(scope: .all, in: conversationDirectory) }
        }
        return .granted
    }

    /// The surface is going away. The bench's draft does not outlive its sheet (as before).
    func release(_ surface: ChatSurface) {
        if surface == .bench { drafts[.bench] = nil }
        if holder == surface { holder = nil }
    }

    func model(for surface: ChatSurface) -> ModelEntry? { models[surface] }

    // MARK: - Drafts

    func draft(for surface: ChatSurface) -> ChatDraft { drafts[surface] ?? ChatDraft() }

    func setDraft(_ draft: ChatDraft, for surface: ChatSurface) {
        drafts[surface] = draft == ChatDraft() ? nil : draft
    }

    // MARK: - Conversations (forwarded to the runner, which keeps the array)

    var conversations: [Conversation] { runner.conversations }
    var activeConversationID: UUID? { runner.activeConversationID }

    func newConversation(on surface: ChatSurface) {
        guard holder == surface else { return }
        runner.newConversation()
    }

    func select(_ id: UUID, on surface: ChatSurface) {
        guard holder == surface else { return }
        runner.selectConversation(id)
    }

    func delete(_ id: UUID, on surface: ChatSurface) {
        guard holder == surface else { return }
        runner.deleteConversation(id)
    }

    // MARK: - Generation

    /// `false` only when the user declined the served-load question (see `ModelRunner.send`).
    @discardableResult
    func send(_ text: String, attachments: [URL] = [], from surface: ChatSurface) async -> Bool {
        guard holder == surface, let model = models[surface] else { return true }
        if !runner.isRunning { startedBy = surface }
        return await runner.send(text, attachments: attachments, for: model)
    }

    /// Stops the runner, unless a different surface owns the turn in flight.
    func stop(from surface: ChatSurface) {
        if let owner = generationOwner, owner != surface { return }
        runner.stop()
    }
}
