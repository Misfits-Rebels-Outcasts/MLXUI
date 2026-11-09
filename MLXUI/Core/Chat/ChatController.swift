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
    /// Another surface is generating; nothing was rebound. The surface is now *waiting*.
    case busy(holder: ChatSurface)
    /// `ChatGate` is closed (the page only).
    case unavailable
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
    /// Surfaces that asked for the runner while another surface was generating; they show a notice and bind
    /// themselves when the turn ends (`bindWhenFree`).
    private(set) var waiting: Set<ChatSurface> = []
    /// Installed chat models as of the last `refreshPageModel` (a selected conversation picks its model from here).
    private var pageCandidates: [ModelEntry] = []

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
        if surface == .page, !ChatGate.isAvailable { return .unavailable }
        if let model { models[surface] = model }
        if let current = holder, current != surface {
            if runner.isRunning {
                waiting.insert(surface)
                return .busy(holder: current)
            }
            parked[current] = runner.release()
        }
        waiting.remove(surface)
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
        waiting.remove(surface)
        if surface == .bench { drafts[.bench] = nil }
        if holder == surface { holder = nil }
    }

    func model(for surface: ChatSurface) -> ModelEntry? { models[surface] }

    /// Whether `surface` currently drives the runner: only that surface presents the approval and load
    /// questions and registers shortcuts.
    func holds(_ surface: ChatSurface) -> Bool { holder == surface }

    /// Called by a view when the runner may have become free: a waiting surface takes it. Returns whether it did.
    @discardableResult
    func bindWhenFree(_ surface: ChatSurface) -> Bool {
        guard waiting.contains(surface), !runner.isRunning else { return false }
        return acquire(surface) == .granted
    }

    /// The user's explicit "stop it" from a waiting surface; `stop(from:)` refuses a non-owner on purpose.
    func stopCurrentGeneration() { runner.stop() }

    /// After the bench goes away the page, if it is on screen, takes the runner back.
    var pageShouldReacquire: Bool { holder == nil }

    // MARK: - The Chat page's model (MT2-2; the model menu is MT2-4)

    /// Choose the page's model from what is installed: keep the current one, else the newest conversation's,
    /// else the first installed. `nil` leaves the page on its empty state.
    func refreshPageModel(installedChatModels: [ModelEntry]) {
        pageCandidates = installedChatModels
        let newest = ConversationStore.allConversations(in: conversationDirectory).map(\.modelID)
        models[.page] = ChatPageModel.resolve(current: models[.page], newestConversationModelIDs: newest,
                                              installed: installedChatModels)
    }

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
        // The page lists every model's chats, so the runner has no model of its own to start one with.
        runner.newConversation(for: surface == .page ? models[.page]?.id : nil)
    }

    func select(_ id: UUID, on surface: ChatSurface) {
        guard holder == surface else { return }
        runner.selectConversation(id)
        // On the page, a selected chat continues with its own model when that model is installed.
        if surface == .page, let chosen = runner.conversations.first(where: { $0.id == id }),
           let match = pageCandidates.first(where: { $0.id == chosen.modelID }) {
            models[.page] = match
        }
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
