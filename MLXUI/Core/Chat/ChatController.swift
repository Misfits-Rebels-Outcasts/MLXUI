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
    /// How the every-model list is read from disk (`ConversationStore.allConversations`); a seam so a test can count
    /// reads. Called only by `loadedIndex()`.
    @ObservationIgnored private let readAll: (URL) -> [Conversation]
    /// The every-model list, loaded once (owner ruling 2026-11-09: no file read on a sidebar click, page appear or
    /// install change) and dropped when something may have changed on disk behind the page's back.
    @ObservationIgnored private var index: [Conversation]?

    private(set) var holder: ChatSurface?
    private var models: [ChatSurface: ModelEntry] = [:]
    private var drafts: [ChatSurface: ChatDraft] = [:]
    private var parked: [ChatSurface: ChatWorkingSet] = [:]
    private var startedBy: ChatSurface?
    /// MT4-5: files an "Ask in Chat" asked the page to start a new conversation with. Applied by `acquire(.page)` (the one
    /// place the page takes the runner), so it lands whether the page is already bound, waiting, or not yet on screen.
    @ObservationIgnored private var pendingAskInChat: [URL] = []
    /// Surfaces that asked for the runner while another surface was generating; they show a notice and bind
    /// themselves when the turn ends (`bindWhenFree`).
    private(set) var waiting: Set<ChatSurface> = []
    /// What the model menu says about each installed model, loaded once per model (file reads) and kept.
    private(set) var modelFacts: [String: ChatModelFacts] = [:]
    @ObservationIgnored private let factsLoader: (ModelEntry) -> ChatModelFacts
    /// Installed chat models as of the last `refreshPageModel` (a selected conversation picks its model from here).
    private var pageCandidates: [ModelEntry] = []
    /// MT3-3: Apple Intelligence's readiness as of the last refresh (`nil` = below macOS 26 / not opted in: the menu row is absent).
    private(set) var appleReadiness: Readiness?
    @ObservationIgnored private let readinessProvider: () -> Readiness?
    /// MT3-3: when nothing is installed and Apple Intelligence is not ready, the catalog model the empty page recommends.
    private(set) var pageRecommendation: ModelEntry?

    init(runner: ModelRunner, conversationDirectory: URL = ConversationStore.defaultDirectory(),
         readAllConversations: @escaping (URL) -> [Conversation] = { ConversationStore.allConversations(in: $0) },
         factsLoader: @escaping (ModelEntry) -> ChatModelFacts = { ChatModelFacts.load(for: $0) },
         readinessProvider: @escaping () -> Readiness? = { AppleFoundationAvailability.currentReadiness() }) {
        self.factsLoader = factsLoader
        self.readinessProvider = readinessProvider
        self.runner = runner
        self.conversationDirectory = conversationDirectory
        self.readAll = readAllConversations
    }

    /// The cached every-model list; the first call (and the first after `invalidateIndex`) reads the directory.
    private func loadedIndex() -> [Conversation] {
        if let index { return index }
        let loaded = readAll(conversationDirectory)
        index = loaded
        return loaded
    }

    /// Something may have written or removed a conversation file without the page seeing it.
    private func invalidateIndex() { index = nil }

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
        let previous = holder
        holder = surface
        switch surface {
        case .bench:
            if let set = parked.removeValue(forKey: .bench), let model = models[.bench], set.scope == .model(model.id) {
                runner.adopt(set)
            } else if let model = models[.bench] {
                runner.prepare(for: model, in: conversationDirectory)
            }
        case .page:
            // Already bound and nothing happened elsewhere: the runner's own state is the truth, so no read.
            if previous == .page, parked[.page] == nil { applyPendingAskInChat(); break }
            if var set = parked.removeValue(forKey: .page) {
                // The sheet may have saved or deleted chats while this set was parked: one read, merged under the
                // live set (live entries win; a blank, memory-only chat is kept).
                if index == nil { set.conversations = ChatConversationList.merge(disk: loadedIndex(), live: set.conversations) }
                runner.adopt(set)
            } else {
                runner.prepare(scope: .all, in: conversationDirectory, preloaded: loadedIndex())
            }
            applyPendingAskInChat()
        }
        return .granted
    }

    // MARK: - Ask in Chat (MT4-5)

    /// A Run UI's "Ask in Chat": the page starts a **new** conversation with `files` waiting in its composer. They become
    /// asset `a1` (…) through the ordinary send path when the user sends (Model Tools are on in the page), so nothing is
    /// registered, loaded or sent until then. Returns `false`, changing nothing, when the gate is closed or a turn is
    /// running (a new conversation would stop it). When another surface still holds the runner the request waits and
    /// is applied when the page binds.
    @discardableResult
    func askInChat(files: [URL]) -> Bool {
        guard ChatGate.isAvailable, !runner.isRunning, !files.isEmpty else { return false }
        pendingAskInChat = files
        if acquire(.page) == .unavailable { pendingAskInChat = []; return false }
        return true
    }

    /// Whether `askInChat` would be accepted right now (the button's enabled state).
    var canAskInChat: Bool { ChatGate.isAvailable && !runner.isRunning }

    private func applyPendingAskInChat() {
        guard !pendingAskInChat.isEmpty else { return }
        let files = pendingAskInChat
        pendingAskInChat = []
        runner.newConversation(for: models[.page]?.id)
        // Whatever the user had half-typed or attached stays; the result joins it.
        var draft = draft(for: .page)
        for file in files where !draft.files.contains(file) { draft.files.append(file) }
        setDraft(draft, for: .page)
    }

    /// The surface is going away. The bench's draft does not outlive its sheet (as before).
    func release(_ surface: ChatSurface) {
        waiting.remove(surface)
        // The sheet may have saved, renamed or deleted chats on disk.
        if surface == .bench { drafts[.bench] = nil; invalidateIndex() }
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

    // MARK: - Open in Chat (MT2-5)

    /// Whether the sheet may hand its conversation to the page: the gate is open, the sheet holds the runner and is not
    /// itself waiting for it.
    var canOpenBenchInChat: Bool {
        ChatGate.isAvailable && holder == .bench && !waiting.contains(.bench)
    }

    /// Pass the runner from the sheet to the page **in place**: the same conversation, transcript, attached folders and
    /// any running turn carry over (nothing is reloaded or stopped), the rest of the chats join the list, the page's
    /// model becomes the conversation's when installed, and the sheet's draft moves to the page. A turn the sheet
    /// started now belongs to the page, so the sheet going away (`stop(from: .bench)`) leaves it running.
    /// The caller dismisses the sheet and selects the Chat item.
    @discardableResult
    func openBenchInChat(installedChatModels: [ModelEntry]) -> Bool {
        guard canOpenBenchInChat else { return false }
        pageCandidates = installedChatModels
        var others = loadedIndex()
        if let page = parked.removeValue(forKey: .page) {
            // The page was on screen under the sheet: keep its chats (including a blank one) in the list.
            others = ChatConversationList.merge(disk: others, live: page.conversations)
        }
        runner.widenToAll(with: others)
        holder = .page
        waiting.remove(.page)
        if runner.isRunning { startedBy = .page }
        let modelID = runner.conversations.first { $0.id == runner.activeConversationID }?.modelID
        if let match = installedChatModels.first(where: { $0.id == modelID }) { models[.page] = match }
        else if models[.page] == nil { models[.page] = models[.bench] }
        let carried = draft(for: .bench)
        if carried != ChatDraft() { setDraft(carried, for: .page) }
        drafts[.bench] = nil
        return true
    }

    // MARK: - The Chat page's model and its menu (MT2-4)

    /// Loads the facts of any model the menu has not shown yet.
    func loadModelFacts(for models: [ModelEntry]) {
        for model in models where modelFacts[model.id] == nil { modelFacts[model.id] = factsLoader(model) }
    }

    /// Switch the page's model from the composer menu. It applies from the **next** turn: `send` reads the model
    /// when a turn starts, and the earlier messages (and their stamps) are untouched. Refused while a turn is
    /// running (one generation app-wide, and the header must not name a model that is not the one answering).
    @discardableResult
    func selectPageModel(_ model: ModelEntry) -> Bool {
        guard !runner.isRunning, holder == .page || holder == nil else { return false }
        // MT3-3: Apple Intelligence can be chosen only while it is ready.
        if AppleIntelligenceChat.isApple(model.id) {
            appleReadiness = readinessProvider()
            guard appleReadiness == .ready else { return false }
        }
        models[.page] = model
        return true
    }

    /// The menu's Apple Intelligence row for the current readiness (re-read, so turning it on in Settings shows when the menu opens).
    func appleMenuEntry() -> AppleIntelligenceMenuEntry? {
        appleReadiness = readinessProvider()
        return AppleIntelligenceMenuEntry.make(readiness: appleReadiness)
    }

    /// Choose the page's model from what is installed: keep the current one, else the newest conversation's,
    /// else the first installed. `nil` leaves the page on its empty state.
    func refreshPageModel(installedChatModels: [ModelEntry], catalog: [ModelEntry] = [], totalRAMGB: Double = 0,
                          isRunnable: (ModelEntry) -> Bool = { _ in true }) {
        pageCandidates = installedChatModels
        appleReadiness = readinessProvider()
        // Keep the current choice while it still works: an installed model, or Apple Intelligence while ready.
        if let current = models[.page] {
            if AppleIntelligenceChat.isApple(current.id) {
                if appleReadiness == .ready { return }
            } else if let kept = installedChatModels.first(where: { $0.id == current.id }) {
                models[.page] = kept
                return
            }
        }
        // Live list when the page holds the runner, else the cached one: never a fresh directory read.
        let source = holder == .page ? runner.conversations : loadedIndex()
        let newest = source.map(\.modelID)
        // MT4-7-FIX-3: with no current choice (a relaunch), reopen on the model the chat last ran on, not the default router's pick.
        if models[.page] == nil {
            let own = holder == .page ? source.first(where: { $0.id == runner.activeConversationID }) : nil
            if let saved = (own ?? source.first)?.modelID {
                if AppleIntelligenceChat.isApple(saved) {
                    if appleReadiness == .ready {
                        models[.page] = AppleIntelligenceChat.entry
                        pageRecommendation = nil
                        return
                    }
                } else if let match = installedChatModels.first(where: { $0.id == saved }) {
                    models[.page] = match
                    pageRecommendation = nil
                    return
                }
            }
        }
        let toolCapable = Set(installedChatModels.filter { modelFacts[$0.id]?.canUseTools == true }.map(\.id))
        let decision = ChatDefaultRouter.decide(
            appleReadiness: appleReadiness, recentRouterIDs: newest, installed: installedChatModels,
            installedToolCapableIDs: toolCapable, catalog: catalog, totalRAMGB: totalRAMGB, isRunnable: isRunnable)
        pageRecommendation = nil
        switch decision {
        case .appleIntelligence: models[.page] = AppleIntelligenceChat.entry
        case .installedModel(let id): models[.page] = installedChatModels.first { $0.id == id }
        case .recommend(let id): models[.page] = nil; pageRecommendation = catalog.first { $0.id == id }
        case .nothing: models[.page] = nil
        }
    }

    // MARK: - Banners (MT3-3)

    /// The banner for the page's last Apple Intelligence failure or overflow, or nil.
    func banner(installedChatModels: [ModelEntry]) -> ChatBanner? {
        let toolCapable = Set(installedChatModels.filter { modelFacts[$0.id]?.canUseTools == true }.map(\.id))
        let windows = modelFacts.compactMapValues(\.usableTokens)
        if let notice = runner.contextNotice {
            return ChatBanner.contextExceeded(limit: notice.limit, installed: installedChatModels, installedToolCapableIDs: toolCapable,
                                              promptTokens: runner.unansweredPromptTokens, windows: windows)
        }
        if let failure = runner.appleFailure {
            return ChatBanner.failure(failure, installed: installedChatModels, installedToolCapableIDs: toolCapable,
                                      promptTokens: runner.unansweredPromptTokens, windows: windows)
        }
        return nil
    }

    /// "Continue with ‹model›": switch the page to that model and send the unanswered message again. A message too big for
    /// the model is refused first (plain sentence in `runner.errorMessage`), before the page model or the transcript changes.
    @discardableResult
    func continueAfterOverflow(with model: ModelEntry) async -> Bool {
        let usable = modelFacts[model.id]?.usableTokens
        if let refusal = runner.continueRefusal(with: model, usableTokens: usable) {
            runner.errorMessage = refusal
            return false
        }
        guard selectPageModel(model) else { return false }
        startedBy = .page
        runner.modelToolsEnabled = Self.modelToolsEnabled(for: .page)
        return await runner.retryUnanswered(with: model, usableTokens: usable)
    }

    /// "Start a new chat" from a banner.
    func startNewChatAfterBanner(on surface: ChatSurface) {
        runner.contextNotice = nil
        runner.appleFailure = nil
        runner.errorMessage = nil
        newConversation(on: surface)
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
        if surface == .page, let chosen = runner.conversations.first(where: { $0.id == id }) {
            if AppleIntelligenceChat.isApple(chosen.modelID) {
                // Back to a conversation Apple Intelligence answered: use it again while it is ready.
                if readinessProvider() == .ready { models[.page] = AppleIntelligenceChat.entry }
            } else if let match = pageCandidates.first(where: { $0.id == chosen.modelID }) {
                models[.page] = match
            }
        }
    }

    func delete(_ id: UUID, on surface: ChatSurface) {
        guard holder == surface else { return }
        runner.deleteConversation(id)
        invalidateIndex()
    }

    // MARK: - Generation

    /// `false` only when the user declined the served-load question (see `ModelRunner.send`).
    @discardableResult
    func send(_ text: String, attachments: [URL] = [], from surface: ChatSurface) async -> Bool {
        guard holder == surface, let model = models[surface] else { return true }
        if !runner.isRunning { startedBy = surface }
        // MT4-2 (ruling D): Model Tools are on in the Chat page and off in the bench sheet.
        runner.modelToolsEnabled = Self.modelToolsEnabled(for: surface)
        return await runner.send(text, attachments: attachments, for: model)
    }

    /// Ruling D: "use my models" is on in the Chat page, off in the bench sheet.
    static func modelToolsEnabled(for surface: ChatSurface) -> Bool { surface == .page }

    /// Stops the runner, unless a different surface owns the turn in flight.
    func stop(from surface: ChatSurface) {
        if let owner = generationOwner, owner != surface { return }
        runner.stop()
    }
}
